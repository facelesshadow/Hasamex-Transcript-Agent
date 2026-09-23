import operator
import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
CHROMA_DIR = "./chroma_db"

class RAGState(MessagesState):
    query: str
    retrieved_docs: Annotated[list[Document], operator.add]
    answer: str | None

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key = OPENAI_API_KEY
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name="expert_interviews",
    embedding_function=embeddings,
)

class RetrieverInput(BaseModel):
    query: str = Field(description="Semantic query to search research paper chunks")
    market: Literal["France", "United Kingdom", "Germany"] = Field(description="The single market to retrieve from.")

@tool(args_schema=RetrieverInput)
def retrieve_market_docs(
    query: str,
    market: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Retrieve the top 3 relevant document chunks for ONE market.
    """

    docs = vectorstore.similarity_search(
        query,
        k=3,
        filter={"market": market},
    )

    summary = f"Retrieved {len(docs)} chunk(s) from the {market} market."

    return Command(
        update={
            "retrieved_docs": docs,
            "messages": [
                ToolMessage(
                    content=summary,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )

model = ChatOpenAI(model='gpt-5.4-mini', openai_api_key=OPENAI_API_KEY, temperature=0)
retrieval_model = model.bind_tools([retrieve_market_docs])

class QueryRewrite(BaseModel):
    query: str = Field(
        description="A self-contained query that preserves all relevant context and nuances from the conversation."
    )

def setter_node(state: RAGState) -> dict:
    structured_llm = model.with_structured_output(QueryRewrite)
    result = structured_llm.invoke(
        [
            SystemMessage(content="""
            Rewrite the user's latest question into one standalone question.

            Use the chat history only to resolve missing references or context.
            Include ONLY what the user is asking.
            Do not answer, explain, summarize, or add information.
            Do not add assumptions or details that are not necessary.
            If the query is already self contained, then leave it as is. 
            Output only the rewritten question.
            """),
        ] + state['messages']
    )


    query = result.query
    return {'query': query, "retrieved_context": []}


RETRIEVAL_PROMPT = """
You are a research assistant capable of retrieving context. 
You have a retriever tool at your diposal which retrieves documents given a query and a market.
There are three markets - 'France', 'United Kingdom' and 'Germany'.
Retrieve important documents in regards to the user query.
You can only retrieve documents from one market in one call.
If multiple markets are present in the user query, you will need to make multiple calls.
If there is no market / country mentioned in the user's query, assume that the query is related to all the markets, in that case, retrieve relevant docs from all the markets. 
You DO NOT need to answer the user's query, just retrieve the relevant context.
Keep the retrieval query AS SIMPLE AS IT CAN GET. Try to only include the keywords provided by the user + based on the history.
When context is retrieved, you will only see a retrieval summary.
DO NOT PRODUCE the FINAL ANSWER, ONLY CALL TOOLS TO COLLECT CONTEXT."""

RETRIEVAL_SYSTEM_PROMPT = SystemMessage(content=RETRIEVAL_PROMPT)

def agent_node(state: RAGState) -> dict:
    messages = [RETRIEVAL_SYSTEM_PROMPT] + state['messages']
    response = retrieval_model.invoke(messages)
    updates: dict = {'messages': [response]}
    return updates


ANSWER_SYSTEM_PROMPT = SystemMessage(
    content="""
You are an answer generation assistant.

Answer the user's question using ONLY the retrieved research chunks.

MARKET -> 'France', 'Germany', 'United Kingdom'

If the user's query mentions a market, write only about that market. 
If two markets are mentioned, then write about both.
IF NO MARKET IS MENTIONED, WRITE ABOUT ALL THREE MARKETS.

For every factual claim, preserve its source and timestamp.

Citations MUST use this format:

[Source: <source> | Timestamp: <timestamp>]

When directly quoting a retrieved chunk, use quotation marks and include
the source and timestamp immediately after the quote.

Do not invent, modify, or omit source metadata.
Do not attribute information to a source unless that source actually
supports the claim.

If the retrieved information is insufficient, say so clearly.
Do not use outside knowledge. Do NOT ADD any knowledge outside from the context provided.
DO NOT USE IRRELEVANT CHUNKS / DOCS. 
Write a VERY SHORT and crisp answer. ONLY INCLUDE THE INFORMATION WHICH DIRECTLY ANSWERS THE QUESTION. 
KEEP THE ANSWER ONE - TWO POINTS / CITATIONS LONG FOR EACH COUNTRY / MARKET. 
"""
)


def answer_node(state: RAGState) -> dict:
    docs = state.get("retrieved_docs", [])

    context_parts = []

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown source")
        timestamp = doc.metadata.get("timestamp", "Unknown timestamp")
        market = doc.metadata.get("market", "Unknown market")

        context_parts.append(
            f"""
[Chunk {i + 1}]
Source: {source}
Timestamp: {timestamp}
Market: {market}

Content:
{doc.page_content}
"""
        )

    context = "\n".join(context_parts)

    prompt = HumanMessage(
        content=f"""
Question:
{state["query"]}

Retrieved research:

{context}

Answer the question using only the retrieved research.
Preserve source attribution and timestamps for every factual claim.
Select only relevant chunks that DIRECTLY answer the question. Ignore all irrelevant or merely related information. 
If there are no relevant chunks, simply return the message "No related data found in database"
"""
    )

    response = model.invoke([
        ANSWER_SYSTEM_PROMPT,
        prompt,
    ])

    return {
        "messages": [response],
        "answer": response.content
    }

def hallucination_checker(state):
    prompt = f"""
You are a hallucination removal step.

User query:
{state["query"]}

Retrieved context:
{state["retrieved_docs"]}

Generated answer:
{state["answer"]}

Rewrite the answer by removing any claim that is not directly supported
by the retrieved context.

Rules:
- Do not add new information.
- Do not use outside knowledge.
- Keep all claims that are supported.
- Preserve the original answer's structure and wording where possible.
- If a statement cannot be verified from the context, remove it.
- If the entire answer is unsupported, say that the retrieved context
  does not contain enough information to answer.
- IF ANY ANSWER DOES NOT HAVE CITATIONS, PUT THE RIGHT SOURCE.

Return only the cleaned answer.
"""

    response = model.invoke(prompt)

    return {"answer": response.content}

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


# Only tool available to the retrieval agent
tools = [retrieve_market_docs]

tool_node = ToolNode(tools)

def should_continue(state: RAGState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "answer"


# Build graph
builder = StateGraph(RAGState)


builder.add_node("answer", answer_node)
builder.add_node("setter_node", setter_node)

builder.add_node("hallucination_checker", hallucination_checker)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "setter_node")
builder.add_edge("setter_node", 'agent')

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "answer": "answer",
    },
)

builder.add_edge("tools", "agent")
builder.add_edge("answer", 'hallucination_checker')
builder.add_edge('hallucination_checker', END)

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
conn = sqlite3.connect(
    "checkpoints.db",
    check_same_thread=False
)

memory = SqliteSaver(conn)

graph = builder.compile(checkpointer=memory)