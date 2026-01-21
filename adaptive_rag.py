from  dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict



load_dotenv()

web_search_tool=TavilySearch(max_results=4)

class RAGState(TypedDict):
    question:str
    context:str
    is_relevant:bool
    answer:str
    enriched_question:str
    web_results:str

embedding_models = OllamaEmbeddings(model="nomic-embed-text")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
file_path = "./hr_manual.pdf"
loader = PyPDFLoader(file_path, mode="page")
text_split = text_splitter.split_documents(loader.load())
vector_store = InMemoryVectorStore(embedding=embedding_models)
vector_store.add_documents(documents=text_split)

model = ChatOpenAI(model="gpt-5-mini-2025-08-07")

def retrieve_context(query:str):
    retrieved_docs = vector_store.similarity_search(query=query, k=4)
    content = "\n\n".join(
        (f"Source : {doc.metadata} \n Content : {doc.page_content}")
        for doc in retrieved_docs
    )

    return content

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant who provides answers using the provided context.
                    Use only the information from the context to answer.If context doesnt have the answer say so"""),

    ("human", "Context: \n {context} \n\n Question: {question}")
])


rag_chain = prompt_template | model

def validate_context(question:str, context:str):

    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a strict validator.
        Return Only 'YES' or 'NO'.
        Answer YES if the context contain information
        that can directly answer the question
        """),
        ("human","""
        Question:
        {question}
        
        Context:
        {context}
        """)
    ])

    validation_chain = validation_prompt | model

    result = validation_chain.invoke({
        "question" : question,
        "context" :context
    })

    return result.content.strip().upper() == "YES"  # return True if answer is YES false if NO

def retrieve_node(state:RAGState):
    context = retrieve_context(state["enriched_question"])
    return {"context" : context}

def validate_node(state:RAGState):
    is_relevant = validate_context(
        state["question"], state["context"]
    )

    return  {"is_relevant" : is_relevant}


def answer_node(state: RAGState):

    response = rag_chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })
    return {"answer": response.content}


def enrich_query_node(state: RAGState):

    enrich_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You rewrite user question to improve document retrieval.
        Make the question more specific and focused.
        DO NOT answer the question
        """),
        ("human", """
        Original Question:
        
        "{question}
        
        Rewrite the question for better search""")
    ])

    enrich_chain = enrich_prompt | model

    enriched_question = enrich_chain.invoke({
        "question" : state["question"]
    })

    return {"enriched_question" : enriched_question.content}

def web_search_node(state:RAGState):
    tavily_resp = web_search_tool.invoke({
        "query" : state["question"]

    })

    results = tavily_resp["results"]

    content = "\n".join(
        [result["content"] for result in results]
    )

    return {"web_results" : content}

def web_answer_node(state:RAGState):

    web_answer_prompt = ChatPromptTemplate.from_messages([
       ("system","""
        You answer questions using web search results.
        If the answer is uncertain, say so
        """),("human","""
        Question :
        
        {question}
        
        Web Search Results:
        
        {web_results}
        """)]
    )

    web_answering_chain = web_answer_prompt | model
    response = web_answering_chain.invoke({
        "question" : state["question"],
        "web_results" : state["web_results"]
    })

    return {"answer": response.content}

def is_relevant_condition(state: RAGState):
    return state["is_relevant"] # return either true or false

graph = StateGraph(RAGState)

graph.add_node("enrich_query_node",enrich_query_node)
graph.add_node("retrieve_node", retrieve_node)
graph.add_node("validate_node", validate_node)
graph.add_node("answer_node",answer_node)
graph.add_node("web_search_node", web_search_node)
graph.add_node("web_answer_node", web_answer_node)


graph.add_edge(START,"enrich_query_node")
graph.add_edge("enrich_query_node","retrieve_node")
graph.add_edge("retrieve_node","validate_node")
graph.add_conditional_edges("validate_node",is_relevant_condition,{True:"answer_node",
                                                                   False:"web_search_node"})

graph.add_edge("web_search_node","web_answer_node")

graph.add_edge("answer_node", END)
graph.add_edge("web_answer_node", END)

adaptive_rag_graph = graph.compile()

while True:
    user_query= input("Enter your Question : ")

    resp = adaptive_rag_graph.invoke({
        "question": user_query
    })
    print(resp["answer"])
