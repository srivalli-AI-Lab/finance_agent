# agents/rag_agent.py
import os
import json
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
import time
import faiss
from langchain_community.docstore import InMemoryDocstore  # Correct import path

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

categories = [
    "budgeting", "investing basics", "stocks", "retirement planning"
]  # Reduced to 4 core finance-related categories to minimize searches and avoid rate limits

def build_kb():
    docs = []
    search = DuckDuckGoSearchAPIWrapper()
    loaded_urls = set()  # To avoid duplicates

    for cat in categories:
        query = f"best financial education articles on {cat} site:investopedia.com OR site:nerdwallet.com OR site:khanacademy.org OR site:gov OR site:edu"
        try:
            results = search.results(query, max_results=2)  # Reduced for faster testing; increase to 10 later
            time.sleep(5)  # Delay to avoid rate limits
        except Exception as e:
            print(f"Search failed for {cat}: {e}. Skipping...")
            continue
        links = [r['link'] for r in results if 'link' in r]

        for link in links:
            if link in loaded_urls:
                continue
            try:
                loader = WebBaseLoader(link)
                doc = loader.load()
                if doc:
                    doc[0].metadata['category'] = cat
                    doc[0].metadata['source'] = link
                    docs.extend(doc)
                    loaded_urls.add(link)
                time.sleep(2)  # Delay between loads
            except Exception as e:
                print(f"Failed to load {link}: {e}")
            if len(loaded_urls) >= 10:  # Reduced cap for faster testing; increase to 100 later
                break
        if len(loaded_urls) >= 10:
            break

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    if not chunks:
        print("Warning: No documents loaded. Creating empty vectorstore with dummy dimension.")
        # Create empty FAISS with dimension from dummy embedding
        dummy_embedding = embeddings.embed_query(" ")
        d = len(dummy_embedding)
        index = faiss.IndexFlatL2(d)
        return FAISS(
            embedding_function=embeddings.embed_query,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Load or build the vectorstore (persisted to disk for efficiency)
index_path = "faiss_index"
if os.path.exists(index_path):
    vectorstore = FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = build_kb()
    vectorstore.save_local(index_path)

def finance_rag_node(state):
    query = state["messages"][-1].content

    # Classify query category for filtering
    category_prompt = f"""
    Classify the following query into EXACTLY ONE of these categories: {', '.join(categories)} or 'general'.
    Respond only with the category name.

    Query: {query}
    """
    query_cat = llm.invoke(category_prompt).content.strip()

    # Retrieve with filter if category is specific
    if query_cat == 'general':
        docs = vectorstore.similarity_search(query, k=5)
    else:
        docs = vectorstore.similarity_search(query, k=5, filter={"category": query_cat})

    # Prepare context with source attribution
    context = "\n\n".join([f"Source: {d.metadata['source']}\n{d.page_content}" for d in docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a finance expert. Use only the provided context to answer. Be concise. Include inline citations to sources where relevant (e.g., [Source: url]). At the end, list all unique sources used."),
        ("human", "Context: {context}\n\nQuestion: {query}")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": query})
    return {"messages": [AIMessage(content=response.content)]}