from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi
import numpy as np

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class HybridRetriever:
    def __init__(self, vectorstore, chunks):
        self.vectorstore = vectorstore
        self.chunks = chunks
        tokenized = [chunk.page_content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def invoke(self, question, k=4):
        # Vector search
        vector_docs = self.vectorstore.similarity_search(question, k=k)
        vector_contents = [d.page_content for d in vector_docs]

        # BM25 keyword search
        tokens = question.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        bm25_docs = [self.chunks[i] for i in top_indices]
        bm25_contents = [d.page_content for d in bm25_docs]

        # Combine and deduplicate
        seen = set()
        combined = []
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)

        return combined[:k]

def process_documents(texts, filenames):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    all_chunks = []
    for text, filename in zip(texts, filenames):
        chunks = splitter.create_documents(
            [text],
            metadatas=[{"source": filename}]
        )
        all_chunks.extend(chunks)

    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    retriever = HybridRetriever(vectorstore, all_chunks)
    return vectorstore, retriever

def get_answer(vectorstore, question, api_key, chat_history=[], retriever=None):
    llm = ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    history_text = ""
    for msg in chat_history[-4:]:
        history_text += f"User: {msg['question']}\nAssistant: {msg['answer']}\n\n"

    # Use hybrid retriever if available, else fall back to vector
    if retriever:
        docs = retriever.invoke(question, k=4)
    else:
        docs = vectorstore.similarity_search(question, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([
        doc.metadata.get("source", "Unknown") for doc in docs
    ]))

    prompt = PromptTemplate.from_template("""
You are a helpful assistant that answers questions based on uploaded documents.

Previous conversation:
{history}

Use the following context from the documents to answer the question.
If the answer is not in the context, say "I could not find this in the uploaded documents."

Context:
{context}

Question: {question}

Answer:""")

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": question,
        "history": history_text
    })

    return answer, sources

def summarize_document(text, filename, api_key):
    llm = ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
    preview = text[:3000]
    prompt = PromptTemplate.from_template("""
You are a helpful assistant. Read the following document content and provide a clear,
structured summary including:
- Main topic
- Key points (as bullet points)
- Important conclusions

Document name: {filename}

Content:
{content}

Summary:""")
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"content": preview, "filename": filename})
    return summary