
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Deployment-friendly paths and API setup

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env or environment variables.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

print("Client created successfully")


# 2. Load your existing documents


with open(BASE_DIR / "txtfile1.txt", "r", encoding="utf-8") as f:
    doc1 = f.read()

with open(BASE_DIR / "txtfile2.txt", "r", encoding="utf-8") as f:
    doc2 = f.read()

with open(BASE_DIR / "txtfile3.txt", "r", encoding="utf-8") as f:
    doc3 = f.read()



# 3. Chunking

def chunk_document(text, source_name):
    paragraphs = text.strip().split("\n\n")
    chunks = []

    for para in paragraphs:
        para = para.strip()

        if len(para) < 20:
            continue

        if para.startswith("=="):
            continue

        chunks.append({
            "text": para,
            "source": source_name
        })

    return chunks


doc1_chunks = chunk_document(doc1, "RAG Architecture and LLM Limitations")
doc2_chunks = chunk_document(doc2, "Data Ingestion, Chunking Strategies, and Vector Embeddings")
doc3_chunks = chunk_document(doc3, "Prompt Augmentation, Generation, and Evaluation Strategies")

all_chunks = doc1_chunks + doc2_chunks + doc3_chunks

print(len(doc1_chunks))
print(len(doc2_chunks))
print(len(doc3_chunks))
print(len(all_chunks))


# 4. Embedding model

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# 5. Original sample similarity test (kept)

sentence1 = "I love chicken Biriyani"
sentence2 = "I love eating flavoured rice with chicken"
sentence3 = "I am doing work from home"

embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
embedding3 = model.encode(sentence3)

print(embedding1.shape)
print(embedding2.shape)
print(embedding3.shape)

similarity1 = cosine_similarity([embedding1], [embedding2])
similarity2 = cosine_similarity([embedding2], [embedding3])
similarity3 = cosine_similarity([embedding1], [embedding3])

print(similarity1)
print(similarity2)
print(similarity3)


# 6. ChromaDB

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="company_docs")

documents = []
ids = []
metadatas = []

for i, chunk in enumerate(all_chunks):
    documents.append(chunk["text"])
    ids.append(f"chunk_{i}")
    metadatas.append({"source": chunk["source"]})

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print(f"Stored {len(documents)} chunks in ChromaDB")


# 7. Search function

def search_docs(query: str) -> str:
    results = collection.query(
        query_texts=[query],
        n_results=min(3, len(documents))
    )

    chunks = results["documents"][0]
    sources = results["metadatas"][0]

    return "\n\n".join(
        f"Source: {sources[i]['source']}\n{chunk}"
        for i, chunk in enumerate(chunks)
    )


# 8. Tool definition

rag_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search the knowledge base for information about "
                "RAG architecture, ingestion, vector embeddings, "
                "generation, and evaluation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

available_tools = {
    "search_docs": search_docs
}

print("RAG tool defined")


# 9. RAG agent

def rag_agent(question, verbose=True):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant who answers questions "
                "about RAG. Use search_docs for questions that need "
                "information from the knowledge base. Base document "
                "answers on retrieved content. If the documents do "
                "not contain the answer, say so clearly. For unrelated "
                "general questions, you may use your general knowledge."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    if verbose:
        print(f"\n{'═' * 60}")
        print(f"🧑 Question: {question}")
        print(f"{'─' * 60}")

    max_steps = 5

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=rag_tools,
        )

        choice = response.choices[0]
        message = choice.message

        if choice.finish_reason == "stop":
            if verbose:
                print(f"🤖 Answer: {message.content}")
                print(f"{'═' * 60}")

            return message.content

        if message.tool_calls:
            messages.append(message.model_dump(exclude_none=True))

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if func_name not in available_tools:
                    result = "Unknown tool requested."
                else:
                    if verbose:
                        print(f"🔍 Searching: {args['query']}")

                    result = available_tools[func_name](**args)

                    if verbose:
                        print(f"📄 Found {len(result)} characters of context")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

    return "Could not answer within step limit."


# 10. Run tests only when this file is executed directly

if __name__ == "__main__":
    print("RAG Agent ready!")

    rag_agent("How many sick leave days do I get per year?")
    rag_agent("What is 25 * 4?")

