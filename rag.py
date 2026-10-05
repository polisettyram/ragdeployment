import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv("/Users/ramsai/rag deployment/rag deployment/.env")

print("API KEY:", os.getenv("GROQ_API_KEY"))


client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

print("client created successfully")

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
)

print(response.choices[0].message.content)


with open("/Users/ramsai/rag deployment/rag deployment/txtfile1.txt","r") as f:
    doc1=f.read()

with open("/Users/ramsai/rag deployment/rag deployment/txtfile2.txt","r") as f:
    doc2=f.read()

with open("/Users/ramsai/rag deployment/rag deployment/txtfile3.txt","r") as f:
    doc3=f.read()


def chunk_document(text, source_name):
    paragraphs = text.strip().split("\n\n")
    chunks = []

    for para in paragraphs:
        if len(para) < 20:
            continue

        if para.startswith("=="):
            continue

        chunks.append({
            "text": para,
            "source": source_name
        })

    return chunks


doc1_chunks = chunk_document(
    doc1,
    "RAG Architecture and LLM Limitations"
)

doc2_chunks = chunk_document(
    doc2,
    "Data Ingestion, Chunking Strategies, and Vector Embeddings"
)

doc3_chunks = chunk_document(
    doc3,
    "Prompt Augmentation, Generation, and Evaluation Strategies"
)

all_chunks = doc1_chunks + doc2_chunks + doc3_chunks

print(len(doc1_chunks))
print(len(doc2_chunks))
print(len(doc3_chunks))
print(len(all_chunks))


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentence1 = ''' I love chicken Biriyani '''
sentence2 = ''' I love eating flaovured rice with chicken'''
sentence3 = ''' I am doing work from Home'''

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


chroma_client = chromadb.Client()

collection = chroma_client.create_collection(
    name="company_docs"
)

documents = []
ids = []
metadatas = []

for i, chunk in enumerate(all_chunks):
    documents.append(chunk["text"])
    ids.append(f"chunk {i}")
    metadatas.append({"source": chunk["source"]})

print(documents[1])
print(ids[1])
print(metadatas[1])

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print(f" Stored {len(documents)} chunks in chroma db")


def search_docs(query: str) -> str:
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    chunks = results["documents"][0]

    return "\n\n".join(chunks)


rag_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "answers about RAG Architecture & Limitations Test,Ingestion & Vector Math Test,Generation & Evaluation Test",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query you want to find about RAG"
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

print("rag tool defined")


def rag_agent(question, verbose=True):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant who answers questions about RAG"
                "answers about RAG Architecture & Limitations Test,Ingestion & Vector Math Test,Generation & Evaluation Test"
                "If the documents dont contian the answer, say so clearly."
                "Always base your answers on the rerieved documets if you decided to use the tool."
                "If tool use is not needed, specifcy that you are not using tool and asnwer the quesiton based on your knowledge."
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

    max_steps = 3

    for step in range(max_steps):

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=rag_tools,
        )

        choice = response.choices[0]

        if choice.finish_reason == "stop":
            if verbose:
                print(f"🤖 Answer: {choice.message.content}")
                print(f"{'═' * 60}")

            return choice.message.content

        if choice.message.tool_calls:
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:

                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if verbose:
                    print(f'  🔍 Searching: "{args["query"]}"')

                result = available_tools[func_name](**args)

                if verbose:
                    print(f"  📄 Found {len(result)} chars of context")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

    return "Could not answer within step limit."


print("✅ RAG Agent ready!")

rag_agent("How many sick leave days do I get per year?")
rag_agent("What is 25*4 ?")
 