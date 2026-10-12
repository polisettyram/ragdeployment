# 🤖 RAG Chatbot | Agentic RAG with Streamlit

An AI-powered chatbot that answers questions using information retrieved from custom text documents. This project combines **Retrieval-Augmented Generation (RAG)**, semantic search, vector embeddings, and an LLM to generate context-aware responses.

The project also explores **Agentic RAG**, where an LLM can decide when to use a retrieval tool to find relevant information before answering a question.

## ✨ Features

- **Document-based Q&A:** Ask questions about your uploaded or configured text documents.
- **Semantic Search:** Retrieve relevant content based on meaning rather than exact keyword matches.
- **Vector Embeddings:** Convert text chunks and user queries into numerical representations using Sentence Transformers.
- **Vector Database:** Use ChromaDB to store document embeddings and retrieve relevant chunks.
- **Agentic Retrieval:** Allow the LLM to invoke a retrieval tool when additional document context is needed.
- **LLM-powered Responses:** Generate natural-language answers using the Groq API.
- **Interactive Chat Interface:** Chat with the system through a Streamlit web application.
- **Conversation History:** Keep track of chats within the app session.

## 🏗️ Architecture

```text
        Text Documents (.txt)
                 |
                 v
        Document Loading
                 |
                 v
        Text Chunking
                 |
                 v
      Sentence Transformer
       (Vector Embeddings)
                 |
                 v
          ChromaDB
                 |
                 v
User Question --> Semantic Retrieval
                         |
                         v
                 Relevant Chunks
                         |
                         v
                   Groq LLM
                         |
                         v
                 Generated Answer
                         |
                         v
                Streamlit Chat UI
```

In the Agentic RAG workflow, the LLM can decide to call the document retrieval tool before generating its final response.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web interface |
| Groq API | LLM inference |
| OpenAI Python SDK | API client compatible with Groq's API |
| Sentence Transformers | Generate text embeddings |
| ChromaDB | Store and retrieve vector embeddings |
| python-dotenv | Load environment variables locally |
| Git & GitHub | Version control and project hosting |

## 📂 Project Structure

```text
ragdeployment/
├── app.py             # Streamlit chatbot interface
├── rag.py             # RAG pipeline and agent logic
├── txtfile1.txt       # Sample knowledge document
├── txtfile2.txt       # Sample knowledge document
├── txtfile3.txt       # Sample knowledge document
├── requirements.txt   # Python dependencies
├── .gitignore         # Excludes secrets and unnecessary files
└── README.md          # Project documentation
```

## ⚙️ How It Works

### 1. Document Loading
The application loads text documents that act as its knowledge base.

### 2. Text Chunking
Documents are split into smaller chunks so that relevant sections can be retrieved efficiently.

### 3. Embedding Generation
A Sentence Transformer model converts text chunks into 384-dimensional vectors.

Model: `sentence-transformers/all-MiniLM-L6-v2`

### 4. Vector Storage
The generated embeddings are stored in ChromaDB for semantic retrieval.

### 5. Semantic Retrieval
When a user asks a question, the system searches for the most relevant document chunks.

### 6. LLM Response Generation
The retrieved context is passed to the Groq-hosted language model to generate a relevant response.

### 7. Agentic RAG
The agent can use a retrieval tool when it needs information from the document collection to answer a question.

### 8. Streamlit Interface
Users interact with the chatbot through a web interface without needing to run commands for every question.

## 🚀 Run Locally

### Prerequisites

- Python 3.10 or a compatible version
- A Groq API key
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/polisettyrami/ragdeployment.git
cd ragdeployment
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure the API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_actual_groq_api_key
```

Replace the placeholder with your own API key.

**Security:** Never commit your `.env` file or expose your API key in source code.

### Step 5: Run the chatbot

```bash
streamlit run app.py
```

Streamlit will display a local URL in your terminal. Open that URL in your browser to use the chatbot.

## ☁️ Deployment

The application can be deployed using **Streamlit Community Cloud**.

1. Push the project files to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app using your GitHub repository.
4. Select the `main` branch and `app.py` as the main file.
5. Add your API key under the app's Secrets settings:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

6. Deploy the application.

Streamlit installs the dependencies from `requirements.txt` and runs the application.

## 🔐 Security Notes

- Keep API keys in environment variables or Streamlit Secrets.
- Do not commit `.env` files.
- Do not publish API keys in screenshots, logs, or documentation.
- If a key is accidentally exposed, revoke it and generate a replacement.

## 🔮 Future Improvements

- Support PDF and DOCX documents.
- Add file uploads directly through the chat interface.
- Use persistent vector storage to retain embeddings across restarts.
- Improve retrieval with hybrid search combining semantic and keyword-based methods.
- Add citations and source snippets to generated answers.
- Support multiple knowledge bases and users.
- Add evaluation metrics for retrieval quality and answer accuracy.

## 🎯 Learning Outcomes

This project explores practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Agentic RAG and tool calling
- Text chunking and vector embeddings
- Semantic search and vector databases
- LLM API integration
- Streamlit application development
- Environment variable management
- GitHub-based deployment workflows

## 👨‍💻 Author

**Ram Sai**

B.Tech Computer Science Engineering student interested in AI, LLMs, RAG systems, and intelligent applications.

---

*Built as a hands-on project to understand how retrieval-based AI applications work and how to deploy them as web apps.*
