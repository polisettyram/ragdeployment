
import streamlit as st
from rag import rag_agent
from datetime import datetime
import uuid


st.set_page_config(
    page_title="RAG chatbot",
    page_icon="🤖",
    layout="wide"
)


# 1. Initialize saved conversations
if "conversations" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.conversations = {
        first_id: {
            "title": "New chat",
            "messages": []
        }
    }
    st.session_state.current_chat = first_id


# 2. Create a new conversation
def new_chat():
    chat_id = str(uuid.uuid4())

    st.session_state.conversations[chat_id] = {
        "title": "New chat",
        "messages": []
    }

    st.session_state.current_chat = chat_id


# 3. Sidebar with chat history
with st.sidebar:
    st.title("🤖 RAG Chatbot")

    if st.button("＋ New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()
    st.subheader("Chat history")

    for chat_id, chat in reversed(
        list(st.session_state.conversations.items())
    ):
        title = chat["title"]

        if st.button(
            title,
            key=f"chat_{chat_id}",
            use_container_width=True
        ):
            st.session_state.current_chat = chat_id
            st.rerun()


# 4. Current conversation
current_id = st.session_state.current_chat
current_chat = st.session_state.conversations[current_id]

st.title(current_chat["title"])
st.caption("Ask questions about your knowledge-base documents.")


# 5. Display messages from the selected conversation
for message in current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# 6. Ask a question
question = st.chat_input("Ask a question about your documents...")

if question:
    current_chat["messages"].append({
        "role": "user",
        "content": question
    })

    if current_chat["title"] == "New chat":
        current_chat["title"] = question[:35]

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                answer = rag_agent(question, verbose=False)
            except Exception as error:
                print(f"Error: {error}")
                answer = "Something went wrong. Please check the terminal."

        st.write(answer)

    current_chat["messages"].append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()


