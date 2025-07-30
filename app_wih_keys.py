import sys
import os
import pysqlite3

# Redirect sqlite3 to use pysqlite3 for compatibility with ChromaDB
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import (
    ChatOpenAI,
    ChatAnthropic,
    ChatCohere,
)

try:
    from langchain_mistralai.chat_models import ChatMistralAI
except ImportError:
    ChatMistralAI = None

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

VECTOR_DB_DIR = "french_labour_code_vectordb"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LLM_OPTIONS = {
    "OpenAI (gpt-3.5-turbo)": ("openai", "gpt-3.5-turbo"),
    "Anthropic (Claude 3 Opus)": ("anthropic", "claude-3-opus-20240229"),
    "Mistral (Tiny)": ("mistral", "mistral-tiny"),
    "Cohere (Command R+)": ("cohere", "command-r-plus"),
}

st.title("French Labour Code Assistant")

llm_label = st.sidebar.selectbox("Choose your LLM model", list(LLM_OPTIONS.keys()))
provider, model_name = LLM_OPTIONS[llm_label]
api_key = st.sidebar.text_input("API Key for the selected model", type="password")

def check_model_change(selected_model: str):
    """
    Reset chat history if the selected LLM model changes.
    """
    if "last_model" not in st.session_state:
        st.session_state.last_model = selected_model
    elif selected_model != st.session_state.last_model:
        st.session_state.chat_history = []
        st.session_state.last_model = selected_model

check_model_change(llm_label)

@st.cache_resource
def load_embeddings():
    """
    Load HuggingFace embeddings model.
    """
    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        st.stop()

@st.cache_resource
def load_vector_db():
    """
    Load the persistent Chroma vector database.
    """
    try:
        return Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=load_embeddings()
        )
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        st.stop()

def initialize_memory():
    """
    Initialize conversation memory for context retention.
    """
    try:
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    except Exception as e:
        st.error(f"Error initializing memory: {e}")
        st.stop()

def load_llm(provider: str, model_name: str, api_key: str):
    """
    Load the selected LLM with the provided API key.
    """
    try:
        if provider == "openai":
            return ChatOpenAI(model_name=model_name, openai_api_key=api_key)
        elif provider == "anthropic":
            return ChatAnthropic(model=model_name, anthropic_api_key=api_key)
        elif provider == "mistral":
            if ChatMistralAI is None:
                raise ImportError("`ChatMistralAI` requires `langchain-mistralai`. Install it with: pip install langchain-mistralai")
            return ChatMistralAI(model=model_name, mistral_api_key=api_key)
        elif provider == "cohere":
            return ChatCohere(model=model_name, cohere_api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        st.stop()

def get_user_input():
    """
    Get user input from the text field and clear it after submission.
    """
    if "last_question" not in st.session_state:
        st.session_state["last_question"] = ""
    user_input = st.text_input("Ask your question:", key="user_question", on_change=save_and_clear)
    return st.session_state["last_question"]

def save_and_clear():
    """
    Save the last user question and clear the input field.
    """
    st.session_state["last_question"] = st.session_state["user_question"]
    st.session_state["user_question"] = ""

def clean_text(text: str) -> str:
    """
    Remove newline characters from the text.
    """
    return text.replace("\n", " ")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Check API key before loading LLM
if not api_key or api_key.strip() == "":
    st.error("Please enter a valid API key for the selected model.")
    st.stop()
llm = load_llm(provider, model_name, api_key)
vector_db = load_vector_db()
memory = initialize_memory()
custom_prompt = PromptTemplate(
    template="""
    You are a legal assistant specialized in the French Labour Code.

    Your behavior depends on the user's question:

    1. If it's a greeting → Respond warmly and invite the user to ask a legal question.
    2. If it's not law-related → Politely explain that you only answer legal questions.
    3. If it's a legal question → Answer only based on the **provided context**, and **cite relevant law articles**.

    Always answer **in French**. Never make up information.

    Chat History:
    {chat_history}

    User question:
    {question}

    Context (excerpts from the Labour Code):
    {context}

    Answer:
    """,
    input_variables=["question", "context", "chat_history"]


try:
    custom_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
except Exception as e:
    st.error(f"Error creating conversational chain: {e}")
    st.stop()

def handle_user_interaction():
    """
    Handle user input, generate a response, and display it with context.
    """
    user_input = get_user_input()
    if user_input:
        cleaned_question = clean_text(user_input)
        try:
            # source
            retriever = vector_db.as_retriever()
            relevant_docs = retriever.get_relevant_documents(cleaned_question)

            result = custom_chain({
                "question": cleaned_question,
                "chat_history": st.session_state.chat_history
            })
            # display answer
            st.session_state.chat_history.append((cleaned_question, result["answer"]))
            st.write(result["answer"])
            # display context
            if relevant_docs:
                st.write("**Context used :**")
                for doc in relevant_docs:
                    st.write(doc.page_content)
            else:
                st.write("No context found.")


        except Exception as e:
            st.error(f"Error generating response: {e}")

def display_chat_history_sidebar():
    """
    Display the conversation history in the sidebar.
    """
    with st.sidebar:
        if st.session_state.chat_history:
            st.markdown("### Conversation History:")
            for q, a in st.session_state.chat_history:
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Assistant:** {a}")

handle_user_interaction()
display_chat_history_sidebar()