import streamlit as st
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI

#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# --- Global variables / constants ---
LLM_CHOICES = ("Local (Ollama/Mistral)", "OpenAI (API key required)")
VECTOR_DB_DIR = "french_labour_code_vectordb"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "mistral:7b-instruct"
OPENAI_MODEL_NAME = "gpt-3.5-turbo"

# --- Streamlit App Title ---
st.title("Labour Code Assistant")

# --- LLM model selection ---
llm_choice = st.sidebar.selectbox("Choose your LLM model", LLM_CHOICES)

# --- OpenAI API key input if needed ---
openai_api_key = None
if llm_choice == "OpenAI (API key required)":
    openai_api_key = st.sidebar.text_input(
        "Please enter your OpenAI API Key",
        type="password",
        help="You must provide your OpenAI API key to use this model."
    )
    if not openai_api_key:
        st.sidebar.warning("⚠️ Please enter your OpenAI API key to proceed.")

def check_and_reset_history(selected_model: str) -> None:
    """
    Check if the selected model has changed since the last interaction.
    If it has changed, reset the conversation history.
    """
    if "last_model" not in st.session_state:
        st.session_state.last_model = selected_model
    elif selected_model != st.session_state.last_model:
        st.session_state.chat_history = []
        st.session_state.last_model = selected_model

check_and_reset_history(llm_choice)

@st.cache_resource
def load_llm(selected_model: str, api_key: str = None):
    """
    Load the LLM model based on the user's choice.
    Uses Ollama locally or OpenAI with API key.
    Raises ValueError if API key is missing for OpenAI.
    """
    if selected_model == "Local (Ollama/Mistral)":
        llm = Ollama(model=OLLAMA_MODEL_NAME)
    else:
        if not api_key or api_key.strip() == "":
            raise ValueError("OpenAI API key is required for this model but was not provided.")
        llm = ChatOpenAI(
            model_name=OPENAI_MODEL_NAME,
            openai_api_key=api_key
        )
    st.sidebar.markdown(f"🔍 **Using LLM model:** `{selected_model}`")
    return llm

@st.cache_resource
def load_embeddings():
    """
    Load the HuggingFace embedding model.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_vector_db():
    """
    Load the persistent Chroma vector database.
    """
    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=load_embeddings()
    )

def initialize_memory():
    """
    Initialize the conversational memory to maintain context.
    """
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

def save_and_clear():
    """
    Save the current user input before clearing the input field.
    """
    st.session_state["last_question"] = st.session_state["user_question"]
    st.session_state["user_question"] = ""

def get_user_input() -> str:
    """
    Display the text input widget, retrieve the user input,
    and clear the input field while keeping the last entered question.
    """
    if "last_question" not in st.session_state:
        st.session_state["last_question"] = ""

    user_input = st.text_input(
        "Your question:",
        key="user_question",
        on_change=save_and_clear
    )
    return st.session_state["last_question"]

def clean_text(text: str) -> str:
    """
    Clean the text by removing newline characters.
    """
    return text.replace("\n", " ")

# --- Initialize chat history if missing ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Try loading the LLM, show error and stop if missing key ---
try:
    llm = load_llm(llm_choice, openai_api_key)
except ValueError as e:
    st.error(str(e))
    st.stop()

vector_db = load_vector_db()
memory = initialize_memory()

# Custom prompt with precise legal assistant instructions
custom_prompt = PromptTemplate(
    template="""
    You are an intelligent, kind, and expert legal assistant specialized in the French Labour Code.

    Your behavior depends on the type of user question:

    ---

    **1. If the question is a greeting (e.g., "hello", "hi"):**  
    → Respond warmly in French, introduce yourself, and invite the user to ask a legal question about the French Labour Code.

    **2. If the question is non-legal or off-topic (e.g., "What day is it?", "Tell me a joke"):**  
    → Respond politely, informing that you only answer questions related to the French Labour Code.

    **3. If the question is legal and concerns the French Labour Code:**  
    → Respond solely based on the **provided context** below.  
    → Explicitly cite the **articles** of the Labour Code mentioned in the context.  
    → If no relevant information is found in the context, clearly say you cannot answer with certainty.

    ---

    **Important:**
    - Always answer in **French**, even if the question is in another language.
    - Never fabricate information. If unsure, say so.

    ---

    **User question:**  
    {question}

    **Legal context (relevant excerpts from the Labour Code):**  
    {context}

    ---

    **Answer:**
    """,
    input_variables=["question", "context"]
)

# Create the conversational retrieval chain with the custom prompt
custom_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

def handle_user_interaction():
    """
    Handle the user input, query the conversational chain,
    and display the answer along with the used context.
    """
    user_input = get_user_input()
    if user_input:
        cleaned_question = clean_text(user_input)
        result = custom_chain({"question": cleaned_question, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((cleaned_question, result["answer"]))
        st.write(result["answer"])
        # Show the context used for debugging
        st.write("**Context used:**", result.get("context", "No context found"))

handle_user_interaction()

def display_chat_history_sidebar():
    """
    Display the conversation history in the Streamlit sidebar.
    """
    with st.sidebar:
        if st.session_state.chat_history:
            st.markdown("### Conversation History:")
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Assistant:** {a}")

display_chat_history_sidebar()
