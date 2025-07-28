import streamlit as st
import os
import json
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

st.title("Assistant Code du Travail ")

llm_choice = st.sidebar.selectbox(
    "Choisissez votre modèle LLM",
    ("Local (Ollama/Mistral)", "OpenAI (clé requise)")
)

if llm_choice == "OpenAI (clé requise)":
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
else:
    openai_api_key = None




def check_and_reset_history(llm_choice: str):
    if "last_model" not in st.session_state:
        st.session_state.last_model = llm_choice
    elif llm_choice != st.session_state.last_model:
        st.session_state.chat_history = []
        st.session_state.last_model = llm_choice


check_and_reset_history(llm_choice)

# load LLMs
@st.cache_resource
def load_llm(llm_choice, openai_api_key=None):
    if llm_choice == "Local (Ollama/Mistral)":
        llm = Ollama(model="mistral:7b-instruct")
    else:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
    st.sidebar.markdown(f"🔍 **Modèle LLM utilisé :** `{llm_choice}`")
    return llm




# load embeddings
@st.cache_resource
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# load vector db
@st.cache_resource
def load_vector_db():
    vector_db = Chroma(
        persist_directory="french_labour_code_vectordb",
        embedding_function=load_embeddings()
    )
    return vector_db

# --- Conversation Memory ---
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Initialiser l'historique de conversation dans Streamlit
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Charger les composants

llm = load_llm(llm_choice, openai_api_key)

embeddings = load_embeddings()
vector_db = load_vector_db()

# Préparer la mémoire conversationnelle
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Créer la chaîne conversationnelle
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(),
    memory=memory
)


def get_user_input():


    return st.text_input("Votre question", key="user_question", value=st.session_state.get("user_input", ""))

def clean_text(text):
    return text.replace("\n", " ")


# Créer la chaîne conversationnelle
custom_prompt = PromptTemplate(
    template="""
    Vous êtes un assistant juridique intelligent, bienveillant, et expert du Code du travail français.

    Votre comportement dépend du type de question posée par l'utilisateur :

    ---

    **1. Si la question est une salutation (ex. "salut", "bonjour", "hi") :**  
    → Répondez chaleureusement, en français, présentez votre rôle, et invitez l’utilisateur à poser une question juridique liée au Code du travail français.

    **2. Si la question n’est pas juridique ou semble hors sujet (ex. "Quel jour sommes-nous ?", "raconte-moi une blague") :**  
    → Répondez poliment, mais informez que vous êtes conçu uniquement pour répondre à des questions relatives au Code du travail français.

    **3. Si la question est juridique et concerne le Code du travail français :**  
    → Répondez uniquement en vous appuyant sur le **contexte fourni** ci-dessous.  
    → Citez **explicitement** les **articles** du Code du travail mentionnés dans le contexte.  
    → Si aucune information pertinente n’est trouvée dans le contexte, dites clairement que vous ne pouvez pas répondre avec certitude.

    ---

    **Important :**
    - Toutes vos réponses doivent être rédigées en **français**, même si la question est posée dans une autre langue.
    - N’inventez jamais d’informations. Si vous ne savez pas, dites-le.

    ---

    **Question de l'utilisateur :**  
    {question}

    **Contexte juridique (extraits pertinents du Code du travail) :**  
    {context}

    ---

    **Réponse :**
    """
    ,
    input_variables=["question", "context"]
)
custom_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)


def response():
    user_input = get_user_input()
    if user_input:
        result = custom_chain({"question": user_input, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((user_input, result["answer"]))
        st.write(result["answer"])
        # Pour debug :
        st.write("**Contexte utilisé :**", result.get("context", "Aucun contexte trouvé"))


# Appel de la fonction centrale pour la gestion de la réponse
response()

# Affichage de l'historique dans la sidebar

def creat_chat_history_sidebar():
    with st.sidebar:
        if st.session_state.chat_history:
            st.markdown("### Historique de la conversation :")
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"**Vous :** {q}")
                st.markdown(f"**Assistant :** {a}")

creat_chat_history_sidebar()
