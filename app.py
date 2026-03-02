import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentType, initialize_agent, tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.utilities import GoogleSerperAPIWrapper, OpenWeatherMapAPIWrapper
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables from .env file
load_dotenv()

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

@st.cache_resource(show_spinner=False)
def LLM_init():
    # Load and process the document
    loader = TextLoader("documents/travel_info.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Chroma vector store
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    retriever = vectordb.as_retriever()

    # Groq LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="LLaMA3-8b-8192"
    )

    # Tools
    weather = OpenWeatherMapAPIWrapper()
    search = GoogleSerperAPIWrapper()

    retriever_tool = create_retriever_tool(
        retriever,
        "Travel Information",
        "answers questions about travel from provided documents"
    )

    tools = [
        Tool(
            name="current search",
            func=search.run,
            description="useful for answering current events or facts"
        ),
        Tool(
            name="weather",
            func=weather.run,
            description="returns current weather data for a location"
        ),
        Tool(
            name="getlength",
            func=get_word_length,
            description="returns the length of a word"
        ),
        retriever_tool
    ]

    memory = ConversationBufferMemory(
        chat_memory=StreamlitChatMessageHistory(key="messages"),
        memory_key="chat_history",
        return_messages=True
    )

    llm_chain = initialize_agent(
        tools,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        llm=llm,
        memory=memory,
        verbose=True,
    )

    return llm_chain


# --- Streamlit UI ---
st.set_page_config(page_title="Travel Assistant", page_icon="✈️", layout="wide")

with st.sidebar:
    st.title("✈️ Travel Assistant")
    st.info(
        "This is a smart travel assistant powered by LangChain and Groq. I can help you with:\n"
        "- **Travel Information:** Answering questions based on provided documents.\n"
        "- **Current Events:** Searching the web for real-time information.\n"
        "- **Weather:** Providing current weather forecasts.\n\n"
        "Ask me anything in the chat!"
    )

st.header("💬 Chat with your Travel Assistant")


# Initial Assistant Message
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, I am your travel consultant. How can I help you?"}]

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant", avatar="🤖").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user", avatar="🧑‍💻").write(msg.content)
    else:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

if prompt := st.chat_input("Ask me about travel, weather, or current events..."):
    st.chat_message("user", avatar="🧑‍💻").write(prompt)

    if all([os.getenv("GROQ_API_KEY"), os.getenv("SERPER_API_KEY"), os.getenv("OPENWEATHERMAP_API_KEY")]):
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                agent = LLM_init()
                response = agent.invoke(
                    {"input": prompt}
                )
                st.write(response["output"])
    else:
        st.warning("Missing required environment variables. Please make sure .env file is configured.")
        st.stop()
