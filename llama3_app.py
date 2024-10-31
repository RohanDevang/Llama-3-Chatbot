import os
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Set up the Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')

# App configuration
st.set_page_config(page_title="Llama 3 Chatbot", page_icon="✨", layout="wide")

# Sidebar for user input
with st.sidebar:
    st.title('🦙💬 Llama 3 Chatbot')
    st.write('This chatbot uses the open-source Llama 3.1 LLM model from Meta.')

    st.sidebar.title("Select an LLM")
    model = st.sidebar.selectbox(
        'Choose a Model',
        ['llama3-70b-8192', 'llama-3.1-70b-versatile']
    )

    # Slider for conversational memory length
    conversational_memory_length = st.sidebar.slider('Conversational Memory Length:', 1, 10, value=5)

# Initialize memory for conversation based on the selected memory length
if 'memory' not in st.session_state or 'conversational_memory_length' not in st.session_state or st.session_state.conversational_memory_length != conversational_memory_length:
    st.session_state.conversational_memory_length = conversational_memory_length
    st.session_state.memory = ConversationBufferWindowMemory(k=conversational_memory_length)

# Initialize ChatGroq and ConversationChain with the updated memory
groq_chat = ChatGroq(
    groq_api_key = groq_api_key,
    model_name = model
)

conversation = ConversationChain(
    llm = groq_chat,
    memory=st.session_state.memory
)

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clear chat history function
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.memory = ConversationBufferWindowMemory(k = st.session_state.conversational_memory_length)  # Reset memory

st.sidebar.button('Clear Chat History', on_click = clear_chat_history)

# User input prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate a new response from Groq if the last message was from the user
    if st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = conversation.predict(input = prompt)  # Use the conversation chain to get the response
                st.write(response)  # Display the assistant's response
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
