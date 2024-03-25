import os

from dotenv import load_dotenv

import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

st.set_page_config(page_title="Streamlit & Groq Test")
st.title("Mixtral-8x7b")


def get_response(query: HumanMessage, chat_history: list) -> str:

    template = f"""You are a helpful AI assistant. Given the following messages: {chat_history} respond to the following query: {query}"""

    prompt = PromptTemplate.from_template(template)

    chat = ChatGroq(
        temperature=0, model="mixtral-8x7b-32768", api_key=GROQ_API_KEY, streaming=True
    )

    chain = prompt | chat | StrOutputParser()

    res = chain.stream({"chat_history": chat_history, "user_question": query})

    return res


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

user_message = st.chat_input("Enter message:")

if user_message is not None and user_message != "":
    st.session_state.chat_history.append(HumanMessage(user_message))

    with st.chat_message("User"):
        st.markdown(user_message)

    with st.chat_message("AI"):
        response = st.write_stream(
            get_response(user_message, st.session_state.chat_history)
        )

    st.session_state.chat_history.append(AIMessage(response))
