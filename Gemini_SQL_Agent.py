import google.generativeai as genai
import streamlit as st
import os
from langchain.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import create_sql_agent
from dotenv import load_dotenv 

st.title("SQL Agent Using Gemini")

# Enter your OpenAI API private access key here. IMPORTANT - don't share your code online if it contains your access key or anyone will be able to access your openai account
if api_key := st.text_input("Enter the API Key for Gemini"):
    os.environ['GEMINI_API_KEY'] = api_key
    load_dotenv()
    # define the database we want to use for our test
    db = SQLDatabase.from_uri('sqlite:///sql_lite_database.db')

    # choose llm model, in this case the default OpenAI model
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    # setup agent
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response=str(agent_executor.invoke(prompt)["output"])
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})