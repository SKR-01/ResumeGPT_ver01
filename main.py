import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("Sourabh\'s ResumeGPT 🌱")
btn = st.button("Create Knowledgebase First! (Click here)")
if btn:
    create_vector_db()
    #pass

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)
    #pass

    st.header("Answer")
    st.write(response["result"])






