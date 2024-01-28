llm_model = "gpt-3.5-turbo"
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


current_directory = os.getcwd()

# Specify the file name (change this to the actual file name)
file_name = "resume.csv"

# Join the current directory with the file name to get the full file path
FILE_PATH = os.path.join(current_directory, file_name)

print("L32: ", FILE_PATH)
llm_2 = ChatOpenAI(
    temperature=0.2,
    openai_api_key=os.environ.get("API_KEY"),
    model=llm_model,
)

instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large"
)
vectordb_file_path = "faiss_index"


def create_vector_db():
    loader = CSVLoader(
        # file_path="/Users/sourabhkumar/langchain/resume.csv", source_column="prompt"
        file_path=FILE_PATH,
        source_column="prompt",
    )
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Generate a concise response (50-70 words) suitable for HR or interviewers call them 'User' 
                            asking about '<b>Sourabh's</b>' detail (here resume is a vectorDB). 
                            Provide relevant information from the 'response' section of the source document. 
                            If the answer is not found, 
                            respond with 'I would need additional details to address this inquiry thoroughly. 
                            Try to respond in a way that passively convinces to hire him.

        CONTEXT: {context}

        QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_2 = RetrievalQA.from_chain_type(
        llm=llm_2,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return chain_2
