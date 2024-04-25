#Initializing libraries
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from IPython.display import Markdown as md

# Title and header for the Streamlit app
st.title("Q&A Chatbot Using RAG System")
st.header("Ask me a Questions  from 'Leave No Context Behind' Paper")

#Loading the data
loader = PyPDFLoader(r"C:\Users\seela\OneDrive\Desktop\KRANTHI\LANGCHAIN.pdf")
embeddings_model = GoogleGenerativeAIEmbeddings(google_api_key='AIzaSyAISGIUD-V6HSWksLgjsu6lsXBde99QUek',
                                               model="models/embedding-001")

#Splitting the data
data = loader.load_and_split()
text_splitter = NLTKTextSplitter(chunk_size=200, chunk_overlap=200)

chunks = text_splitter.split_documents(data)

#Creating Chroma Database
db = Chroma.from_documents(chunks, embeddings_model, persist_directory="./chroma_db_")

# Persist the database on drive
db.persist()

#Connected to the Database
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embeddings_model)
retriever = db_connection.as_retriever(search_kwargs={"k": 1})

#Creating Chat Templete
chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Aswer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#Creating Chat Model
chat_model = ChatGoogleGenerativeAI(google_api_key="AIzaSyAISGIUD-V6HSWksLgjsu6lsXBde99QUek", 
                                   model="gemini-1.5-pro-latest")

#Creating Output Parser
output_parser = StrOutputParser()

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)


User_input = st.text_input("Ask your Question: ")
if st.button("Search"):
    response = rag_chain.invoke(User_input)
    st.write(response)
