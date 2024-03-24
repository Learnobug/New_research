import os
import streamlit as st
import pickle
from langchain import hub
from langchain.globals import set_debug
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.url_selenium import SeleniumURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv

load_dotenv()
             
llm = GoogleGenerativeAI(model="models/text-bison-001",google_api_key=os.getenv("API_KEY") )
st.title("News Research Tool 🔎")
prompt = hub.pull("efriis/my-first-prompt")
st.sidebar.title("News Article URLs ")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

main_placefolder = st.empty()

process_url_clicked = st.sidebar.button("Process URLs")
file_path = 'vectorstore_openai.pkl'

if process_url_clicked:
    # 1. Loading data
    loader = SeleniumURLLoader(urls=urls)
    main_placefolder.text("Data Loading...started... ✅✅✅")
    data= loader.load()
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n', '.', ','],
        chunk_size=1000
    )
    main_placefolder.text("Text Splitter...started... ✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save it to faiss index
    if not docs:
        st.error("No valid documents found after splitting. Please check the content of the URLs.")
    else:
        # Create embeddings and save it to faiss index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placefolder.text("Embedding Vector Started Building...started... ✅✅✅")
        
        # Save the vectorstore_openai to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)

query = main_placefolder.text_input("Question:   ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            ret=vectorstore.as_retriever()
            chain=RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": prompt})
            try:
                # Perform retrieval and get the answer
                result = chain({'query': query},return_only_outputs=True)
                st.write(result['result'])
                               
            except Exception as e:
                st.error(f"An error occurred: {e}")
