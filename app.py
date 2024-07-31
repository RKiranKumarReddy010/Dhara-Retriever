from flask import Flask, request, render_template
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from numba import cuda,jit
import cupy as cp
import bs4
import chromadb
import json

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    query = request.form['query']
    with cp.cuda.Device(0):
        res = after_rag_chain.invoke(query)
    response = f"\t\t{res}" 
    return response




if __name__ == '__main__':
    model_local = ChatOllama(model='llama3')
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    loaded_vectorstore = Chroma(persist_directory="./chroma_db1", embedding_function=embeddings)
    loaded_vectorstore.get()
    retriever = loaded_vectorstore.as_retriever()
    after_rag_template = """answer the question based on the following context:
    {context}
    Question:{question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context":retriever, "question":RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    app.run(debug=True)
