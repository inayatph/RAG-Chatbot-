from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os


class ChatBot():
    load_dotenv()
    file_loader=PyPDFDirectoryLoader('document/')
    documents=file_loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=50)
    docs=text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    vector_store=FAISS.from_documents(docs, embedding=embeddings)
    vector_store.save_local("faiss_index")

    new_db = FAISS.load_local("faiss_index", embeddings , allow_dangerous_deserialization=True)

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50}, huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )


    template = """
    You are a seer. These Human will ask you a questions about their life. Use following piece of context to answer the question. 
    If you don't know the answer, just say you don't know. 
    You answer with short and concise answer, no longer than2 sentences.

    Context: {context}
    Question: {question}
    Answer: 
    
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

    rag_chain = (
    {"context": new_db.as_retriever(),  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
    
    )


bot = ChatBot()
input = input("Ask me anything: ")
result = bot.rag_chain.invoke(input)
print(result)

