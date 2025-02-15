import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    return splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def load_qa_model():
    template = ("""
    Answer the question using the given context. If the answer is not available, say "answer is not available in the context."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """)
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def answer_question(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings)
    docs = vector_store.similarity_search(question)
    qa_chain = load_qa_model()
    response = qa_chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]

def process_pdfs(pdf_files):
    text = extract_text_from_pdfs(pdf_files)
    chunks = split_text_into_chunks(text)
    create_vector_store(chunks)
    return "PDFs processed successfully!"

# Gradio Interface
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=gr.Textbox(label="Reply"),
    title="Multi-PDF Chatbot",
    description="Upload PDFs and ask questions based on their content."
)

upload_section = gr.Interface(
    fn=process_pdfs,
    inputs=gr.File(file_types=[".pdf"], label="Upload PDFs", multiple=True),
    outputs="text",
    title="Upload PDF Files",
    description="Upload PDF files and process them before asking questions."
)

gr.TabbedInterface([upload_section, demo], ["Upload PDFs", "Ask Questions"]).launch()
