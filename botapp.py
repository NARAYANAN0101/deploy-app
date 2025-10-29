import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import tempfile

# --- Streamlit UI ---
st.set_page_config(page_title="📄 PDF RAG QA App", layout="centered")
st.title("📄 PDF-Powered Question Answering App")
st.markdown("Upload a PDF file and ask any question — powered by FAISS + Ollama + LangChain.")

# --- File upload ---
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    st.success("✅ PDF uploaded successfully!")

    # --- Load and split the document ---
    st.info("📚 Processing PDF document...")
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # --- Create embeddings & vector store ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

 from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # example model
    task="text-generation",
    huggingfacehub_api_token=st.secrets["HF_TOKEN"]  # add your token in Streamlit secrets
)

    # --- Define prompt and RAG chain ---
    prompt = ChatPromptTemplate.from_template("""
    Use the following context to answer the question:
    {context}

    Question: {question}
    """)

    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- User question input ---
    st.markdown("### ❓ Ask a question about your PDF:")
    user_query = st.text_input("Enter your question here:")

    if st.button("Get Answer"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("💭 Thinking..."):
                response = rag_chain.invoke(user_query)
            st.success("🧠 **Answer:**")
            st.write(response)

else:
    st.info("Please upload a PDF file to start.")
    

