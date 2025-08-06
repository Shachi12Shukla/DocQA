from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone as SDKPinecone
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("DocQAKey")
PINECONE_API_KEY=os.environ["PINECONE_API_KEY"] = os.getenv("docqa_pinecone")

# Step 1: Load and split documents
def load_docs(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    return splitter.split_documents(docs)

pdf_path = "PDFs/Doc1_policy.pdf"
docs = load_docs(pdf_path)

# Step 2: Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5',
    encode_kwargs={'normalize_embeddings': False}
)

# Step 3: Pinecone vector store
index_name = "docqa-project"
pc = SDKPinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# index already exists
vector_store = LangchainPinecone.from_documents(
    documents=docs,
    embedding=embedding_model,
    index_name=index_name
)

# Step 4: LLM and Prompt Template
llm = ChatGroq(
    temperature=0,
    api_key=GROQ_API_KEY,
    model_name="llama3-8b-3.1-instant"
)

# 🧠 Custom Prompt Template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert PDF assistant. Use the below context to answer the question.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question:
{question}

Answer:
"""
)

llm_chain = LLMChain(llm=llm, prompt=custom_prompt)

# Step 5: Streamlit UI
st.title("Chat with PDF")

user_question = st.text_input("Ask a question about the PDF:")
if user_question:
    # Embed + Search in Vector DB
    relevant_docs = vector_store.similarity_search(user_question, k=4)

    # Prepare context for the prompt
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Pass context + user question to LLM chain
    response = llm_chain.invoke({"context": context_text, "question": user_question})
    
    # Output
    st.write("### Answer:")
    st.write(response["text"])

    # Show source
    with st.expander("See Sources"):
        for doc in relevant_docs:
            st.write(doc.page_content[:300] + "...")

    
