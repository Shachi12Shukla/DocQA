import os
from dotenv import load_dotenv
import streamlit as st

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_groq import ChatGroq

# FAISS 
from langchain_community.vectorstores import FAISS

# --- 🔐 Load environment variables ---
GROQ_API_KEY = os.getenv("DocQAKey")
PINECONE_API_KEY = os.getenv("docqa_pinecone")
index_name = "docqa-project"

# --- 📄 Step 1: Load and split PDF ---
def load_docs(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    # length of pages
    print(len(docs))
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_documents(docs)

pdf_path = "PDFs/Doc1_policy.pdf"
docs = load_docs(pdf_path)
print("Length of text chunks " ,len(pdf_path))

# ---  Step 2: Embedding model ---
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": False}
    )
    return embedding_model

embedding_model = get_embedding_model()

# --- 🤖 Step 4: LLM + Prompt Template ---
# llm = ChatGroq(
#     temperature=0,
#     api_key=GROQ_API_KEY,
#     model_name="llama3-8b-3.1-instant"
# )

# custom_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are an expert PDF assistant. Use the below context to answer the question.
# If the answer is not in the context, say "I couldn't find that in the document."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
# )

# llm_chain = LLMChain(llm=llm, prompt=custom_prompt)

# # --- 💬 Step 5: Streamlit UI ---
# st.title("📄 Chat with PDF")

# user_question = st.text_input("Ask a question about the PDF:")
# if user_question:
#     with st.spinner("Searching and generating answer..."):
#         relevant_docs = vector_store.similarity_search(user_question, k=4)
#         context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
#         response = llm_chain.invoke({"context": context_text, "question": user_question})

#         st.write("### ✅ Answer:")
#         st.write(response["text"])

#         with st.expander("📚 See Sources"):
#             for doc in relevant_docs:
#                 st.write(doc.page_content[:300] + "...")
