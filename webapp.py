import os
import streamlit as st
import tempfile
from datetime import datetime
import uuid

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Vector DB imports
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables 
GROQ_API_KEY = os.getenv("DocQAKey")
PINECONE_API_KEY = os.getenv("docqa_pinecone")

class SimpleChatWithPDF:
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.chain = None
        self.setup_components()
    
    def setup_components(self):
        """Initialize all components"""
        # Setup embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings": False}
        )
        
        # Setup LLM
        llm = ChatGroq(
            temperature=0,
            api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant"
        )
        
        # Setup prompt template
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an expert PDF assistant. Use the below context to answer the question with facts, dates and numbers (if these exist)
            If the answer is not in the context, say "I couldn't find that in the document."

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )
        
        
        self.chain = custom_prompt | llm
        
        # Connect to existing Pinecone index
        self.setup_existing_vector_store()
    
    def setup_existing_vector_store(self):
        """Connect to your existing Pinecone index"""
        try:
            # Connect to Pinecone
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Connect to existing index 
            existing_index = pc.Index("docqa-project2")
            
            # Create vector store with existing index
            self.vector_store = PineconeVectorStore(
                index=existing_index, 
                embedding=self.embedding_model
            )
            
            st.success("✅ Connected to existing Pinecone index: docqa-project2")
            
        except Exception as e:
            st.error(f"Error connecting to Pinecone: {str(e)}")
            self.vector_store = None
    
    def load_and_process_pdf(self, uploaded_file):
        """Load and process uploaded PDF file"""
        # Create temporary file from uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        try:
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            st.success(f"📄 Loaded PDF with {len(docs)} pages")
            
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            st.info(f"📝 Split into {len(chunks)} text chunks")
            
            return chunks
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    def add_document_to_vector_store(self, uploaded_file):
        """Add uploaded document to existing vector store"""
        if not self.vector_store:
            st.error("Vector store not available!")
            return None
        
        # Process the uploaded PDF
        docs = self.load_and_process_pdf(uploaded_file)
        
        # Create unique document ID for this upload
        doc_id = str(uuid.uuid4())
        
        # Add metadata to identify this specific document
        for doc in docs:
            doc.metadata['document_id'] = doc_id
            doc.metadata['filename'] = uploaded_file.name
            doc.metadata['upload_time'] = datetime.now().isoformat()
        
        # Upload documents to existing vector store
        with st.spinner("🔄 Adding documents to vector store..."):
            try:
                self.vector_store.add_documents(documents=docs)
                st.success("✅ Documents added to vector store!")
                return doc_id
            except Exception as e:
                st.error(f"Error adding documents: {str(e)}")
                return None
    
    def query_document(self, question, doc_id):
        """Query the specific document using document ID filter"""
        if not self.vector_store:
            st.error("Vector store not available!")
            return None, []
        
        with st.spinner("🔍 Searching and generating answer..."):
            try:
                # Search for relevant documents with filter for this specific document
                relevant_docs = self.vector_store.similarity_search(
                    question, 
                    k=3,
                    filter={"document_id": doc_id}
                )
                
                if not relevant_docs:
                    return "I couldn't find any relevant information in the document.", []
                
                # Combine context
                context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Generate response
                response = self.chain.invoke({"context": context_text, "question": question})
                
                return response.content, relevant_docs
                
            except Exception as e:
                st.error(f"Error during query: {str(e)}")
                return None, []

def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="📄")
    st.title("📄 Chat with PDF - Simple Version")
    st.markdown("Upload a PDF document and ask questions about its content!")
    
    # Initialize the app
    if 'chat_app' not in st.session_state:
        with st.spinner("Initializing app..."):
            st.session_state.chat_app = SimpleChatWithPDF()
    
    chat_app = st.session_state.chat_app
    
    # Check if vector store is available
    if not chat_app.vector_store:
        st.error("❌ Could not connect to Pinecone. Please check your API key and index name.")
        return
    
    # File upload section
    st.header("1️⃣ Upload Your PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Upload a PDF document to chat with"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        # Check if this is a new file
        if ('current_file' not in st.session_state or 
            st.session_state.current_file != uploaded_file.name):
            
            st.session_state.current_file = uploaded_file.name
            
            # Add document to vector store
            doc_id = chat_app.add_document_to_vector_store(uploaded_file)
            
            if doc_id:
                st.session_state.doc_id = doc_id
                st.session_state.pdf_processed = True
            else:
                st.session_state.pdf_processed = False
        
        # Show file info and chat interface
        if st.session_state.get('pdf_processed', False):
            st.success(f"✅ Currently loaded: **{uploaded_file.name}**")
            
            # Chat section
            st.header("2️⃣ Ask Questions")
            user_question = st.text_input(
                "Ask a question about your PDF:",
                placeholder="What is this document about?"
            )
            
            if user_question:
                try:
                    answer, relevant_docs = chat_app.query_document(
                        user_question, 
                        st.session_state.doc_id
                    )
                    
                    if answer:
                        st.write("### ✅ Answer:")
                        st.write(answer)
                        
                        # Show sources
                        if relevant_docs:
                            with st.expander("📚 See Sources"):
                                for i, doc in enumerate(relevant_docs, 1):
                                    st.write(f"**Source {i}:**")
                                    st.write(doc.page_content[:500] + "...")
                                    if hasattr(doc, 'metadata') and doc.metadata:
                                        st.caption(f"Metadata: {doc.metadata}")
                                    st.divider()
                
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
    
    else:
        st.info("👆 Please upload a PDF file to get started!")
    
    # Sidebar with info and cleanup
    with st.sidebar:
        st.header("📊 Status")
        if st.session_state.get('pdf_processed', False):
            st.success("✅ PDF Loaded")
            st.info(f"File: {st.session_state.get('current_file', 'Unknown')}")
            st.info(f"Document ID: {st.session_state.get('doc_id', 'Unknown')[:8]}...")
        else:
            st.warning("❌ No PDF loaded")
        
        st.header("🗑️ Cleanup")
        if st.button("Clear Current Document"):
            # Clear session state
            for key in ['doc_id', 'current_file', 'pdf_processed']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Document cleared!")
            st.rerun()
        
        st.header("ℹ️ Info")
        st.info("This app uses your existing Pinecone index 'docqa-project2' without creating new indexes.")

if __name__ == "__main__":
    main()