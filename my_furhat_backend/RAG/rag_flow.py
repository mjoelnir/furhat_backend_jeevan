"""
RAG (Retrieval-Augmented Generation) Module

This module implements a sophisticated document retrieval system using Chroma vector store.
It provides functionality for loading, chunking, and retrieving documents with semantic search
and reranking capabilities.

Key Features:
    - Document loading and chunking
    - Vector store management with Chroma
    - Semantic search with reranking
    - GPU-accelerated embeddings
    - Persistent storage
"""

import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import LlamaCppEmbeddings
from my_furhat_backend.config.settings import config
from my_furhat_backend.utils.gpu_utils import print_gpu_status, clear_gpu_cache
from typing import List

# Set up logging configuration for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAG:
    """
    Manages Retrieval-Augmented Generation (RAG) tasks using a Chroma vector store.
    
    This class provides a comprehensive interface for:
    - Loading and chunking documents (PDFs)
    - Managing a persistent Chroma vector store
    - Performing semantic search with reranking
    - GPU-accelerated document processing
    
    Attributes:
        hf (bool): Whether to use HuggingFace embeddings
        persist_directory (str): Path to store the vector database
        path_to_document (str): Path to the source document
        vector_store (Chroma): The vector store instance
        documents (List[Document]): Loaded document chunks
        embeddings: The embedding model instance
    """

    def __init__(self, hf: bool = True, persist_directory: str = None, path_to_document: str = None):
        """
        Initialize the RAG system.
        
        Args:
            hf (bool): Whether to use HuggingFace embeddings
            persist_directory (str): Path to store the vector database
            path_to_document (str): Path to the source document
        """
        print_gpu_status()
        
        self.hf = hf
        self.persist_directory = persist_directory or config["VECTOR_STORE_PATH"]
        self.path_to_document = path_to_document
        
        logger.info(f"Initializing RAG with:")
        logger.info(f"- HF embeddings: {hf}")
        logger.info(f"- Persist directory: {self.persist_directory}")
        logger.info(f"- Document path: {self.path_to_document}")
        
        # Initialize embeddings first
        logger.info("Initializing embeddings...")
        self.embeddings = self._initialize_embeddings()
        
        # Then initialize vector store with the embeddings
        logger.info("Initializing vector store...")
        self.vector_store = self._initialize_vector_store()
        
        # Load documents and populate vector store if needed
        logger.info("Loading documents...")
        self.documents = self.__load_docs()
        if self.documents:
            logger.info(f"Found {len(self.documents)} documents, populating vector store...")
            self.__populate_chroma(self.vector_store)
        else:
            logger.warning("No documents loaded, vector store will be empty")
        
        print_gpu_status()
        
    def _initialize_embeddings(self):
        """
        Initialize the embedding model.
        
        Returns:
            The initialized embedding model
        """
        if self.hf:
            # Auto-detect device: prefer MPS (macOS), then CUDA, then CPU
            import torch
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
            logger.info(f"Using device: {device} for embeddings")
            
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            return LlamaCppEmbeddings(
                model_path=os.path.join(config["GGUF_MODELS_PATH"], "all-MiniLM-L6-v2-Q4_K_M.gguf"),
                n_ctx=2048,
                n_batch=512,
                n_gpu_layers=32
            )
        
    def _initialize_vector_store(self) -> Chroma:
        """
        Initialize the vector store with appropriate settings.
        
        Returns:
            Chroma: Initialized vector store
        """
        if os.path.exists(self.persist_directory):
            logger.info("Loading existing vector store...")
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            logger.info("Creating new vector store...")
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
    
    def __load_docs(self) -> List[Document]:
        """
        Load documents from a PDF file.
        
        Returns:
            List[Document]: List of Document objects loaded from the file
        """
        try:
            logger.info(f"Attempting to load document from: {self.path_to_document}")
            logger.info(f"File exists: {os.path.exists(self.path_to_document)}")
            if os.path.exists(self.path_to_document):
                logger.info(f"File size: {os.path.getsize(self.path_to_document)} bytes")
                logger.info(f"File permissions: {oct(os.stat(self.path_to_document).st_mode)[-3:]}")
            
            if not os.path.exists(self.path_to_document):
                logger.error(f"Document file does not exist at: {self.path_to_document}")
                return []
                
            loader = PyPDFLoader(self.path_to_document)
            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} document(s) from {self.path_to_document}")
            for i, doc in enumerate(docs):
                logger.info(f"Document {i+1} length: {len(doc.page_content)} characters")
                logger.info(f"Document {i+1} metadata: {doc.metadata}")
            return docs
        except Exception as e:
            logger.error(f"Error loading documents from {self.path_to_document}: {e}")
            logger.error(f"Full error details: {str(e)}")
            return []
    
    def __load_and_chunk_docs(self) -> List[Document]:
        """
        Load documents and split them into chunks.
        
        Returns:
            List[Document]: List of document chunks
        """
        docs = self.__load_docs()
        if not docs:
            logger.error("No documents loaded; cannot perform chunking.")
            return []
            
        logger.info("Starting document chunking...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Successfully split documents into {len(chunks)} chunk(s)")
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i+1} length: {len(chunk.page_content)} characters")
        return chunks
    
    def __populate_chroma(self, vector_store: Chroma) -> None:
        """
        Populate the Chroma vector store with document chunks.
        
        Args:
            vector_store (Chroma): The vector store to populate
        """
        chunks = self.__load_and_chunk_docs()
        if chunks:
            logger.info(f"Adding {len(chunks)} chunks to vector store...")
            try:
                _ = vector_store.add_documents(documents=chunks)
                logger.info("Successfully populated vector store with document chunks")
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {e}")
        else:
            logger.warning("No document chunks available to populate the vector store.")
        
    def __create_and_populate_chroma(self) -> Chroma:
        """
        Create or load a Chroma vector store and populate it with documents.
        
        Returns:
            Chroma: The initialized vector store
        """
        if os.path.exists(self.persist_directory):
            logger.info("Persist directory exists. Loading existing vector store.")
            return self.__load_db()
        else:
            logger.info("Creating new Chroma vector store.")
            vector_store = Chroma.from_documents(
                self.__load_and_chunk_docs(),
                self.embeddings,
                persist_directory=self.persist_directory
            )
            return vector_store
    
    def __load_db(self) -> Chroma:
        """
        Load an existing Chroma vector store.
        
        Returns:
            Chroma: The loaded vector store
        """
        logger.info(f"Loading vector store from {self.persist_directory}.")
        return Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
    
    def add_docs_to_db(self, path_to_docs: str) -> None:
        """
        Add new documents to the existing vector store.
        
        Args:
            path_to_docs (str): Path to the new document(s)
        """
        self.path_to_document = path_to_docs
        self.__populate_chroma(self.vector_store)
    
    def retrieve_similar(self, query_text: str, top_n: int = 5, search_kwargs: int = 20, rerank: bool = True) -> List[Document]:
        """
        Retrieve document chunks similar to the query text.
        
        Args:
            query_text (str): The query string
            top_n (int): Number of top documents to return after reranking
            search_kwargs (int): Number of documents to retrieve initially
            rerank (bool): Whether to rerank results using cross-encoder
            
        Returns:
            List[Document]: List of relevant document chunks
        """
        if rerank:
            logger.info("Reranking documents...")
            return self.__rerank(query_text, top_n=top_n, search_kwargs=search_kwargs)
        return self.vector_store.similarity_search(query_text)
    
    def get_document_context(self, document: str) -> str:
        """
        Retrieve the context of a specific document.
        
        Args:
            document (str): Name or identifier of the document
            
        Returns:
            str: The document's context and main themes
        """
        prompt = (
            "Extract the overarching context and main themes from the document titled "
            f"'{document}'. Focus on summarizing the key topics, narrative, and any major findings "
            "without including extraneous details."
        )
        return self.retrieve_similar(prompt)
    
    def __rerank(self, prompt: str, top_n: int = 5, search_kwargs: int = 20) -> List[Document]:
        """
        Rerank retrieved documents using a cross-encoder.
        
        Args:
            prompt (str): The query text
            top_n (int): Number of top documents to return
            search_kwargs (int): Number of documents to retrieve initially
            
        Returns:
            List[Document]: Reranked document chunks
        """
        model = HuggingFaceCrossEncoder(model_name="mixedbread-ai/mxbai-rerank-base-v1")
        compressor = CrossEncoderReranker(model=model, top_n=top_n)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.vector_store.as_retriever(search_kwargs={"k": search_kwargs})
        )
        return compression_retriever.invoke(prompt)

    def __del__(self):
        """Cleanup method to clear GPU cache when the RAG instance is destroyed."""
        clear_gpu_cache()

    def get_list_docs(self) -> List[str]:
        """
        Get a list of all available documents in the vector store.
        
        Returns:
            List[str]: List of document names/identifiers
        """
        try:
            # Get all documents from the vector store
            if not self.vector_store:
                self.vector_store = self.__load_db()
            
            # Get unique document names from metadata
            doc_names = set()
            for doc in self.vector_store.get()["metadatas"]:
                if "source" in doc:
                    doc_name = os.path.basename(doc["source"])
                    doc_names.add(doc_name)
            
            return list(doc_names)
            
        except Exception as e:
            logger.error(f"Error getting document list: {e}")
            return []
