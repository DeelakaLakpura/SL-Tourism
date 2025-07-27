import os
import json
import hashlib
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pymongo
from pymongo import MongoClient
from bson import ObjectId

# LangChain imports
from langchain_community.vectorstores import FAISS, MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    CSVLoader, 
    JSONLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

# Custom imports
from config import Config

# Initialize MongoDB client
def get_mongo_client():
    return MongoClient(Config.MONGODB_URI)

class DocumentLoader:
    """Handles loading and preprocessing of various document types."""
    
    @staticmethod
    def load_documents(folder_path: str = "data/") -> List[Document]:
        """Load and process documents from the specified directory."""
        folder = Path(folder_path)
        all_docs = []
        
        # Create directory if it doesn't exist
        if not folder.exists():
            print(f"Warning: Directory '{folder.absolute()}' does not exist. Creating it...")
            try:
                folder.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {folder.absolute()}")
                return all_docs  # Return empty list since there are no documents to load yet
            except Exception as e:
                raise RuntimeError(f"Failed to create directory '{folder}': {str(e)}")
        
        # Check if directory is empty
        if not any(folder.iterdir()):
            print(f"Warning: Directory '{folder.absolute()}' is empty. No documents to load.")
            return all_docs
        
        # Configure text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        
        # Process each file in the directory
        for file_path in folder.glob("*"):
            try:
                if file_path.suffix.lower() == '.csv':
                    loader = CSVLoader(str(file_path))
                    docs = loader.load()
                elif file_path.suffix.lower() == '.json':
                    loader = JSONLoader(
                        file_path=str(file_path),
                        jq_schema='.',
                        text_content=False
                    )
                    docs = loader.load()
                elif file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                elif file_path.suffix.lower() in ['.md', '.markdown']:
                    loader = UnstructuredMarkdownLoader(str(file_path))
                    docs = loader.load()
                else:
                    continue
                
                # Split documents into chunks
                split_docs = text_splitter.split_documents(docs)
                
                # Add metadata
                for doc in split_docs:
                    doc.metadata.update({
                        'source': str(file_path.name),
                        'load_time': datetime.utcnow().isoformat(),
                        'doc_type': file_path.suffix.lower()
                    })
                
                all_docs.extend(split_docs)
                
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue
                
        return all_docs

class VectorStoreManager:
    """Manages vector store operations with caching and hybrid search capabilities."""
    
    def __init__(self, use_mongodb: bool = False):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        self.use_mongodb = use_mongodb
        self.vector_store = None
        self.cache = {}
        
        # Initialize cache
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    
    def create_vector_store(self, docs: List[Document] = None) -> Any:
        """Create or load a vector store."""
        if docs is None:
            try:
                docs = DocumentLoader.load_documents("data/")
                if not docs:
                    print("Warning: No documents found in the 'data/' directory. Using an empty vector store.")
                    # Create an empty list of documents if none found
                    docs = [Document(page_content="No documents found in the data directory.")]
            except Exception as e:
                print(f"Error loading documents: {str(e)}")
                # Create an empty list of documents in case of error
                docs = [Document(page_content="Error loading documents. Please check the data directory.")]
        
        try:
            if self.use_mongodb:
                return self._create_mongodb_store(docs)
            else:
                return self._create_faiss_store(docs)
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            raise
    
    def _create_faiss_store(self, docs: List[Document]) -> FAISS:
        """Create a FAISS vector store."""
        return FAISS.from_documents(docs, self.embeddings)
    
    def _create_mongodb_store(self, docs: List[Document]) -> MongoDBAtlasVectorSearch:
        """Create a MongoDB Atlas vector store."""
        client = get_mongo_client()
        collection = client[Config.DB_NAME][Config.VECTOR_COLLECTION]
        
        # Clear existing data if needed
        collection.delete_many({})
        
        # Create and return the vector store
        return MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection=collection,
            index_name="vector_index"
        )
    
    def get_retriever(self, search_type: str = "similarity", k: int = 5, **kwargs):
        """Get a retriever with the specified configuration."""
        if self.vector_store is None:
            self.vector_store = self.create_vector_store()
        
        # Configure search parameters
        search_kwargs = {"k": k, **kwargs}
        
        if search_type == "mmr":
            search_kwargs["fetch_k"] = min(20, k * 3)
        
        # Get base retriever
        retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        # Add contextual compression if needed
        if kwargs.get("use_compression", False):
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            llm = ChatGoogleGenerativeAI(
                model=Config.MODEL_NAME,  # Using the model name from Config
                temperature=0,
                google_api_key=os.getenv("GEMINI_API_KEY")
            )
            
            # Instead of using a custom compressor, we'll modify the retriever to ensure it returns Document objects
            from langchain_core.retrievers import BaseRetriever
            from langchain_core.documents import Document
            from typing import List, Dict, Any, Optional, Union
            from pydantic import Field
            
            class DocumentEnsuringRetriever(BaseRetriever):
                """Wrapper retriever that ensures Document objects are returned."""
                
                base_retriever: Any = Field(exclude=True)
                
                class Config:
                    arbitrary_types_allowed = True
                
                def __init__(self, base_retriever):
                    super().__init__(base_retriever=base_retriever)
                
                def _ensure_document(self, doc: Any, index: int = 0) -> Document:
                    """Convert any input to a Document object."""
                    if doc is None:
                        return Document(page_content="No content", metadata={"source": "none"})
                    if isinstance(doc, Document):
                        doc.metadata = getattr(doc, 'metadata', {})
                        return doc
                    if isinstance(doc, str):
                        return Document(page_content=doc, metadata={"source": "generated"})
                    if hasattr(doc, 'page_content'):
                        return Document(
                            page_content=doc.page_content,
                            metadata=getattr(doc, 'metadata', {"source": "converted"})
                        )
                    return Document(page_content=str(doc), metadata={"source": "converted"})
                
                def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
                    """Sync document retrieval with error handling."""
                    try:
                        if hasattr(self.base_retriever, 'invoke'):
                            result = self.base_retriever.invoke(query, **kwargs)
                        elif hasattr(self.base_retriever, 'get_relevant_documents'):
                            result = self.base_retriever.get_relevant_documents(query, **kwargs)
                        else:
                            raise ValueError("Unsupported retriever")
                        
                        if not isinstance(result, list):
                            result = [result]
                        return [self._ensure_document(doc, i) for i, doc in enumerate(result)]
                    except Exception as e:
                        print(f"Retrieval error: {str(e)}")
                        return []
                
                async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
                    """Async version of _get_relevant_documents."""
                    try:
                        # Try the async invoke method first
                        if hasattr(self.base_retriever, 'ainvoke'):
                            result = await self.base_retriever.ainvoke(query, **kwargs)
                        # Fall back to sync version if async not available
                        else:
                            result = self._get_relevant_documents(query, **kwargs)
                        
                        # Handle both single document and list of documents
                        if isinstance(result, list):
                            return [self._ensure_document(doc, i) for i, doc in enumerate(result)]
                        else:
                            return [self._ensure_document(result)]
                            
                    except Exception as e:
                        print(f"Error in _aget_relevant_documents: {str(e)}")
                        return []
            
            # Create a simple compressor without custom logic
            compressor = LLMChainExtractor.from_llm(llm)
            
            # First create the compression retriever
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )
            
            # Then wrap it with our document-ensuring retriever
            retriever = DocumentEnsuringRetriever(compression_retriever)
        
        return retriever

def create_vector_store(use_mongodb: bool = False):
    """Create and return a vector store instance."""
    manager = VectorStoreManager(use_mongodb=use_mongodb)
    return manager.create_vector_store()

def get_vector_manager() -> VectorStoreManager:
    """Get a vector store manager instance."""
    return VectorStoreManager()
