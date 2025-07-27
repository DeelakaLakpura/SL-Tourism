import os
import json
import base64
from io import BytesIO
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
import google.generativeai as genai
import tempfile
import asyncio
import nest_asyncio
import re

# Custom imports
from config import Config
from flight_service import get_flight_service

# Core LangChain imports
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import format_document
from langchain_core.documents import Document

# Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

# Memory and History
from langchain.memory import (
    ConversationBufferMemory,
    FileChatMessageHistory,
    ConversationSummaryMemory
)
from langchain.memory.vectorstore import VectorStoreRetrieverMemory

# Chains and Retrievers
from langchain.schema import BaseRetriever
from langchain.chains import (
    ConversationalRetrievalChain,
    RetrievalQA,
    LLMChain
)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain

# Tools and Utilities
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Custom imports
from dotenv import load_dotenv
from vector_store import get_vector_manager, VectorStoreManager

# Load environment variables
load_dotenv()

# Configuration
class Config:
    # Model Configuration
    # Using the latest supported Gemini models
    MODEL_NAME = "gemini-1.5-flash"  # Default model to use
    EMBEDDING_MODEL = "models/embedding-001"  # Using the standard embedding model
    
    # Generation parameters for the model
    GENERATION_CONFIG = {
        "temperature": 0.7,  # Controls randomness in the response (0.0 to 1.0)
        "top_p": 0.95,       # Nucleus sampling parameter
        "top_k": 40,         # Limits the number of highest probability tokens to consider
        "max_output_tokens": 2048,  # Maximum length of the generated response
    }

    # Memory Configuration
    MAX_TOKENS = 4000
    MEMORY_FILE = "chat_history.json"

    # Retrieval Configuration
    MAX_SOURCE_DOCS = 5
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Conversation Settings
    SYSTEM_PROMPT = """
    You are a knowledgeable and friendly Sri Lanka Tourism Assistant. Your goal is to provide 
    accurate, helpful, and engaging information about Sri Lanka's tourism attractions, 
    culture, history, accommodations, travel tips, and flight information.
    
    When responding:
    1. Be informative but concise
    2. Include relevant cultural context
    3. Provide practical tips when appropriate
    4. Be enthusiastic about Sri Lanka's offerings
    5. If you don't know something, say so and offer to help with related information
    
    For flight-related queries, you can help with:
    - Flight schedules and timetables
    - Flight status and tracking
    - Airport information
    - Airline information
    - General flight booking inquiries
    
    Always respond in a warm, welcoming tone that reflects Sri Lankan hospitality.
    """

class Chatbot:
    def __init__(self):
        # Initialize components
        self.llm = self._init_llm()
        self.vector_manager = get_vector_manager()
        self.vector_manager.vector_store = self.vector_manager.create_vector_store()  # Properly assign vector_store
        self.retriever = self._init_retriever()
        self.memory = self._init_memory()
        self.qa_chain = self._init_qa_chain()
        
        # Initialize the vision model for image analysis
        self.vision_model = genai.GenerativeModel('gemini-pro-vision')
        
        # Initialize flight service with API key from environment
        self.flight_service = get_flight_service()  # Will use AVIATIONSTACK_API_KEY from .env

    def _init_llm(self):
        """Initialize the language model with streaming support."""
        try:
            print("Initializing LLM...")
            
            # Ensure the API key is set
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set in the environment variables")
            
            print(f"Using model: {Config.MODEL_NAME}")
            print(f"API Key: {'Set' if api_key else 'Not set'}")
            
            # For older versions, we need to use the model name without the 'models/' prefix
            # and set the model name in the client config
            import google.generativeai as genai
            
            try:
                # Configure the API
                genai.configure(api_key=api_key)
                print("Successfully configured Google Generative AI API")
                
                # List available models and filter for Gemini models
                models = genai.list_models()
                gemini_models = [m for m in models if 'gemini' in m.name.lower()]
                
                # Print available Gemini models for debugging
                print("\nAvailable Gemini models:")
                for model in gemini_models:
                    print(f"- {model.name} (Supports: {', '.join(model.supported_generation_methods)})")
                
                # Try to find a suitable model - prioritize gemini-1.5-flash or gemini-1.5-pro
                model_name = None
                for name in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
                    if any(name in m.name for m in gemini_models):
                        model_name = name
                        break
                
                if not model_name and gemini_models:
                    # Fallback to first available Gemini model if preferred ones not found
                    model_name = gemini_models[0].name.split('/')[-1]  # Extract just the model name
                
                if not model_name:
                    raise ValueError("No compatible Gemini models found. Please check your API access.")
                
                print(f"\nUsing model: {model_name}")
                
                # Initialize the model with the configuration from Config class
                llm = ChatGoogleGenerativeAI(
                    model=model_name,  # Using just the model name, not the full path
                    google_api_key=api_key,
                    # Use generation parameters from Config
                    temperature=Config.GENERATION_CONFIG["temperature"],
                    top_p=Config.GENERATION_CONFIG["top_p"],
                    top_k=Config.GENERATION_CONFIG["top_k"],
                    max_output_tokens=Config.GENERATION_CONFIG["max_output_tokens"],
                    streaming=True,
                    callbacks=[StreamingStdOutCallbackHandler()],
                    convert_system_message_to_human=True,  # Convert system messages to human messages
                )
                
                print("Successfully initialized ChatGoogleGenerativeAI")
                return llm
                
            except Exception as e:
                print(f"Error during Google Generative AI initialization: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
                
                print("Successfully initialized ChatGoogleGenerativeAI")
                return llm
                
            except Exception as e:
                print(f"Error during Google Generative AI initialization: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            print(f"Model name: {Config.MODEL_NAME}")
            print(f"API Key: {'Set' if os.getenv('GEMINI_API_KEY') else 'Not set'}")
            # Print more detailed error information
            import traceback
            traceback.print_exc()
            raise

    def _init_retriever(self):
        """Initialize the document retriever with hybrid search."""
        print("[_init_retriever] Initializing document retriever...")
        try:
            print(f"[_init_retriever] Using vector manager: {type(self.vector_manager).__name__}")
            print(f"[_init_retriever] Vector store: {type(getattr(self.vector_manager, 'vector_store', None)).__name__ if hasattr(self.vector_manager, 'vector_store') else 'Not found'}")
            
            retriever = self.vector_manager.get_retriever(
                search_type="mmr",
                k=Config.MAX_SOURCE_DOCS,
                fetch_k=min(20, Config.MAX_SOURCE_DOCS * 3),
                use_compression=True
            )
            
            print(f"[_init_retriever] Retriever initialized: {type(retriever).__name__}")
            return retriever
            
        except Exception as e:
            print(f"[_init_retriever] Error initializing retriever: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _init_memory(self):
        """Initialize conversation memory with persistent storage."""
        print("[_init_memory] Initializing conversation memory...")
        try:
            # Ensure directory exists only if needed
            memory_file = Config.MEMORY_FILE
            memory_dir = os.path.dirname(memory_file)
            print(f"[_init_memory] Memory file path: {os.path.abspath(memory_file)}")
            
            if memory_dir:
                print(f"[_init_memory] Creating memory directory if it doesn't exist: {memory_dir}")
                os.makedirs(memory_dir, exist_ok=True)

            # Initialize chat history memory
            print("[_init_memory] Initializing ConversationBufferMemory...")
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                chat_memory=FileChatMessageHistory(memory_file)
            )
            print("[_init_memory] ConversationBufferMemory initialized")

            # Add summary memory for long-term context
            print("[_init_memory] Initializing ConversationSummaryMemory...")
            summary_memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key="summary",
                input_key="question"
            )
            print("[_init_memory] ConversationSummaryMemory initialized")

            # Combine memories
            print("[_init_memory] Combining memories...")
            memory.chat_memory.messages.extend(summary_memory.chat_memory.messages)
            print(f"[_init_memory] Combined {len(memory.chat_memory.messages)} messages from summary memory")
            
            print("[_init_memory] Memory initialization complete")
            return memory
            
        except Exception as e:
            print(f"[_init_memory] Error initializing memory: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _init_qa_chain(self):
        """Initialize the question-answering chain with retrieval and memory."""
        print("[_init_qa_chain] Initializing QA chain...")
        
        try:
            # Define a document formatter that ensures we have Document objects
            def format_document(doc):
                if isinstance(doc, str):
                    return Document(page_content=doc, metadata={"source": "generated"})
                elif hasattr(doc, 'page_content'):
                    # If it's already a Document, ensure it has metadata
                    if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                        doc.metadata = getattr(doc, 'metadata', {})
                    return doc
                else:
                    return Document(page_content=str(doc), metadata={"source": "converted"})
            
            # Define the prompt template for the document chain
            print("[_init_qa_chain] Creating document prompt template...")
            document_prompt = PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            )
            
            # Define the prompt template for the QA chain
            print("[_init_qa_chain] Creating QA prompt template...")
            
            # Use MessagesPlaceholder for chat history
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", Config.SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Context: {context}\n\nQuestion: {question}")
            ])
            print("[_init_qa_chain] QA prompt template created")

            # Create a custom document chain to ensure proper document handling
            class SafeStuffDocumentsChain(StuffDocumentsChain):
                def _get_inputs(self, docs, **kwargs):
                    # Ensure all docs are Document objects with proper attributes
                    formatted_docs = []
                    for doc in docs:
                        if isinstance(doc, str):
                            formatted_docs.append(Document(page_content=doc, metadata={"source": "generated"}))
                        elif hasattr(doc, 'page_content'):
                            if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                                doc.metadata = {}
                            formatted_docs.append(doc)
                        else:
                            formatted_docs.append(Document(page_content=str(doc), metadata={"source": "converted"}))
                    return super()._get_inputs(formatted_docs, **kwargs)
                
                async def _acombine_docs(self, docs, **kwargs):
                    # Ensure all docs are properly formatted before combining
                    formatted_docs = []
                    for doc in docs:
                        if isinstance(doc, str):
                            formatted_docs.append(Document(page_content=doc, metadata={"source": "generated"}))
                        elif hasattr(doc, 'page_content'):
                            if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                                doc.metadata = {}
                            formatted_docs.append(doc)
                        else:
                            formatted_docs.append(Document(page_content=str(doc), metadata={"source": "converted"}))
                    return await super()._acombine_docs(formatted_docs, **kwargs)

            # Create the document chain with proper input variables
            print("[_init_qa_chain] Creating document chain...")
            
            # First create the LLM chain that will be used by our custom chain
            llm_chain = LLMChain(
                llm=self.llm,
                prompt=qa_prompt,
                verbose=True
            )
            
            # Create our custom document chain directly
            document_chain = SafeStuffDocumentsChain(
                llm_chain=llm_chain,
                document_variable_name="context",
                document_prompt=document_prompt,
                verbose=True
            )
            
            # Create the final QA chain with our custom document chain
            document_chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_prompt=document_prompt,
                document_variable_name="context",
                verbose=True
            )
                
            print("[_init_qa_chain] Document chain created with safe document handling")

            # Create a question generator chain
            print("[_init_qa_chain] Creating question generator chain...")
            question_generator = LLMChain(
                llm=self.llm,
                prompt=ChatPromptTemplate.from_template(
                    "Given the following conversation and a follow up question, "
                    "rephrase the follow up question to be a standalone question.\n\n"
                    "Chat History:\n{chat_history}\n"
                    "Follow Up Input: {question}\n"
                    "Standalone question:"
                ),
                verbose=True
            )
            print("[_init_qa_chain] Question generator chain created")

            # Create a custom retriever chain to ensure proper document handling
            class SafeRetrieverChain(ConversationalRetrievalChain):
                async def _aget_docs(self, question, inputs):
                    try:
                        # Get documents from the retriever
                        docs = await self.retriever.aget_relevant_documents(question)
                        # Ensure all docs are Document objects
                        return [format_document(doc) for doc in docs]
                    except Exception as e:
                        print(f"[SafeRetrieverChain] Error getting documents: {str(e)}")
                        return []

            # Create the QA chain with retrieval and memory
            print("[_init_qa_chain] Creating ConversationalRetrievalChain...")
            print(f"[_init_qa_chain] Retriever type: {type(self.retriever).__name__}")
            print(f"[_init_qa_chain] Memory type: {type(self.memory).__name__}")
            
            # Ensure the retriever returns proper document objects
            async def safe_retriever(query):
                try:
                    docs = await self.retriever.aget_relevant_documents(query)
                    # Ensure all docs are properly formatted
                    formatted_docs = []
                    for doc in docs:
                        if isinstance(doc, str):
                            formatted_docs.append(Document(page_content=doc, metadata={"source": "generated"}))
                        elif hasattr(doc, 'page_content'):
                            if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                                doc.metadata = {}
                            formatted_docs.append(doc)
                        else:
                            formatted_docs.append(Document(page_content=str(doc), metadata={"source": "converted"}))
                    return formatted_docs
                except Exception as e:
                    print(f"[safe_retriever] Error in retriever: {str(e)}")
                    return []
            
            # Create a custom retriever class
            class CustomRetriever(BaseRetriever):
                async def _aget_relevant_documents(self, query: str, **kwargs):
                    return await safe_retriever(query)
                
                def _get_relevant_documents(self, query: str, **kwargs):
                    raise RuntimeError("_get_relevant_documents cannot be called from within an event loop. Use _aget_relevant_documents instead.")
            
            # Create the QA chain with our custom retriever and document chain
            qa_chain = ConversationalRetrievalChain(
                question_generator=question_generator,
                retriever=CustomRetriever(),
                memory=self.memory,
                combine_docs_chain=document_chain,
                return_source_documents=True,
                output_key="answer",
                verbose=True,
                get_chat_history=lambda h: h,  # Pass through the chat history
                return_generated_question=True,
                rephrase_question=True,  # Let the model rephrase the question if needed
                max_tokens_limit=4000  # Set a token limit to prevent context overflow
            )
            
            print("[_init_qa_chain] QA chain created successfully")
            return qa_chain
            
        except Exception as e:
            print(f"[_init_qa_chain] Error creating QA chain: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _process_image(self, image_data: str) -> Image.Image:
        """Process base64 encoded image data into a PIL Image."""
        try:
            # Remove the data URL prefix if present
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            
            # Decode the base64 data
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            return image.convert("RGB")
        except Exception as e:
            print(f"[_process_image] Error processing image: {str(e)}")
            raise ValueError("Invalid image data provided")
    
    async def _analyze_image(self, image: Image.Image, prompt: str = None) -> str:
        """Analyze an image using the vision model."""
        try:
            if prompt is None:
                prompt = """Analyze this image and provide a detailed description. 
                If this appears to be a plant, identify any visible diseases or issues. 
                Provide care recommendations if applicable."""
            
            response = await self.vision_model.generate_content_async([prompt, image])
            return response.text
        except Exception as e:
            print(f"[_analyze_image] Error analyzing image: {str(e)}")
            return f"I couldn't analyze this image. Error: {str(e)}"

    async def _handle_flight_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Handle flight-related queries using the flight service.
        
        Args:
            query: The user's query about flights
            
        Returns:
            Dict with flight information if it's a flight query, None otherwise
        """
        # Check if the query is about flights
        flight_keywords = [
            'flight', 'flights', 'airline', 'airport', 'schedule', 'timetable',
            'departure', 'arrival', 'airplane', 'aircraft', 'fly', 'flying',
            'plane', 'airline'
        ]
        
        query_lower = query.lower()
        is_flight_query = any(keyword in query_lower for keyword in flight_keywords)
        
        if not is_flight_query:
            return None
            
        try:
            # Use the flight service to search for flights
            result = self.flight_service.search_flights(query)
            
            # Format the response
            if 'error' in result:
                return {
                    'answer': f"I couldn't find any flight information. {result['error']}",
                    'is_flight_info': False,
                    'sources': []
                }
                
            if 'data' in result and result['data']:
                flights = result['data'][:5]  # Limit to 5 flights
                response = ["Here are some flight options:\n"]
                
                for i, flight in enumerate(flights, 1):
                    dep = flight.get('departure', {})
                    arr = flight.get('arrival', {})
                    airline = flight.get('airline', {}).get('name', 'Unknown Airline')
                    flight_num = flight.get('flight', {}).get('iata', 'Unknown')
                    
                    response.append(
                        f"{i}. {airline} {flight_num}\n"
                        f"   From: {dep.get('airport', 'Unknown')} ({dep.get('iata', '?')})\n"
                        f"   To: {arr.get('airport', 'Unknown')} ({arr.get('iata', '?')})\n"
                        f"   Departure: {dep.get('scheduled', 'TBD')}\n"
                        f"   Arrival: {arr.get('scheduled', 'TBD')}\n"
                        f"   Status: {flight.get('flight_status', 'unknown').title()}\n"
                    )
                
                return {
                    'answer': '\n'.join(response),
                    'is_flight_info': True,
                    'sources': [],
                    'flight_data': result
                }
            else:
                return {
                    'answer': "I couldn't find any flights matching your query. "
                               "Please try being more specific with your request.",
                    'is_flight_info': False,
                    'sources': []
                }
                
        except Exception as e:
            print(f"Error processing flight query: {e}")
            return {
                'answer': "I encountered an error while processing your flight request. "
                           "Please try again later.",
                'is_flight_info': False,
                'sources': []
            }

    async def _aprocess_query(self, query: str, image_data: str = None) -> Dict[str, Any]:
        """Async helper to process a user query with optional image data.
        
        Args:
            query: The user's text query
            image_data: Optional base64 encoded image data
            
        Returns:
            Dict containing the response and metadata
        """
        print(f"[_aprocess_query] Starting to process query: {query}")
        print(f"[_aprocess_query] Image data: {'Provided' if image_data else 'Not provided'}")
        
        # First, check if this is a flight-related query
        flight_response = await self._handle_flight_query(query)
        if flight_response:
            print("[_aprocess_query] Processed as flight query")
            return flight_response
            
        try:
            # Handle image analysis if image is provided
            if image_data:
                try:
                    image = self._process_image(image_data)
                    analysis = await self._analyze_image(image, query if query else None)
                    
                    # If there was no text query, just return the analysis
                    if not query or query.strip() == "":
                        return {
                            "answer": analysis,
                            "sources": [],
                            "is_image_analysis": True
                        }
                    
                    # If there was a query, combine it with the image analysis
                    query = f"{query}\n\nHere's what I see in the image:\n{analysis}"
                    
                except Exception as e:
                    print(f"[_aprocess_query] Error processing image: {str(e)}")
                    return {
                        "answer": "I had trouble processing the image. Please try again with a different image.",
                        "sources": [],
                        "error": True,
                        "debug": str(e)
                    }
            
            # Verify memory is initialized
            print("[_aprocess_query] Verifying memory initialization...")
            if not hasattr(self, 'memory') or not hasattr(self.memory, 'chat_memory'):
                error_msg = "Memory not properly initialized"
                print(f"[_aprocess_query] ERROR: {error_msg}")
                raise ValueError(error_msg)
            print("[_aprocess_query] Memory verified")
            
            # Prepare the input for the QA chain
            print("[_aprocess_query] Preparing chat history...")
            chat_history = getattr(self.memory.chat_memory, 'messages', [])
            print(f"[_aprocess_query] Chat history length: {len(chat_history)}")
            
            # Format chat history as a list of (role, content) tuples
            formatted_history = []
            for msg in chat_history:
                role = msg.type
                content = msg.content
                formatted_history.append((role, content))
            
            inputs = {
                "question": query,
                "chat_history": formatted_history
            }
            print("[_aprocess_query] Inputs prepared:", {
                "question": query[:100] + ("..." if len(query) > 100 else ""),
                "chat_history_length": len(chat_history)
            })

            # Verify QA chain is initialized
            print("[_aprocess_query] Verifying QA chain initialization...")
            if not hasattr(self, 'qa_chain'):
                error_msg = "QA chain not properly initialized"
                print(f"[_aprocess_query] ERROR: {error_msg}")
                raise ValueError(error_msg)
            print("[_aprocess_query] QA chain verified")

            # Process the query through the QA chain asynchronously
            print("[_aprocess_query] Invoking QA chain...")
            try:
                result = await self.qa_chain.ainvoke(inputs)
                print("[_aprocess_query] QA chain invocation successful")
                print(f"[_aprocess_query] Result keys: {list(result.keys())}")
                
                # Log the answer and source documents
                answer = result.get("answer", "No answer generated")
                source_docs = result.get("source_documents", [])
                print(f"[_aprocess_query] Answer length: {len(answer) if answer else 0} characters")
                print(f"[_aprocess_query] Source documents found: {len(source_docs)}")
                
                # Log first 200 chars of answer for debugging
                print(f"[_aprocess_query] Answer preview: {answer[:200]}...")
                
                # Log source document previews
                for i, doc in enumerate(source_docs):
                    content = getattr(doc, 'page_content', str(doc))
                    metadata = getattr(doc, 'metadata', {})
                    print(f"[_aprocess_query] Source doc {i+1}:")
                    print(f"  Content preview: {content[:200]}...")
                    print(f"  Metadata: {metadata}")
                
            except Exception as e:
                error_msg = str(e)
                print(f"[_aprocess_query] ERROR during QA chain invocation: {error_msg}")
                import traceback
                traceback.print_exc()
                
                # Return a helpful error message while preserving chat history
                return {
                    "answer": "I apologize, but I encountered an issue processing your request. Let me try that again. " \
                             "Could you please rephrase your question?",
                    "sources": [],
                    "error": True,
                    "debug": error_msg
                }

            # Format the response with enhanced error handling
            print("[_aprocess_query] Formatting response...")
            try:
                answer = result.get("answer", "I'm sorry, I couldn't generate a response. Please try again.")
                source_docs = result.get("source_documents", [])
                
                # Safely extract content and metadata from source documents
                sources = []
                for doc in source_docs:
                    try:
                        content = getattr(doc, 'page_content', str(doc))
                        metadata = getattr(doc, 'metadata', {})
                        sources.append({
                            "content": content,
                            "metadata": metadata
                        })
                    except Exception as doc_error:
                        print(f"[_aprocess_query] Error processing source document: {str(doc_error)}")
                        sources.append({
                            "content": "[Error processing document content]",
                            "metadata": {}
                        })
                
                response = {
                    "answer": answer,
                    "sources": sources,
                    "context_used": bool(source_docs)
                }
                
                print(f"[_aprocess_query] Response formatted successfully. Answer length: {len(answer)}")
                print(f"[_aprocess_query] Number of sources included: {len(sources)}")
                print(f"[_aprocess_query] Context used: {response['context_used']}")
                
            except Exception as format_error:
                print(f"[_aprocess_query] ERROR formatting response: {str(format_error)}")
                import traceback
                traceback.print_exc()
                
                # Fallback response in case of formatting errors
                response = {
                    "answer": "I'm sorry, I encountered an error processing your request. The error has been logged.",
                    "sources": [],
                    "context_used": False
                }

            # Update the chat history with the response
            print("[_aprocess_query] Updating chat history with response...")
            try:
                self.memory.save_context(
                    {"question": query},
                    {"answer": response["answer"]}
                )
                print("[_aprocess_query] Chat history updated successfully")
            except Exception as mem_error:
                print(f"[_aprocess_query] ERROR updating chat history: {str(mem_error)}")
                import traceback
                traceback.print_exc()
                # Continue with the response even if memory update fails
            
            print("[_aprocess_query] Query processing completed successfully")
            return response

        except Exception as e:
            import traceback
            error_msg = f"I encountered an error: {str(e)}\n\n{traceback.format_exc()}"
            return {
                "answer": "I'm sorry, I encountered an error processing your request. Please try again.",
                "sources": [],
                "error": True,
                "debug": str(e)
            }

    async def process_query_async(self, query: str, image_data: str = None) -> Dict[str, Any]:
        """Process a user query with optional image data asynchronously.
        
        Args:
            query: The user's text query
            image_data: Optional base64 encoded image data (with data URL prefix)
            
        Returns:
            Dict containing the response, sources, and metadata
        """
        print(f"[process_query_async] Starting to process query")
        print(f"[process_query_async] Query: {query}")
        print(f"[process_query_async] Image data: {'Provided' if image_data else 'Not provided'}")
        
        try:
            # Process the query using the async method
            response = await self._aprocess_query(query, image_data)
            print(f"[process_query_async] Query processed successfully. Response length: {len(str(response))}")
            return response
            
        except asyncio.CancelledError:
            print("[process_query_async] Query was cancelled")
            return {
                "answer": "The request was cancelled. Please try again.",
                "sources": [],
                "error": True
            }
            
        except Exception as e:
            import traceback
            error_msg = f"Error in process_query_async: {str(e)}\n\n{traceback.format_exc()}"
            print(f"[process_query_async] {error_msg}")
            return {
                "answer": "I'm sorry, I encountered an error processing your request. Please try again.",
                "sources": [],
                "error": True,
                "debug": str(e)
            }
    
    def process_query(self, query: str, image_data: str = None) -> Dict[str, Any]:
        """Synchronous wrapper for process_query_async.
        
        Args:
            query: The user's text query
            image_data: Optional base64 encoded image data (with data URL prefix)
            
        Returns:
            Dict containing the response, sources, and metadata
        """
        from async_utils import run_async
        try:
            return run_async(self.process_query_async(query, image_data), timeout=120)
        except Exception as e:
            import traceback
            print(f"[process_query] Error: {e}\n{traceback.format_exc()}")
            return {
                "answer": "I'm sorry, I encountered an error processing your request. Please try again.",
                "sources": [],
                "error": True,
                "debug": str(e)
            }

    def clear_memory(self) -> None:
        """Clear the conversation history."""
        self.memory.clear()
        if os.path.exists(Config.MEMORY_FILE):
            try:
                os.remove(Config.MEMORY_FILE)
            except Exception as e:
                print(f"[clear_memory] Error removing memory file: {str(e)}")

# Singleton instance
_chatbot_instance = None

def get_chatbot():
    """Get or create the chatbot instance."""
    global _chatbot_instance
    if _chatbot_instance is None:
        print("[get_chatbot] Creating new Chatbot instance")
        try:
            # Configure the Gemini API
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
                
            genai.configure(api_key=api_key)
            _chatbot_instance = Chatbot()
            print("[get_chatbot] Chatbot instance created successfully")
        except Exception as e:
            print(f"[get_chatbot] Error initializing chatbot: {str(e)}")
            raise
    return _chatbot_instance

def clear_chat_history() -> None:
    """Clear the chat history."""
    chatbot = get_chatbot()
    chatbot.clear_memory()
