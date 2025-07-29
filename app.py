import os
import json
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Optional
import base64
import requests
from PIL import Image
import io
import threading
import asyncio
import concurrent.futures
import grpc

from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.web.server.websocket_headers import _get_websocket_headers
# Custom imports
from ai_model import get_chatbot, clear_chat_history
from vector_store import get_vector_manager

# Page configuration
st.set_page_config(
    page_title="Sri Lanka AI Travel Guide",
    page_icon="🇱🇰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def load_css():
    st.markdown("""
    <style>
        /* Main container */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Chat container */
        .chat-container {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
            height: 70vh;
            overflow-y: auto;
        }
        
        /* Message bubbles */
        .message {
            margin: 10px 0;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 80%;
            word-wrap: break-word;
        }
        
        .user-message {
            background-color: #007bff;
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        
        .bot-message {
            background-color: #f1f3f5;
            margin-right: auto;
            border-bottom-left-radius: 4px;
        }
        
        /* Input area */
        .stTextInput>div>div>input {
            border-radius: 20px !important;
            padding: 10px 20px !important;
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 20px;
            padding: 8px 20px;
            font-weight: 500;
        }
        
        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        
        /* Source documents */
        .source-doc {
            font-size: 0.85em;
            color: #6c757d;
            border-left: 3px solid #dee2e6;
            padding-left: 10px;
            margin: 5px 0;
        }
    </style>
    """, unsafe_allow_html=True)

# Save session state to URL
def save_session_to_url():
    session_data = {
        'messages': st.session_state.get('messages', []),
        'sources': st.session_state.get('sources', {}),
        'uploaded_images': st.session_state.get('uploaded_images', {})
    }
    st.query_params['session'] = base64.b64encode(json.dumps(session_data).encode()).decode()

# Load session state from URL
def load_session_from_url():
    try:
        if 'session' in st.query_params:
            session_data = json.loads(base64.b64decode(st.query_params['session']).decode())
            return session_data
    except Exception as e:
        print(f"Error loading session from URL: {e}")
    return None

# Initialize session state
def init_session_state():
    # Try to load from URL first
    saved_session = load_session_from_url()
    
    if saved_session:
        # Only update if we don't have messages yet to prevent overwriting
        if 'messages' not in st.session_state or not st.session_state.messages:
            st.session_state.messages = saved_session.get('messages', [])
            st.session_state.sources = saved_session.get('sources', {})
            st.session_state.uploaded_images = saved_session.get('uploaded_images', {})
    
    # Initialize with default values if no messages in session
    if 'messages' not in st.session_state or not st.session_state.messages:
        st.session_state.messages = [
            {
                'role': 'assistant',
                'content': 'Welcome to Sri Lanka AI Travel Guide! 🇱🇰\n\nI can help you discover the best of Sri Lanka - from ancient temples to beautiful beaches, delicious food to cultural experiences. How can I assist you today?',
                'timestamp': datetime.now().isoformat()
            }
        ]

    if 'sources' not in st.session_state:
        st.session_state.sources = {}

    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = {}
    
    # Save the session to URL after initialization
    save_session_to_url()

# Display chat history
def display_chat():
    # Ensure messages exist in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        role = message.get('role', 'assistant')
        with st.chat_message(role):
            # Display message content
            content = message.get('content', '')
            if content:
                st.markdown(content)
            
            # Display images if any
            if 'images' in message and message['images']:
                for img_data in message['images']:
                    try:
                        if img_data.startswith('data:image'):
                            st.image(img_data, use_column_width=True)
                        else:
                            # Handle base64 encoded images
                            st.image(Image.open(io.BytesIO(base64.b64decode(img_data))), use_column_width=True)
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")

# Process user input
def process_user_input(query: str, images: List[Any] = None):
    # Save session state after processing each input
    save_session_to_url()
    try:
        # Initialize messages if not exists
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            
        # Add user message to chat
        user_msg = {
            'role': 'user',
            'content': query,
            'timestamp': datetime.now().isoformat(),
            'images': []
        }

        # Process images if any
        if images:
            for img in images:
                try:
                    if hasattr(img, 'read'):
                        # Handle file upload
                        img_bytes = img.read()
                        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                        user_msg['images'].append(f"data:image/jpeg;base64,{img_base64}")
                    elif isinstance(img, str) and img.startswith('data:image'):
                        # Handle base64 string
                        user_msg['images'].append(img)
                except Exception as e:
                    print(f"Error processing image: {e}")

        # Add user message to chat history
        st.session_state.messages.append(user_msg)

        # Process the query using threading to avoid event loop conflicts
        with st.spinner('Thinking...'):
            chatbot = get_chatbot()
            
            # Use ThreadPoolExecutor to run async code in a separate thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async_query, chatbot, query)
                response = future.result()

            # Add bot response to chat
            bot_msg = {
                'role': 'assistant',
                'content': response.get('answer', 'I apologize, but I encountered an error processing your request.'),
                'sources': response.get('sources', []),
                'timestamp': datetime.now().isoformat()
            }

            # Add bot response to chat history
            st.session_state.messages.append(bot_msg)

            # Update sources in session state
            if 'sources' not in st.session_state:
                st.session_state.sources = {}
            if response.get('sources'):
                st.session_state.sources[bot_msg['timestamp']] = response['sources']

    except Exception as e:
        print(f"Error in process_user_input: {str(e)}")
        error_msg = {
            'role': 'assistant',
            'content': 'I encountered an error processing your request. Please try again.',
            'sources': [],
            'timestamp': datetime.now().isoformat(),
            'error': True
        }
        if 'messages' in st.session_state:
            st.session_state.messages.append(error_msg)

    except Exception as e:
        print(f"Unexpected error in process_user_input: {str(e)}")
        st.error("An unexpected error occurred. Please try again.")
        st.error(f"Error details: {str(e)}")

import asyncio
import threading
import queue
from functools import partial

# Global flag to track if we're in a notebook environment
_in_notebook = False
try:
    from IPython import get_ipython
    _in_notebook = get_ipython() is not None
except ImportError:
    pass

class AsyncExecutor:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._init_executor()
        return cls._instance
    
    def _init_executor(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            self.loop = asyncio.new_event_loop()
            self.queue = queue.Queue()
            self._stop_event = threading.Event()
            self._initialized = True
            
            # Start the event loop thread
            self.thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="AsyncExecutorThread"
            )
            self.thread.start()
    
    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._process_queue())
    
    async def _process_queue(self):
        while not self._stop_event.is_set():
            try:
                # Get the next task with a timeout to allow checking _stop_event
                try:
                    future, coro, callback = await asyncio.wait_for(
                        self.loop.run_in_executor(None, self.queue.get, True, 0.1),
                        timeout=0.1
                    )
                except (queue.Empty, asyncio.TimeoutError):
                    continue
                    
                try:
                    result = await coro
                    if future.done():
                        continue
                    future.set_result(result)
                    if callback:
                        callback(result)
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
            except Exception as e:
                print(f"Error in async executor: {e}")
    
    def submit(self, coro, callback=None):
        """Submit a coroutine to be executed in the async executor."""
        if not self._initialized:
            self._init_executor()
            
        future = asyncio.Future()
        self.queue.put((future, coro, callback))
        return future
        
    def shutdown(self):
        """Shutdown the executor cleanly."""
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=5.0)

def run_async_query(chatbot, query, image_data=None):
    """Run an async query using the async executor.
    
    Args:
        chatbot: Instance of the Chatbot class
        query: The user's query string
        image_data: Optional image data to process with the query
        
    Returns:
        Dictionary containing the response data or error information
    """
    try:
        # Use the synchronous process_query method which internally handles async execution
        return chatbot.process_query(query, image_data)
    except Exception as e:
        import traceback
        error_msg = f"Error in async query: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            "answer": "I'm sorry, I encountered an error processing your request. Please try again.",
            "sources": [],
            "error": True,
            "debug": str(e)
        }

# Sidebar with additional options
def sidebar():
    with st.sidebar:
        st.title("🇱🇰 Sri Lanka Guide")
        st.markdown("### Explore Sri Lanka")

        # Quick action buttons
        if st.button("🗺️ Top Attractions"):
            process_user_input("What are the top tourist attractions in Sri Lanka?")

        if st.button("🏨 Best Hotels"):
            process_user_input("Can you recommend some luxury hotels in Sri Lanka?")

        if st.button("🍛 Local Cuisine"):
            process_user_input("Tell me about traditional Sri Lankan food")

        st.markdown("---")

        # Image upload
        st.markdown("### Upload Images")
        uploaded_files = st.file_uploader(
            "Upload images for analysis",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.session_state.uploaded_images = uploaded_files

        st.markdown("---")

        # Clear chat button
        if st.button("🧹 Clear Chat"):
            clear_chat_history()
            st.session_state.messages = [
                {
                    'role': 'assistant',
                    'content': 'Chat history cleared. How can I help you now?',
                    'timestamp': datetime.now().isoformat()
                }
            ]
            st.rerun()

        st.markdown("---")
        st.markdown("*Your AI-powered travel companion for exploring the wonders of Sri Lanka!*")

# Main app
def main():
    # Load CSS
    load_css()

    # Initialize session state
    init_session_state()

    # Sidebar
    sidebar()

    # Main content
    st.title("Sri Lanka AI Travel Guide")
    st.markdown("*Ask me anything about traveling in Sri Lanka!*")

    # Chat container
    with st.container():
        image_urls = [
            "https://www.bluelankatours.com/wp-content/uploads/2022/08/Sembuwatta-Lake.jpg",
            "https://static01.nyt.com/images/2019/02/03/travel/03frugal-srilanka01/merlin_148552275_74c0d250-949c-46e0-b8a1-e6d499e992cf-superJumbo.jpg",
            "https://www.andbeyond.com/wp-content/uploads/sites/5/colombo-sri-lanka.jpg",
            "https://www.latexforless.com/cdn/shop/articles/Sri_Lanka_1400x.progressive.jpg"
        ]
        
        # Add CSS for the carousel
        carousel_style = """
        <style>
            .carousel-container {
                position: relative;
                width: 100%;
                height: 300px;
                overflow: hidden;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .carousel-slide {
                position: absolute;
                width: 100%;
                height: 100%;
                opacity: 0;
                transition: opacity 1s ease-in-out;
                object-fit: cover;
            }
            .carousel-slide.active {
                opacity: 1;
            }
            @keyframes fadeInOut {
                0% { opacity: 0; }
                20% { opacity: 1; }
                80% { opacity: 1; }
                100% { opacity: 0; }
            }
        </style>
        """
        
        # Add HTML for the carousel
        carousel_html = f"""
        {carousel_style}
        <div class="carousel-container">
            {''.join([
                f'<img class="carousel-slide" src="{url}" style="animation: fadeInOut 16s infinite {i*4}s; width: 100%; height: 100%; object-fit: cover;">' 
                for i, url in enumerate(image_urls)
            ])}
        </div>
        """
        
        st.markdown(carousel_html, unsafe_allow_html=True)
        
        # Display chat messages
        display_chat()

    # Input area
    with st.form("chat_input", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])

        with col1:
            user_input = st.text_input(
                "Your message",
                placeholder="Ask about places to visit, things to do, or get travel tips...",
                label_visibility="collapsed",
                key=f"user_input_{len(st.session_state.messages)}"
            )

        with col2:
            submit_button = st.form_submit_button("Send", type="primary")
    
    # Handle form submission
    if 'last_submit' not in st.session_state:
        st.session_state.last_submit = None
    
    if submit_button and user_input.strip():
        # Only process if this is a new submission
        if user_input != st.session_state.last_submit:
            st.session_state.last_submit = user_input
            process_user_input(user_input, st.session_state.get('uploaded_images', []))
            # Clear uploaded images after processing
            if 'uploaded_images' in st.session_state:
                st.session_state.uploaded_images = []
            # Force a rerun to update the chat
            st.rerun()
        st.session_state.uploaded_images = []  # Clear uploaded images after processing

if __name__ == "__main__":
    main()
