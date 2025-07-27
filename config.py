import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Model Configuration
    MODEL_NAME = "gemini-1.5-flash-latest"  # Using a stable model from the available list
    FULL_MODEL_NAME = f"models/{MODEL_NAME}"  # Full model path for API calls
    EMBEDDING_MODEL = "models/embedding-001"  # Using the standard embedding model
    
    # Generation parameters for the model
    GENERATION_CONFIG = {
        "temperature": 0.7,  # Controls randomness in the response (0.0 to 1.0)
        "top_p": 0.95,       # Nucleus sampling parameter
        "top_k": 40,         # Limits the number of highest probability tokens to consider
        "max_output_tokens": 2048,  # Maximum length of the generated response
    }
    
    # Memory and Processing
    MAX_TOKENS = 4000
    MEMORY_FILE = "chat_history.json"
    MAX_SOURCE_DOCS = 5
    
    # Text Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Database
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    DB_NAME = "sri_lanka_tourism"
    VECTOR_COLLECTION = "tourism_vectors"
    CACHE_COLLECTION = "vector_cache"
    CACHE_EXPIRY_DAYS = 7
    
    # System Prompt
    SYSTEM_PROMPT = """
    You are a knowledgeable and friendly Sri Lanka Tourism Assistant. Your goal is to provide 
    accurate, helpful, and engaging information about Sri Lanka's tourism attractions, 
    culture, history, accommodations, and travel tips. 
    
    When responding:
    1. Be informative but concise
    2. Include relevant cultural context
    3. Provide practical tips when appropriate
    4. Be enthusiastic about Sri Lanka's offerings
    5. If you don't know something, say so and offer to help with related information
    
    Always respond in a warm, welcoming tone that reflects Sri Lankan hospitality.
    """
