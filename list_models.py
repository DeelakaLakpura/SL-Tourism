import os
import google.generativeai as genai

# Get the API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable is not set")
    exit(1)

# Configure the API
try:
    genai.configure(api_key=api_key)
    print("Successfully configured Google Generative AI API")
    
    # List all available models
    print("\nListing all available models:")
    print("-" * 50)
    models = genai.list_models()
    
    # Print detailed information about each model
    for model in models:
        print(f"\nModel Name: {model.name}")
        print(f"Display Name: {model.display_name}")
        print(f"Description: {model.description}")
        print(f"Supported Generation Methods: {model.supported_generation_methods}")
        print(f"Input Token Limit: {getattr(model, 'input_token_limit', 'N/A')}")
        print(f"Output Token Limit: {getattr(model, 'output_token_limit', 'N/A')}")
        print("-" * 50)
    
    # Print a summary of available models
    print("\nSummary of available models:")
    print("-" * 50)
    for model in models:
        print(f"- {model.name} (Generation methods: {', '.join(model.supported_generation_methods)})")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
