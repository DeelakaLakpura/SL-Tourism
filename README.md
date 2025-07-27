# Sri Lanka AI Travel Guide

An advanced AI-powered chatbot for Sri Lanka tourism, providing personalized travel recommendations, cultural insights, and travel planning assistance.

## Features

- **Natural Language Understanding**: Advanced NLP for understanding complex travel queries
- **Multimodal Support**: Process both text and images for richer interactions
- **Contextual Responses**: Maintains conversation context for more natural interactions
- **Source Citation**: Provides sources for all information
- **Interactive UI**: Modern, responsive interface with chat history
- **Personalization**: Remembers user preferences and past interactions

## Tech Stack

- **Backend**: Python, LangChain, Google Gemini Pro
- **Vector Store**: FAISS (with MongoDB Atlas support)
- **Frontend**: Streamlit
- **Deployment**: Docker, Streamlit Cloud

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sri-lanka-travel-bot.git
   cd sri-lanka-travel-bot
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on the example:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your Google Gemini API key.

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Data

The chatbot uses the following data sources stored in the `data/` directory:
- Tourism destinations
- Hotel information
- Restaurant recommendations
- Cultural insights
- Transportation options
- Weather information

## Usage

1. Start the application and open the provided URL in your browser
2. Ask questions about traveling in Sri Lanka, for example:
   - "What are the must-visit places in Sri Lanka?"
   - "Can you suggest a 7-day itinerary?"
   - "What's the best time to visit Yala National Park?"
   - "Show me luxury hotels in Colombo"

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Docker
```bash
docker build -t sri-lanka-travel-bot .
docker run -p 8501:8501 sri-lanka-travel-bot
```

### Streamlit Cloud
1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app" and connect your repository
4. Set the main file to `app.py`
5. Add your environment variables
6. Deploy!

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Gemini for the powerful language model
- LangChain for the LLM orchestration
- Streamlit for the beautiful UI framework
- Sri Lanka Tourism for the wonderful destination data
