
# 🦜🔗 Travel Chatbot with RAG Chain

This project implements a travel chatbot powered by the RAG (Retrieve and Generate) chain, providing real-time information retrieval using various tools and the ability to fetch weather reports.

## Features

- **Conversational Interface**: Engage in a conversation with the travel chatbot.
- **RAG Chain Integration**: Utilizes RAG (Retrieve and Generate) chain for natural language processing.
- **Real-time Tool Search**: Access to real-time information through integrated tools.
- **Weather Report**: Fetch current weather data based on the user's location.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Groq API key
- Google Serper API key
- OpenWeatherMap API key

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/ArijitAcharya/Tour-Travel-Chatbot
    cd Chatbot-App
    ```

2. Install dependencies:

    ```bash
    pip install -r requirement.txt
    ```

3. Set up environment variables:
    Create a `.env` file and add the following:
    ```
    GROQ_API_KEY="YOUR_GROQ_API_KEY"
    SERPER_API_KEY="YOUR_SERPER_API_KEY"
    OPENWEATHERMAP_API_KEY="YOUR_OPENWEATHERMAP_API_KEY"
    ```

4. Run the application:

    ```bash
    streamlit run app.py
    ```
### Running with Docker
1. Build the Docker image:
    ```bash
    docker build -t travel-chatbot .
    ```
2. Run the Docker container:
    ```bash
    docker run -p 8501:8501 travel-chatbot
    ```
### Usage

- Access the application via the provided Streamlit URL.
- Input your query or engage in a conversation with the travel chatbot.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for any enhancements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

