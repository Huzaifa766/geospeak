# 🌍 GeoSpeak - AI-Powered Language Translation

GeoSpeak is an intelligent, real-time web application that leverages Google Gemini AI to provide accurate, context-aware language translations. Built with Flask and modern web technologies, it offers a seamless translation experience for over 20 languages.

## ✨ Features

- **AI-Powered Translation**: Uses Google Gemini AI for accurate, contextually-aware translations
- **Multi-Language Support**: Supports 20+ languages including Spanish, French, German, Japanese, Korean, Arabic, Hindi, and more
- **Real-time Processing**: Fast translation with minimal latency
- **Context-Aware**: Understands idioms, cultural nuances, and context
- **Language Detection**: Automatic detection of input text language
- **Modern UI**: Responsive, user-friendly interface
- **Error Handling**: Robust error handling and user feedback
- **Health Monitoring**: Service status monitoring and diagnostics

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AunSyedShah/geospeak.git
   cd geospeak
   ```

2. **Run the setup script**
   ```bash
   ./run.sh
   ```
   
   Or manually:
   
3. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your Google Gemini API key
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Open your browser**
   Navigate to `http://localhost:5000`

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_GEMINI_API_KEY=your_actual_api_key_here
FLASK_ENV=development
FLASK_DEBUG=True
```

### API Key Setup

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## 🏗️ Architecture

The application follows a modular architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API     │    │  Google Gemini  │
│   (HTML/JS)     │───▶│   (Python)      │───▶│      AI         │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

1. **User Interface**: Modern, responsive web interface
2. **Flask Backend**: RESTful API handling translation requests
3. **Translation Service**: Core logic for text processing and AI integration
4. **Context Engine**: Creates context-aware prompts for better translations
5. **Error Handler**: Comprehensive error handling and user feedback

## 📝 API Endpoints

### POST `/translate`
Translate text from English to target language.

**Request:**
```json
{
    "text": "Hello, how are you?",
    "target_language": "es"
}
```

**Response:**
```json
{
    "original_text": "Hello, how are you?",
    "translated_text": "Hola, ¿cómo estás?",
    "target_language": "Spanish",
    "timestamp": "2025-08-12T10:30:00"
}
```

### POST `/detect-language`
Detect the language of input text.

**Request:**
```json
{
    "text": "Bonjour, comment allez-vous?"
}
```

**Response:**
```json
{
    "detected_language": "French",
    "text": "Bonjour, comment allez-vous?"
}
```

### GET `/health`
Check service health status.

**Response:**
```json
{
    "status": "healthy",
    "service": "GeoSpeak Translation Service",
    "timestamp": "2025-08-12T10:30:00"
}
```

## 🌐 Supported Languages

- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Japanese (ja)
- Korean (ko)
- Chinese Simplified (zh)
- Arabic (ar)
- Hindi (hi)
- Russian (ru)
- Dutch (nl)
- Swedish (sv)
- Norwegian (no)
- Danish (da)
- Polish (pl)
- Turkish (tr)
- Hebrew (he)
- Thai (th)
- Vietnamese (vi)

## 🧪 Testing

### Manual Testing
1. Open the application in your browser
2. Enter text in English
3. Select a target language
4. Click "Translate"
5. Verify the translation quality

### API Testing
```bash
# Test translation endpoint
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "target_language": "es"}'

# Test language detection
curl -X POST http://localhost:5000/detect-language \
  -H "Content-Type: application/json" \
  -d '{"text": "Hola mundo"}'

# Test health endpoint
curl http://localhost:5000/health
```

## 📁 Project Structure

```
geospeak/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .env                  # Environment variables (create this)
├── run.sh                # Setup and run script
├── README.md             # Project documentation
├── documentation.txt     # Project requirements (SRS)
├── templates/
│   └── index.html        # Main UI template
└── .venv/                # Virtual environment (auto-created)
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your Google Gemini API key is correctly set in `.env`
   - Verify the API key is active and has proper permissions

2. **Module Not Found**
   - Activate the virtual environment: `source .venv/bin/activate`
   - Install dependencies: `pip install -r requirements.txt`

3. **Port Already in Use**
   - Change the port in `app.py`: `app.run(port=5001)`
   - Or kill the process using port 5000

4. **Translation Errors**
   - Check your internet connection
   - Verify API key quotas and limits
   - Check application logs for detailed error messages

## 🔒 Security Considerations

- Never commit API keys to version control
- Use environment variables for sensitive configuration
- Implement rate limiting for production use
- Validate and sanitize user inputs
- Use HTTPS in production environments

## 🚀 Deployment

### Local Development
```bash
source .venv/bin/activate
python app.py
```

### Production Deployment
1. Set `FLASK_ENV=production` in `.env`
2. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini AI for providing the translation capabilities
- Flask framework for the web application structure
- The open-source community for various tools and libraries

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---

**Made with ❤️ for breaking down language barriers worldwide**