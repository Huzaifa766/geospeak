from flask import Flask, render_template, request, jsonify
from google import genai
import os
import json
import logging
from datetime import datetime
import re
from dotenv import load_dotenv
from corpus_manager import CorpusManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Gemini client with API key
api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
if not api_key:
    logger.error("GOOGLE_GEMINI_API_KEY not found in environment variables")
    logger.error("Please set your API key in a .env file or environment variable")
    client = None
else:
    try:
        client = genai.Client(api_key=api_key)
        logger.info("Google Gemini client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        client = None

# Initialize Corpus Manager
corpus_manager = None
try:
    corpus_manager = CorpusManager()
    logger.info("Initializing corpus manager (this may take a moment)...")
    corpus_manager.initialize_all_corpora()
    logger.info("Corpus manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize corpus manager: {e}")
    corpus_manager = None

# Supported languages for translation
SUPPORTED_LANGUAGES = {
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese (Simplified)',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'ur': 'Urdu',
    'ru': 'Russian',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'no': 'Norwegian',
    'da': 'Danish',
    'pl': 'Polish',
    'tr': 'Turkish',
    'he': 'Hebrew',
    'th': 'Thai',
    'vi': 'Vietnamese'
}

class TranslationService:
    def __init__(self):
        self.model_name = "gemini-2.0-flash-exp"
        
    def create_context_aware_prompt(self, text, target_language, context_examples=None):
        """Create a context-aware prompt for translation with corpus examples"""
        
        # Get context examples from corpus if not provided
        if not context_examples and corpus_manager:
            # Map language codes to full names for corpus search
            lang_code_map = {
                'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Italian': 'it',
                'Portuguese': 'pt', 'Japanese': 'ja', 'Korean': 'ko', 'Chinese (Simplified)': 'zh',
                'Arabic': 'ar', 'Hindi': 'hi', 'Urdu': 'ur', 'Russian': 'ru', 'Dutch': 'nl',
                'Swedish': 'sv', 'Norwegian': 'no', 'Danish': 'da', 'Polish': 'pl',
                'Turkish': 'tr', 'Hebrew': 'he', 'Thai': 'th', 'Vietnamese': 'vi'
            }
            
            # Find language code
            target_lang_code = None
            for lang_name, code in lang_code_map.items():
                if target_language == lang_name:
                    target_lang_code = code
                    break
            
            if target_lang_code:
                context_examples = corpus_manager.get_context_examples(text, target_lang_code, max_examples=3)

        base_prompt = f"""You are a professional translator. Translate the text from English to {target_language}.

{context_examples if context_examples else ""}

INSTRUCTIONS:
- Provide ONLY the direct translation
- Choose the most common/natural interpretation
- No explanations, alternatives, or additional commentary
- No quotes around the translation
- Single line response only

Text to translate: "{text}"

Translation:"""
        
        return base_prompt
    
    def translate_text(self, text, target_language):
        """Translate text using Gemini AI with corpus context"""
        try:
            if not client:
                return {"error": "Gemini AI client not initialized. Please check your API key."}
                
            if not text.strip():
                return {"error": "Empty text provided"}
            
            # Create context-aware prompt with corpus examples
            prompt = self.create_context_aware_prompt(text, target_language)
            
            # Generate translation using Gemini
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            if response and response.text:
                # Clean up the response - extract only the translation
                translated_text = response.text.strip()
                
                # Remove common prefixes that AI might add
                prefixes_to_remove = [
                    "Translation:", "Translated text:", "The translation is:",
                    "Here is the translation:", "The text translates to:",
                    "In Arabic:", "In Spanish:", "In French:", "In German:",
                    "In Italian:", "In Portuguese:", "In Japanese:", "In Korean:",
                    "In Chinese:", "In Hindi:", "In Urdu:", "In Russian:", "In Dutch:",
                    "In Swedish:", "In Norwegian:", "In Danish:", "In Polish:",
                    "In Turkish:", "In Hebrew:", "In Thai:", "In Vietnamese:"
                ]
                
                for prefix in prefixes_to_remove:
                    if translated_text.startswith(prefix):
                        translated_text = translated_text[len(prefix):].strip()
                
                # Remove quotes if the entire response is quoted
                if (translated_text.startswith('"') and translated_text.endswith('"')) or \
                   (translated_text.startswith("'") and translated_text.endswith("'")):
                    translated_text = translated_text[1:-1].strip()
                
                # Take only the first line if multiple lines (to avoid explanations)
                lines = translated_text.split('\n')
                if lines:
                    translated_text = lines[0].strip()
                
                # Remove any remaining asterisks or markdown formatting
                translated_text = translated_text.replace('*', '').replace('#', '').strip()
                
                # Ensure we have actual content
                if not translated_text or len(translated_text) < 1:
                    translated_text = response.text.strip()[:200]  # Fallback to first 200 chars
                
                # Get similar examples from corpus for reference
                similar_examples = []
                if corpus_manager:
                    lang_code_map = {
                        'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Italian': 'it',
                        'Portuguese': 'pt', 'Japanese': 'ja', 'Korean': 'ko', 'Chinese (Simplified)': 'zh',
                        'Arabic': 'ar', 'Hindi': 'hi', 'Urdu': 'ur', 'Russian': 'ru', 'Dutch': 'nl',
                        'Swedish': 'sv', 'Norwegian': 'no', 'Danish': 'da', 'Polish': 'pl',
                        'Turkish': 'tr', 'Hebrew': 'he', 'Thai': 'th', 'Vietnamese': 'vi'
                    }
                    
                    target_lang_code = None
                    for lang_name, code in lang_code_map.items():
                        if target_language == lang_name:
                            target_lang_code = code
                            break
                    
                    if target_lang_code:
                        search_results = corpus_manager.search_similar_texts(text, k=3)
                        for result in search_results:
                            if target_lang_code in result.get('translations', {}):
                                similar_examples.append({
                                    'source': result['translations']['en'],
                                    'target': result['translations'][target_lang_code],
                                    'corpus': result['corpus_name'],
                                    'similarity': result['similarity_score']
                                })
                
                # Log translation
                logger.info(f"Translation completed: {text[:50]}... -> {translated_text[:50]}...")
                
                return {
                    "original_text": text,
                    "translated_text": translated_text,
                    "target_language": target_language,
                    "timestamp": datetime.now().isoformat(),
                    "similar_examples": similar_examples,
                    "corpus_used": corpus_manager is not None
                }
            else:
                return {"error": "No translation received from the model"}
                
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return {"error": f"Translation failed: {str(e)}"}
    
    def get_language_detection(self, text):
        """Detect the language of input text"""
        try:
            if not client:
                return "Client not initialized"
                
            prompt = f"""Detect the language of the following text and respond with just the language name:

Text: "{text}"

Language:"""
            
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            if response and response.text:
                return response.text.strip()
            return "Unknown"
            
        except Exception as e:
            logger.error(f"Language detection error: {str(e)}")
            return "Unknown"

# Initialize translation service
translation_service = TranslationService()

@app.route('/')
def index():
    """Main page with translation interface"""
    return render_template('index.html', languages=SUPPORTED_LANGUAGES)

@app.route('/translate', methods=['POST'])
def translate():
    """Handle translation requests"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        text = data.get('text', '').strip()
        target_language_code = data.get('target_language')
        
        if not text:
            return jsonify({"error": "No text provided for translation"}), 400
        
        if not target_language_code or target_language_code not in SUPPORTED_LANGUAGES:
            return jsonify({"error": "Invalid target language"}), 400
        
        target_language = SUPPORTED_LANGUAGES[target_language_code]
        
        # Perform translation
        result = translation_service.translate_text(text, target_language)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Translation endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/detect-language', methods=['POST'])
def detect_language():
    """Detect the language of input text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        detected_language = translation_service.get_language_detection(text)
        
        return jsonify({
            "detected_language": detected_language,
            "text": text
        })
        
    except Exception as e:
        logger.error(f"Language detection endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    corpus_status = "available" if corpus_manager else "unavailable"
    return jsonify({
        "status": "healthy",
        "service": "GeoSpeak Translation Service",
        "timestamp": datetime.now().isoformat(),
        "corpus_status": corpus_status,
        "gemini_status": "available" if client else "unavailable"
    })

@app.route('/corpus/stats')
def corpus_stats():
    """Get corpus statistics"""
    if not corpus_manager:
        return jsonify({"error": "Corpus manager not available"}), 500
    
    try:
        stats = corpus_manager.get_corpus_stats()
        return jsonify({
            "corpus_stats": stats,
            "total_corpora": len(stats),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Corpus stats error: {str(e)}")
        return jsonify({"error": "Failed to get corpus statistics"}), 500

@app.route('/corpus/search', methods=['POST'])
def search_corpus():
    """Search for similar texts in corpus"""
    if not corpus_manager:
        return jsonify({"error": "Corpus manager not available"}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        corpus_name = data.get('corpus_name')  # Optional: search specific corpus
        k = min(data.get('k', 5), 20)  # Limit to max 20 results
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        results = corpus_manager.search_similar_texts(query, corpus_name, k)
        
        return jsonify({
            "query": query,
            "results": results,
            "total_results": len(results),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Corpus search error: {str(e)}")
        return jsonify({"error": "Corpus search failed"}), 500

@app.route('/corpus/examples', methods=['POST'])
def get_translation_examples():
    """Get translation examples for a text"""
    if not corpus_manager:
        return jsonify({"error": "Corpus manager not available"}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        target_language = data.get('target_language')
        
        if not text or not target_language:
            return jsonify({"error": "Text and target language required"}), 400
        
        # Map language names to codes
        lang_code_map = {
            'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Italian': 'it',
            'Portuguese': 'pt', 'Japanese': 'ja', 'Korean': 'ko', 'Chinese (Simplified)': 'zh',
            'Arabic': 'ar', 'Hindi': 'hi', 'Urdu': 'ur', 'Russian': 'ru', 'Dutch': 'nl',
            'Swedish': 'sv', 'Norwegian': 'no', 'Danish': 'da', 'Polish': 'pl',
            'Turkish': 'tr', 'Hebrew': 'he', 'Thai': 'th', 'Vietnamese': 'vi'
        }
        
        target_lang_code = lang_code_map.get(target_language)
        if not target_lang_code:
            return jsonify({"error": "Unsupported target language"}), 400
        
        context_examples = corpus_manager.get_context_examples(text, target_lang_code, max_examples=5)
        
        # Also get raw search results
        search_results = corpus_manager.search_similar_texts(text, k=5)
        examples = []
        for result in search_results:
            if target_lang_code in result.get('translations', {}):
                examples.append({
                    'source': result['translations']['en'],
                    'target': result['translations'][target_lang_code],
                    'corpus': result['corpus_name'],
                    'similarity': result['similarity_score']
                })
        
        return jsonify({
            "text": text,
            "target_language": target_language,
            "context_examples": context_examples,
            "raw_examples": examples,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Translation examples error: {str(e)}")
        return jsonify({"error": "Failed to get translation examples"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    if not client:
        logger.error("Cannot start application: Gemini client not initialized")
        logger.error("Please set GOOGLE_GEMINI_API_KEY in your environment or .env file")
        logger.error("Get your API key from: https://makersuite.google.com/app/apikey")
        exit(1)
    
    # Test Gemini connection
    try:
        test_response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Hello"
        )
        logger.info("Gemini AI connection test successful")
    except Exception as e:
        logger.error(f"Failed to connect to Gemini AI: {str(e)}")
        logger.error("Please check your API key and internet connection")
    
    logger.info("Starting GeoSpeak application...")
    app.run(debug=True, host='0.0.0.0', port=5000)