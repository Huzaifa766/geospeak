"""
GeoSpeak Corpus Manager
Handles multiple corpus sources for context-aware translation
"""

import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset
import logging
from typing import List, Dict, Tuple, Optional
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)

class CorpusManager:
    def __init__(self, corpus_dir="./corpus_data", vector_dim=384):
        self.corpus_dir = corpus_dir
        self.vector_dim = vector_dim
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.corpora = {}
        self.indexes = {}
        self.metadata = {}
        
        # Create corpus directory
        os.makedirs(corpus_dir, exist_ok=True)
        
        # Initialize corpus sources
        self.corpus_sources = {
            'opus_opensubtitles': {
                'name': 'OpenSubtitles',
                'description': 'Movie/TV subtitles - conversational language',
                'domain': 'conversational',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'medical_terminology': {
                'name': 'Medical Terms',
                'description': 'Medical terminology and healthcare translations',
                'domain': 'medical',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'business_common': {
                'name': 'Business Common',
                'description': 'Common business phrases and terminology',
                'domain': 'business',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'technical_computing': {
                'name': 'Technical Computing',
                'description': 'Programming and technical terminology',
                'domain': 'technical',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'news_current': {
                'name': 'News & Current Events',
                'description': 'News articles and current affairs',
                'domain': 'news',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            }
        }
    
    def create_sample_corpora(self):
        """Create sample corpus data for demonstration"""
        logger.info("Creating sample corpora...")
        
        # OpenSubtitles - Conversational
        opensubtitles_data = [
            {"en": "Hello, how are you?", "es": "Hola, ¿cómo estás?", "fr": "Salut, comment ça va?", "de": "Hallo, wie geht es dir?", "ur": "ہیلو، آپ کیسے ہیں؟"},
            {"en": "Nice to meet you!", "es": "¡Encantado de conocerte!", "fr": "Ravi de vous rencontrer!", "de": "Freut mich, dich kennenzulernen!", "ur": "آپ سے مل کر خوشی ہوئی!"},
            {"en": "What's your name?", "es": "¿Cómo te llamas?", "fr": "Comment tu t'appelles?", "de": "Wie heißt du?", "ur": "آپ کا نام کیا ہے؟"},
            {"en": "I'm fine, thank you", "es": "Estoy bien, gracias", "fr": "Je vais bien, merci", "de": "Mir geht es gut, danke", "ur": "میں ٹھیک ہوں، شکریہ"},
            {"en": "See you later!", "es": "¡Hasta luego!", "fr": "À plus tard!", "de": "Bis später!", "ur": "بعد میں ملیں گے!"},
            {"en": "Good morning!", "es": "¡Buenos días!", "fr": "Bonjour!", "de": "Guten Morgen!", "ur": "صبح بخیر!"},
            {"en": "Have a great day!", "es": "¡Que tengas un buen día!", "fr": "Passez une excellente journée!", "de": "Hab einen schönen Tag!", "ur": "آپ کا دن اچھا گزرے!"},
            {"en": "I love this movie", "es": "Me encanta esta película", "fr": "J'adore ce film", "de": "Ich liebe diesen Film", "ur": "مجھے یہ فلم پسند ہے"},
            {"en": "What time is it?", "es": "¿Qué hora es?", "fr": "Quelle heure est-il?", "de": "Wie spät ist es?", "ur": "کیا وقت ہے؟"},
            {"en": "I'm sorry", "es": "Lo siento", "fr": "Je suis désolé", "de": "Es tut mir leid", "ur": "میں معافی چاہتا ہوں"}
        ]
        
        # Medical terminology
        medical_data = [
            {"en": "Blood pressure", "es": "Presión arterial", "fr": "Tension artérielle", "de": "Blutdruck", "ur": "بلڈ پریشر"},
            {"en": "Heart rate", "es": "Frecuencia cardíaca", "fr": "Rythme cardiaque", "de": "Herzfrequenz", "ur": "دل کی رفتار"},
            {"en": "Take this medication twice daily", "es": "Tome este medicamento dos veces al día", "fr": "Prenez ce médicament deux fois par jour", "de": "Nehmen Sie dieses Medikament zweimal täglich", "ur": "یہ دوا روزانہ دو بار لیں"},
            {"en": "You need to rest", "es": "Necesitas descansar", "fr": "Vous devez vous reposer", "de": "Du musst dich ausruhen", "ur": "آپ کو آرام کرنا چاہیے"},
            {"en": "Emergency room", "es": "Sala de emergencias", "fr": "Salle d'urgence", "de": "Notaufnahme", "ur": "ایمرجنسی روم"},
            {"en": "Medical history", "es": "Historia médica", "fr": "Antécédents médicaux", "de": "Krankengeschichte", "ur": "طبی تاریخ"},
            {"en": "Prescription", "es": "Receta médica", "fr": "Ordonnance", "de": "Rezept", "ur": "نسخہ"},
            {"en": "Symptoms", "es": "Síntomas", "fr": "Symptômes", "de": "Symptome", "ur": "علامات"},
            {"en": "Diagnosis", "es": "Diagnóstico", "fr": "Diagnostic", "de": "Diagnose", "ur": "تشخیص"},
            {"en": "Treatment", "es": "Tratamiento", "fr": "Traitement", "de": "Behandlung", "ur": "علاج"}
        ]
        
        # Business terminology
        business_data = [
            {"en": "Schedule a meeting", "es": "Programar una reunión", "fr": "Planifier une réunion", "de": "Ein Meeting planen", "ur": "میٹنگ کا وقت مقرر کریں"},
            {"en": "Business proposal", "es": "Propuesta comercial", "fr": "Proposition commerciale", "de": "Geschäftsvorschlag", "ur": "کاروباری تجویز"},
            {"en": "Revenue growth", "es": "Crecimiento de ingresos", "fr": "Croissance des revenus", "de": "Umsatzwachstum", "ur": "آمدنی میں اضافہ"},
            {"en": "Market analysis", "es": "Análisis de mercado", "fr": "Analyse de marché", "de": "Marktanalyse", "ur": "مارکیٹ کا تجزیہ"},
            {"en": "Customer service", "es": "Servicio al cliente", "fr": "Service client", "de": "Kundendienst", "ur": "کسٹمر سروس"},
            {"en": "Financial report", "es": "Informe financiero", "fr": "Rapport financier", "de": "Finanzbericht", "ur": "مالی رپورٹ"},
            {"en": "Project deadline", "es": "Fecha límite del proyecto", "fr": "Date limite du projet", "de": "Projekttermin", "ur": "پروجیکٹ کی آخری تاریخ"},
            {"en": "Budget allocation", "es": "Asignación de presupuesto", "fr": "Allocation budgétaire", "de": "Budgetzuweisung", "ur": "بجٹ کی تقسیم"},
            {"en": "Strategic planning", "es": "Planificación estratégica", "fr": "Planification stratégique", "de": "Strategische Planung", "ur": "اسٹریٹجک پلاننگ"},
            {"en": "Quality assurance", "es": "Aseguramiento de calidad", "fr": "Assurance qualité", "de": "Qualitätssicherung", "ur": "معیار کی یقین دہانی"}
        ]
        
        # Technical/Computing
        technical_data = [
            {"en": "Database connection", "es": "Conexión a base de datos", "fr": "Connexion à la base de données", "de": "Datenbankverbindung", "ur": "ڈیٹابیس کنکشن"},
            {"en": "Software development", "es": "Desarrollo de software", "fr": "Développement logiciel", "de": "Softwareentwicklung", "ur": "سافٹ ویئر ڈیولپمنٹ"},
            {"en": "Machine learning algorithm", "es": "Algoritmo de aprendizaje automático", "fr": "Algorithme d'apprentissage automatique", "de": "Algorithmus für maschinelles Lernen", "ur": "مشین لرننگ الگورتھم"},
            {"en": "User interface", "es": "Interfaz de usuario", "fr": "Interface utilisateur", "de": "Benutzeroberfläche", "ur": "یوزر انٹرفیس"},
            {"en": "System administrator", "es": "Administrador del sistema", "fr": "Administrateur système", "de": "Systemadministrator", "ur": "سسٹم ایڈمنسٹریٹر"},
            {"en": "Cloud computing", "es": "Computación en la nube", "fr": "Informatique en nuage", "de": "Cloud-Computing", "ur": "کلاؤڈ کمپیوٹنگ"},
            {"en": "Artificial intelligence", "es": "Inteligencia artificial", "fr": "Intelligence artificielle", "de": "Künstliche Intelligenz", "ur": "مصنوعی ذہانت"},
            {"en": "Data security", "es": "Seguridad de datos", "fr": "Sécurité des données", "de": "Datensicherheit", "ur": "ڈیٹا سیکیورٹی"},
            {"en": "Network protocol", "es": "Protocolo de red", "fr": "Protocole réseau", "de": "Netzwerkprotokoll", "ur": "نیٹ ورک پروٹوکول"},
            {"en": "Version control", "es": "Control de versiones", "fr": "Contrôle de version", "de": "Versionskontrolle", "ur": "ورژن کنٹرول"}
        ]
        
        # News/Current events
        news_data = [
            {"en": "Breaking news", "es": "Noticias de última hora", "fr": "Dernières nouvelles", "de": "Eilmeldung", "ur": "تازہ خبریں"},
            {"en": "Economic forecast", "es": "Pronóstico económico", "fr": "Prévisions économiques", "de": "Wirtschaftsprognose", "ur": "معاشی پیشن گوئی"},
            {"en": "Climate change", "es": "Cambio climático", "fr": "Changement climatique", "de": "Klimawandel", "ur": "موسمیاتی تبدیلی"},
            {"en": "Election results", "es": "Resultados electorales", "fr": "Résultats des élections", "de": "Wahlergebnisse", "ur": "انتخابی نتائج"},
            {"en": "International relations", "es": "Relaciones internacionales", "fr": "Relations internationales", "de": "Internationale Beziehungen", "ur": "بین الاقوامی تعلقات"},
            {"en": "Scientific research", "es": "Investigación científica", "fr": "Recherche scientifique", "de": "Wissenschaftliche Forschung", "ur": "سائنسی تحقیق"},
            {"en": "Public health", "es": "Salud pública", "fr": "Santé publique", "de": "Öffentliche Gesundheit", "ur": "عوامی صحت"},
            {"en": "Educational system", "es": "Sistema educativo", "fr": "Système éducatif", "de": "Bildungssystem", "ur": "تعلیمی نظام"},
            {"en": "Environmental protection", "es": "Protección ambiental", "fr": "Protection de l'environnement", "de": "Umweltschutz", "ur": "ماحولیاتی تحفظ"},
            {"en": "Technology advancement", "es": "Avance tecnológico", "fr": "Avancement technologique", "de": "Technologischer Fortschritt", "ur": "ٹیکنالوجی کی ترقی"}
        ]
        
        # Save corpora
        corpus_data = {
            'opus_opensubtitles': opensubtitles_data,
            'medical_terminology': medical_data,
            'business_common': business_data,
            'technical_computing': technical_data,
            'news_current': news_data
        }
        
        for corpus_name, data in corpus_data.items():
            self.save_corpus(corpus_name, data)
            logger.info(f"Created {corpus_name} corpus with {len(data)} entries")
    
    def save_corpus(self, corpus_name: str, data: List[Dict]):
        """Save corpus data to file"""
        filepath = os.path.join(self.corpus_dir, f"{corpus_name}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'name': corpus_name,
                'data': data,
                'created_at': datetime.now().isoformat(),
                'count': len(data)
            }, f, ensure_ascii=False, indent=2)
    
    def load_corpus(self, corpus_name: str) -> List[Dict]:
        """Load corpus data from file"""
        filepath = os.path.join(self.corpus_dir, f"{corpus_name}.json")
        if not os.path.exists(filepath):
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            corpus_info = json.load(f)
            return corpus_info.get('data', [])
    
    def build_vector_index(self, corpus_name: str, source_lang: str = 'en'):
        """Build FAISS vector index for a corpus"""
        logger.info(f"Building vector index for {corpus_name}...")
        
        data = self.load_corpus(corpus_name)
        if not data:
            logger.warning(f"No data found for corpus {corpus_name}")
            return
        
        # Extract source language texts
        texts = []
        metadata = []
        
        for idx, entry in enumerate(data):
            if source_lang in entry:
                texts.append(entry[source_lang])
                metadata.append({
                    'corpus': corpus_name,
                    'index': idx,
                    'translations': entry
                })
        
        if not texts:
            logger.warning(f"No texts found for language {source_lang} in corpus {corpus_name}")
            return
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        # Create FAISS index
        index = faiss.IndexFlatIP(self.vector_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        # Store index and metadata
        self.indexes[corpus_name] = index
        self.metadata[corpus_name] = metadata
        
        # Save to disk
        index_path = os.path.join(self.corpus_dir, f"{corpus_name}_index.faiss")
        metadata_path = os.path.join(self.corpus_dir, f"{corpus_name}_metadata.pkl")
        
        faiss.write_index(index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Vector index built for {corpus_name} with {index.ntotal} vectors")
    
    def load_vector_index(self, corpus_name: str):
        """Load vector index from disk"""
        index_path = os.path.join(self.corpus_dir, f"{corpus_name}_index.faiss")
        metadata_path = os.path.join(self.corpus_dir, f"{corpus_name}_metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.indexes[corpus_name] = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.metadata[corpus_name] = pickle.load(f)
            return True
        return False
    
    def search_similar_texts(self, query: str, corpus_name: str = None, k: int = 5) -> List[Dict]:
        """Search for similar texts in corpus"""
        if corpus_name and corpus_name not in self.indexes:
            if not self.load_vector_index(corpus_name):
                return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        results = []
        
        # Search in specific corpus or all corpora
        corpora_to_search = [corpus_name] if corpus_name else list(self.indexes.keys())
        
        for corpus in corpora_to_search:
            if corpus not in self.indexes:
                continue
                
            # Search in FAISS index
            scores, indices = self.indexes[corpus].search(query_embedding.astype('float32'), k)
            
            # Collect results
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata[corpus]):
                    result = self.metadata[corpus][idx].copy()
                    result['similarity_score'] = float(score)
                    result['corpus_name'] = corpus
                    results.append(result)
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:k]
    
    def get_context_examples(self, query: str, target_language: str, max_examples: int = 3) -> str:
        """Get context examples for translation"""
        similar_texts = self.search_similar_texts(query, k=max_examples * 2)
        
        context_examples = []
        for result in similar_texts:
            translations = result.get('translations', {})
            if 'en' in translations and target_language in translations:
                example = f"English: \"{translations['en']}\"\n{self.get_language_name(target_language)}: \"{translations[target_language]}\""
                context_examples.append(example)
                
                if len(context_examples) >= max_examples:
                    break
        
        if context_examples:
            return "Translation examples:\n" + "\n\n".join(context_examples) + "\n\n"
        return ""
    
    def get_language_name(self, code: str) -> str:
        """Get full language name from code"""
        language_names = {
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
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
        return language_names.get(code, code)
    
    def initialize_all_corpora(self):
        """Initialize all corpora with sample data and build indexes"""
        logger.info("Initializing all corpora...")
        
        # Create sample corpora if they don't exist
        self.create_sample_corpora()
        
        # Build vector indexes for all corpora
        for corpus_name in self.corpus_sources.keys():
            if not self.load_vector_index(corpus_name):
                self.build_vector_index(corpus_name)
        
        logger.info("All corpora initialized successfully")
    
    def get_corpus_stats(self) -> Dict:
        """Get statistics about all corpora"""
        stats = {}
        for corpus_name in self.corpus_sources.keys():
            data = self.load_corpus(corpus_name)
            stats[corpus_name] = {
                'name': self.corpus_sources[corpus_name]['name'],
                'description': self.corpus_sources[corpus_name]['description'],
                'domain': self.corpus_sources[corpus_name]['domain'],
                'entries': len(data),
                'indexed': corpus_name in self.indexes
            }
        return stats

if __name__ == "__main__":
    # Test the corpus manager
    logging.basicConfig(level=logging.INFO)
    
    corpus_manager = CorpusManager()
    corpus_manager.initialize_all_corpora()
    
    # Test search
    query = "I need to see a doctor"
    results = corpus_manager.search_similar_texts(query, k=3)
    print(f"Search results for '{query}':")
    for result in results:
        print(f"- {result['translations']['en']} (Score: {result['similarity_score']:.3f})")
    
    # Test context examples
    context = corpus_manager.get_context_examples("Good morning", "es")
    print(f"\nContext examples:\n{context}")
