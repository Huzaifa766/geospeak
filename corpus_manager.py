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
            },
            'legal_formal': {
                'name': 'Legal & Formal',
                'description': 'Legal documents, contracts, and formal communications',
                'domain': 'legal',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'academic_research': {
                'name': 'Academic & Research',
                'description': 'Academic papers, research terminology, and scholarly language',
                'domain': 'academic',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'travel_tourism': {
                'name': 'Travel & Tourism',
                'description': 'Travel phrases, tourism, hotels, and transportation',
                'domain': 'travel',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'culinary_food': {
                'name': 'Culinary & Food',
                'description': 'Food, cooking, restaurants, and culinary terminology',
                'domain': 'culinary',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'education_learning': {
                'name': 'Education & Learning',
                'description': 'Educational content, classroom language, and learning materials',
                'domain': 'education',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'finance_banking': {
                'name': 'Finance & Banking',
                'description': 'Financial services, banking, investments, and economic terms',
                'domain': 'finance',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'sports_fitness': {
                'name': 'Sports & Fitness',
                'description': 'Sports terminology, fitness, and athletic activities',
                'domain': 'sports',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'automotive_transport': {
                'name': 'Automotive & Transport',
                'description': 'Cars, transportation, mechanics, and automotive industry',
                'domain': 'automotive',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'real_estate': {
                'name': 'Real Estate',
                'description': 'Property, real estate, housing, and rental terminology',
                'domain': 'realestate',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'arts_culture': {
                'name': 'Arts & Culture',
                'description': 'Art, music, literature, and cultural expressions',
                'domain': 'arts',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'government_politics': {
                'name': 'Government & Politics',
                'description': 'Political terminology, government services, and civic language',
                'domain': 'politics',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'social_media': {
                'name': 'Social Media',
                'description': 'Social media language, internet slang, and digital communication',
                'domain': 'social',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'environmental_science': {
                'name': 'Environmental Science',
                'description': 'Environment, climate, ecology, and sustainability terminology',
                'domain': 'environment',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'religious_spiritual': {
                'name': 'Religious & Spiritual',
                'description': 'Religious texts, spiritual concepts, and faith-based language',
                'domain': 'religious',
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ar', 'hi', 'ur', 'ru']
            },
            'pharmaceutical': {
                'name': 'Pharmaceutical',
                'description': 'Drug names, pharmacy, medication instructions, and pharmaceutical industry',
                'domain': 'pharmaceutical',
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
        
        # Medical terminology - Enhanced with temperature, health conditions, and conversational medical phrases
        medical_data = [
            # Basic vital signs and measurements
            {"en": "Blood pressure", "es": "Presión arterial", "fr": "Tension artérielle", "de": "Blutdruck", "ur": "بلڈ پریشر"},
            {"en": "Heart rate", "es": "Frecuencia cardíaca", "fr": "Rythme cardiaque", "de": "Herzfrequenz", "ur": "دل کی رفتار"},
            {"en": "Body temperature", "es": "Temperatura corporal", "fr": "Température corporelle", "de": "Körpertemperatur", "ur": "جسم کا درجہ حرارت"},
            {"en": "Check your temperature", "es": "Revisa tu temperatura", "fr": "Vérifiez votre température", "de": "Überprüfen Sie Ihre Temperatur", "ur": "اپنا درجہ حرارت چیک کریں"},
            
            # Temperature and fever related (addressing "you are hot" context)
            {"en": "You have a fever", "es": "Tienes fiebre", "fr": "Vous avez de la fièvre", "de": "Sie haben Fieber", "ur": "آپ کو بخار ہے"},
            {"en": "Feeling hot and feverish", "es": "Sintiéndose caliente y febril", "fr": "Se sentir chaud et fiévreux", "de": "Heiß und fiebrig fühlen", "ur": "بخار اور گرمی محسوس کرنا"},
            {"en": "You feel warm", "es": "Te sientes caliente", "fr": "Vous vous sentez chaud", "de": "Sie fühlen sich warm", "ur": "آپ گرم محسوس کر رہے ہیں"},
            {"en": "Running a temperature", "es": "Teniendo temperatura alta", "fr": "Avoir de la température", "de": "Temperatur haben", "ur": "تیز بخار ہے"},
            
            # Mental health and mood (addressing "feeling blue" context)
            {"en": "Feeling depressed", "es": "Sintiéndose deprimido", "fr": "Se sentir déprimé", "de": "Sich deprimiert fühlen", "ur": "ڈپریشن محسوس کرنا"},
            {"en": "Mental health support", "es": "Apoyo de salud mental", "fr": "Soutien en santé mentale", "de": "Unterstützung für psychische Gesundheit", "ur": "ذہنی صحت کی مدد"},
            {"en": "Feeling sad or blue", "es": "Sintiéndose triste", "fr": "Se sentir triste ou déprimé", "de": "Sich traurig oder niedergeschlagen fühlen", "ur": "اداس یا پریشان محسوس کرنا"},
            {"en": "Mood disorders", "es": "Trastornos del estado de ánimo", "fr": "Troubles de l'humeur", "de": "Stimmungsstörungen", "ur": "موڈ کی خرابی"},
            
            # Medical instructions and care
            {"en": "Take this medication twice daily", "es": "Tome este medicamento dos veces al día", "fr": "Prenez ce médicament deux fois par jour", "de": "Nehmen Sie dieses Medikament zweimal täglich", "ur": "یہ دوا روزانہ دو بار لیں"},
            {"en": "You need to rest", "es": "Necesitas descansar", "fr": "Vous devez vous reposer", "de": "Du musst dich ausruhen", "ur": "آپ کو آرام کرنا چاہیے"},
            {"en": "Stay hydrated", "es": "Manténgase hidratado", "fr": "Restez hydraté", "de": "Bleiben Sie hydratisiert", "ur": "پانی پیتے رہیں"},
            {"en": "Get plenty of rest", "es": "Descanse mucho", "fr": "Reposez-vous beaucoup", "de": "Viel Ruhe bekommen", "ur": "خوب آرام کریں"},
            
            # Medical facilities and processes
            {"en": "Emergency room", "es": "Sala de emergencias", "fr": "Salle d'urgence", "de": "Notaufnahme", "ur": "ایمرجنسی روم"},
            {"en": "Medical history", "es": "Historia médica", "fr": "Antécédents médicaux", "de": "Krankengeschichte", "ur": "طبی تاریخ"},
            {"en": "Prescription", "es": "Receta médica", "fr": "Ordonnance", "de": "Rezept", "ur": "نسخہ"},
            {"en": "Symptoms", "es": "Síntomas", "fr": "Symptômes", "de": "Symptome", "ur": "علامات"},
            {"en": "Diagnosis", "es": "Diagnóstico", "fr": "Diagnostic", "de": "Diagnose", "ur": "تشخیص"},
            {"en": "Treatment", "es": "Tratamiento", "fr": "Traitement", "de": "Behandlung", "ur": "علاج"},
            
            # Health conditions and complaints
            {"en": "Not feeling well", "es": "No me siento bien", "fr": "Je ne me sens pas bien", "de": "Fühle mich nicht wohl", "ur": "طبیعت ٹھیک نہیں"},
            {"en": "Under the weather", "es": "Sintiéndose mal", "fr": "Pas dans son assiette", "de": "Nicht ganz fit", "ur": "طبیعت خراب ہے"},
            {"en": "Health checkup", "es": "Chequeo médico", "fr": "Bilan de santé", "de": "Gesundheitscheck", "ur": "طبی معائنہ"}
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
        
        # Legal & Formal
        legal_data = [
            {"en": "Terms and conditions", "es": "Términos y condiciones", "fr": "Conditions générales", "de": "Geschäftsbedingungen", "ur": "شرائط و ضوابط"},
            {"en": "Legal agreement", "es": "Acuerdo legal", "fr": "Accord juridique", "de": "Rechtsvereinbarung", "ur": "قانونی معاہدہ"},
            {"en": "Court hearing", "es": "Audiencia judicial", "fr": "Audience du tribunal", "de": "Gerichtsverhandlung", "ur": "عدالتی سماعت"},
            {"en": "Power of attorney", "es": "Poder legal", "fr": "Procuration", "de": "Vollmacht", "ur": "وکالت نامہ"},
            {"en": "Contract negotiation", "es": "Negociación de contrato", "fr": "Négociation de contrat", "de": "Vertragsverhandlung", "ur": "کنٹریکٹ کی بات چیت"},
            {"en": "Intellectual property", "es": "Propiedad intelectual", "fr": "Propriété intellectuelle", "de": "Geistiges Eigentum", "ur": "دانشورانہ املاک"},
            {"en": "Legal liability", "es": "Responsabilidad legal", "fr": "Responsabilité légale", "de": "Rechtliche Haftung", "ur": "قانونی ذمہ داری"},
            {"en": "Due diligence", "es": "Diligencia debida", "fr": "Diligence raisonnable", "de": "Sorgfaltsprüfung", "ur": "محتاط جانچ"},
            {"en": "Compliance requirements", "es": "Requisitos de cumplimiento", "fr": "Exigences de conformité", "de": "Compliance-Anforderungen", "ur": "تعمیل کی ضروریات"},
            {"en": "Legal documentation", "es": "Documentación legal", "fr": "Documentation juridique", "de": "Rechtsdokumentation", "ur": "قانونی دستاویزات"}
        ]

        # Academic & Research
        academic_data = [
            {"en": "Research methodology", "es": "Metodología de investigación", "fr": "Méthodologie de recherche", "de": "Forschungsmethodik", "ur": "تحقیقی طریقہ کار"},
            {"en": "Peer review", "es": "Revisión por pares", "fr": "Évaluation par les pairs", "de": "Peer-Review", "ur": "ہم مرتبہ جائزہ"},
            {"en": "Literature review", "es": "Revisión de literatura", "fr": "Revue de littérature", "de": "Literaturübersicht", "ur": "ادبی جائزہ"},
            {"en": "Statistical analysis", "es": "Análisis estadístico", "fr": "Analyse statistique", "de": "Statistische Analyse", "ur": "شماریاتی تجزیہ"},
            {"en": "Academic conference", "es": "Conferencia académica", "fr": "Conférence académique", "de": "Akademische Konferenz", "ur": "علمی کانفرنس"},
            {"en": "Dissertation defense", "es": "Defensa de tesis", "fr": "Soutenance de thèse", "de": "Dissertationsverteidigung", "ur": "مقالے کا دفاع"},
            {"en": "Citation format", "es": "Formato de cita", "fr": "Format de citation", "de": "Zitierformat", "ur": "حوالہ جات کا انداز"},
            {"en": "Hypothesis testing", "es": "Prueba de hipótesis", "fr": "Test d'hypothèse", "de": "Hypothesentest", "ur": "فرضیے کی جانچ"},
            {"en": "Data collection", "es": "Recolección de datos", "fr": "Collecte de données", "de": "Datensammlung", "ur": "ڈیٹا اکٹھا کرنا"},
            {"en": "Scholarly publication", "es": "Publicación académica", "fr": "Publication académique", "de": "Wissenschaftliche Publikation", "ur": "علمی اشاعت"}
        ]

        # Travel & Tourism
        travel_data = [
            {"en": "Hotel reservation", "es": "Reserva de hotel", "fr": "Réservation d'hôtel", "de": "Hotelreservierung", "ur": "ہوٹل کی بکنگ"},
            {"en": "Flight booking", "es": "Reserva de vuelo", "fr": "Réservation de vol", "de": "Flugbuchung", "ur": "پرواز کی بکنگ"},
            {"en": "Tourist attraction", "es": "Atracción turística", "fr": "Attraction touristique", "de": "Touristenattraktion", "ur": "سیاحتی مقام"},
            {"en": "Local cuisine", "es": "Cocina local", "fr": "Cuisine locale", "de": "Lokale Küche", "ur": "مقامی کھانا"},
            {"en": "Tour guide", "es": "Guía turístico", "fr": "Guide touristique", "de": "Reiseführer", "ur": "سیاحتی گائیڈ"},
            {"en": "Currency exchange", "es": "Cambio de moneda", "fr": "Change de devises", "de": "Geldwechsel", "ur": "کرنسی تبدیلی"},
            {"en": "Travel insurance", "es": "Seguro de viaje", "fr": "Assurance voyage", "de": "Reiseversicherung", "ur": "سفری انشورنس"},
            {"en": "Public transportation", "es": "Transporte público", "fr": "Transport public", "de": "Öffentliche Verkehrsmittel", "ur": "عوامی ٹرانسپورٹ"},
            {"en": "Cultural experience", "es": "Experiencia cultural", "fr": "Expérience culturelle", "de": "Kulturelle Erfahrung", "ur": "ثقافتی تجربہ"},
            {"en": "Travel itinerary", "es": "Itinerario de viaje", "fr": "Itinéraire de voyage", "de": "Reiseroute", "ur": "سفری منصوبہ"}
        ]

        # Culinary & Food
        culinary_data = [
            {"en": "Menu selection", "es": "Selección del menú", "fr": "Sélection du menu", "de": "Menüauswahl", "ur": "مینو کا انتخاب"},
            {"en": "Cooking technique", "es": "Técnica de cocina", "fr": "Technique de cuisine", "de": "Kochtechnik", "ur": "پکانے کی تکنیک"},
            {"en": "Fresh ingredients", "es": "Ingredientes frescos", "fr": "Ingrédients frais", "de": "Frische Zutaten", "ur": "تازہ اجزاء"},
            {"en": "Dietary restrictions", "es": "Restricciones dietéticas", "fr": "Restrictions alimentaires", "de": "Diätbeschränkungen", "ur": "غذائی پابندیاں"},
            {"en": "Food allergy", "es": "Alergia alimentaria", "fr": "Allergie alimentaire", "de": "Nahrungsmittelallergie", "ur": "کھانے سے الرجی"},
            {"en": "Recipe instructions", "es": "Instrucciones de receta", "fr": "Instructions de recette", "de": "Rezeptanweisungen", "ur": "ترکیب کی ہدایات"},
            {"en": "Restaurant review", "es": "Reseña de restaurante", "fr": "Critique de restaurant", "de": "Restaurantbewertung", "ur": "ریستوران کا جائزہ"},
            {"en": "Wine pairing", "es": "Maridaje de vinos", "fr": "Accord mets et vins", "de": "Weinpaarung", "ur": "شراب کا جوڑ"},
            {"en": "Kitchen equipment", "es": "Equipo de cocina", "fr": "Équipement de cuisine", "de": "Küchenausstattung", "ur": "باورچی خانے کا سامان"},
            {"en": "Food presentation", "es": "Presentación de comida", "fr": "Présentation culinaire", "de": "Speisenpräsentation", "ur": "کھانے کی پیشکش"}
        ]

        # Education & Learning
        education_data = [
            {"en": "Lesson plan", "es": "Plan de lección", "fr": "Plan de cours", "de": "Unterrichtsplan", "ur": "سبق کا منصوبہ"},
            {"en": "Student assessment", "es": "Evaluación de estudiantes", "fr": "Évaluation des étudiants", "de": "Schülerbewertung", "ur": "طلباء کا جائزہ"},
            {"en": "Learning objectives", "es": "Objetivos de aprendizaje", "fr": "Objectifs d'apprentissage", "de": "Lernziele", "ur": "سیکھنے کے مقاصد"},
            {"en": "Educational resources", "es": "Recursos educativos", "fr": "Ressources éducatives", "de": "Bildungsressourcen", "ur": "تعلیمی وسائل"},
            {"en": "Classroom management", "es": "Gestión del aula", "fr": "Gestion de classe", "de": "Klassenmanagement", "ur": "کلاس روم کا انتظام"},
            {"en": "Academic achievement", "es": "Logro académico", "fr": "Réussite académique", "de": "Akademische Leistung", "ur": "تعلیمی کامیابی"},
            {"en": "Distance learning", "es": "Aprendizaje a distancia", "fr": "Apprentissage à distance", "de": "Fernunterricht", "ur": "فاصلاتی تعلیم"},
            {"en": "Educational technology", "es": "Tecnología educativa", "fr": "Technologie éducative", "de": "Bildungstechnologie", "ur": "تعلیمی ٹیکنالوجی"},
            {"en": "Study schedule", "es": "Horario de estudio", "fr": "Horaire d'étude", "de": "Stundenplan", "ur": "مطالعے کا شیڈول"},
            {"en": "Parent-teacher meeting", "es": "Reunión de padres y maestros", "fr": "Réunion parents-enseignants", "de": "Eltern-Lehrer-Gespräch", "ur": "والدین اور استاد کی ملاقات"}
        ]

        # Finance & Banking
        finance_data = [
            {"en": "Investment portfolio", "es": "Cartera de inversiones", "fr": "Portefeuille d'investissement", "de": "Investmentportfolio", "ur": "سرمایہ کاری کا پورٹ فولیو"},
            {"en": "Credit score", "es": "Puntuación crediticia", "fr": "Score de crédit", "de": "Kreditwürdigkeit", "ur": "کریڈٹ سکور"},
            {"en": "Mortgage application", "es": "Solicitud de hipoteca", "fr": "Demande d'hypothèque", "de": "Hypothekenantrag", "ur": "رہن کی درخواست"},
            {"en": "Interest rate", "es": "Tasa de interés", "fr": "Taux d'intérêt", "de": "Zinssatz", "ur": "سود کی شرح"},
            {"en": "Financial planning", "es": "Planificación financiera", "fr": "Planification financière", "de": "Finanzplanung", "ur": "مالی منصوبہ بندی"},
            {"en": "Stock market", "es": "Mercado de valores", "fr": "Marché boursier", "de": "Aktienmarkt", "ur": "سٹاک مارکیٹ"},
            {"en": "Retirement savings", "es": "Ahorros para la jubilación", "fr": "Épargne retraite", "de": "Altersvorsorge", "ur": "ریٹائرمنٹ کی بچت"},
            {"en": "Banking fees", "es": "Tarifas bancarias", "fr": "Frais bancaires", "de": "Bankgebühren", "ur": "بینک کی فیسیں"},
            {"en": "Currency trading", "es": "Comercio de divisas", "fr": "Trading de devises", "de": "Devisenhandel", "ur": "کرنسی ٹریڈنگ"},
            {"en": "Risk management", "es": "Gestión de riesgos", "fr": "Gestion des risques", "de": "Risikomanagement", "ur": "خطرات کا انتظام"}
        ]

        # Sports & Fitness
        sports_data = [
            {"en": "Training program", "es": "Programa de entrenamiento", "fr": "Programme d'entraînement", "de": "Trainingsprogramm", "ur": "تربیتی پروگرام"},
            {"en": "Athletic performance", "es": "Rendimiento atlético", "fr": "Performance athlétique", "de": "Sportliche Leistung", "ur": "کھیلوں کی کارکردگی"},
            {"en": "Killer performance", "es": "Rendimiento increíble", "fr": "Performance fantastique", "de": "Fantastische Leistung", "ur": "شاندار کارکردگی"},
            {"en": "Absolutely killed it", "es": "Lo hizo increíblemente bien", "fr": "C'était fantastique", "de": "Hat es fantastisch gemacht", "ur": "شاندار کارکردگی کی"},
            {"en": "Crushing the game", "es": "Dominando el juego", "fr": "Dominant le jeu", "de": "Das Spiel beherrschen", "ur": "کھیل میں غالب"},
            {"en": "Smashing records", "es": "Rompiendo récords", "fr": "Battre des records", "de": "Rekorde brechen", "ur": "ریکارڈ توڑنا"},
            {"en": "Beast mode activated", "es": "Modo bestia activado", "fr": "Mode bête activé", "de": "Biest-Modus aktiviert", "ur": "شیر کی طرح کھیل"},
            {"en": "On fire today", "es": "En llamas hoy", "fr": "En feu aujourd'hui", "de": "Heute on fire", "ur": "آج زبردست فارم میں"},
            {"en": "Absolutely crushing it", "es": "Absolutamente aplastándolo", "fr": "Absolument écrasant", "de": "Absolut zermalmen", "ur": "مکمل طور پر دبدبہ"},
            {"en": "Dominating the field", "es": "Dominando el campo", "fr": "Dominant le terrain", "de": "Das Feld beherrschen", "ur": "میدان میں غالب"},
            {"en": "That was insane", "es": "Eso fue increíble", "fr": "C'était dingue", "de": "Das war wahnsinnig", "ur": "یہ تو کمال تھا"},
            {"en": "Totally nailed it", "es": "Lo clavó totalmente", "fr": "Totalement réussi", "de": "Voll getroffen", "ur": "مکمل طور پر کامیاب"},
            {"en": "Playing out of their mind", "es": "Jugando increíblemente", "fr": "Joue de façon incroyable", "de": "Unglaublich spielen", "ur": "شاندار کھیل رہا ہے"},
            {"en": "Fitness goal", "es": "Objetivo de fitness", "fr": "Objectif de remise en forme", "de": "Fitnessziel", "ur": "فٹنس کا ہدف"},
            {"en": "Sports equipment", "es": "Equipo deportivo", "fr": "Équipement sportif", "de": "Sportausrüstung", "ur": "کھیل کا سامان"},
            {"en": "Team strategy", "es": "Estrategia de equipo", "fr": "Stratégie d'équipe", "de": "Teamstrategie", "ur": "ٹیم کی حکمت عملی"},
            {"en": "Championship tournament", "es": "Torneo de campeonato", "fr": "Tournoi de championnat", "de": "Meisterschaftsturnier", "ur": "چیمپئن شپ ٹورنامنٹ"},
            {"en": "Injury prevention", "es": "Prevención de lesiones", "fr": "Prévention des blessures", "de": "Verletzungsprävention", "ur": "چوٹ سے بچاؤ"},
            {"en": "Nutrition plan", "es": "Plan de nutrición", "fr": "Plan nutritionnel", "de": "Ernährungsplan", "ur": "غذائی منصوبہ"},
            {"en": "Exercise routine", "es": "Rutina de ejercicios", "fr": "Routine d'exercices", "de": "Trainingsroutine", "ur": "ورزش کا طریقہ"},
            {"en": "Physical therapy", "es": "Fisioterapia", "fr": "Kinésithérapie", "de": "Physiotherapie", "ur": "فزیو تھراپی"}
        ]

        # Additional categories data (continuing with remaining categories)
        automotive_data = [
            {"en": "Engine maintenance", "es": "Mantenimiento del motor", "fr": "Entretien moteur", "de": "Motorwartung", "ur": "انجن کی دیکھ بھال"},
            {"en": "Brake system", "es": "Sistema de frenos", "fr": "Système de freinage", "de": "Bremssystem", "ur": "بریک سسٹم"},
            {"en": "Vehicle inspection", "es": "Inspección del vehículo", "fr": "Inspection du véhicule", "de": "Fahrzeuginspektion", "ur": "گاڑی کا معائنہ"},
            {"en": "Car insurance", "es": "Seguro de automóvil", "fr": "Assurance automobile", "de": "Autoversicherung", "ur": "کار انشورنس"},
            {"en": "Fuel efficiency", "es": "Eficiencia de combustible", "fr": "Efficacité énergétique", "de": "Kraftstoffeffizienz", "ur": "ایندھن کی بچت"}
        ]

        real_estate_data = [
            {"en": "Property valuation", "es": "Valuación de propiedad", "fr": "Évaluation immobilière", "de": "Immobilienbewertung", "ur": "جائیداد کی قیمت"},
            {"en": "Rental agreement", "es": "Acuerdo de alquiler", "fr": "Contrat de location", "de": "Mietvertrag", "ur": "کرایے کا معاہدہ"},
            {"en": "Home inspection", "es": "Inspección de vivienda", "fr": "Inspection domiciliaire", "de": "Hausinspektion", "ur": "گھر کا معائنہ"},
            {"en": "Mortgage calculator", "es": "Calculadora de hipoteca", "fr": "Calculateur d'hypothèque", "de": "Hypothekenrechner", "ur": "گروی کیلکولیٹر"},
            {"en": "Property management", "es": "Gestión de propiedades", "fr": "Gestion immobilière", "de": "Immobilienverwaltung", "ur": "جائیداد کا انتظام"}
        ]

        arts_culture_data = [
            {"en": "Art exhibition", "es": "Exposición de arte", "fr": "Exposition d'art", "de": "Kunstausstellung", "ur": "آرٹ کی نمائش"},
            {"en": "Musical performance", "es": "Actuación musical", "fr": "Performance musicale", "de": "Musikaufführung", "ur": "موسیقی کی پرفارمنس"},
            {"en": "Cultural heritage", "es": "Patrimonio cultural", "fr": "Patrimoine culturel", "de": "Kulturerbe", "ur": "ثقافتی ورثہ"},
            {"en": "Theater production", "es": "Producción teatral", "fr": "Production théâtrale", "de": "Theaterproduktion", "ur": "تھیٹر پروڈکشن"},
            {"en": "Creative writing", "es": "Escritura creativa", "fr": "Écriture créative", "de": "Kreatives Schreiben", "ur": "تخلیقی تحریر"},
            {"en": "Dance choreography", "es": "Coreografía de danza", "fr": "Chorégraphie de danse", "de": "Tanzchoreografie", "ur": "ڈانس کوریوگرافی"},
            {"en": "Film production", "es": "Producción cinematográfica", "fr": "Production cinématographique", "de": "Filmproduktion", "ur": "فلم پروڈکشن"},
            {"en": "Literary criticism", "es": "Crítica literaria", "fr": "Critique littéraire", "de": "Literaturkritik", "ur": "ادبی تنقید"},
            {"en": "Museum collection", "es": "Colección de museo", "fr": "Collection de musée", "de": "Museumssammlung", "ur": "عجائب گھر کا مجموعہ"},
            {"en": "Artistic technique", "es": "Técnica artística", "fr": "Technique artistique", "de": "Künstlerische Technik", "ur": "فنکارانہ تکنیک"}
        ]

        # Government & Politics
        government_data = [
            {"en": "Public policy", "es": "Política pública", "fr": "Politique publique", "de": "Öffentliche Politik", "ur": "عوامی پالیسی"},
            {"en": "Legislative process", "es": "Proceso legislativo", "fr": "Processus législatif", "de": "Gesetzgebungsverfahren", "ur": "قانون سازی کا عمل"},
            {"en": "Voter registration", "es": "Registro de votantes", "fr": "Inscription des électeurs", "de": "Wählerregistrierung", "ur": "ووٹر رجسٹریشن"},
            {"en": "Government services", "es": "Servicios gubernamentales", "fr": "Services gouvernementaux", "de": "Regierungsdienstleistungen", "ur": "سرکاری خدمات"},
            {"en": "Political campaign", "es": "Campaña política", "fr": "Campagne politique", "de": "Politischer Wahlkampf", "ur": "سیاسی مہم"},
            {"en": "Civic duty", "es": "Deber cívico", "fr": "Devoir civique", "de": "Bürgerpflicht", "ur": "شہری فریضہ"},
            {"en": "Electoral system", "es": "Sistema electoral", "fr": "Système électoral", "de": "Wahlsystem", "ur": "انتخابی نظام"},
            {"en": "Constitutional rights", "es": "Derechos constitucionales", "fr": "Droits constitutionnels", "de": "Verfassungsrechte", "ur": "آئینی حقوق"},
            {"en": "Public administration", "es": "Administración pública", "fr": "Administration publique", "de": "Öffentliche Verwaltung", "ur": "عوامی انتظامیہ"},
            {"en": "Democratic process", "es": "Proceso democrático", "fr": "Processus démocratique", "de": "Demokratischer Prozess", "ur": "جمہوری عمل"}
        ]

        # Social Media - Enhanced with conversational and colloquial expressions
        social_media_data = [
            # Compliments and appearance
            {"en": "You look amazing", "es": "Te ves increíble", "fr": "Tu es magnifique", "de": "Du siehst fantastisch aus", "ur": "آپ بہت خوبصورت لگ رہے ہیں"},
            {"en": "You are so hot", "es": "Estás muy guapo/a", "fr": "Tu es très sexy", "de": "Du bist so heiß", "ur": "آپ بہت خوبصورت ہیں"},
            {"en": "Looking good today", "es": "Te ves bien hoy", "fr": "Tu es beau aujourd'hui", "de": "Du siehst heute gut aus", "ur": "آج آپ اچھے لگ رہے ہیں"},
            {"en": "Stunning outfit", "es": "Atuendo impresionante", "fr": "Tenue magnifique", "de": "Atemberaubendes Outfit", "ur": "شاندار لباس"},
            
            # Emotions and feelings (casual)
            {"en": "I'm feeling blue", "es": "Me siento triste", "fr": "Je me sens triste", "de": "Mir ist traurig", "ur": "میں اداس محسوس کر رہا ہوں"},
            {"en": "Feeling down today", "es": "Me siento mal hoy", "fr": "Je me sens déprimé aujourd'hui", "de": "Heute fühle ich mich schlecht", "ur": "آج پریشان ہوں"},
            {"en": "On cloud nine", "es": "En las nubes", "fr": "Aux anges", "de": "Im siebten Himmel", "ur": "بہت خوش ہوں"},
            {"en": "Living my best life", "es": "Viviendo mi mejor vida", "fr": "Je vis ma meilleure vie", "de": "Ich lebe mein bestes Leben", "ur": "اپنی بہترین زندگی گزار رہا ہوں"},
            
            # Performance and achievements (casual)
            {"en": "That was fire", "es": "Eso estuvo genial", "fr": "C'était génial", "de": "Das war der Hammer", "ur": "یہ شاندار تھا"},
            {"en": "Absolutely killed it", "es": "Lo mataste", "fr": "Tu as assuré", "de": "Du hast es gerockt", "ur": "بہترین کارکردگی"},
            {"en": "Nailed the performance", "es": "Clavaste la actuación", "fr": "Tu as réussi ta performance", "de": "Die Leistung war perfekt", "ur": "کامیاب پرفارمنس"},
            {"en": "Epic win", "es": "Victoria épica", "fr": "Victoire épique", "de": "Epischer Sieg", "ur": "شاندار کامیابی"},
            
            # Technology and digital life
            {"en": "Phone died on me", "es": "Se me murió el teléfono", "fr": "Mon téléphone est mort", "de": "Mein Handy ist kaputt", "ur": "فون کی بیٹری ختم ہو گئی"},
            {"en": "Need to charge up", "es": "Necesito cargar", "fr": "Je dois recharger", "de": "Muss aufladen", "ur": "چارج کرنا ہے"},
            {"en": "Going viral", "es": "Se está volviendo viral", "fr": "Ça devient viral", "de": "Wird viral", "ur": "وائرل ہو رہا ہے"},
            {"en": "Posted a story", "es": "Publiqué una historia", "fr": "J'ai posté une story", "de": "Story gepostet", "ur": "اسٹوری پوسٹ کی"},
            
            # Social interactions
            {"en": "Slide into DMs", "es": "Escribir al privado", "fr": "Envoyer un message privé", "de": "Private Nachricht senden", "ur": "پرائیویٹ میسج بھیجنا"},
            {"en": "Double tap this", "es": "Dale doble tap", "fr": "Double-cliquez", "de": "Doppelt antippen", "ur": "ڈبل ٹیپ کریں"},
            {"en": "Drop a comment", "es": "Deja un comentario", "fr": "Laisse un commentaire", "de": "Kommentier das", "ur": "کمنٹ کریں"},
            {"en": "Tag your squad", "es": "Etiqueta a tu grupo", "fr": "Tague tes potes", "de": "Markiere deine Crew", "ur": "اپنے دوستوں کو ٹیگ کریں"},
            
            # Slang and expressions
            {"en": "No cap", "es": "Sin mentir", "fr": "Sans mentir", "de": "Echt jetzt", "ur": "سچ میں"},
            {"en": "It's giving vibes", "es": "Da buenas vibras", "fr": "Ça donne des bonnes vibes", "de": "Hat gute Vibes", "ur": "اچھا لگ رہا ہے"},
            {"en": "Periodt", "es": "Punto final", "fr": "Point final", "de": "Punkt aus", "ur": "بس اتنا ہی"},
            {"en": "Slay queen", "es": "Reina poderosa", "fr": "Reine puissante", "de": "Starke Königin", "ur": "شاندار ملکہ"},
            
            # Events and activities
            {"en": "The party was dead", "es": "La fiesta estuvo aburrida", "fr": "La fête était morte", "de": "Die Party war tot", "ur": "پارٹی بورنگ تھی"},
            {"en": "Had a blast", "es": "Me divertí mucho", "fr": "Je me suis éclaté", "de": "Hatte einen Riesenspaß", "ur": "بہت مزہ آیا"},
            {"en": "FOMO is real", "es": "El miedo a perdérselo es real", "fr": "La peur de rater quelque chose", "de": "Angst etwas zu verpassen", "ur": "کچھ چھوٹ جانے کا ڈر"},
            {"en": "Living for this", "es": "Vivo para esto", "fr": "Je vis pour ça", "de": "Dafür lebe ich", "ur": "یہی میری زندگی ہے"}
        ]

        # Environmental Science
        environmental_data = [
            {"en": "Carbon footprint", "es": "Huella de carbono", "fr": "Empreinte carbone", "de": "CO2-Fußabdruck", "ur": "کاربن فوٹ پرنٹ"},
            {"en": "Renewable energy", "es": "Energía renovable", "fr": "Énergie renouvelable", "de": "Erneuerbare Energie", "ur": "قابل تجدید توانائی"},
            {"en": "Ecosystem preservation", "es": "Preservación del ecosistema", "fr": "Préservation de l'écosystème", "de": "Ökosystemerhaltung", "ur": "ماحولیاتی نظام کا تحفظ"},
            {"en": "Sustainable development", "es": "Desarrollo sostenible", "fr": "Développement durable", "de": "Nachhaltige Entwicklung", "ur": "پائیدار ترقی"},
            {"en": "Climate adaptation", "es": "Adaptación climática", "fr": "Adaptation climatique", "de": "Klimaanpassung", "ur": "آب و ہوا کے ساتھ ہم آہنگی"},
            {"en": "Biodiversity conservation", "es": "Conservación de la biodiversidad", "fr": "Conservation de la biodiversité", "de": "Biodiversitätsschutz", "ur": "حیاتیاتی تنوع کا تحفظ"},
            {"en": "Pollution control", "es": "Control de la contaminación", "fr": "Contrôle de la pollution", "de": "Umweltverschmutzungskontrolle", "ur": "آلودگی کنٹرول"},
            {"en": "Waste management", "es": "Gestión de residuos", "fr": "Gestion des déchets", "de": "Abfallwirtschaft", "ur": "فضلہ کا انتظام"},
            {"en": "Environmental impact", "es": "Impacto ambiental", "fr": "Impact environnemental", "de": "Umweltauswirkung", "ur": "ماحولیاتی اثرات"},
            {"en": "Green technology", "es": "Tecnología verde", "fr": "Technologie verte", "de": "Grüne Technologie", "ur": "سبز ٹیکنالوجی"}
        ]

        # Religious & Spiritual
        religious_data = [
            {"en": "Spiritual guidance", "es": "Orientación espiritual", "fr": "Guidance spirituelle", "de": "Spirituelle Führung", "ur": "روحانی رہنمائی"},
            {"en": "Religious ceremony", "es": "Ceremonia religiosa", "fr": "Cérémonie religieuse", "de": "Religiöse Zeremonie", "ur": "مذہبی تقریب"},
            {"en": "Faith community", "es": "Comunidad de fe", "fr": "Communauté de foi", "de": "Glaubensgemeinschaft", "ur": "ایمانی برادری"},
            {"en": "Sacred text", "es": "Texto sagrado", "fr": "Texte sacré", "de": "Heiliger Text", "ur": "مقدس کتاب"},
            {"en": "Prayer service", "es": "Servicio de oración", "fr": "Service de prière", "de": "Gebetsgottesdienst", "ur": "نماز کی خدمت"},
            {"en": "Meditation practice", "es": "Práctica de meditación", "fr": "Pratique de méditation", "de": "Meditationspraxis", "ur": "مراقبے کی مشق"},
            {"en": "Religious holiday", "es": "Festividad religiosa", "fr": "Fête religieuse", "de": "Religiöser Feiertag", "ur": "مذہبی تہوار"},
            {"en": "Moral teaching", "es": "Enseñanza moral", "fr": "Enseignement moral", "de": "Moralische Lehre", "ur": "اخلاقی تعلیم"},
            {"en": "Spiritual journey", "es": "Viaje espiritual", "fr": "Voyage spirituel", "de": "Spirituelle Reise", "ur": "روحانی سفر"},
            {"en": "Divine blessing", "es": "Bendición divina", "fr": "Bénédiction divine", "de": "Göttlicher Segen", "ur": "الہی برکت"}
        ]

        # Pharmaceutical
        pharmaceutical_data = [
            {"en": "Drug interaction", "es": "Interacción de medicamentos", "fr": "Interaction médicamenteuse", "de": "Arzneimittelwechselwirkung", "ur": "دوا کا باہمی اثر"},
            {"en": "Clinical trial", "es": "Ensayo clínico", "fr": "Essai clinique", "de": "Klinische Studie", "ur": "کلینیکل ٹرائل"},
            {"en": "Prescription label", "es": "Etiqueta de prescripción", "fr": "Étiquette d'ordonnance", "de": "Verschreibungsetikett", "ur": "نسخے کا لیبل"},
            {"en": "Generic medication", "es": "Medicamento genérico", "fr": "Médicament générique", "de": "Generikum", "ur": "عام دوا"},
            {"en": "Side effects", "es": "Efectos secundarios", "fr": "Effets secondaires", "de": "Nebenwirkungen", "ur": "ضمنی اثرات"},
            {"en": "Dosage instructions", "es": "Instrucciones de dosificación", "fr": "Instructions de dosage", "de": "Dosierungsanweisungen", "ur": "خوراک کی ہدایات"},
            {"en": "Pharmacy consultation", "es": "Consulta farmacéutica", "fr": "Consultation pharmaceutique", "de": "Apothekenberatung", "ur": "فارمیسی مشاورت"},
            {"en": "Drug safety", "es": "Seguridad del medicamento", "fr": "Sécurité du médicament", "de": "Arzneimittelsicherheit", "ur": "دوا کی حفاظت"},
            {"en": "Medication management", "es": "Gestión de medicamentos", "fr": "Gestion des médicaments", "de": "Medikamentenmanagement", "ur": "دوا کا انتظام"},
            {"en": "Pharmaceutical research", "es": "Investigación farmacéutica", "fr": "Recherche pharmaceutique", "de": "Pharmazeutische Forschung", "ur": "دوائی تحقیق"}
        ]

        # Save corpora
        corpus_data = {
            'opus_opensubtitles': opensubtitles_data,
            'medical_terminology': medical_data,
            'business_common': business_data,
            'technical_computing': technical_data,
            'news_current': news_data,
            'legal_formal': legal_data,
            'academic_research': academic_data,
            'travel_tourism': travel_data,
            'culinary_food': culinary_data,
            'education_learning': education_data,
            'finance_banking': finance_data,
            'sports_fitness': sports_data,
            'automotive_transport': automotive_data,
            'real_estate': real_estate_data,
            'arts_culture': arts_culture_data,
            'government_politics': government_data,
            'social_media': social_media_data,
            'environmental_science': environmental_data,
            'religious_spiritual': religious_data,
            'pharmaceutical': pharmaceutical_data
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
