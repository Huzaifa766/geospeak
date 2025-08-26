"""
GeoSpeak Translation Accuracy Testing System
Evaluates translation quality across different domains and languages
"""

import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Tuple, Optional
from corpus_manager import CorpusManager
import time

logger = logging.getLogger(__name__)

class TranslationAccuracyTester:
    def __init__(self, corpus_dir="./corpus_data"):
        self.corpus_manager = CorpusManager(corpus_dir)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.multilingual_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Test languages
        self.test_languages = ['es', 'fr', 'de', 'ur']
        
        # Ground truth test cases for different domains
        self.ground_truth_tests = {
            'medical': [
                {
                    'en': 'Take this medication twice daily',
                    'es': 'Tome este medicamento dos veces al día',
                    'fr': 'Prenez ce médicament deux fois par jour',
                    'de': 'Nehmen Sie dieses Medikament zweimal täglich',
                    'ur': 'یہ دوا روزانہ دو بار لیں'
                },
                {
                    'en': 'Blood pressure measurement',
                    'es': 'Medición de presión arterial',
                    'fr': 'Mesure de la tension artérielle',
                    'de': 'Blutdruckmessung',
                    'ur': 'بلڈ پریشر کی پیمائش'
                }
            ],
            'business': [
                {
                    'en': 'Schedule a meeting for tomorrow',
                    'es': 'Programar una reunión para mañana',
                    'fr': 'Programmer une réunion pour demain',
                    'de': 'Ein Meeting für morgen planen',
                    'ur': 'کل کے لیے میٹنگ کا وقت مقرر کریں'
                },
                {
                    'en': 'Quarterly financial report',
                    'es': 'Informe financiero trimestral',
                    'fr': 'Rapport financier trimestriel',
                    'de': 'Vierteljährlicher Finanzbericht',
                    'ur': 'سہ ماہی مالی رپورٹ'
                }
            ],
            'legal': [
                {
                    'en': 'Legal agreement and contract',
                    'es': 'Acuerdo legal y contrato',
                    'fr': 'Accord juridique et contrat',
                    'de': 'Rechtsvereinbarung und Vertrag',
                    'ur': 'قانونی معاہدہ اور کنٹریکٹ'
                }
            ],
            'travel': [
                {
                    'en': 'Hotel reservation confirmation',
                    'es': 'Confirmación de reserva de hotel',
                    'fr': 'Confirmation de réservation d\'hôtel',
                    'de': 'Bestätigung der Hotelreservierung',
                    'ur': 'ہوٹل ریزرویشن کی تصدیق'
                }
            ]
        }
        
    def calculate_semantic_similarity(self, text1: str, text2: str, lang1='en', lang2='en') -> float:
        """Calculate semantic similarity between two texts using multilingual embeddings"""
        try:
            # Use multilingual model for cross-language similarity
            if lang1 != lang2:
                embeddings = self.multilingual_model.encode([text1, text2])
            else:
                embeddings = self.embedding_model.encode([text1, text2])
            
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def calculate_bleu_score_simple(self, reference: str, candidate: str) -> float:
        """Simple BLEU-like score based on n-gram overlap"""
        import re
        
        # Simple tokenization
        ref_tokens = re.findall(r'\w+', reference.lower())
        cand_tokens = re.findall(r'\w+', candidate.lower())
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        # Calculate 1-gram, 2-gram overlap
        ref_1grams = set(ref_tokens)
        cand_1grams = set(cand_tokens)
        
        overlap_1 = len(ref_1grams & cand_1grams) / max(len(cand_1grams), 1)
        
        # 2-grams
        ref_2grams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
        cand_2grams = set(zip(cand_tokens[:-1], cand_tokens[1:]))
        
        if ref_2grams and cand_2grams:
            overlap_2 = len(ref_2grams & cand_2grams) / max(len(cand_2grams), 1)
        else:
            overlap_2 = 0.0
        
        # Simple geometric mean
        if overlap_1 > 0 and overlap_2 > 0:
            return (overlap_1 * overlap_2) ** 0.5
        else:
            return overlap_1
    
    def test_domain_accuracy(self, domain: str, corpus_name: str) -> Dict:
        """Test translation accuracy for a specific domain"""
        logger.info(f"Testing domain: {domain} using corpus: {corpus_name}")
        
        results = {
            'domain': domain,
            'corpus_name': corpus_name,
            'language_scores': {},
            'average_semantic_similarity': 0.0,
            'average_bleu_score': 0.0,
            'test_cases_count': 0,
            'errors': []
        }
        
        # Load corpus
        try:
            self.corpus_manager.load_corpus(corpus_name)
            self.corpus_manager.load_vector_index(corpus_name)
        except Exception as e:
            results['errors'].append(f"Failed to load corpus: {e}")
            return results
        
        if domain not in self.ground_truth_tests:
            results['errors'].append(f"No ground truth tests for domain: {domain}")
            return results
        
        test_cases = self.ground_truth_tests[domain]
        results['test_cases_count'] = len(test_cases)
        
        all_semantic_scores = []
        all_bleu_scores = []
        
        for lang in self.test_languages:
            lang_semantic_scores = []
            lang_bleu_scores = []
            
            for test_case in test_cases:
                if lang not in test_case:
                    continue
                
                en_text = test_case['en']
                ground_truth = test_case[lang]
                
                # Find similar translation in corpus
                try:
                    similar_entries = self.corpus_manager.find_similar(en_text, corpus_name, top_k=3)
                    
                    if similar_entries:
                        best_match = similar_entries[0]
                        corpus_translation = best_match['translations'].get(lang, '')
                        
                        if corpus_translation:
                            # Calculate semantic similarity
                            semantic_score = self.calculate_semantic_similarity(
                                ground_truth, corpus_translation, lang, lang
                            )
                            lang_semantic_scores.append(semantic_score)
                            all_semantic_scores.append(semantic_score)
                            
                            # Calculate BLEU-like score
                            bleu_score = self.calculate_bleu_score_simple(ground_truth, corpus_translation)
                            lang_bleu_scores.append(bleu_score)
                            all_bleu_scores.append(bleu_score)
                            
                            logger.debug(f"Test: {en_text}")
                            logger.debug(f"Ground truth ({lang}): {ground_truth}")
                            logger.debug(f"Corpus translation ({lang}): {corpus_translation}")
                            logger.debug(f"Semantic: {semantic_score:.3f}, BLEU: {bleu_score:.3f}")
                        
                except Exception as e:
                    results['errors'].append(f"Error testing {en_text} -> {lang}: {e}")
            
            if lang_semantic_scores:
                results['language_scores'][lang] = {
                    'semantic_similarity': np.mean(lang_semantic_scores),
                    'bleu_score': np.mean(lang_bleu_scores),
                    'test_count': len(lang_semantic_scores)
                }
        
        if all_semantic_scores:
            results['average_semantic_similarity'] = np.mean(all_semantic_scores)
            results['average_bleu_score'] = np.mean(all_bleu_scores)
        
        return results
    
    def test_corpus_retrieval_accuracy(self, test_queries: List[str]) -> Dict:
        """Test how well the system retrieves appropriate domain-specific corpora"""
        logger.info("Testing corpus retrieval accuracy")
        
        # Load several corpora for comparison
        test_corpora = ['medical_terminology', 'business_common', 'legal_formal', 'travel_tourism', 'technical_computing']
        
        for corpus_name in test_corpora:
            try:
                self.corpus_manager.load_corpus(corpus_name)
                self.corpus_manager.load_vector_index(corpus_name)
            except Exception as e:
                logger.error(f"Failed to load {corpus_name}: {e}")
        
        results = {
            'queries_tested': len(test_queries),
            'retrieval_results': [],
            'domain_classification_accuracy': 0.0
        }
        
        expected_domains = {
            'I need a prescription': 'medical',
            'Schedule a meeting': 'business', 
            'Court hearing': 'legal',
            'Hotel reservation': 'travel',
            'Database connection': 'technical',
            'Blood pressure': 'medical',
            'Legal contract': 'legal',
            'Flight booking': 'travel',
            'Software development': 'technical'
        }
        
        correct_classifications = 0
        
        for query in test_queries:
            query_results = {
                'query': query,
                'corpus_scores': {},
                'best_match': None,
                'expected_domain': expected_domains.get(query, 'unknown')
            }
            
            best_score = 0
            best_corpus = None
            
            for corpus_name in test_corpora:
                if corpus_name in self.corpus_manager.corpora:
                    try:
                        matches = self.corpus_manager.find_similar(query, corpus_name, top_k=1)
                        if matches:
                            score = matches[0]['similarity']
                            query_results['corpus_scores'][corpus_name] = score
                            
                            if score > best_score:
                                best_score = score
                                best_corpus = corpus_name
                    except Exception as e:
                        logger.error(f"Error testing query '{query}' on {corpus_name}: {e}")
            
            if best_corpus:
                predicted_domain = self.corpus_manager.corpus_sources[best_corpus]['domain']
                query_results['best_match'] = {
                    'corpus': best_corpus,
                    'domain': predicted_domain,
                    'score': best_score
                }
                
                # Check if classification is correct
                if query in expected_domains and expected_domains[query] == predicted_domain:
                    correct_classifications += 1
            
            results['retrieval_results'].append(query_results)
        
        if test_queries:
            results['domain_classification_accuracy'] = correct_classifications / len(test_queries)
        
        return results
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive accuracy testing"""
        logger.info("Starting comprehensive translation accuracy test")
        start_time = time.time()
        
        # Test individual domains
        domain_tests = [
            ('medical', 'medical_terminology'),
            ('business', 'business_common'),
            ('legal', 'legal_formal'),
            ('travel', 'travel_tourism')
        ]
        
        domain_results = []
        for domain, corpus_name in domain_tests:
            result = self.test_domain_accuracy(domain, corpus_name)
            domain_results.append(result)
        
        # Test corpus retrieval
        test_queries = [
            'I need a prescription', 'Schedule a meeting', 'Court hearing',
            'Hotel reservation', 'Database connection', 'Blood pressure',
            'Legal contract', 'Flight booking', 'Software development'
        ]
        
        retrieval_results = self.test_corpus_retrieval_accuracy(test_queries)
        
        # Calculate overall metrics
        overall_semantic = np.mean([r['average_semantic_similarity'] for r in domain_results if r['average_semantic_similarity'] > 0])
        overall_bleu = np.mean([r['average_bleu_score'] for r in domain_results if r['average_bleu_score'] > 0])
        
        final_results = {
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_duration_seconds': time.time() - start_time,
            'overall_metrics': {
                'semantic_similarity': float(overall_semantic) if not np.isnan(overall_semantic) else 0.0,
                'bleu_score': float(overall_bleu) if not np.isnan(overall_bleu) else 0.0,
                'domain_classification_accuracy': retrieval_results['domain_classification_accuracy']
            },
            'domain_results': domain_results,
            'retrieval_results': retrieval_results,
            'system_info': {
                'total_corpora': len(self.corpus_manager.corpus_sources),
                'languages_tested': self.test_languages,
                'embedding_model': 'all-MiniLM-L6-v2',
                'multilingual_model': 'paraphrase-multilingual-MiniLM-L12-v2'
            }
        }
        
        return final_results
    
    def save_results(self, results: Dict, filename: str = 'translation_accuracy_results.json'):
        """Save test results to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filename}")
    
    def print_summary(self, results: Dict):
        """Print a human-readable summary of test results"""
        print("\n" + "="*80)
        print("🧪 GEOSPEAK TRANSLATION ACCURACY TEST RESULTS")
        print("="*80)
        
        overall = results['overall_metrics']
        print(f"⏱️  Test Duration: {results['test_duration_seconds']:.2f} seconds")
        print(f"📊 Overall Semantic Similarity: {overall['semantic_similarity']:.3f}")
        print(f"📝 Overall BLEU Score: {overall['bleu_score']:.3f}")
        print(f"🎯 Domain Classification Accuracy: {overall['domain_classification_accuracy']:.3f}")
        
        print(f"\n📚 Domain-Specific Results:")
        print("-" * 50)
        
        for domain_result in results['domain_results']:
            domain = domain_result['domain']
            sem_score = domain_result['average_semantic_similarity']
            bleu_score = domain_result['average_bleu_score']
            test_count = domain_result['test_cases_count']
            
            print(f"🏥 {domain.upper()}: Semantic={sem_score:.3f}, BLEU={bleu_score:.3f} ({test_count} tests)")
            
            for lang, scores in domain_result['language_scores'].items():
                sem = scores['semantic_similarity']
                bleu = scores['bleu_score']
                count = scores['test_count']
                print(f"   └─ {lang}: {sem:.3f}/{bleu:.3f} ({count} tests)")
        
        print(f"\n🔍 Corpus Retrieval Test:")
        print("-" * 50)
        
        retrieval = results['retrieval_results']
        for query_result in retrieval['retrieval_results']:
            query = query_result['query']
            if query_result['best_match']:
                domain = query_result['best_match']['domain']
                score = query_result['best_match']['score']
                expected = query_result['expected_domain']
                status = "✅" if domain == expected else "❌"
                print(f"{status} \"{query}\" → {domain} ({score:.3f})")
        
        print(f"\n🏆 System Configuration:")
        print("-" * 50)
        sys_info = results['system_info']
        print(f"Total Corpora: {sys_info['total_corpora']}")
        print(f"Languages Tested: {', '.join(sys_info['languages_tested'])}")
        print(f"Embedding Model: {sys_info['embedding_model']}")
        print("="*80)

def main():
    """Main testing function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Initializing Translation Accuracy Tester...")
    tester = TranslationAccuracyTester()
    
    logger.info("Running comprehensive accuracy tests...")
    results = tester.run_comprehensive_test()
    
    # Save results
    tester.save_results(results)
    
    # Print summary
    tester.print_summary(results)
    
    return results

if __name__ == "__main__":
    main()
