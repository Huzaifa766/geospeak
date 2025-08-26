"""
Final working translation accuracy test for GeoSpeak
"""

from corpus_manager import CorpusManager
import logging
import os

def test_geospeak_accuracy():
    """Test translation accuracy with proper error handling"""
    
    logging.basicConfig(level=logging.WARNING)
    
    print('🧪 GEOSPEAK TRANSLATION ACCURACY TEST')
    print('='*60)
    
    # Initialize
    cm = CorpusManager()
    
    # Test corpora
    test_corpora = ['legal_formal', 'medical_terminology', 'business_common', 'travel_tourism']
    
    print('📚 Loading corpora...')
    loaded_corpora = []
    
    for corpus_name in test_corpora:
        try:
            data = cm.load_corpus(corpus_name)
            if data:
                cm.corpora[corpus_name] = {'data': data}
                if cm.load_vector_index(corpus_name):
                    loaded_corpora.append(corpus_name)
                    print(f'✅ {corpus_name}: {len(data)} entries')
                else:
                    print(f'⚠️  {corpus_name}: loaded data but no index')
            else:
                print(f'❌ {corpus_name}: no data found')
        except Exception as e:
            print(f'❌ {corpus_name}: {e}')
    
    if not loaded_corpora:
        print('❌ No corpora loaded successfully')
        return
    
    print(f'\\n🎯 Domain Classification Testing:')
    print('-'*50)
    
    # Test queries with expected domains
    domain_tests = [
        ('legal agreement', 'legal_formal'),
        ('blood pressure', 'medical_terminology'), 
        ('schedule meeting', 'business_common'),
        ('hotel reservation', 'travel_tourism'),
        ('court hearing', 'legal_formal'),
        ('medical prescription', 'medical_terminology')
    ]
    
    correct_classifications = 0
    total_tests = 0
    
    for query, expected_corpus in domain_tests:
        print(f'\\n🔍 \"{query}\"')
        
        best_score = -1
        best_corpus = None
        best_result = None
        
        # Test against all loaded corpora
        for corpus_name in loaded_corpora:
            try:
                results = cm.search_similar_texts(query, corpus_name, k=1)
                if results:
                    score = results[0].get('similarity_score', 0)
                    if score > best_score:
                        best_score = score
                        best_corpus = corpus_name
                        best_result = results[0]
            except Exception as e:
                print(f'   Error with {corpus_name}: {e}')
                continue
        
        total_tests += 1
        
        if best_result and best_corpus:
            is_correct = best_corpus == expected_corpus
            if is_correct:
                correct_classifications += 1
            
            domain = cm.corpus_sources[best_corpus]['domain']
            status = '✅' if is_correct else '⚠️ '
            
            print(f'   {status} {best_corpus} ({domain}) - Score: {best_score:.3f}')
            print(f'      Match: "{best_result.get("text", "N/A")}"')
            
            # Show translations if available
            translations = best_result.get('translations', {})
            if 'es' in translations:
                print(f'      Spanish: "{translations["es"]}"')
        else:
            print('   ❌ No matches found')
    
    print(f'\\n📊 Translation Quality Testing:')
    print('-'*40)
    
    # Test translation consistency within each corpus
    for corpus_name in loaded_corpora[:2]:  # Test first 2 corpora
        print(f'\\n📚 {corpus_name}:')
        
        data = cm.corpora[corpus_name]['data']
        test_entries = data[:2]  # Test first 2 entries
        
        for i, entry in enumerate(test_entries):
            en_text = entry['en']
            print(f'   Test {i+1}: "{en_text}"')
            
            try:
                # Search for similar entries (should find itself and others)
                results = cm.search_similar_texts(en_text, corpus_name, k=3)
                
                if results:
                    # Check the top matches
                    for j, result in enumerate(results[:2]):
                        score = result.get('similarity_score', 0)
                        matched_text = result.get('text', 'N/A')
                        
                        print(f'     Match {j+1}: {score:.3f} - "{matched_text}"')
                        
                        # Compare Spanish translations
                        orig_es = entry.get('es', '')
                        match_es = result.get('translations', {}).get('es', '')
                        
                        if orig_es and match_es:
                            print(f'       Original ES: "{orig_es}"')
                            print(f'       Matched ES:  "{match_es}"')
                            
                            # Check if it's an exact match or similar
                            if orig_es.lower().strip() == match_es.lower().strip():
                                print('       ✅ Exact translation match')
                            else:
                                print('       📝 Different translation')
                
            except Exception as e:
                print(f'     Error: {e}')
    
    print(f'\\n🏆 FINAL RESULTS:')
    print('='*40)
    print(f'Corpora successfully loaded: {len(loaded_corpora)}/{len(test_corpora)}')
    print(f'Domain classification accuracy: {correct_classifications}/{total_tests} ({correct_classifications/total_tests*100:.1f}%)')
    print(f'Total available categories: {len(cm.corpus_sources)}')
    print(f'Vector search working: {"✅" if best_score > 0 else "❌"}')
    
    print(f'\\n📈 Quality Assessment:')
    if correct_classifications / total_tests >= 0.8:
        print('🏆 Excellent: >80% accuracy')
    elif correct_classifications / total_tests >= 0.6:
        print('👍 Good: 60-80% accuracy') 
    elif correct_classifications / total_tests >= 0.4:
        print('⚠️  Fair: 40-60% accuracy')
    else:
        print('❌ Poor: <40% accuracy')
    
    print(f'\\n🌐 Translation Coverage:')
    sample_corpus = loaded_corpora[0] if loaded_corpora else None
    if sample_corpus and sample_corpus in cm.corpora:
        sample_entry = cm.corpora[sample_corpus]['data'][0]
        supported_langs = [lang for lang in sample_entry.keys() if lang != 'en']
        print(f'Languages per entry: {len(supported_langs)+1} ({", ".join(["en"] + supported_langs)})')
    
    return {
        'loaded_corpora': len(loaded_corpora),
        'total_corpora': len(test_corpora),
        'accuracy': correct_classifications / total_tests if total_tests > 0 else 0,
        'vector_search_working': best_score > 0 if 'best_score' in locals() else False
    }

if __name__ == '__main__':
    results = test_geospeak_accuracy()
    print(f'\\n✨ Test completed successfully!')
