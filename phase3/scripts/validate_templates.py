import sys
import sentencepiece as spm
sys.path.append('c:/Users/arped/OneDrive/Desktop/LMA/phase3/src')
sys.stdout.reconfigure(encoding='utf-8')
from reasoning_solvers import ReasoningSolver

hindi_sp = spm.SentencePieceProcessor(model_file='c:/Users/arped/OneDrive/Desktop/LMA/hindi/tokenizer/models/hindi_bpe_10k_final.model')
nepali_sp = spm.SentencePieceProcessor(model_file='c:/Users/arped/OneDrive/Desktop/LMA/nepali/tokenizer/models/nepali_bpe_10k_assignment.model')

def check_tokenization(sp, text):
    ids = sp.encode(text)
    unks = [id_ for id_ in ids if sp.is_unknown(id_)]
    return len(ids), len(unks)

def validate_examples():
    solver = ReasoningSolver()
    
    print("--- HINDI VALIDATION ---")
    
    # 1. Height (Direct)
    relations = [('राहुल', '>', 'अंजलि'), ('अंजलि', '>', 'विक्रम')]
    ans = solver.solve_direct_comparison(relations, 'max')
    text = f"प्रश्न: राहुल, अंजलि से लंबा है। अंजलि, विक्रम से लंबी है। सबसे लंबा कौन है? उत्तर: {ans}"
    t_cnt, u_cnt = check_tokenization(hindi_sp, text)
    print(f"Text: {text} | Answer: {ans} | Tokens: {t_cnt} | UNKs: {u_cnt}")

    # 2. Age (Attribute) - Fixed phrasing
    attrs = {'सुमित': 25, 'नेहा': 22, 'अंजलि': 30}
    ans = solver.solve_attribute_comparison(attrs, 'min')
    text = f"प्रश्न: सुमित उम्र में 25 साल का है। नेहा उम्र में 22 साल की है। अंजलि उम्र में 30 साल की है। उम्र में सबसे छोटा कौन है? उत्तर: {ans}"
    t_cnt, u_cnt = check_tokenization(hindi_sp, text)
    print(f"Text: {text} | Answer: {ans} | Tokens: {t_cnt} | UNKs: {u_cnt}")
    
    # 3. Price (Attribute) - Fixed phrasing
    prices = {'किताब': 150, 'कलम': 20, 'बस्ता': 500}
    ans = solver.solve_attribute_comparison(prices, 'max')
    text = f"प्रश्न: किताब की कीमत 150 रुपये है। कलम की कीमत 20 रुपये है। बस्ता की कीमत 500 रुपये है। सबसे महंगा क्या है? उत्तर: {ans}"
    t_cnt, u_cnt = check_tokenization(hindi_sp, text)
    print(f"Text: {text} | Answer: {ans} | Tokens: {t_cnt} | UNKs: {u_cnt}")

    # 4. Multi-hop (Transitive)
    rel = [('राहुल', '>', 'अंजलि'), ('अंजलि', '=', 'सुमित')]
    ans = solver.solve_transitive(rel, 'राहुल', 'सुमित')
    ans_text = "बड़ा" if ans == '>' else "छोटा"
    text = f"प्रश्न: राहुल उम्र में अंजलि से बड़ा है। अंजलि और सुमित की उम्र बराबर है। राहुल, सुमित से उम्र में क्या है? उत्तर: {ans_text}"
    t_cnt, u_cnt = check_tokenization(hindi_sp, text)
    print(f"Text: {text} | Answer: {ans_text} | Tokens: {t_cnt} | UNKs: {u_cnt}")


    print("\n--- NEPALI VALIDATION ---")
    
    # 1. Height (Direct)
    relations = [('राम', '>', 'सिता'), ('सिता', '>', 'हरि')]
    ans = solver.solve_direct_comparison(relations, 'max')
    text = f"प्रश्न: राम, सिता भन्दा अग्लो छ। सिता, हरि भन्दा अग्लो छ। सबैभन्दा अग्लो को हो? उत्तर: {ans}"
    t_cnt, u_cnt = check_tokenization(nepali_sp, text)
    print(f"Text: {text} | Answer: {ans} | Tokens: {t_cnt} | UNKs: {u_cnt}")

    # 2. Age (Attribute) - Fixed phrasing
    attrs = {'राम': 25, 'हरि': 22, 'सिता': 30}
    ans = solver.solve_attribute_comparison(attrs, 'min')
    text = f"प्रश्न: राम उमेरमा 25 वर्षको छ। हरि उमेरमा 22 वर्षको छ। सिता उमेरमा 30 वर्षकी छे। उमेरमा सबैभन्दा कान्छो को हो? उत्तर: {ans}"
    t_cnt, u_cnt = check_tokenization(nepali_sp, text)
    print(f"Text: {text} | Answer: {ans} | Tokens: {t_cnt} | UNKs: {u_cnt}")
    
    # 3. Price (Attribute) - Fixed phrasing (Genuine price)
    prices = {'किताब': 150, 'कलम': 20, 'झोला': 500}
    ans = solver.solve_attribute_comparison(prices, 'max')
    text = f"प्रश्न: किताबको मूल्य 150 रुपैयाँ छ। कलमको मूल्य 20 रुपैयाँ छ। झोलाको मूल्य 500 रुपैयाँ छ। सबैभन्दा महँगो के हो? उत्तर: {ans}"
    t_cnt, u_cnt = check_tokenization(nepali_sp, text)
    print(f"Text: {text} | Answer: {ans} | Tokens: {t_cnt} | UNKs: {u_cnt}")

    # 4. Multi-hop (Transitive)
    rel = [('राम', '>', 'सिता'), ('सिता', '=', 'हरि')]
    ans = solver.solve_transitive(rel, 'राम', 'हरि')
    ans_text = "जेठो" if ans == '>' else "कान्छो"
    text = f"प्रश्न: राम उमेरमा सिता भन्दा जेठो छ। सिता र हरि को उमेर बराबर छ। राम, हरि भन्दा उमेरमा के छ? उत्तर: {ans_text}"
    t_cnt, u_cnt = check_tokenization(nepali_sp, text)
    print(f"Text: {text} | Answer: {ans_text} | Tokens: {t_cnt} | UNKs: {u_cnt}")

if __name__ == "__main__":
    validate_examples()
