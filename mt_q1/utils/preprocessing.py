
import re
import string
from typing import List, Tuple
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextPreprocessor:
    
    def __init__(self, 
                 max_length: int = 256,
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False):
        self.max_length = max_length
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        
        # Noktalama işaretleri
        self.punctuation = string.punctuation
        
    def clean_text(self, text: str) -> str:
     
        text = re.sub(r'<[^>]+>', '', text)
        
        if self.lowercase:
            text = text.lower()
        
        
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
       
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', self.punctuation))
        
        
        text = re.sub(r'\s+', ' ', text)
        
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        
        tokens = word_tokenize(text)
        return tokens
    
    def truncate_or_pad(self, tokens: List[str]) -> List[str]:
       
        if len(tokens) > self.max_length:
            
            tokens = tokens[:self.max_length]
        elif len(tokens) < self.max_length:
            
            tokens = tokens + ['<PAD>'] * (self.max_length - len(tokens))
        
        return tokens
    
    def preprocess(self, text: str, pad: bool = True) -> List[str]:
        
        text = self.clean_text(text)
        
      
        tokens = self.tokenize(text)
        
        if pad:
            tokens = self.truncate_or_pad(tokens)
        
        return tokens
    
    def batch_preprocess(self, texts: List[str], pad: bool = True) -> List[List[str]]:
       
        return [self.preprocess(text, pad=pad) for text in texts]


def create_vocabulary(tokenized_texts: List[List[str]], 
                     min_freq: int = 2,
                     max_vocab_size: int = None) -> Tuple[dict, dict]:

    word_freq = {}
    for tokens in tokenized_texts:
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
    
    filtered_words = [word for word, freq in word_freq.items() if freq >= min_freq]
    
    filtered_words.sort(key=lambda w: word_freq[w], reverse=True)
    
    if max_vocab_size is not None:
        filtered_words = filtered_words[:max_vocab_size]
    
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    vocab = special_tokens + filtered_words
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    print(f"Vocabulary oluşturuldu:")
    print(f"  - Toplam kelime: {len(word_to_idx):,}")
    print(f"  - Min frekans: {min_freq}")
    if max_vocab_size:
        print(f"  - Max boyut: {max_vocab_size:,}")
    
    return word_to_idx, idx_to_word


if __name__ == "__main__":
    preprocessor = TextPreprocessor(max_length=20)
    
    sample_texts = [
        "This movie was absolutely AMAZING! I loved it so much!!!",
        "Worst film ever. Complete waste of time and money.",
        "The acting was good, but the plot was confusing."
    ]
    
    print("=" * 60)
    print("TEXT PREPROCESSING TEST")
    print("=" * 60)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{i}. Orijinal:")
        print(f"   {text}")
        
        tokens = preprocessor.preprocess(text, pad=False)
        print(f"\n   İşlenmiş ({len(tokens)} token):")
        print(f"   {' '.join(tokens)}")
    
    print("\n" + "=" * 60)

