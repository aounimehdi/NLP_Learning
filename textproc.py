import time
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.corpus import names
from nltk import FreqDist
import pandas as pd


def unusual_words(text):
	text_vocab= set(w.lower() for w in text if w.isalpha())
	english_vocab= set(w.lower() for w in words.words())
	unusual= text_vocab.difference(english_vocab)
	return(sorted(unusual))

def content_fraction(text):
	stopsFr= stopwords.words("french")
	stopsEn= stopwords.words("english")
	stopwords= stopsFr + stopsEn
	content= [w for w in text if w.lower() not in stopwords]
	return len(content) / len(text)

def summary_corpus(data, column, language="english"):
    """
    Return summary info for the frequency of words in the corpus
    example: tokens, vocab, frequency_dist= summary_corpus(data= df, column= 'reviews', language="english")
    """
    tokens= [word for text in data[column] for word in word_tokenize(text, language= language) ]
    vocab= set(tokens)
    frequency_dist= FreqDist(tokens)
    
    keys, values= [], []
    for key, value in frequency_dist.items():
        keys.append(key)
        values.append(value)
        
    frequency_dist= {"word": keys, "frequency": values}
    frequency_dist= pd.DataFrame.from_dict(frequency_dist)
    frequency_dist.sort_values(by= 'frequency', ascending= False, inplace= True, axis= 0)
    
    print('Number of tokens in the corpus :' , len(tokens))
    print('Vocabulary size                :' , len(vocab))
    
    return tokens, vocab, frequency_dist



from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import contractions
import unidecode
import string
import re 

class TextPreProcess():
    def __init__(self, language='english'):
        self.language= language
    
    def remove_punct(self, text):
       return re.sub(r'[^\w\s]', '', text)

    def remove_digits(self, text):
        return re.sub(r'[\d]', '', text)
    
    def remove_punct_digits(self, text):
        return re.sub(r'[^a-zA-Z\s]', '', text)
    
    def remove_outside_spaces(self, text):
        return text.strip()
    
    def remove_double_spaces(self, text):
        pattern = re.compile(r"\s+")
        return re.sub(pattern, ' ', text)
        
    def remove_stops(self, text, stopwords):
        tokens = word_tokenize(text, self.language)
        return ' '.join([w for w in tokens if not w in stopwords])
    
    def expand_contractions(self, text):
        """
        Replace contractions in string of text
        """
        return contractions.fix(text)       
    
    def remove_repeated_characters(self, text):
        """
        Limitation example: happiness ==> hapines
        """
        # using regex backreference
        pattern= re.compile(r'(\w*)(\w)\2(\w*)')
        tokens = word_tokenize(text, self.language)
        new_tokens= []
        for word in tokens: 
            # look if word exist in wordnet dictionnary
            if wordnet.synsets(word):
                new_tokens.append(word)
            else:
                new_word= pattern.sub(r'\1\2\3', word)
                if new_word != word:
                    new_tokens.append(self.remove_repeated_characters(new_word))
                else:
                    new_tokens.append(new_word)
    
        return ' '.join(new_tokens)

    
    def word_regex_replacement(self, text, replacement_patterns):
        """
        ex:
            replacement_patterns = [
                                    (r'won\'t', 'will not'),
                                    (r'can\'t', 'cannot'),
                                    (r'i\'m', 'i am'),
                                    (r'ain\'t', 'is not'),
                                    (r'(\w+)\'ll', '\g<1> will'),
                                    (r'(\w+)n\'t', '\g<1> not'),
                                    (r'(\w+)\'ve', '\g<1> have'),
                                    (r'(\w+)\'s', '\g<1> is'),
                                    (r'(\w+)\'re', '\g<1> are'),
                                    (r'(\w+)\'d', '\g<1> would')
                                    ]
        """
        s= text
        patterns= [(re.compile(regex), repl) for (regex, repl) in
        replacement_patterns]
        for (pattern, repl) in patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s


    def synonym_replacement(self, text, mapping_dict):
        tokens = word_tokenize(text, self.language) 
        return ' '.join([mapping_dict.get(word, word) for word in tokens])

  #  def remove_non_ascii(self, text):
   # 	"""
   # 	Remove non-ASCII characters from list of tokenized words
   #  	"""
   #  	return unidecode.unidecode(text)
   







#stopsFr = stopwords("french")
#stopsEn = stopwords("english")

#male_names= names.words("male.txt")
#female_names= names.words("female.txt")
