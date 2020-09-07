from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import re

def tokenize(text):
    '''
    Tokanize and clean sentences to a list of words.

    ARGS:
    messages: str. Messages as one string.
       
    OUTPUT:
    Tokenized and cleaned list of words: list. Messages are cleaned by removing white spaces, urls and numbers, lower cased and appended word by word to a list. 
    
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    number_regex = '\{\d+:\d+\}'
        
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    detected_numbers = re.findall(number_regex, text)
    for url in detected_numbers:
        text = text.replace(url, "numberplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        '''
        Identify if first word is a verb.

        ARGS:
        messages: str. Filepath to messages file.

        OUTPUT:
        1: int. If first word is a verb, present tense, not 3rd person singular or 'RT'.
        0: int. Else.
        '''
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
                           
        return pd.DataFrame(X_tagged)