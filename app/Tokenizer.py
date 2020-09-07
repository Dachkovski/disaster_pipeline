import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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

        return pd.Series(X).apply(tokenize).values