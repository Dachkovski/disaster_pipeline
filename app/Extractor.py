from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
import pandas as pd


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