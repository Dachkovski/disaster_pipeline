import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import pickle
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

def load_data(database_filepath):
    '''
    Load data from SQL Database.

    ARGS:
    database_filepath: str. Filepath to SQL Database file.
       
    OUTPUT:
    X: Pandas Series. Input data containing message stings.
    Y: Pandas DataFrame. Fitting data containing categories for messages.
    category_names: list. Category names. 
    '''

    # Load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('data/DisasterResponse.db', engine)  
    
    # Assign columns to input and target variables
    X = df.message.values
    Y = df[df.columns[4:]]
    category_names = Y.columns.values
                           
    return X, Y, category_names

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
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
                           
        return pd.DataFrame(X_tagged)


                         
def build_model():
    '''
    Build machine learning model with parallel feature processing and optimized parameter set.

    ARGS:
    NONE

    OUTPUT:
    model: object. Machine learning model.
    '''
    # Build pipeline with Feature Union 
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # parameters to fit for GridSearch                       
    parameters = {
     'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
     'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
     'features__text_pipeline__vect__max_features': (None, 5000, 10000),
     'features__text_pipeline__tfidf__use_idf': (True, False),
     'clf__estimator__n_estimators': [50, 100, 200],
     'clf__estimator__learning_rate': [1.0, 0.8, 0.5],
     'features__transformer_weights': (
       {'text_pipeline': 1, 'starting_verb': 0.5},
       {'text_pipeline': 0.5, 'starting_verb': 1},
       {'text_pipeline': 0.8, 'starting_verb': 1},
       {'text_pipeline': 1, 'starting_verb': 0.0},
        )
    }

    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate models precision, recall and F-score.

    ARGS:
    model: object. Machine learning model.
    X_test: Pandas Series. Input data containing message stings.
    Y_test: Pandas DataFrame. Fitting data containing categories for messages.
    category_names: list. Category names. 

    OUTPUT:
    NONE
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Save model to pickle serualized file.

    ARGS:
    model: object. Machine learning model.
    model_filepath: str. model filepath.
    
    OUTPUT:
    *.pkl: Pickle file.  
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()