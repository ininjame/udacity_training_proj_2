"""
This file contains the code base for the ML pipeline.
The functionalities are as follows:
    - Retrieve data from db
    - Using NLTK to process and featurize the message text
    - Build a multi-output classifier for this problem
    - Build a pipeline
    - Utilize gridsearch
"""
from sqlalchemy import create_engine
import sys
import pandas as pd
import numpy as np

import nltk
nltk.download(["wordnet", "punkt", "stopwords"])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline, FeatureUnion
import pickle

#Function for tokenizing text
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        cleaned = lemmatizer.lemmatize(tok.strip().lower())
        clean_tokens.append(cleaned)
    
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]
    
    return clean_tokens
    
#Custom estimator class to identify if 1st word of response is a verb
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, tokenize_func=None):
        self.tokenize_func = tokenize_func

    def starting_verb(self, text):
        try:
            pos_tags = nltk.pos_tag(self.tokenize_func(text))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
            return False
        except:
            return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb).values
        return pd.DataFrame(X_tagged)

#Custom classifier for Disaster report
class NewMultiOutput(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, random_state=42):
        self.estimator = estimator
        self.random_state = random_state
    
    def _more_tags(self):
        return {'multioutput': True}
    
    def fit(self, X_train, y_train):
        return self.estimator.fit(X_train, y_train.iloc[:,1:])

    def predict(self, X_test):
        y_pred_ = self.estimator.predict(X_test)
        constant_col = np.array([[1] for i in range(len(y_pred_))])
        return np.concatenate((constant_col,y_pred_), axis=1)
    
    def get_params(self, deep=True):
        return self.estimator.get_params(deep=deep)

    def set_params(self, **parameters):
        for param, val in parameters.items():
            if hasattr(self.estimator, param):
                setattr(self.estimator, param, val)
            else:
                self.estimator.kwargs[param] = val
        return self

class TrainClassifier(object):

    def __init__(self):
        self.df = None
        self.pipeline = None
    
    def load_data(self):
        if len(sys.argv) >1:
            db_name = sys.argv[1]
        else:
            db_name = "data\DisasterResponse.db"
        db_url = "".join(["sqlite+pysqlite:///", db_name])
        engine = create_engine(db_url)
        with engine.connect() as conn:
            df = pd.read_sql("disres", con=conn)
        self.df = df
        return self.df

    #Function for tokenizing text  
    def tokenize(self,text):
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for tok in tokens:
            cleaned = lemmatizer.lemmatize(tok.strip().lower())
            clean_tokens.append(cleaned)
        
        clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]
        
        return clean_tokens
    
    def create_pipe(self):
        pipeline = Pipeline([
                ("features", 
                    FeatureUnion([
                        ("text_pipeline", 
                            Pipeline([
                                ("vect", CountVectorizer(tokenizer = self.tokenize)),
                                ("tfidf", TfidfTransformer())
                            ])
                        ),
                        ("verb_extract", StartingVerbExtractor(self.tokenize))
                    ])
                ),
                ("clf", NewMultiOutput(estimator=MultiOutputClassifier(estimator=RandomForestClassifier(), n_jobs=2)))
                # ("clf", MultiOutputClassifier(estimator=RandomForestClassifier(), n_jobs=2))
                ])


        X = self.df["message"]
        y = self.df.iloc[:,4:]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        pipeline.fit(X_train, y_train)
        
        print("Score of the model on test data: {}".format(r2_score(y_test, pipeline.predict(X_test)))) 

        self.pipeline = pipeline
        return self.pipeline

    def save_model(self,filename):
        with open(filename,"wb") as file:
            pickle.dump(self, file)
        
    
    def load_model(self, filename):
        with open(filename, "rb") as file:
            model = pickle.load(file)
        self.pipeline = model.pipeline
        

    
if __name__ == "__main__":
    new_trainer = TrainClassifier()
    new_trainer.load_data()
    new_trainer.create_pipe()
    
