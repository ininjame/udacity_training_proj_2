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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline, FeatureUnion

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

    def starting_verb(self, text):
        try:
            pos_tags = nltk.pos_tag(tokenize(text))
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

def main():
    
    #STEP 1: Load data from DB
    if len(sys.argv) >1:
        db_name = sys.argv[1]
    else:
        db_name = "data\DisasterResponse.db"
    db_url = "".join(["sqlite+pysqlite:///", db_name])
    engine = create_engine(db_url)
    with engine.connect() as conn:
        df = pd.read_sql("disres", con=conn)

    
    
    #STEP 2: Create Pipeline
    pipeline = Pipeline([
                ("features", 
                    FeatureUnion([
                        ("text_pipeline", 
                            Pipeline([
                                ("vect", CountVectorizer(tokenizer = tokenize)),
                                ("tfidf", TfidfTransformer())
                            ])
                        ),
                        ("verb_extract", StartingVerbExtractor())
                    ])
                ),
                ("clf", MultiOutputClassifier(estimator=RandomForestClassifier(), n_jobs=2))
                ])

    pipe2 = Pipeline([
                ("vect", CountVectorizer(tokenizer = tokenize)),
                ("tfidf", TfidfTransformer()),
                ("clf", MultiOutputClassifier(estimator=RandomForestClassifier(), n_jobs=3))
                ])

    X = df["message"]
    y = df.iloc[:,4:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipeline.fit(X_train, y_train)
    print(pipeline.score(X_test, y_test))

    # pipe2.fit(X_train, y_train)
    # print(pipe2.score(X_train, y_train))
    # print(X.head())
    # print(X.shape)

if __name__=="__main__":
    main()

