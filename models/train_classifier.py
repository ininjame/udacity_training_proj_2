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

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion



def main():
    
    #STEP 1: Load data from DB
    if len(sys.argv) >1:
        db_name = sys.argv[1]
    else:
        db_name = "..\data\DisasterResponse.db"
    db_url = "".join(["sqlite+pysqlite:///", db_name])
    engine = create_engine(db_url)
    with engine.connect() as conn:
        df = pd.read_sql("disres", con=conn)
    
    #STEP 2: Processing text
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
    



