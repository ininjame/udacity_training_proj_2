import json
from disasterReportapp import app
from flask import render_template, request
from sqlalchemy import create_engine
from models.gen_fig_data import return_figures
import plotly
import json
from sklearn.externals import joblib
import pandas as pd
import models.train_classifier as train_classifier
from models.train_classifier import TrainClassifier
import sys

sys.modules["train_classifier"] = train_classifier

# new_trainer = TrainClassifier()
model = joblib.load("models/model.pkl")
db_name = "data/DisasterResponse.db"
db_url = "".join(["sqlite+pysqlite:///", db_name])
engine = create_engine(db_url)
with engine.connect() as conn:
    df = pd.read_sql("disres", con=conn)

@app.route("/")
@app.route("/index")
def homepage():
    #read data to prepare for front-end injection
    figures = return_figures()

    #Make ids for html id tag
    ids = ["figure-{}".format(i) for i,_ in enumerate(figures)]
    #Convert figure data to JSON
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("index.html",
                            ids = ids,
                            figuresJSON = figuresJSON)

@app.route("/dashboard")
def dashboard():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.pipeline.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    return render_template("dashboard.html",
                            query = query,
                            classification_results = classification_results)