import json
from disasterReportapp import app
from flask import render_template, request
from sqlalchemy import create_engine
from models.gen_fig_data import create_base, return_figures
import plotly
import json
import joblib
import pickle
import pandas as pd
import models.train_classifier as train_classifier
from models.train_classifier import TrainClassifier, StartingVerbExtractor, NewMultiOutput
import sys
import os

sys.modules["train_classifier"] = train_classifier

if "graph_1_x.npy" not in os.listdir("./models"):
    create_base()

if "model.pickle" not in os.listdir("./models"):
    new_trainer = TrainClassifier()
    new_trainer.load_data()
    new_trainer.create_pipe()
    new_trainer.save_model("models/model.pickle")

model = joblib.load("models/model.pickle")
# with open("models/model.pickle", "rb") as file:
#     from models.train_classifier import TrainClassifier, StartingVerbExtractor, NewMultiOutput
#     model = pickle.load(file)
db_name = "data/DisasterResponse.db"
db_url = "".join(["sqlite+pysqlite:///", db_name])
engine = create_engine(db_url)
with engine.connect() as conn:
    df = pd.read_sql("select * from disres", con=conn)

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
    classification_result = dict(zip(df.columns[4:], classification_labels))
    return render_template("dashboard.html",
                            query = query,
                            classification_result = classification_result)