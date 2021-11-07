import json
from disasterReportapp import app
from flask import render_template
from sqlalchemy import create_engine
from models.gen_fig_data import return_figures
import plotly
import json



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
    return render_template("dashboard")