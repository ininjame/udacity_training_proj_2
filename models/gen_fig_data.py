from sqlalchemy import create_engine
import plotly.graph_objs as go
import pandas as pd
# from train_classifier import TrainClassifier
from models.train_classifier import TrainClassifier
import numpy as np

def create_base():
    """
    Create base for plotly visualizations:
    Will produce base data for 2 below graphs, as separate .npy array files (separate for x and y)
    - Visualize number of messages in each class.
    - Also visualizes the words with the highest number of appearance.
    """
    db_name = "data/DisasterResponse.db"
    db_url = "".join(["sqlite+pysqlite:///", db_name])
    engine = create_engine(db_url)
    with engine.connect() as conn:
        df = pd.read_sql("select * from disres", con=conn)
    
    graph_1_x = df.iloc[:,4:].sum().sort_values(ascending=False).index.tolist()[:10]
    graph_1_y = df.iloc[:,4:].sum().sort_values(ascending=False).values.tolist()[:10]

    new_trainer = TrainClassifier()
    word_counts = []
    for item in df.message.values:
        word_counts.extend(new_trainer.tokenize(item))

    word_list = pd.Series(word_counts)
    word_list.value_counts()
    count_words = pd.DataFrame(word_list.value_counts()).reset_index().rename(columns={"index":"word", 0:"count"})
    filt = count_words["word"].apply(lambda x: True if len(x)>2 else False)
    graph_2_x = count_words[filt].head(10)["word"].values.tolist()
    graph_2_y = count_words[filt].head(10)["count"].values.tolist()

    np.save("models/graph_1_x", graph_1_x)
    np.save("models/graph_1_y", graph_1_y)
    np.save("models/graph_2_x", graph_2_x)
    np.save("models/graph_2_y", graph_2_y)

def return_figures():
    """
    Return Plotly graph objects for displaying in the web app,
    using data generated from create_base()
    """
    # db_name = "data/DisasterResponse.db"
    # db_url = "".join(["sqlite+pysqlite:///", db_name])
    # engine = create_engine(db_url)
    # with engine.connect() as conn:
    #     df = pd.read_sql("disres", con=conn)
    
    #Graph for top 10 categories and count of messages
    graph_one = []
    #top 10 categories
    # x_val = df.iloc[:,4:].sum().sort_values(ascending=False).index.tolist()[:10]
    #Count of items per each
    # y_val = df.iloc[:,4:].sum().sort_values(ascending=False).values.tolist()[:10]
    graph_one.append(
        go.Bar(
            x = np.load("models/graph_1_x.npy"),
            y = np.load("models/graph_1_y.npy")
        )
    )
    layout_one = dict(title = "Top 10 categories and message count in each",
                    xaxis = dict(title="Categories"),
                    yaxis = dict(title="Message count"))
    
    #Graph for top 10 popular words
    new_trainer = TrainClassifier()
    graph_two = []
    # word_counts = []
    # for item in df.message.values:
    #     word_counts.extend(new_trainer.tokenize(item))

    # word_list = pd.Series(word_counts)
    # word_list.value_counts()
    # count_words = pd.DataFrame(word_list.value_counts()).reset_index().rename(columns={"index":"word", 0:"count"})
    # filt = count_words["word"].apply(lambda x: True if len(x)>2 else False)
    graph_two.append(
        go.Bar(
            # x = count_words[filt].head(10)["word"].values.tolist(),
            x = np.load("models/graph_2_x.npy"),
            # y = count_words[filt].head(10)["count"].values.tolist(),
            y = np.load("models/graph_2_y.npy")
        )
    )
    layout_two = dict(title="Top 10 words used in terms of appearance",
                    xaxis= dict(title="Word"),
                    yaxis= dict(title="Count of appearance")
                    )

    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))

    return figures

if __name__ == "__main__":
    #Create base data when running script independently
    create_base()