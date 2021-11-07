from sqlalchemy import create_engine
import plotly.graph_objs as go
import pandas as pd
from models.train_classifier import tokenize

def return_figures():
    """
    Create base for plotly visualizations
    Visualize number of messages in each class.
    Also visualizes the words with the highest number of 
    """
    db_name = "data/DisasterResponse.db"
    db_url = "".join(["sqlite+pysqlite:///", db_name])
    engine = create_engine(db_url)
    with engine.connect() as conn:
        df = pd.read_sql("disres", con=conn)
    
    #Graph for top 10 categories and count of messages
    graph_one = []
    #top 10 categories
    x_val = df.iloc[:,4:].sum().sort_values(ascending=False).index.tolist()[:10]
    #Count of items per each
    y_val = df.iloc[:,4:].sum().sort_values(ascending=False).values.tolist()[:10]
    graph_one.append(
        go.Bar(
            x = x_val,
            y = y_val,
        )
    )
    layout_one = dict(title = "Top 10 categories and message count in each",
                    xaxis = dict(title="Categories"),
                    yaxis = dict(title="Message count"))
    
    #Graph for top 10 popular words
    graph_two = []
    word_counts = []
    for item in df.message.values:
        word_counts.extend(tokenize(item))

    word_list = pd.Series(word_counts)
    word_list.value_counts()
    count_words = pd.DataFrame(word_list.value_counts()).reset_index().rename(columns={"index":"word", 0:"count"})
    filt = count_words["word"].apply(lambda x: True if len(x)>2 else False)
    graph_two.append(
        go.Bar(
            x = count_words[filt].head(10)["word"].values.tolist(),
            y = count_words[filt].head(10)["count"].values.tolist(),
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