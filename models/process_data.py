"""
This is the ETL Pipeline for the Disaster Report model.
The pipeline consists of the following features:
    - Accepts user inputs for data source file names, as well as targeted DB
    via command line input
    - Process source files, clean class targets, drop duplicates to prepare input
    for ML pipeline
    - Output both as a pickle file and to the designated SQL DB
"""

import pandas as pd
import sys
import re
from sqlalchemy import create_engine

def load_combine_data(data_source1, data_source2):
    #STEP 1: Ingest data and create dataframes
    messages = pd.read_csv(data_source1)
    categories = pd.read_csv(data_source2)

    #STEP 2: Merge feature data and target classes
    df_merged = messages.merge(categories, on="id")
    return df_merged

def clean_data(df_merged):

    #STEP 3: cleaning
    #Function for returning dataframe of cleaned and binarized class targets
    def categorize(series):
        columns = []
        for col in series.iloc[0].split(";"):
            columns.append(col.split("-")[0])
        arr = series.apply(lambda x: re.sub(r'\w+_*\w+-', '', x).split(";")).values
        
        return pd.DataFrame(list(arr), columns = columns).applymap(lambda x: int(x))
    
    #Combine class target dataframe with original dataframe
    df_all = pd.concat([df_merged, categorize(df_merged["categories"])], axis=1).drop(columns=["categories","original"])
    
    #Drop duplicate messages
    df_all.drop_duplicates(subset="message", inplace=True)
    #Drop lines where message has no content
    df_all = df_all[df_all.message.apply(lambda x: True if len(x.split())>0 else False)]

    return df_all

def save_data(df, db_url):
    #Write to existing DB:
    engine = create_engine(db_url)
    df.to_sql("disres", con=engine, if_exists="replace")
    return 


def main():
    # Accepts 3 inputs from the user via command lines for data sources and target DB
    # Use default values if inputs are not available
    if len(sys.argv) >1:
        data_source1 = "\\".join([".\data",sys.argv[1]])
        data_source2 = "\\".join([".\data", sys.argv[2]])
        db_name = "\\".join([".\data", sys.argv[3]])
    else:
        data_source1 = ".\data\messages.csv"
        data_source2 = ".\data\categories.csv"
        db_name = ".\data\DisasterResponse.db"

    #Ingest data and create dataframes
    print("Loading and combining data...")
    df_merged = load_combine_data(data_source1, data_source2)

    #cleaning
    print("Cleaning data...")
    df_all = clean_data(df_merged)
    
    #Write to existing DB:
    print("Saving data to DB {}...".format(db_name))
    db_url = "".join(["sqlite+pysqlite:///", db_name])
    save_data(df_all, db_url)


    
if __name__ == "__main__":
    main()

    