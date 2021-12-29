import os
from typing import List, Dict, Union
import argparse
import requests
import pandas as pd


def download(url: str) -> Union[List, Dict]:
    """Get json data from the given url"""
    r = requests.get(url)
    return r.json()


def get_df(data: Dict) -> pd.DataFrame:
    """Creates a data frame and flattens the `content` column"""
    df = pd.DataFrame(data)
    res = df['content'].apply(pd.Series)
    df = df.drop('content', axis=1)
    new_df = pd.concat([df, res], axis=1)
    return new_df


def main(args):
    data = download(args.url)
    print("Downloaded push data from the server")

    df = get_df(data)
    print(f"Dataframe -- created with columns {df.columns}")
    print(f"Dataframe -- number of rows {len(df)}")

    df.to_csv(args.save_file_name, encoding='utf-8')
    print(f"Dataframe saved as csv at {os.path.abspath(args.save_file_name)}")
    

if __name__ == '__main__':
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Download and save push data')
    parser.add_argument("--url", type=str)
    parser.add_argument("--save_file_name", type=str, default="push.csv")
    args = parser.parse_args()

    main(args)