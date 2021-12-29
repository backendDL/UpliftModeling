import os
import argparse
import datetime
from typing import Tuple, Union, Optional

import pandas as pd

from .load_login_logs import get_login_df
from .load_push_logs import get_push_df

def prepare_login_df(
    url: str,
    start_date: Union[datetime.datetime, str], 
    end_date: Union[datetime.datetime, str],
    save_path: str,
    prefix: Optional[str] = None, 
    overwrite: bool = False,
) -> pd.DataFrame:

    start_date = datetime.datetime.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    end_date = datetime.datetime.fromisoformat(end_date) if isinstance(end_date, str) else end_date

    dfs = []

    dates = pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()
    file_names = [date.strftime("%Y-%m-%d") + ".csv" for date in dates]
    if prefix is not None:
        file_names = [prefix + "_" + f for f in file_names]
    file_names = [os.path.join(save_path, f) for f in file_names]

    for date, file in zip(dates, file_names):
        if os.path.isfile(file) and not overwrite:
            print(f"Login data on {date.strftime('%Y-%m-%d')} already in {save_path}. Load the existing csv file.")
            df = pd.read_csv(file, encoding='utf-8')
        else:
            print(f"Login data on {date.strftime('%Y-%m-%d')} not in {save_path}. Download it from the server.")
            df = get_login_df(url, date)
            df.to_csv(file)
        dfs.append(df)

    return pd.concat(dfs, axis=0)

def prepare_push_df(
    url: str,
    save_path: str,
    file_name: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    file_path = os.path.join(save_path, file_name)
    if os.path.isfile(file_path) and not overwrite:
        print(f"Push data file ({file_path}) exists. Load the existing csv file.")
        df = pd.read_csv(file_path, encoding='utf-8')
    else:
        print(f"{file_path} does not exist. Download it from the server.")
        df = get_push_df(url)
        df.to_csv(file_path)
    return df