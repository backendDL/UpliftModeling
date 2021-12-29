import os
import json
import argparse
import datetime
from datetime import timedelta
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from data_update.prepare_data import prepare_login_df, prepare_push_df

def prepare_data(args) -> Tuple[pd.DataFrame]:
    login_df = prepare_login_df(args.login_url, args.start_date, args.end_date, args.save_path, args.login_prefix, args.overwrite)
    push_df = prepare_push_df(args.push_url, args.save_path, args.push_file_name, args.overwrite)

    login_df["inDate"] = pd.to_datetime(login_df["inDate"].apply(lambda x: x[:-1])) + timedelta(hours=9) # KST
    push_df["pushTime"] = pd.to_datetime(push_df["pushTime"].apply(lambda x: x[:-1])) # already KST

    push_df = push_df[(push_df["pushTime"] >= args.start_date) & (push_df["pushTime"] <= args.end_date)]

    return login_df, push_df

def get_available_pushes(
    push_df: pd.DataFrame, 
    window_before_in_hours: int=48, 
    window_after_in_hours: int=24, 
    interval_in_hours: int=168,
    sampling_type: str="before",
):

    available_types = ["before", "after", "both"]
    if sampling_type not in available_types:
        raise ValueError(f"type not in available types {available_types}")

    available_pushes = []

    game_ids = push_df["game_id"].unique()
    print(f"Total of {len(game_ids)} available games")

    for game_id in game_ids:
        subset = push_df[push_df["game_id"] == game_id]

        for push_id in range(len(subset)):
            
            time_1 = subset["pushTime"].iloc[push_id]
            start_1 = time_1 - timedelta(hours=window_before_in_hours)
            end_1 = time_1 + timedelta(hours=window_after_in_hours)
            
            count_1 = np.sum((subset["pushTime"] >= start_1) & (subset["pushTime"] <= end_1))

            time_0 = time_1 - timedelta(hours=interval_in_hours)
            start_0 = time_0 - timedelta(hours=window_before_in_hours)
            end_0 = time_0 + timedelta(hours=window_after_in_hours)

            count_0 = np.sum((subset["pushTime"] >= start_0) & (subset["pushTime"] <= end_0))

            if (count_0 == 0) and (count_1 >= 1):
                result = {k: str(push_df.iloc[push_id, i]) for i, k in enumerate(push_df.iloc[push_id].index)}
                result.update({"count_1": int(count_1)})
                result.update({"count_0": int(count_0)})
                available_pushes.append(result)
            
    return available_pushes


def main(args):
    login_df, push_df = prepare_data(args)
    print(f"total length of login df: {len(login_df)}, total length length of push df: {len(push_df)}")
    print()

    available_pushes = get_available_pushes(push_df, args.window_before, args.window_after, args.interval, sampling_type="before")
    print(f"Number of available pushes: {len(available_pushes)}")

    with open("available_pushes.json", "w") as f:
        json.dump(available_pushes, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do analysis on push and log data')

    parser.add_argument('--push_url', type=str, metavar='URL', required=True, help='url to push data API')
    parser.add_argument('--login_url', type=str, metavar='URL', required=True, help='url to login data ES server')

    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--download', action='store_true')

    parser.add_argument('--save_path', type=str, metavar='PATH', default='./data', help='save path (default: ./data)')
    parser.add_argument("--push_file_name", type=str, metavar='X.csv', default="push.csv", help='save file name (default: push.csv)')
    parser.add_argument("--login_prefix", type=str, metavar='X', default="login", help="prefix to the saved csv file (default: login)")

    parser.add_argument('--start_date', type=datetime.datetime.fromisoformat, metavar='DATE', default='2021-12-01', help='start date for the login query of format yyyy-mm-dd (default: 2021-12-01)')
    parser.add_argument('--end_date', type=datetime.datetime.fromisoformat, metavar='DATE', default='2021-12-01', help='end date for login query of format yyyy-mm-dd (default: 2021-12-01)')

    parser.add_argument('--window_before', type=int, metavar='HOURS', default=48, help='window size before the push time in hour. Look for the game_id/time available to sample (default: 48)')
    parser.add_argument('--window_after', type=int, metavar='HOURS', default=24, help='window size after the push time in hour. Look for the game_id/time available to sample (default: 24)')
    parser.add_argument('--interval', type=int, metavar='HOURS', default=168, help='interval between t=0 and t=1. Look for the game_id/time available to sample (default: 168)')

    args = parser.parse_args()

    main(args)