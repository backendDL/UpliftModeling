import json
import hashlib
from typing import List, Dict, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from data_update.prepare_data import *


def load_pushes(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding='utf-8') as f:
        pushes = json.load(f)
    return pushes


def calculate_window(push_time: Union[datetime.datetime, str], window_before: int, window_after: int):
    push_time = push_time if isinstance(push_time, datetime.datetime) else datetime.datetime.fromisoformat(push_time)
    start_time = push_time - datetime.timedelta(hours=window_before)
    end_time = push_time + datetime.timedelta(hours=window_after)
    return start_time, push_time, end_time


def main(args):

    if not os.path.isdir(args.save_path):
        print("The `save_path` given does not exist. Create it.")
        os.makedirs(args.save_path)

    pushes = load_pushes(args.json_path)

    prev_game_id = ""
    results = []

    for push in pushes:
        
        game_id = push["game_id"]
        push_date_t1 = datetime.datetime.fromisoformat(push["pushTime"])
        push_date_t0 = push_date_t1 - timedelta(hours=args.interval)
        window_t1 = calculate_window(push_date_t1, args.window_size_before, args.window_size_after)
        window_t0 = calculate_window(push_date_t0, args.window_size_before, args.window_size_after)

        if window_t0[0] < datetime.datetime.fromisoformat("2021-12-01"):
            if args.verbose: print("Not appropriate to sample. Skip to next iteration.")
            continue

        if args.verbose:
            print(f"Game_id: {game_id}, push_date: {push_date_t1}")
            print(f"Window for t=1 {window_t1}")
            print(f"Window for t=0 {window_t0}")

        windows = [window_t0, window_t1]

        for t, window in enumerate(windows):

            start_date, push_date, end_date = window
            if args.verbose: print(window)

            login_df = prepare_login_df(
                url=args.login_url,
                start_date=start_date,
                end_date=end_date,
                save_path="./data",
                prefix="login",
            )
            
            login_by_game_id = login_df[login_df["game_id"].astype(str) == str(game_id)].copy()
            login_by_game_id["inDate"] = pd.to_datetime(login_by_game_id["inDate"])

            if prev_game_id != game_id:
                # To avoid loading huge CRUD data every time...
                crud_by_game_id = prepare_crud_df(
                    save_path=args.crud_path,
                    game_id=game_id,
                )
                prev_game_id = game_id
                crud_by_game_id["inDate"] = crud_by_game_id["inDate"].apply(lambda x: x[:-1])
                crud_by_game_id["inDate"] = pd.to_datetime(crud_by_game_id["inDate"])

            columns = ["game_id", "gamer_id", "inDate"]
            login_by_game_id = login_by_game_id[columns]
            crud_by_game_id = crud_by_game_id[columns]

            login_by_game_id = login_by_game_id[(login_by_game_id["inDate"] >= start_date) & (login_by_game_id["inDate"] < end_date)]
            login_by_game_id["gamer_id"] = login_by_game_id["gamer_id"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

            crud_by_game_id = crud_by_game_id[(crud_by_game_id["inDate"] >= start_date) & (crud_by_game_id["inDate"] < end_date)].copy()

            if args.verbose: print(f"Length of login_by_game_id {len(login_by_game_id)}, crud_by_game_id {len(crud_by_game_id)}")

            # login_by_game_id["type"] = "login"
            # crud_by_game_id["type"] = "crud"

            gamer_ids = pd.concat([login_by_game_id["gamer_id"], crud_by_game_id["gamer_id"]], axis=0)
            gamer_ids = gamer_ids.unique().tolist()

            bins = pd.date_range(start=start_date, end=end_date, freq=args.frequency)

            if args.verbose: print(f"Number of data points {len(gamer_ids)}")

            for gamer_id in tqdm(gamer_ids):

                login_by_gamer_id = login_by_game_id[login_by_game_id["gamer_id"] == gamer_id].copy()
                crud_by_gamer_id = crud_by_game_id[crud_by_game_id["gamer_id"] == gamer_id].copy()

                login_cut = pd.cut(login_by_gamer_id["inDate"], bins=bins, right=False)
                crud_cut = pd.cut(crud_by_gamer_id["inDate"], bins=bins, right=False)

                login_cut = login_cut.value_counts().sort_index()
                crud_cut = crud_cut.value_counts().sort_index()

                cuts = pd.concat([login_cut, crud_cut], axis=1)
                cuts.columns = ["login", "crud"]
                cuts.index = [item.left for item in cuts.index]

                X = cuts[cuts.index < push_date]
                y = 1 if cuts[cuts.index >= push_date]["login"].sum() > 0 else 0


                _file_name = f"logs_{game_id}_{push_date.strftime('%Y-%m-%d')}_{gamer_id}.csv"
                _individual_path = os.path.join(args.save_path, _file_name)
                X.to_csv(_individual_path)
                results.append({"file_name": _file_name, "t": t, "y": y})

    with open(os.path.join(args.save_path, "dataset.json"), "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Login & CRUD data and create data frames for train/test set")

    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--crud_path', type=str, default='./data/crud')
    parser.add_argument('--json_path', type=str, default='available_pushes_168hr.json')
    parser.add_argument('--save_path', type=str, default='./dataset')

    parser.add_argument('--login_url', type=str)

    parser.add_argument('--window_size_before', type=int, default=24)
    parser.add_argument('--window_size_after',  type=int, default=12)
    parser.add_argument('--interval', type=int, default=168)
    parser.add_argument('--frequency', type=str, default="10T", help="Sampling frequency. 1H (1 hour), 10T (10 min) ... (default: 10T)")

    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()

    main(args)