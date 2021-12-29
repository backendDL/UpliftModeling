import os
import glob
import argparse
import datetime
from typing import Tuple

import pandas as pd

from data_update.load_login_logs import get_login_df
from data_update.load_push_logs import get_push_df

def prepare_login_df(args) -> pd.DataFrame:

    dfs = []

    dates = pd.date_range(start=args.start_date, end=args.end_date).to_pydatetime().tolist()
    file_names = [date.strftime("%Y-%m-%d") + ".csv" for date in dates]
    if args.login_prefix is not None:
        file_names = [args.login_prefix + "_" + f for f in file_names]
    file_names = [os.path.join(args.save_path, f) for f in file_names]

    for date, file in zip(dates, file_names):
        if os.path.isfile(file) and not args.overwrite:
            df = pd.read_csv(file, encoding='utf-8')
        else:
            df = get_login_df(args.login_url, date)
            df.to_csv(file)
        dfs.append(df)

    return pd.concat(dfs, axis=0)

def prepare_push_df(args) -> pd.DataFrame:
    file_path = os.path.join(args.save_path, args.push_file_name)
    if os.path.isfile(file_path) and not args.overwrite:
        df = pd.read_csv(file_path, encoding='utf-8')
    else:
        df = get_push_df(file_path)
        df.to_csv(file_path)
    return df

def prepare_data(args) -> Tuple[pd.DataFrame]:
    login_df = prepare_login_df(args)
    push_df = prepare_push_df(args)

    return login_df, push_df

def main(args):
    login_df, push_df = prepare_data(args)
    print(len(login_df))
    print(len(push_df))

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

    args = parser.parse_args()

    main(args)