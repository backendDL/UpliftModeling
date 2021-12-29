import os
import argparse
import datetime
from typing import Tuple, Optional

import pandas as pd

from data_update.prepare_data import prepare_login_df, prepare_push_df

def prepare_data(args) -> Tuple[pd.DataFrame]:
    login_df = prepare_login_df(args.login_url, args.start_date, args.end_date, args.save_path, args.login_prefix, args.overwrite)
    push_df = prepare_push_df(args.push_url, args.save_path, args.push_file_name, args.overwrite)
    return login_df, push_df

def main(args):
    login_df, push_df = prepare_data(args)
    print(f"total length of login df: {len(login_df)}, total length length of push df: {len(push_df)}")
    print()

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