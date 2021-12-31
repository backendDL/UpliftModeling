import os
import glob
import csv
import sqlite3
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm

def main(args):
    filelist = glob.glob(os.path.join(args.save_path, "*.db"))
    print(f"Found {len(filelist)} file(s)")

    dfs = []

    print("Work in progress")
    for f in tqdm(filelist):
        
        con=sqlite3.connect(f)
        cursor=con.cursor()

        cursor.execute(f"SELECT * FROM inDates WHERE game_id={args.game_id}")
        
        rows = cursor.fetchall()
        cols = [column[0] for column in cursor.description]
        df = pd.DataFrame.from_records(data=rows, columns=cols)
        dfs.append(df)
        con.close()
    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"Length of CRUD logs of game_id {args.game_id}: {len(df)}")

    df.to_csv(os.path.join(args.save_path, f'crud_{args.game_id}.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load and extract queries from the CRUD database")

    parser.add_argument('--save_path', type=str, default="./data/crud")
    parser.add_argument('--game_id', type=int)

    args = parser.parse_args()
    main(args)