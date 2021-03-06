import os
import json
import gzip
import shutil
import requests
from typing import Iterable, Optional, Union, Dict, List

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, Subset

from tqdm import tqdm


def daily_embedding(x):
    # unit is hour in [0, 24)
    return np.stack([np.sin(2 * np.pi / 24 * x), np.cos(2 * np.pi / 24 * x)])

def weekly_embedding(x):
    # unit is dayofweek in [0, 6]
    return np.stack([np.sin(2 * np.pi / 7 * x), np.cos(2 * np.pi / 7 * x)])


def transform(X):
    # X is inDate(pandas datetime64 Series).
    
    # 요일과 접속 시간대를 삼각함수로 임베딩
    day_of_week = np.array(X.dt.dayofweek) # (L,)
    hour_of_day = np.array(X.dt.hour + X.dt.minute / 60)
    embedding_D = daily_embedding(hour_of_day) # (2, L)
    embedding_W = weekly_embedding(day_of_week) # (2, L)

    # in_date = np.array(X).astype('datetime64') # (N,)

    # # 접속 간격 계산
    # # datetime 타입을 numerical type으로 casting하면 micro second로 계산됨
    # interval = (in_date[1:] - in_date[:-1]).astype('float') / 1000000 # (N - 1,)
    # interval = np.array([0, *(interval.tolist())]) # (N,)
    # interval_D = interval // (3600 * 24)
    # interval_H = (interval % (3600 * 24)) // 3600
    # interval_S = interval % 3600

    # interval = np.stack([interval_D, interval_H, interval_S]) # (3, N)
    x = torch.tensor(np.concatenate([embedding_W, embedding_D], axis=0).T, dtype=torch.float32) # (L, 4)

    return x



class UpliftDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        in_features: int,
        t_idx: Optional[int] = None,
        y_idx: Optional[int] = None,
    ):
        t_idx = in_features if t_idx is None else t_idx
        y_idx = in_features + 1 if y_idx is None else y_idx

        self.in_features = in_features
        self.t_idx = t_idx
        self.y_idx = y_idx
        
        self.df = df
        self.X  = self.df.iloc[:, 0:in_features]
        self.t  = self.df.iloc[:, t_idx]
        self.y  = self.df.iloc[:, y_idx]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(self.X.iloc[idx, :].to_numpy(), dtype=torch.float32)
        t = torch.tensor(self.t.iloc[idx], dtype=torch.float32)
        y = torch.tensor(self.y.iloc[idx], dtype=torch.float32)
        return (X, t, y)


class BackendDataset(Dataset):
    def __init__(
        self,
        json_path: str = "./dataset/dataset.json",
        data_path: str = "./dataset",
    ):
        self.data_path = data_path

        with open(json_path, "r") as f:
            _json = json.load(f)
        
        self.df = pd.DataFrame(_json)
        
        self.df["game_id"] = self.df["file_name"].apply(lambda x: x.split("_")[1])
        self.df["push_date"] = pd.to_datetime(self.df["file_name"].apply(lambda x: x.split("_")[2]))
        self.df["gamer_id"] = self.df["file_name"].apply(lambda x: x.split("_")[3].replace(".csv", ""))

        self.game_ids = self.df["game_id"].unique().tolist()
    
    def get_game_ids(self):
        return self.game_ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        file_name = self.df.file_name[idx]
        t = torch.tensor(self.df.t[idx])
        y = torch.tensor(self.df.y[idx])

        data = pd.read_csv(os.path.join(self.data_path, file_name))
        data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])

        emb = transform(data.iloc[:, 0])
        login = torch.tensor(data.login).unsqueeze(1)
        crud = torch.tensor(data.crud).unsqueeze(1)

        X = torch.cat([emb, login, crud], axis=1)

        return (X, t, y)


def prepare_data(data: str, **kwargs) -> pd.DataFrame:

    if data == 'simulated_observational_data':
        """
        # Generate simulated data
        # "Sleeping dogs" (a.k.a. "do-not-disturb"; people who will "buy" if not 
        treated but will not "buy" if treated) can be simulated by negative values 
        in tau parameter.
        # Observational data which includes confounding can be simulated by 
        non-zero values in propensity_coef parameter.  
        # A/B Test (RCT) with a 50:50 split can be simulated by all-zeros values 
        in propensity_coef parameter (default).
        # The first element in each list parameter specifies the intercept.
        """
        from causallift import generate_data

        seed = 42

        df = generate_data(
            N=20000, 
            n_features=3, 
            beta=[0,-2,3,-5], # Effect of [intercept and features] on outcome 
            error_std=0.1, 
            tau=[1,-5,-5,10], # Effect of [intercept and features] on treated outcome
            tau_std=0.1, 
            discrete_outcome=True, 
            seed=seed, 
            feature_effect=0, # Effect of beta on treated outcome
            propensity_coef=[0,-1,1,-1], # Effect of [intercept and features] on propensity log-odds for treatment
            index_name='index',
        )
        
    elif data == 'lalonde':
        r""" 
            Lalonde dataset was used to evaluate propensity score in the paper:
            Dehejia, R., & Wahba, S. (1999). Causal Effects in Nonexperimental 
            Studies: Reevaluating the Evaluation of Training Programs. Journal of 
            the American Statistical Association, 94(448), 1053-1062. 
            doi:10.2307/2669919

            Lalonde dataset is now included in R package named "Matching."
            http://sekhon.berkeley.edu/matching/lalonde.html
        """
        import numpy as np
        
        def get_lalonde():
            r""" Load datasets, concatenate, and create features to get data frame 
            similar to 'lalonde' that comes with "Matching.")
            """
            cols = ['treat', 'age', 'educ', 'black', 'hisp', 'married', 'nodegr','re74','re75','re78']
            control_df = pd.read_csv('http://www.nber.org/~rdehejia/data/nswre74_control.txt', sep=r'\s+', header = None, names = cols)
            treated_df = pd.read_csv('http://www.nber.org/~rdehejia/data/nswre74_treated.txt', sep=r'\s+', header = None, names = cols)
            lalonde_df = pd.concat([control_df, treated_df], ignore_index=True)
            lalonde_df['u74'] = np.where(lalonde_df['re74'] == 0, 1.0, 0.0)
            lalonde_df['u75'] = np.where(lalonde_df['re75'] == 0, 1.0, 0.0)
            return lalonde_df
        lalonde_df = get_lalonde()
        
        """ Prepare the input Data Frame. """
        df = lalonde_df.copy()
        df.rename(columns={'treat':'Treatment', 're78':'Outcome'}, inplace=True)
        df['Outcome'] = np.where(df['Outcome'] > 0, 1.0, 0.0)
        
        # categorize age to 20s, 30s, 40s, and 50s and then one-hot encode age
        df.loc[:,'age'] = df['age'].apply(lambda x:'{:.0f}'.format(x)[:-1]+'0s') 
        df = pd.get_dummies(df, columns=['age'], drop_first=True) 
        
        cols = ['nodegr', 'black', 'hisp', 'age_20s', 'age_30s', 'age_40s', 'age_50s', 
                'educ', 'married', 'u74', 'u75', 'Treatment', 'Outcome']
        df = df[cols]

    elif data == 'criteo':
        save_dir = "./raw_data"
        os.makedirs(save_dir, exist_ok=True)

        criteo_url = "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
        zip_file_name = "raw_data/criteo-research-uplift-v2.1.csv.gz"
        unzip_file_name = "raw_data/criteo-research-uplift-v2.1.csv"
        
        if os.path.isfile(unzip_file_name):
            print("The downloaded file already exists!")
        
        else:
            print("Try to download the raw data from the server...")
            response = requests.get(criteo_url, stream=True)
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(zip_file_name, "wb") as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()
            print("Finished downloading!!!")
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("Error, something went wrong!")
                return

            print("Try to unzip the downloaded file")
            with gzip.open(zip_file_name, "rb") as f_in:
                with open (unzip_file_name, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(zip_file_name)
            print("Zip file removed from disk")

        print("Import the csv file into pd.DataFrame")
        df = pd.read_csv(unzip_file_name)
    
    else:
        raise ValueError("No corresponding data found")

    return df