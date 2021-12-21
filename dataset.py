import os
import json
import gzip
import shutil
import requests
from typing import Iterable, Optional, Union, Dict, List

import numpy as np
import pandas as pd

import torch
from torch._C import Value
from torch.utils.data import Dataset

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


class BackendJsonDataset:
    def __init__(
        self,
        json_path,
    ):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.game_ids = list(self.data.keys())

    def get_game_ids(self):
        return self.game_ids

    def get_df(self, game_id: Union[str, int]) -> Dict[str, List]:
        game_id = str(game_id)
        if game_id not in self.game_ids:
            raise ValueError("game_id not in self.data")
        jsons = self.data[game_id]

        X = []
        t = []
        y = []
        errors = []

        for idx, j in enumerate(jsons):
            X_t0 = pd.read_json(j["X_t0"], orient='table')
            if np.sum(X_t0["Login"]) > 0:
                X_t0 = X_t0.reset_index()
                X_t0["index"] = pd.to_datetime(X_t0["index"])
                X.append(X_t0)
                t.append(0)
                y.append(j["y_t0"])
            else:
                errors.append(idx)
            
            X_t1 = pd.read_json(j["X_t1"], orient='table')
            if np.sum(X_t1["Login"]) > 0:
                X_t1 = X_t1.reset_index()
                X_t1["index"] = pd.to_datetime(X_t1["index"])
                X.append(X_t1)
                t.append(1)
                y.append(j["y_t1"])
            else:
                errors.append(idx)
        
        return {"X": X, "t": t, "y": y, "errors": errors}

    def get_all_dfs(self) -> Dict[str, List]:
        
        results = {"X": [], "t": [], "y": [], "errors": []}

        for game_id in self.game_ids:
            temp = self.get_df(game_id)
            for k in results.keys():
                results[k].extend(temp[k])

        return results
        


class BackendDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, List],
        num_features: int = 1,
    ):
        assert len(data["X"]) == len(data["y"]), f"X's and t's must have the same length"
        assert len(data["t"]) == len(data["y"]), f"t's and y's must have the same length"

        self.data = data
        self.num_features = num_features

        self.X = self.data["X"]
        self.t = self.data["t"]
        self.y = self.data["y"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        emb = transform(self.X[idx]["index"]) # (L, 4)
        _X = torch.tensor(self.X[idx].iloc[:, 1:(1+self.num_features)].to_numpy(), dtype=torch.float32) # (L, num_features)
        X = torch.cat([emb, _X], dim=1) # (L, 4+num_features)
        t = torch.tensor(self.t[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

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