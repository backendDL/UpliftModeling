from collections import defaultdict
from datetime import timedelta
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import torch
import os
import random

from typing import Union, Any, Dict, Tuple, Optional

from utils import *

class Dataset:
    def __init__(self, cfg: Dict[str, Any]=None):
        self.cfg = cfg
        if cfg is not None:
            self.game_id = self.cfg['game_id']
            self.cut_length = self.cfg['cut_length']
            self.test_game_ids = self.cfg['test_game_id']

    def load_train_data(self, game_id: Union[str, int]) -> Union[Tuple[pd.DataFrame, Dict[str, Any]], None]:
        if os.path.isfile('./data/dataset.pkl'):
            print('전처리된 파일이 존재합니다. 전처리 파일 불러오는 중...', end='')
            with open('./data/dataset.pkl', 'rb') as f:
                data = pickle.load(f)

            self.train_X = data['train_X']
            self.train_y = data['train_y']
            self.train_game_ids = data['train_game_ids']
            self.val_X = data['val_X']
            self.val_y = data['val_y']
            self.val_game_ids = data['val_game_ids']
            self.test_X = data['test_X']
            self.test_y = data['test_y']
            self.test_game_ids = data['test_game_ids']
            self.errors = data['errors']
            print('완료.')

            return None
        else:
            print('전처리된 파일이 존재하지 않습니다. raw data 파일 불러오는 중...', end='')
            login_log_path = os.path.join('data', 'login_logs', str(game_id) + '.csv')
            push_log_path = os.path.join('data', 'push.csv')
            assert os.path.isfile(login_log_path), "It is not a valid path: " + str(login_log_path)
            assert os.path.isfile(push_log_path), "It is not a valid path: " + str(push_log_path)
            
            login_logs = pd.read_csv(login_log_path)
            push_logs = pd.read_csv(push_log_path)
            login_logs.inDate = pd.to_datetime(login_logs.inDate.apply(lambda x: x[:-1])) + timedelta(hours=9) # KST
            push_logs.pushTime = pd.to_datetime(push_logs.pushTime.apply(lambda x: x[:-1])) # already KST
            push_logs = {game_id: x.sort_values(by='pushTime') for game_id, x in push_logs.groupby('game_id')}
            print('완료.')

            return login_logs, push_logs

    def prepare_train_data(self):
        self.errors = []
        data = self.load_train_data(self.game_id)
        if data is None: 
            print('\n' + '-' * 100 + '\n')
            return None
        else:
            login_logs, push_logs = data

            total_max_date = login_logs.inDate.max()
            total_min_date = login_logs.inDate.min()
            print("Dataset span: from {} // to {}".format(total_min_date, total_max_date))

            print('전처리 시작. 최초 1회에 한하여 시간이 다소 걸릴 수 있습니다.')
            # TODO: Churner / Non-Churner 비율 밸런싱 필요(레이블의 분포)
            # 데이터 언밸런스를 해결하는 법
            # 1. Non-Churner의 중복 샘플링: 꾸준히 플레이하는 유저를 대상으로 기간을 달리하여 샘플링한다. -> prepared_gamer_id 체크만 안하면 됨
            # 2. SMOTE: Non-Churner 데이터를 augmentation

            # 데이터셋 생성하는 간격을 설정하는 것은 중요
            # 짧은 주기 -> 중복되는 인풋, 긴 주기 -> 인풋 데이터량이 감소
            # 7의 배수(7, 14, ...) 주기는 금지 -> 특정 요일에 오버피팅 발생
            interval = 5
            point_date_count = (total_max_date - total_min_date).days // interval
            point_dates = [(total_max_date - timedelta(days=7 + interval * i)).replace(hour=6, minute=0, second=0, microsecond=0) for i in range(1, point_date_count + 1)]
            
            prepared_gamer_id = set()
            prepared_counts = defaultdict(lambda: 3)
            data = []
            test_data = []
            for point_date in tqdm(point_dates, leave=False):
                # D-day인 point_date 기준으로 D-14 ~ D-day를 input으로 활용
                # D+1 ~ D+7은 레이블링에 활용
                min_date = point_date - timedelta(days=14)
                max_date = point_date + timedelta(days=7)
                sliced_logs = login_logs[(login_logs.inDate >= min_date) & (login_logs.inDate <= max_date)]

                for gamer_id, X in tqdm(sliced_logs.groupby('gamer_id'), leave=False):
                    # 한 명의 게이머는 최신 데이터만 활용하며 중복 샘플링하지 않는다.
                    if gamer_id in prepared_gamer_id: 
                        if prepared_counts[gamer_id] > 0:
                            prepared_counts[gamer_id] -= 1
                            continue

                    # 문제가 있는 gamer_id는 데이터로 활용하지 않는다.
                    if len(X.game_id.unique()) > 1: 
                        self.errors.append({'game_id': X.game_id.unique().tolist(), 'gamer_id': gamer_id, 'length': len(X)})
                        continue

                    # 로그가 너무 적은 gamer_id는 데이터로 활용하지 않는다.
                    if len(X) < 5: continue

                    # 기준일 이전에 로그가 없으면 판단 불가
                    if X.inDate.min() >= point_date: continue

                    sample = self.extract_sample(X, push_logs, point_date, (min_date, max_date))
                    
                    prepared_gamer_id.add(gamer_id)

                    if X.game_id.iloc[0] in self.test_game_ids:
                        test_data.append(sample)
                    else:
                        data.append(sample)

            random.shuffle(data)
            X = torch.from_numpy(np.stack([x[0] for x in data])).float()
            y = torch.from_numpy(np.array([x[2] for x in data])).float()
            game_ids = torch.from_numpy(np.array([x[3] for x in data]))

            self.train_X = X[:int(len(X) * 0.7)]
            self.train_y = y[:int(len(X) * 0.7)]
            self.train_game_ids = game_ids[:int(len(X) * 0.7)]
            self.val_X = X[int(len(X) * 0.7):]
            self.val_y = y[int(len(X) * 0.7):]
            self.val_game_ids = game_ids[int(len(X) * 0.7):]
            
            self.test_X = torch.from_numpy(np.stack([x[0] for x in test_data])).float()
            self.test_y = torch.from_numpy(np.array([x[2] for x in test_data])).float()
            self.test_game_ids = torch.from_numpy(np.array([x[3] for x in test_data]))

            with open('./data/dataset.pkl', 'wb+') as f:
                pickle.dump({
                    'train_X': self.train_X,
                    'train_y': self.train_y,
                    'train_game_ids': self.train_game_ids,
                    'val_X': self.val_X,
                    'val_y': self.val_y,
                    'val_game_ids': self.val_game_ids,
                    'test_X': self.test_X,
                    'test_y': self.test_y,
                    'test_game_ids': self.test_game_ids,
                    'errors': self.errors, # 오류 있는 데이터 확인용
                }, f)

    def extract_sample(
        self, 
        login_logs: pd.DataFrame, 
        push_logs: Dict[str, pd.DataFrame], 
        point_date: pd.Timestamp, 
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]]=None
    ):
        if date_range is None:
            date_range = (login_logs.inDate.min(), login_logs.inDate.max())

        in_date = login_logs.inDate
        game_id = login_logs.game_id.iloc[0]
        game_id = str(game_id)
        push_time = push_logs[game_id].pushTime if game_id in push_logs.keys() else None
        # Dict[game_id, push_time]
        total_login_count = len(in_date)

        in_date = in_date.sort_values()

        y = label(point_date, in_date)
        # Whether a user is loggined from D+1 to D+7

        input_in_date = in_date[in_date <= point_date]
        if push_time is not None:
            push_time = push_time[(push_time >= date_range[0]) & (push_time <= point_date)]
            # within D-14 ~ D-day
            input_in_date = np.concatenate([input_in_date, push_time])
            log_type = np.zeros((len(input_in_date) + len(push_time), 2))
            log_type[:len(input_in_date), 0] = 1
            log_type[len(input_in_date):, 1] = 1
            args = np.argsort(input_in_date)
            input_in_date = pd.Series(input_in_date[args])
            log_type = log_type[args]
        else:
            log_type = np.zeros((len(input_in_date), 2))
            log_type[:, 0] = 1


        log_type = np.concatenate([np.zeros((self.cut_length, 2)), log_type], axis=0)[-self.cut_length:]
        x = transform(input_in_date, self.cut_length)
        x = torch.cat([x, torch.from_numpy(log_type).T], dim=0)
        sample = (x, total_login_count, y, game_id)
        return sample

    def train_mini_batch(self, batch_size=32, return_game_id=False):
        for i in range(len(self.train_X) // batch_size):
            X = self.train_X[i * batch_size:(i + 1) * batch_size]
            y = self.train_y[i * batch_size:(i + 1) * batch_size]
            if return_game_id: gid = self.train_game_ids[i * batch_size:(i + 1) * batch_size]
            yield (X, y, gid) if return_game_id else (X, y)

    def val_mini_batch(self, batch_size=32, return_game_id=False):
        for i in range(len(self.val_X) // batch_size):
            X = self.val_X[i * batch_size:(i + 1) * batch_size]
            y = self.val_y[i * batch_size:(i + 1) * batch_size]
            if return_game_id: gid = self.val_game_ids[i * batch_size:(i + 1) * batch_size]
            yield (X, y, gid) if return_game_id else (X, y)

    def test_mini_batch(self, batch_size=32, return_game_id=False):
        for i in range(len(self.test_X) // batch_size):
            X = self.test_X[i * batch_size:(i + 1) * batch_size]
            y = self.test_y[i * batch_size:(i + 1) * batch_size]
            if return_game_id: gid = self.test_game_ids[i * batch_size:(i + 1) * batch_size]
            yield (X, y, gid) if return_game_id else (X, y)
            
#region dataset utils
def daily_embedding(x):
    # unit is hour in [0, 24)
    return np.stack([np.sin(2 * np.pi / 24 * x), np.cos(2 * np.pi / 24 * x)])

def weekly_embedding(x):
    # unit is dayofweek in [0, 6]
    return np.stack([np.sin(2 * np.pi / 7 * x), np.cos(2 * np.pi / 7 * x)])

def label(point_date, in_date):
    d1 = point_date + timedelta(days=1)
    d7 = point_date + timedelta(days=7)
    y = ((in_date > d1) & (in_date < d7)).sum() > 0
    return [y]

def transform(X, cut_length=0):
    # X is inDate(pandas datetime64 Series).
    
    # 요일과 접속 시간대를 삼각함수로 임베딩
    day_of_week = np.array(X.dt.dayofweek) # (N,)
    hour_of_day = np.array(X.dt.hour + X.dt.minute / 60)
    embedding_D = daily_embedding(hour_of_day) # (2, N)
    embedding_W = weekly_embedding(day_of_week)


    in_date = np.array(X).astype('datetime64') # (N,)

    # 접속 간격 계산
    # datetime 타입을 numerical type으로 casting하면 micro second로 계산됨
    interval = (in_date[1:] - in_date[:-1]).astype('float') / 1000000 # (N - 1,)
    interval = np.array([0, *(interval.tolist())]) # (N,)
    interval_D = interval // (3600 * 24)
    interval_H = (interval % (3600 * 24)) // 3600
    interval_S = interval % 3600

    interval = np.stack([interval_D, interval_H, interval_S]) # (3, N)
    x = torch.from_numpy(np.concatenate([interval, embedding_W, embedding_D], axis=0)) # (7, N)

    if cut_length:
        _x = torch.zeros((x.shape[0], cut_length))
        if cut_length > x.shape[1]:
            _x[:, -x.shape[1]:] = x
        else: # cut_length <= x.shape[1]
            _x[:] = x[:, -cut_length:]
        x = _x

    return x
#endregion

if __name__ == '__main__':
    import yaml

    with open('config.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    dataset = Dataset(cfg['Dataset'])
    dataset.prepare_train_data()

    for X, t, y in tqdm(dataset.train_mini_batch(32)):
        print(X.shape, t.shape, y.shape)