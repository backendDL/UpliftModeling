# -*- coding: utf-8 -*-
from typing import Tuple, Dict, Any, List, Union, Optional
import numpy as np
from numpy.lib.npyio import save
import pandas as pd
import json
import hashlib
from datetime import timedelta
from tqdm import tqdm
import json
from data_update.prepare_data import prepare_login_df, prepare_crud_df


def process(
    json_path: str,
    start_date: str,
    end_date: str,
    window_before_in_hours: int=48, 
    window_after_in_hours: int=24, 
    interval_in_hours: int=168,
):
    tqdm.pandas()
    with open(json_path) as f:
        pushes = json.load(f)

    login = prepare_login_df(
        url='https://search-game-log-73v4ao67bzgafxhgcxse6ue7ou.ap-northeast-2.es.amazonaws.com', 
        start_date=start_date, 
        end_date=end_date,
        save_path='./data',
        prefix='login'
    )
    
    crud = prepare_crud_df(
        save_path='./data/crud',
        game_id=2195
    )

    
    login.inDate = pd.to_datetime(login.inDate.apply(lambda x: x[:-1])) + timedelta(hours=9) # KST
    login['Login'] = 1
    df = login[['game_id', 'gamer_id', 'inDate', 'Login']]
    df['gamer_id'] = df['gamer_id'].progress_apply(lambda x : hashlib.sha256(x.encode()).hexdigest())
    print("Loaded login data")
    
    crud.inDate=pd.to_datetime(crud.inDate)
    crud['Crud']=1
    crud_df=crud[['inDate','game_id', 'gamer_id', 'Crud']]
    print('Loaded crud data')
    


    #전체 게임, 유저 리스트
    game_gamer_ls=list(set([(game, gamer_id) for (game, gamer_id) in zip(df.game_id, df.gamer_id)]))

    #게임{유저:데이터프레임} 형식 만들기
    for i in range(len(game_gamer_ls)):
        item=game_gamer_ls[i]
        if i==0:
            to_dict={item[0]:{item[1]:np.nan}}

        else:
            #game_id만 있을 때
            if item[0] in to_dict.keys():
                to_dict[item[0]].update({item[1]:np.nan})

            #game_id가 없을 때:
            elif item[0] not in to_dict.keys():
                to_dict[item[0]]={item[1]:np.nan}
                
    groupby=df.groupby(['game_id','gamer_id','inDate']).count()[['Login']].reset_index()
    groupby_crud=crud_df.groupby(['game_id','gamer_id', 'inDate']).count()[['Crud']].reset_index()
    groupby['Crud']=0
    groupby_crud['Login']=0
    
    print('---processing---')
    merged_df=pd.concat([groupby, groupby_crud], axis=0, join='outer')
    merged_df['inDate']=merged_df['inDate'].apply(lambda t:t.tz_localize(None)) #pandas 0.15.0 이하는 주석처리
    merged_df_gb=merged_df.groupby(['game_id','gamer_id','inDate']).sum().reset_index(level=2)
                

    for idx in tqdm(merged_df_gb.index):
        to_dict[idx[0]][idx[1]]=merged_df_gb.loc[idx[0],idx[1]].reset_index()[['inDate','Login','Crud']].copy()

    game_and_push = [[d['pushTime'], int(d['game_id'])] for d in pushes]
    
    dic = {}
    for push_date, game in game_and_push:
        users = to_dict[game].keys()
        sample_list = []
        for user in tqdm(users, total=len(users)):
            # Init
            t1_sample = {"gamer_id" : user, "X" : {}}

            data = to_dict[game][user][['Login', 'CRUD']]
            temp = pd.to_datetime(data.index)

            ## t == 1 group
            start_t1 = temp.searchsorted(pd.to_datetime(push_date) - timedelta(hours=interval_in_hours))
            push = temp.searchsorted(pd.to_datetime(push_date))
            end_t1 = temp.searchsorted(pd.to_datetime(push_date) + timedelta(hours=window_after_in_hours))
            
            ## t == 0 group
            start_t0 = temp.searchsorted(pd.to_datetime(push_date) - timedelta(hours=interval_in_hours) - timedelta(hours=interval_in_hours))
            not_push = temp.searchsorted(pd.to_datetime(push_date) - timedelta(hours=interval_in_hours))
            end_t0 = temp.searchsorted(pd.to_datetime(push_date) - timedelta(hours=interval_in_hours) + timedelta(hours=window_after_in_hours))
            

            # data.index = data.index.strftime('%Y-%m-%d %H:00:00')

            X_t1 = data.iloc[start_t1:push]
            y_t1 = data.iloc[push:end_t1]
            
            response = y_t1[y_t1['Login'] > 0]
            y_t1 = 1 if len(response) > 0 else 0

            t0_sample = {"gamer_id" : user, "X" : {}}

            X_t0 = data.iloc[start_t0:not_push]
            y_t0 = data.iloc[not_push:end_t0]
            
            
            response = y_t0[y_t0['Login'] > 0]
            y_t0 = 1 if len(response) > 0 else 0
            
            t1_sample['X']['login'] = X_t1['Login'].tolist()
            t1_sample['X']['crud'] = X_t1['CRUD'].tolist()
            t1_sample['y'] = y_t1
            t1_sample['start'] = (pd.to_datetime(push_date) - timedelta(hours=interval_in_hours)).strftime("%Y-%m-%d %H:%M:%S")
            t1_sample['end'] = (pd.to_datetime(push_date)).strftime("%Y-%m-%d %H:%M:%S")

            t0_sample['X']['login'] = X_t0['Login'].tolist()
            t0_sample['X']['crud'] = X_t0['CRUD'].tolist()
            t0_sample['y'] = y_t0
            t0_sample['start'] = (pd.to_datetime(push_date) - timedelta(hours=interval_in_hours) - timedelta(hours=interval_in_hours)).strftime("%Y-%m-%d %H:%M:%S")
            t0_sample['end'] = (pd.to_datetime(push_date) - timedelta(hours=interval_in_hours)).strftime("%Y-%m-%d %H:%M:%S")

            if len(X_t1[X_t1['Login'] != 0]) > 0:
                sample_list.append(t1_sample)
            if len(X_t0[X_t0['Login'] != 0]) > 0:
                sample_list.append(t0_sample)
        dic[game] = sample_list
    with open('data/input_data_login.json','w')as f:
        json.dump(dic, f, indent=4)
    return dic

if __name__ == '__main__':
    process(
        'available_pushes_24hr.json',
        start_date='2021-12-01',
        end_date='2021-12-20'
    )