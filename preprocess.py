from typing import int, str, list
import numpy as np
import pandas as pd
import glob
import json
import hashlib
from datetime import timedelta
from tqdm import tqdm
import json

df = pd.read_csv('total.csv')
df.inDate = pd.to_datetime(df.inDate.apply(lambda x: x[:-1])) + timedelta(hours=9) # KST
# df['gamer_id'] = df['gamer_id'].apply(lambda x : hashlib.sha256(x.split('-')[1].encode()).hexdigest())
df['Login'] = 1
df = df[['game_id', 'gamer_id', 'inDate', 'Login']]


def make_dateframe(start=None ,end=None, freq=None, periods=None):
    
    dates=pd.date_range(start=start ,end=end, periods=periods, freq=freq)
    str_dates=[str(d) for d in dates]
    
    #행:날짜(시간) 열:범주화한 url 2개, IAP, Login
    date_frame=pd.DataFrame(np.zeros((len(str_dates),4), dtype=int), \
                columns=['social_url', 'non_social_url', 'IAP', 'Login'], index=str_dates)
    return date_frame

def process(
    json_path : str,
    login_path : str,
    
):
lag='1H' #일단 지금은 데이터가 별로 없어서 1시간 단위로 설정

#나중에 더 추가할 예정
if 'H' in lag:
    if len(lag)==2:
        hour_hop=int(lag[0])
        df['time']=df['inDate'].apply(lambda x:x.strftime('%Y-%m-%d %H:00:00'))

date_frame=make_dateframe(start='2021-07-01', end='2021-10-13', freq=lag)

#전체 게임, 유저 리스트
game_gamer_ls=list(set([(game, gamer_id) for (game, gamer_id) in zip(df.game_id, df.gamer_id)]))

#게임-유저의 시간대별 합
groupby=df.groupby(['game_id','gamer_id', 'time']).sum()[['Login']].reset_index()

#게임{유저:데이터프레임} 형식 만들기
for i in range(len(game_gamer_ls)):
    item=game_gamer_ls[i]
    if i==0:
        to_dict={item[0]:{item[1]:date_frame.copy()}}

    else:
        
        #game_id만 있을 때
        if item[0] in to_dict.keys():
            to_dict[item[0]].update({item[1]:date_frame.copy()})
            
        #game_id가 없을 때:
        elif item[0] not in to_dict.keys():
            to_dict[item[0]]={item[1]:date_frame.copy()}
        

for i in tqdm(range(len(groupby))):
    login=groupby.iloc[i]
    temp=to_dict[login['game_id']][login['gamer_id']].copy()
    
    if login['time'] in temp.index:
        temp.loc[login['time'], 'Login']=login['Login']
        to_dict[login[0]][login[1]]=temp
    
    else:
        pass   

game_and_push = [
                ('2021-08-24', 2195),
                ('2021-09-21', 1773),
                ('2021-09-24', 1773),
                ('2021-09-27', 1802),
                ('2021-09-29', 2516)
                ]
dic = {}
for push_date, game in game_and_push:
    users = to_dict[game].keys()
    user_list = []
    for user in tqdm(users, total=len(users)):
        cur_user = {}
        data = to_dict[game][user][['Login']]
        temp = pd.to_datetime(data.index)

        ## t == 1 group
        start_t1 = temp.searchsorted(pd.to_datetime(push_date) - timedelta(days=14))
        push = temp.searchsorted(pd.to_datetime(push_date))
        end_t1 = temp.searchsorted(pd.to_datetime(push_date) + timedelta(days=7))

        ## t == 0 group
        start_t0 = temp.searchsorted(pd.to_datetime(push_date) - timedelta(days=30) - timedelta(days=14))
        not_push = temp.searchsorted(pd.to_datetime(push_date) - timedelta(days=30))
        end_t0 = temp.searchsorted(pd.to_datetime(push_date) - timedelta(days=30) + timedelta(days=7))
        

        # data.index = data.index.strftime('%Y-%m-%d %H:00:00')

        X_t1 = data.iloc[start_t1:push]
        y_t1 = data.iloc[push:end_t1]

        X_t1['treatment'] = 1
        
        
        response = y_t1[y_t1['Login'] > 0]
        y_t1 = 1 if len(response) > 0 else 0

        X_t0 = data.iloc[start_t0:not_push]
        y_t0 = data.iloc[not_push:end_t0]

        X_t0['treatment'] = 1
        
        
        response = y_t0[y_t0['Login'] > 0]
        y_t0 = 1 if len(response) > 0 else 0

        # X_t1.index.name = 'time'
        # X_t0.index.name = 'time'

        # X_t1.reset_index(inplace=True)
        # X_t0.reset_index(inplace=True)
        
        cur_user['X_t1'] = X_t1.to_json(orient='table', indent=4)
        cur_user['y_t1'] = y_t1
        cur_user['X_t0'] = X_t0.to_json(orient='table', indent=4)
        cur_user['y_t0'] = y_t0



        user_list.append(cur_user)
    dic[game] = user_list

#json 타입으로 저장하기
#dataframe -> dictionary
out_dict={}
for idx in game_gamer_ls:
    temp=to_dict[idx[0]][idx[1]]
    dic=temp.to_dict()
    out_dict[idx[0]]={idx[1]:dic}

with open('input_data_login.json','w')as f:
    json.dump(out_dict, f, indent=4)




