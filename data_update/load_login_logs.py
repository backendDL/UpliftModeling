from elasticsearch import Elasticsearch
from datetime import timedelta
import pandas as pd
import datetime

with open("../game_log_url.txt") as f:
    es_game_log_url = f.read()

es = Elasticsearch(es_game_log_url)

def searchAPI(date):
    date = date.replace(hour=15, minute=0, second=0, microsecond=0)
    year = date.strftime('%Y')
    month = date.strftime('%m')
    start_date = (date - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S.000')
    end_date = date.strftime('%Y-%m-%dT%H:%M:%S.000')

    print(start_date)
    print(end_date)
    
    body = {
        'size': 10000, #scrool api / from
        'query': {
            'bool': {
                'filter': [
                    {
                        'range': {
                            'inDate': {
                                'gte': start_date,
                                'lt': end_date,
                            }
                        }
                    }, 
                    {
                        'exists': {'field': 'gamer_id'}
                    },
                ]
            }
        },
        'sort': [
            {'inDate': {'order':'asc'}}
        ],
    }

    res = es.search(index=f'dau_mau-{year}-{month}', body=body, scroll='5m')
    return res

def get_login_log(date):
    resp = searchAPI(date)
    old_scroll_id = [resp['_scroll_id']]
    result = []
    for doc in resp['hits']['hits']:
        x = doc['_source']
        result.append({
            'gamer_id' : x['gamer_id'],
            'game_id': x['game_id'],
            'inDate': x['inDate'],
        })
    
    while len(resp['hits']['hits']):    
        
        resp = es.scroll(
            scroll_id = old_scroll_id[-1],
            scroll = '5m'
        )
        old_scroll_id.append(resp['_scroll_id'])
        for doc in resp['hits']['hits']:
            x = doc['_source']
            result.append({
                'gamer_id' : x['gamer_id'],
                'game_id': x['game_id'],
                'inDate': x['inDate'],
            })
            
    result = pd.DataFrame(result)
    result.inDate = (pd.to_datetime(result.inDate) + datetime.timedelta(hours=9)).dt.strftime('%Y-%m-%dT%H:%M:%S.000') # 원본 데이터는 UTC. KST 처리 필요.
    return result

if __name__ == '__main__':
    # 특정 날짜의 로그인 데이터를 다운로드합니다.
    now = datetime.datetime.now()
    if now.hour > 6 and now.hour < 23:
        date = datetime.datetime.fromisoformat('2021-10-25')
        x = get_login_log(date)
        x.to_csv(f'data/{date.strftime("%Y-%m-%d")}.csv')
    else:
        print('6시부터 23시 사이까지만 쿼리를 넣어주세요.')