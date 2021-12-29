import os
import argparse
import datetime
from datetime import timedelta

from elasticsearch import Elasticsearch
import pandas as pd

def _is_server_available():
    now = datetime.datetime.now()
    return now.hour > 6 and now.hour < 23

def get_es(url: str):
    es = Elasticsearch(url)
    return es

def searchAPI(es, date):

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

def get_login_log(es, date):
    resp = searchAPI(es, date)
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

def get_login_df(url: str, date: datetime.datetime):
    if _is_server_available():
        es = get_es(url)
        print("Successfully connected to Elastic Search server")

        df = get_login_log(es, date)
        print(f"Queried login data at {date}")
        print(f"Length of the data frame {len(df)}")
        return df

    else:
        print("Server not available. Please request it later in between 6 am to 23 pm.")
        return None

def main(args):
    df = get_login_df(args.url, args.date)
    if df is None: 
        return
    
    file_name = args.date.strftime("%Y-%m-%d") + ".csv"
    if args.prefix[-1] != "_":
        args.prefix += "_"
    file_name = args.prefix + file_name if args.prefix is not None else file_name

    save_path = os.path.join(args.save_path, file_name)

    df.to_csv(save_path)
    print(f"Saved the data frame at {os.path.abspath(save_path)}")


if __name__ == '__main__':
    # 특정 날짜의 로그인 데이터를 다운로드합니다.
    parser = argparse.ArgumentParser(description='Download login data at a specific date')

    parser.add_argument('--url', type=str, metavar='URL', required=True, help='url to the elastic search server')
    parser.add_argument('--date', type=datetime.datetime.fromisoformat, metavar='DATE', default='2021-12-01', help='date of format yyyy-mm-dd (default: 2021-12-01)')
    parser.add_argument('--save_path', type=str, metavar='PATH', default='./data', help='save path (default: ./data)')
    parser.add_argument('--prefix', type=str, metavar='X', default='login', help="prefix to the saved csv file (default: login)")

    args = parser.parse_args()
    main(args)