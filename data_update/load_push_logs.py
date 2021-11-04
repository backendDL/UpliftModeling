from collections import defaultdict
import requests 
import json
import pandas as pd


with open("../game_push_url.txt") as f:
    url = f.read()

if __name__ == '__main__':
    response = requests.get(url)
    data = pd.read_json(response.text)
    data.to_csv('../data/push.csv', index=False)