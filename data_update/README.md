# Data Update

## How to use

To download and load all data in the given time frame, please refer to `python analysis.py --help`. 

The files in this folder contain scripts that handle downloading, concatenating, and subsetting the database.

If you download the data files seperately (not using the scripts), you should put them in `root/data` folder to use default arguments.

For example, the file structure should be like below:

```bash
.
├── README.md
├── analysis.py
├── data
│   ├── crud
│       ├── 0.db
│       ├── 1.db
│       ...
│       └── xxx.db
│   ├── login_2021-12-01.csv
│   ├── login_2021-12-02.csv
│   ...
│   └── push.csv
├── data_update
│   ├── README.md
│   ├── load_crud_logs.py
│   ├── load_login_logs.py
│   ├── load_push_logs.py
│   └── prepare_data.py
│   ...
```

`load_login_logins.py` and `load_push_logs.py` are executable files, but `prepare_data.py` is not. 
You can use those two executables to download the data file individually. 

Use `analysis.py` if you want to download all files at once.

You have to pass the URL with the `--url` flag (**required!!!**).

### `data_update/load_push_logs.py`

```bash
python data_update/load_crud_logs.py --help
```

### `data_update/load_login_logs.py`

```bash
python data_update/load_login_logs.py --help
usage: load_login_logs.py [-h] --url URL [--date DATE] [--save_path PATH] [--prefix X]

Download login data at a specific date

optional arguments:
  -h, --help        show this help message and exit
  --url URL         url to the elastic search server
  --date DATE       date of format yyyy-mm-dd (default: 2021-12-01)
  --save_path PATH  save path (default: ./data)
  --prefix X        prefix to the saved csv file (default: login)
```

### `data_update/load_push_logs.py`

```bash
python data_update/load_push_logs.py --help
usage: load_push_logs.py [-h] --url URL [--save_path PATH] [--save_file X.csv]

Download and save push data

optional arguments:
  -h, --help         show this help message and exit
  --url URL          url to the push log API
  --save_path PATH   save path (default: ./data)
  --save_file X.csv  save file name (default: push.csv)

```

## Examples

Please change the working directory to the root and run the commands below.

```bash
python data_update/load_crud_logs.py --game_id 1572
python data_update/load_login_logs.py --url https://xxx.com --date 2021-12-03 --prefix login
python data_update/load_push_logs.py --url http://yyy.com --save_file push.csv
```