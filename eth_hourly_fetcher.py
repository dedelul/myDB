import sqlite3
import requests
import time
from datetime import datetime
import argparse

DB_PATH = 'eth_data.db'
API_URL = 'https://api.coingecko.com/api/v3/coins/ethereum/market_chart'

# create table if not exists
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS hourly_data (
    timestamp INTEGER PRIMARY KEY,
    price REAL,
    volume REAL
)''')
conn.commit()
conn.close()

def fetch_and_store():
    params = {
        'vs_currency': 'usd',
        'days': 1,
        'interval': 'hourly'
    }
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        # data['prices'] and data['total_volumes'] are lists of [timestamp, value]
        prices = data.get('prices', [])
        volumes = {v[0]: v[1] for v in data.get('total_volumes', [])}
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        for ts, price in prices:
            volume = volumes.get(ts, None)
            cur.execute('INSERT OR REPLACE INTO hourly_data VALUES (?, ?, ?)', (ts // 1000, price, volume))
        conn.commit()
        conn.close()
        print(f"Fetched {len(prices)} records at {datetime.utcnow()} UTC")
    except Exception as e:
        print(f"Error fetching data: {e}")

def main():
    parser = argparse.ArgumentParser(description="Fetch hourly ETH data")
    parser.add_argument('--once', action='store_true', help='Run fetch once and exit')
    args = parser.parse_args()

    if args.once:
        fetch_and_store()
        return

    while True:
        fetch_and_store()
        time.sleep(3600)

if __name__ == '__main__':
    main()
