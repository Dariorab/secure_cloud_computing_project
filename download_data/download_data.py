import yfinance as yf
from datetime import datetime
from joblib import dump
import argparse
from pathlib import Path


def downlaoad_data(args):

    '''
    param: args object containing command-line arguments
    brief: Download historical stock data, save it to a file, and print the DataFrame information.

    '''

    # Set up End and Start times for data grab
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)

    # Stock history dump
    df = yf.download("ETH-USD", start, end)
    df = df.reset_index()
    dump(df, args.data)
    df.info()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help="Insert path to save data")
    args = parser.parse_args()

    Path(args.data).parent.mkdir(parents=True, exist_ok=True)

    downlaoad_data(args)