import os
import shutil

import pandas as pd

OTHER_SPLIT_COUNT = 1
directory = 'external_data/FNSPID/'
output_directory = directory + 'full_news/'
hist_directory = directory + 'full_history/'


def split_dataframe_by_column(df, column_name):
    # Replace blank symbols with 'Other'
    df.fillna(value={'Stock_symbol': 'Other'}, inplace=True)

    # Get unique values in the specified column
    unique_values = df[column_name].unique()

    # Create a dictionary to store the split DataFrames
    split_dfs = {}

    # Split the DataFrame for each unique value
    for value in unique_values:
        split_dfs[value] = df[df[column_name] == value]

    return split_dfs


def check_csv_exists(directory, filename):
    if not filename.lower().endswith('.csv'):
        filename += '.csv'

    file_path = os.path.join(directory, filename)

    return os.path.isfile(file_path)


def read_append_export(filepath, symbol, to_append):
    df = pd.read_csv(filepath)
    # if len(df) > 10 * 10 ** 5:
    #     new_filepath = os.path.splitext(filepath)[0] + f'_{OTHER_SPLIT_COUNT}.csv'
    try:
        # last_row = df.iloc[-1]
        # first_to_append = to_append.iloc[0]
        #
        # # Convert timestamps to datetime objects
        # last_timestamp = pd.to_datetime(last_row['Date'])
        # new_timestamp = pd.to_datetime(first_to_append['Date'])
        #
        # # Return if timestamp of new dataframe is newer
        # if new_timestamp > last_timestamp or last_row['Article_title'] == first_to_append['Article_title']:
        #     return

        df = pd.concat([df, to_append], ignore_index=True)
        df.drop_duplicates(subset=['Date', 'Article_title'], inplace=True)
        df = df.sort_values(by=['Date'], ascending=False).reset_index(drop=True)

        df.to_csv(filepath, index=False)
        print(f'Appending to {symbol}.csv (+{len(to_append)} rows)...')
    except Exception as e:
        print('ERROR: ' + str(e))
        print('ERROR: For symbol ' + symbol)
        print(df.tail(1))
        print(to_append.head(1))


def process_chunks_and_split(filename):
    print('----------------- Processing file {} -----------------'.format(filename))
    chunk_size = 10 ** 5
    # if os.path.getsize(filename) < 3 * 1024 * 1024 * 1024: # 3GB
    #     chunk_size = None
    counter = 1
    column_types = {
        'Article_title': 'string',
        'Stock_symbol': 'string',
        'Url': 'string',
        'Publisher': 'string',
        'Author': 'string',
        'Article': 'string',
        'Lsa_summary': 'string',
        'Luhn_summary': 'string',
        'Textrank_summary': 'string',
        'Lexrank_summary': 'string'
    }

    with pd.read_csv(directory + filename, chunksize=chunk_size, dtype=column_types) as reader:
        for chunk in reader:
            print(f'******* Processing chunk #{counter} *******')
            split_dfs = split_dataframe_by_column(chunk, 'Stock_symbol')
            # For each symbol found in the chunk, check if an existing CSV exists
            for symbol in split_dfs.keys():
                if len(split_dfs[symbol]) == 0:
                    print('Empty dataframe for symbol {}?'.format(symbol))
                    continue
                if check_csv_exists(output_directory, str(symbol)):
                    # If CSV exists, read, append and export
                    read_append_export(output_directory + f'{symbol}.csv', symbol, split_dfs[symbol])
                else:
                    # If CSV doesn't exist, export new
                    split_dfs[symbol].to_csv(output_directory + f'{symbol}.csv', index=False)
                    print(f'Creating new {symbol}.csv (+{len(split_dfs[symbol])} rows)...')

            counter += 1

nasdaq = 'nasdaq_external_data.csv'
all_ext = 'All_external.csv'
process_chunks_and_split(nasdaq)
process_chunks_and_split(all_ext)