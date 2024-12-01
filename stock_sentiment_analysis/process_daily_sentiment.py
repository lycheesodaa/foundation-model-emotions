import os

import pandas as pd

for news_type in ['content', 'headlines']:
    directory = f'external_data/FNSPID/top50_news_w_{news_type}/with_sentiment/'
    output_dir = f'external_data/FNSPID/candle_w_sentiment/day_average_{news_type}/'
    candle_dir = 'external_data/FNSPID/full_history/'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    files = os.listdir(directory)
    files.sort()

    for file in files:
        # Read the CSV file
        df = pd.read_csv(directory + file)

        if len(df) < 512:
            continue

        # print(df.columns)
        cols = ['Date',
                'neutral', 'positive', 'negative',
                'sadness', 'neutral_emotion', 'fear', 'anger', 'disgust', 'surprise', 'joy']

        df = df[cols]

        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df.rename(columns={'Date': 'date'}, inplace=True)

        # Set timestamp as index
        df.set_index('date', inplace=True)

        # Resample to daily frequency and calculate mean
        daily_avg = df.resample('B').mean()

        # Interpolate values for all sentiments
        daily_avg = daily_avg.interpolate()

        # Reset index to make date a column
        daily_avg.reset_index(inplace=True)
        daily_avg['date'] = pd.to_datetime(daily_avg['date'])

        # print('% missing:', len(daily_avg[daily_avg['neutral'].isna()]) / len(daily_avg) * 100)

        try:
            candlesticks = pd.read_csv(candle_dir + file)
        except FileNotFoundError:
            candlesticks = pd.read_csv(candle_dir + file.lower())

        candlesticks['date'] = pd.to_datetime(candlesticks['date'], utc=True)
        candlesticks.set_index('date', inplace=True)

        if pd.infer_freq(candlesticks.index) is None or pd.infer_freq(candlesticks.index) != 'B':
            print('Candlestick frequency not B. Casting to B freq and ffill nulls...')
            candlesticks = candlesticks.resample('B').ffill()
        else:
            print('Already in B frequency, skipping cast.')

        combined = pd.merge(candlesticks, daily_avg, how='inner', on='date')
        combined['date'] = pd.to_datetime(combined['date'])
        combined.set_index('date', inplace=True)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Save to CSV
        combined.to_csv(output_dir + file)

        print(f"Daily averaged sentiment scores have been saved to {output_dir + file}")