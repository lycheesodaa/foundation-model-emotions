# Foundation model stock forecasting with sentiment

This folder contains the code and scripts for running the stock forecasting task.

The process is as follows:
- Download the FNSPID dataset to a folder -- I use `external_data/` for this.
- Run `process_fnspid.py` to process the dataset in chunks, as it is fairly large.
- Run `sentiment_embeddings.py` for the extraction of classical and emotional sentiment, then run `process_daily_sentiment.py` to obtain the daily aggregations in the final format for predictions.

Once done, you may run the shell script for MOIRAI forecasting as such:
```shell
sh stock_forecasting.sh
```
This runs the `MOIRAI_stocks_all.py` file internally. You may change the output directory as required. 

After all tests have concluded, you may run the `calc_loss.py` function to calculate the MAPE loss for each run.
