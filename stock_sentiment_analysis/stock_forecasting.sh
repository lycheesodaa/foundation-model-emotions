gpu_id=1

for news_type in content headlines
do
  for emotion_type in historical sentiment emotion
  do
    run_name=${news_type}_${emotion_type}
    folder_path=./external_data/FNSPID/candle_w_sentiment/day_average_${news_type}/
    output_dir=./results/stocks/stocks_${run_name}/

    # Zero-shot evaluation
    python MOIRAI_stocks_all.py \
    --folder_path $folder_path \
    --output_dir $output_dir \
    --run_name $run_name \
    --gpu_id $gpu_id
  done
done
