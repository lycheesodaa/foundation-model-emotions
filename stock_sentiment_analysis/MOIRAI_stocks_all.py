import warnings
from pathlib import Path

import numpy as np
import argparse
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from tqdm import tqdm
import os

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

parser = argparse.ArgumentParser(description='MOIRAI')
parser.add_argument('--folder_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='./results/data/')
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--desc_prefix', type=str, default='')
parser.add_argument('--finetuned', type=bool, default=False)
parser.add_argument('--yaml_cfg', type=str)
parser.add_argument('--run_name', type=str, default='headlines',
                    choices=['headlines_sentiment', 'content_sentiment',
                             'headlines_emotion', 'content_emotion',
                             'headlines_historical', 'content_historical'])
args = parser.parse_args()

SIZE = "large"  # model size: choose from {'small', 'base', 'large'}
PDT_LIST = [1, 3, 7, 14, 30]  # prediction length: any positive integer
CTX = 512  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32 # batch size: any positive integer

# setting up the covariates for model/data configuration
base_covars = ['open', 'high', 'low', 'close', 'volume']
num_features_map = {
    'sentiment': base_covars + ['positive', 'negative', 'neutral'],
    'emotion': base_covars + ['sadness', 'neutral_emotion', 'fear', 'anger', 'disgust', 'surprise', 'joy'],
    'historical': base_covars
}
past_dynamic_real_cols = num_features_map[str(args.run_name).split('_')[-1]]
num_past_feat_dynamic_real = len(past_dynamic_real_cols)


def get_test_data(csv_path, pred_len):
    # data formatting into GluonTS dataset
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if len(df) == 0:
        raise IndexError(f'Dataset {Path(csv_path).stem} is empty.')
    elif len(df) <= CTX + pred_len:
        raise IndexError(f'Dataset {Path(csv_path).stem} is shorter than the required length {CTX + pred_len}.')

    if pd.infer_freq(df.index) is None:
        print('Casting to B freq...')
        df = df.asfreq('B')

    test_ratio = 0.2
    total_len = len(df)
    test_len = int(test_ratio * total_len)

    ds = PandasDataset(df, target='adj close', past_feat_dynamic_real=past_dynamic_real_cols, freq='B')

    # dummy split the last 3 rows to align with the rest of the results
    dummy_split, _ = split(
        ds, offset=-3
    )
    # assign last test_len time steps as test set
    _, test_template = split(
        dummy_split, offset=-test_len + 1
    )

    # Construct rolling window evaluation data
    return test_template.generate_instances(
        prediction_length=pred_len,  # number of time steps for each prediction
        windows=test_len - pred_len,  # number of windows in rolling window evaluation (TEST//PDT for non-overlap)
        distance=1,  # number of time steps between each window (distance=PDT for non-overlapping windows)
    )


# evaluation
for PDT in PDT_LIST:
    print(f"Evaluating for PDT={PDT}...")
    if args.finetuned:
        ckpt_path = f'outputs/finetune/moirai_1.0_R_large/{args.yaml_cfg}/{args.run_name}/checkpoints/'
        filename = os.listdir(ckpt_path)[0]

        # load from FT'd checkpoint
        model = MoiraiForecast.load_from_checkpoint(
            prediction_length=PDT,
            context_length=CTX,
            patch_size=PSZ,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=num_past_feat_dynamic_real,
            checkpoint_path=ckpt_path + filename
        )
    else:
        # Prepare pre-trained model by downloading model weights from huggingface hub
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}"),
            prediction_length=PDT,
            context_length=CTX,
            patch_size=PSZ,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=num_past_feat_dynamic_real,
        )
    predictor = model.create_predictor(batch_size=BSZ, device=f'cuda:{args.gpu_id}')

    dataframes = []
    error_files = {}
    for i, file in enumerate(os.listdir(args.folder_path)):
        print(f'Forecasting for {file}...')
        try:
            warnings.filterwarnings("ignore")
            test_data = get_test_data(Path(args.folder_path) / file, PDT)
        except IndexError as e:
            error_files[file] = str(e)
            continue

        forecasts = predictor.predict(test_data.input)

        input_it = iter(test_data.input)
        label_it = iter(test_data.label)
        forecast_it = iter(forecasts)

        forecast_out = []
        forecast_out_median = []
        forecast_var = []
        forecast_q10 = []
        forecast_q50 = []
        forecast_q90 = []
        label_out = []
        dates = []
        for _ in tqdm(range(test_data.windows)):
            inp = next(input_it)
            label = next(label_it)
            forecast = next(forecast_it)

            label_out.extend(label['target'])
            forecast_out.extend(forecast.mean)
            forecast_out_median.extend(forecast.median)
            forecast_var.extend([np.var(samples, ddof=1) for samples in forecast.samples.T])
            forecast_q10.extend(forecast.quantile(.1))
            forecast_q50.extend(forecast.quantile(.5))
            forecast_q90.extend(forecast.quantile(.9))
            dates.extend(forecast.index.to_timestamp())

        to_export = pd.DataFrame({
            'date': dates,
            'pred_mean': forecast_out,
            'pred_median': forecast_out_median,
            'var': forecast_var,
            'q10': forecast_q10,
            'q50': forecast_q50,
            'q90': forecast_q90,
            'true': label_out,
            'stock': (Path(args.folder_path) / file).stem,
        })
        dataframes.append(to_export)

    # concat all dataframes
    result = pd.concat(dataframes)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.finetuned:
        result.to_csv(args.output_dir + f'MOIRAI_pl{PDT}_finetuned.csv', index=False)
    else:
        result.to_csv(args.output_dir + f'MOIRAI_pl{PDT}_zero_shot.csv', index=False)

    print('Errors occurred with files:')
    print(error_files)