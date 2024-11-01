import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from utils.utils import check_null_values


@dataclass
class UtteranceInfo:
    """Data class to store utterance information."""
    name: str
    start_time: float
    end_time: float
    # label: str
    # avg_val: float
    # avg_act: float
    # avg_dom: float

class UtteranceDataProcessor:
    """Base class for processing utterance forecasting data."""
    
    def __init__(self, data_dir, base_path: Path = Path("Files")):
        self.data_dir = data_dir
        self.base_path = Path(base_path)
        print("Initializing utterance data processor...")

    def parse_iemocap_info_from_raw(self, info_file: str = "IEMOCAP_EmoEvaluation.txt") -> pd.DataFrame:
        """Parse IEMOCAP timing information into a structured format."""
        utterance_data: List[UtteranceInfo] = []

        for session in range(1,6):
            file_path = Path(f'Processed/IEMOCAP_full_release/Session{session}/dialog/EmoEvaluation')

            text_files = [file for file in os.listdir(file_path) if file.endswith('.txt')]

            for text_file in text_files:
                # Read all lines at once
                with open(file_path / text_file) as f:
                    lines = f.readlines()

                for line in lines:
                    if "Ses" in line:
                        fields = line.strip().split('\t')
                        times = re.findall(r'\d+\.\d+', fields[0])
                        assert len(times) == 2
                        start_time, end_time = map(float, times)
                        utterance_data.append(UtteranceInfo(
                            name=fields[1],
                            start_time=start_time,
                            end_time=end_time
                        ))

        return pd.DataFrame.from_records([vars(u) for u in utterance_data])

    def parse_iemocap_info(self, info_file: str = "iemocap_full_dataset_w_dims.csv"):
        return pd.read_csv(Path(self.base_path) / info_file)[['name', 'start_time', 'end_time']]

    def calculate_time_distance(self, lookup_table: pd.DataFrame,
                              current_utt: str, future_utt: str) -> float:
        """Calculate time distance between utterances using vectorized operations."""
        def get_utterance_midpoint(utt_name: str) -> float:
            utt_data = lookup_table[lookup_table['name'].str.contains(utt_name)].iloc[0]
            return (utt_data['start_time'] + utt_data['end_time']) / 2

        return get_utterance_midpoint(future_utt) - get_utterance_midpoint(current_utt)

    def normalize_features(self, features: List[NDArray]) -> List[NDArray]:
        """Normalize features using vectorized operations."""
        stacked_features = np.vstack(features)
        mean_feat = np.mean(stacked_features, axis=0, keepdims=True)
        std_feat = np.std(stacked_features, axis=0, keepdims=True)
        return [(feat - mean_feat) / std_feat for feat in features]

    def save_processed_data(self,
                            data: Dict[str, pd.DataFrame],
                            utt_lengths: List[int],
                            step: int,
                            feature_type: str = "dynamic",
                            mean_only=False) -> None:
        """Save processed data efficiently."""
        if mean_only:
            output_base = self.base_path / f"{self.data_dir}/step_{step}_mean/{feature_type}"
        else:
            output_base = self.base_path / f"{self.data_dir}/step_{step}/{feature_type}"

        output_base.mkdir(parents=True, exist_ok=True)
        
        max_length = max(utt_lengths)
        print(f'Total of {len(utt_lengths)} valid utterances.')
        print(f'Padding to length {max_length} windows per utterance.')

        for speaker_id, df in data.items():
            print(f"Saving data for speaker {speaker_id}...")
            if feature_type == "dynamic":
                df['features'] = df['features'].apply(
                    lambda x: np.pad(x, ((0, max_length - x.shape[0]), (0, 0)),
                                   mode='constant')
                )
            
            output_path = output_base / f"audio_visual_speaker_{speaker_id}.csv"
            df.to_pickle(output_path)

class UFCurrentDataProcessor(UtteranceDataProcessor):
    """Process current utterance forecasting data."""
    
    def __init__(self):
        super().__init__(data_dir="UF_Cur_data")
    
    def process_dataset(self, step: int = 0, 
                        feature_type: str = "dynamic",
                        normalize: bool = False,
                        mean_only=False) -> tuple[dict[str, DataFrame], list[Any]]:
        """Process the current utterance dataset with optimized operations."""
        print('Processing Current Data...')
        speaker_data = {}
        utt_lengths = []
        lookup_table = self.parse_iemocap_info()
        
        for speaker in range(1, 6):
            for gender in ['F', 'M']:
                speaker_id = f"{speaker}{gender}"
                
                if mean_only:
                    data_path = self.base_path / "statistical" / feature_type / 'historical_mean' / f"audio_visual_speaker_{speaker_id}.csv"
                else:
                    data_path = self.base_path / "statistical" / feature_type / f"audio_visual_speaker_{speaker_id}.csv"
                audio_data = pd.read_pickle(data_path)
                
                # Pre-allocate DataFrame with estimated size
                uf_data = []
                
                # Process data in chunks for better memory efficiency
                for idx in range(len(audio_data) - step):
                    current_name = audio_data['name'].iloc[idx]
                    target_name = audio_data['name'].iloc[idx + step]
                    
                    if (current_name[:-5] == target_name[:-5] and 
                        audio_data['label'].iloc[idx + step] is not None):
                        
                        utt_lengths.append(audio_data['stat_features'].iloc[idx].shape[0])
                        
                        uf_data.append({
                            'current_name': current_name,
                            'features': audio_data['stat_features'].iloc[idx],
                            'UF_label': audio_data['label'].iloc[idx + step],
                            'forecasted_utt_name': target_name,
                            'time_distance': self.calculate_time_distance(
                                lookup_table, current_name, target_name
                            )
                        })
                
                # Convert to DataFrame efficiently
                uf_df = pd.DataFrame(uf_data)
                
                if normalize:
                    uf_df['features'] = self.normalize_features(uf_df['features'].tolist())
                
                speaker_data[speaker_id] = uf_df
        
        return speaker_data, utt_lengths


class UFHistoryDataProcessor(UtteranceDataProcessor):
    """Process historical utterance forecasting data."""
    
    def __init__(self):
        super().__init__(data_dir="UF_His_data")

    def process_dataset(self, step: int = 0,
                        feature_type: str = "dynamic",
                        normalize: bool = False,
                        mean_only=False) -> tuple[dict[str, DataFrame], list[Any]]:
        """Process the historical dataset with optimized operations."""
        print('Processing Historical Data...')
        speaker_data = {}
        utt_lengths = []
        lookup_table = self.parse_iemocap_info()
        
        for speaker in range(1, 6):
            for gender in ['F', 'M']:
                speaker_id = f"{speaker}{gender}"
                print(f'Processing speaker {speaker_id}...')
                
                if mean_only:
                    data_path = self.base_path / "statistical" / feature_type / 'historical_mean' / f"audio_visual_speaker_{speaker_id}.csv"
                else:
                    data_path = self.base_path / "statistical" / feature_type / f"audio_visual_speaker_{speaker_id}.csv"
                audio_data = pd.read_pickle(data_path)

                uf_data = []
                
                for idx in range(len(audio_data) - step):
                    current_name = audio_data['name'].iloc[idx]
                    target_name = audio_data['name'].iloc[idx + step]

                    # Skip if not in same dialog or invalid label
                    if (pd.isna(audio_data['label'].iloc[idx + step])
                        or current_name[:-5] != target_name[:-5]):
                        continue

                    features = audio_data['stat_features'].iloc[idx]

                    # Include history if idx > 0 and same speaker in history
                    if idx > 0:
                        history_name = audio_data['name'].iloc[idx - 1]
                        if current_name[:-5] == history_name[:-5]:
                            features = np.vstack((
                                audio_data['stat_features'].iloc[idx - 1],
                                features
                            ))

                    # assert _validate_features(features), f'feature vectors have nulls'

                    utt_lengths.append(features.shape[0])
                    uf_data.append({
                        'current_name': current_name,
                        'features': features,
                        'UF_label': audio_data['label'].iloc[idx + step],
                        'forecasted_utt_name': target_name,
                        'time_distance': self.calculate_time_distance(
                            lookup_table, current_name, target_name
                        )
                    })
                
                uf_df = pd.DataFrame(uf_data)

                # Check for null values with detailed reporting
                check_null_values(uf_df, speaker_id)

                # assert uf_df.notnull().all().all(), f"Null values found in {speaker_id} data"
                
                if normalize:
                    uf_df['features'] = self.normalize_features(uf_df['features'].tolist())
                
                speaker_data[speaker_id] = uf_df
        
        return speaker_data, utt_lengths

def main():
    """Main function to run the data processing."""
    # Process current utterance data
    # uf_cur = UFCurrentDataProcessor()
    # cur_data, cur_lengths = uf_cur.process_dataset(
    #     step=1, 
    #     feature_type='dynamic',
    #     normalize=True
    # )
    # uf_cur.save_processed_data(cur_data, cur_lengths, step=1)
    
    # Process historical utterance data
    uf_his = UFHistoryDataProcessor()
    his_data, his_lengths = uf_his.process_dataset(step=1, normalize=True, mean_only=True)
    uf_his.save_processed_data(his_data, his_lengths, step=1, mean_only=True)

if __name__ == "__main__":
    main()