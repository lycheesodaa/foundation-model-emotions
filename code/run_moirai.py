import sys, os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import recall_score, confusion_matrix
from tqdm import tqdm
from uni2ts.model.moirai import MoiraiModule

from utils.utils import prepare_data, confusion_matrix_plot
from models.emotion_predictor import EmotionForecastModel, EmotionPredictor
from models.emotion_predictor_svm import EmotionPredictorSVM, EmotionForecastSVMModel
from data_provider.emotion_dataset import EmotionDataset


def train_emotion_forecast_model(features, next_features, labels, speaker_ids, batch_size=32, num_epochs=50,
                                 patience=10, freeze_backbone=False, gpu_id=0, ckpt_path='checkpoints/',
                                 accumulation_steps=1, use_svm=False):
    # Model parameters
    target_dim = features.shape[2]  # Feature dimension
    context_length = features.shape[1]  # Sequence length
    prediction_length = next_features.shape[1]  # Sequence length
    num_emotions = len(set(labels))

    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    # Initialize model
    if use_svm:
        model = EmotionForecastSVMModel(
            prediction_length=prediction_length,
            target_dim=target_dim,
            feat_dynamic_real_dim=0,  # adjust if you have dynamic features
            past_feat_dynamic_real_dim=0,  # adjust if you have past dynamic features
            context_length=context_length,
            num_emotions=num_emotions,
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-small"),
            patch_size=8, # this argument is necessary to override 'auto' so MOIRAI processes only CTX, not CTX+PDT
        )
    else:
        model = EmotionForecastModel(
            prediction_length=prediction_length,
            target_dim=target_dim,
            feat_dynamic_real_dim=0,  # adjust if you have dynamic features
            past_feat_dynamic_real_dim=0,  # adjust if you have past dynamic features
            context_length=context_length,
            num_emotions=num_emotions,
            hidden_dim=256,
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-small"),
            patch_size=8, # this argument is necessary to override 'auto' so MOIRAI processes only CTX, not CTX+PDT
            freeze_backbone=freeze_backbone
        )

    # Setup cross-validation
    logo = LeaveOneGroupOut()
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    results = {}

    # Leave-one-speaker-out cross validation
    for fold, (train_idx, test_idx) in enumerate(logo.split(features, labels, speaker_ids)):
        print(f"\nTraining fold {fold} (Speaker {speaker_ids[test_idx[0]]})")

        # Create train/val split
        train_size = int(0.8 * len(train_idx))
        train_subset = train_idx[:train_size]
        val_subset = train_idx[train_size:]

        # Create data loaders
        train_loader = DataLoader(
            EmotionDataset(features[train_subset], next_features[train_subset], labels[train_subset], verbose=True),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            EmotionDataset(features[val_subset], next_features[val_subset], labels[val_subset]),
            batch_size=batch_size
        )
        test_loader = DataLoader(
            EmotionDataset(features[test_idx], next_features[test_idx], labels[test_idx]),
            batch_size=batch_size
        )

        # Initialize training components
        if use_svm:
            predictor = EmotionPredictorSVM(model, device=device)
            predictor.fit_svm_on_best_model(train_loader)
        else:
            predictor = EmotionPredictor(model, device=device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

            torch.cuda.empty_cache()

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(num_epochs):
                train_loss = predictor.train_epoch(train_loader, optimizer, criterion,
                                                   accumulation_steps=accumulation_steps)
                val_loss, val_uwr = predictor.validate(val_loader, criterion)
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val UWR = {val_uwr:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), Path(ckpt_path) / f'best_model_speaker_{speaker_ids[test_idx[0]]}.pt')
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Load best model and evaluate on test set
            model.load_state_dict(torch.load(Path(ckpt_path) / f'best_model_speaker_{speaker_ids[test_idx[0]]}.pt'))
            predictor = EmotionPredictor(model, device=device)

        # Test loop - get predictions
        all_forecasts = []
        all_true = []
        all_probs = []
        all_labels = []

        for batch in tqdm(test_loader, total=len(test_loader)):
            features_batch, next_features_batch, labels_batch = [b.to(device) for b in batch]
            forecasts, probs = predictor.predict(features_batch)
            all_forecasts.append(forecasts)
            all_true.append(next_features_batch)
            all_probs.append(probs)
            all_labels.extend(labels_batch.cpu().numpy())

        # Store results
        results[speaker_ids[test_idx[0]]] = {
            'forecasts': torch.cat(all_forecasts, dim=0).numpy(),
            'true': torch.cat(all_true, dim=0).cpu().numpy(),
            'probabilities': torch.cat(all_probs, dim=0).numpy(),
            'true_labels': all_labels
        }

    return results


# Example usage
if __name__ == "__main__":
    batch_size = 8
    accumulation_steps = 4
    freeze_backbone=True
    use_svm=True
    run_name='run10'

    export_path = Path(f'results/moirai_{run_name}/')
    export_path.mkdir(parents=True, exist_ok=True)

    # Prepare your data - Moirai uses the mean only data, so 179 features
    features, next_features, labels, speaker_ids = prepare_data(
        directory='Files/UF_His_data/step_1_mean', mean_only=True, verbose=True)

    # Train model and get results
    results = train_emotion_forecast_model(features, next_features, labels, speaker_ids,
                                           batch_size=batch_size,
                                           patience=3, freeze_backbone=freeze_backbone, gpu_id=0,
                                           accumulation_steps=accumulation_steps, use_svm=use_svm)

    all_recall = []
    prediction_length = next_features.shape[1]

    # Print results for each speaker
    for speaker, result in results.items():
        predictions = np.argmax(result['probabilities'], axis=1)
        accuracy = np.mean(predictions == result['true_labels'])
        uwr = recall_score(result['true_labels'], predictions, average='macro')
        print(f"\nResults for Speaker {speaker}:")
        print(f"Accuracy - {accuracy:.4f}")
        print(f"UWR - {uwr:.4f}")

        # Confusion matrix
        try:
            print("Confusion Matrix:")
            print(confusion_matrix(result['true_labels'], predictions))
            confusion_matrix_plot(
                str(speaker), result['true_labels'], predictions,
                ['Anger', 'Happy', 'Neutral', 'Sad'], normalize=False,
                savedir=f'Images/pl{prediction_length}_{run_name}'
            )
        except Exception as e:
            print('Error plotting confusion matrix:', e)

        # Recall score
        all_recall.append(uwr)

        # Export the results
        pd.DataFrame.from_dict({
            k: [v] for k, v in result.items()
        }).to_pickle(export_path / f'results_speaker_{speaker}')
        # when loading, use:
        # df = pd.read_pickle(file)
        # result = {col: df[col].iloc[0] for col in df.columns}

    print('Average UWR:', np.mean(all_recall * 100))