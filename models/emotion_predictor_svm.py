import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from jaxtyping import Float, Bool
from tqdm import tqdm
from uni2ts.loss.packed import PackedMSELoss
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule, MoiraiFinetune


class EmotionForecastSVMModel(nn.Module):
    def __init__(
            self,
            prediction_length: int,
            target_dim: int,
            feat_dynamic_real_dim: int,
            past_feat_dynamic_real_dim: int,
            context_length: int,
            num_emotions: int,
            module_kwargs: Optional[dict] = None,
            module: Optional[MoiraiModule] = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-base"),
            patch_size: int | str = "auto",
            num_samples: int = 100,
            svm_kernel: str = 'rbf',
            svm_C: float = 1.0,
    ):
        super().__init__()

        # Initialize MOIRAI base model
        self.moirai = MoiraiForecast(
            prediction_length=prediction_length,
            target_dim=target_dim,
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
            context_length=context_length,
            module_kwargs=module_kwargs,
            module=module,
            patch_size=patch_size,
            num_samples=num_samples
        ).requires_grad_(False)

        # Initialize SVM classifier
        self.svm = SVC(
            kernel=svm_kernel,
            C=svm_C,
            probability=True  # Enable probability estimates
        )
        self.scaler = StandardScaler()

        # Flag to track if SVM has been fitted
        self.is_svm_fitted = False

    def forward(
            self,
            past_target: Float[torch.Tensor, "batch past_time tgt"],
            past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
            past_is_pad: Bool[torch.Tensor, "batch past_time"],
            feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
            observed_feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
            past_feat_dynamic_real: Optional[Float[torch.Tensor, "batch past_time past_feat"]] = None,
            past_observed_feat_dynamic_real: Optional[Float[torch.Tensor, "batch past_time past_feat"]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Get predictions from MOIRAI
        # ! forecast function is modified to return mean instead of sampling from distr.
        forecasts = self.moirai(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
        )

        # Flatten the forecasts
        batch_size = forecasts.shape[0]
        flattened_forecasts = forecasts.reshape(batch_size, -1)

        # During training, we don't use the SVM here
        # SVM predictions are handled separately in the EmotionPredictor class
        return forecasts, flattened_forecasts

    def fit_svm(self, features: np.ndarray, labels: np.ndarray):
        """Fit the SVM classifier on the given features and labels"""
        # Scale the features
        scaled_features = self.scaler.fit_transform(features)

        # Fit the SVM
        self.svm.fit(scaled_features, labels)
        self.is_svm_fitted = True

    def predict_svm(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get SVM predictions and probabilities"""
        if not self.is_svm_fitted:
            raise RuntimeError("SVM must be fitted before making predictions")

        # Scale the features
        scaled_features = self.scaler.transform(features)

        # Get predictions and probabilities
        predictions = self.svm.predict(scaled_features)
        probabilities = self.svm.predict_proba(scaled_features)

        return predictions, probabilities



class EmotionPredictorSVM:
    def __init__(
            self,
            model: EmotionForecastSVMModel,
            device: torch.device,
            batch_size: int = 32,
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size

    def prepare_input_tensors(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_target = features
        past_observed_target = (features != 0.0)
        # past_observed_target = torch.ones_like(features, dtype=torch.bool)
        past_is_pad = (features.sum(dim=2) == 0)
        return past_target, past_observed_target, past_is_pad

    def train_epoch(
            self,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            accumulation_steps: int = 1
    ) -> float:
        """
        Training epoch for SVM-based model.

        We only train the model based on the next_features with e.g. MSELoss, and fit the SVM later on with the best
        performing model based on validation performance.
        """
        self.model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            features, next_features, emotion_labels = [b.to(self.device) for b in batch]

            # Prepare input tensors
            past_target, past_observed_target, past_is_pad = self.prepare_input_tensors(features)

            # Forward pass through MOIRAI only
            forecasts, _ = self.model(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad
            )

            loss = criterion(forecasts, next_features) / accumulation_steps

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            total_loss += loss.item() * accumulation_steps

            # Only update weights after accumulating enough gradients
            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        return total_loss / len(train_loader)

    def validate(
            self,
            val_loader: torch.utils.data.DataLoader,
            criterion: nn.Module,
    ) -> Tuple[float, float]:
        """
        Validation for SVM-based model
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                features, next_features, emotion_labels = [b.to(self.device) for b in batch]

                # Prepare input tensors
                past_target, past_observed_target, past_is_pad = self.prepare_input_tensors(features)

                # Forward pass through MOIRAI only
                forecasts, _ = self.model(
                    past_target=past_target,
                    past_observed_target=past_observed_target,
                    past_is_pad=past_is_pad
                )

                loss = criterion(forecasts, next_features)
                total_loss += loss.item()

        return total_loss / len(val_loader), 0.0

    def fit_svm_on_best_model(
            self,
            train_loader: torch.utils.data.DataLoader
    ) -> None:
        """
        Training epoch for SVM-based model. Collects MOIRAI features for SVM training.
        """
        print("Fitting SVM on best model...")
        self.model.eval()
        all_features = []
        all_labels = []

        for batch in tqdm(train_loader, total=len(train_loader)):
            features, next_features, emotion_labels = [b.to(self.device) for b in batch]

            # Prepare input tensors
            past_target, past_observed_target, past_is_pad = self.prepare_input_tensors(features)

            # Forward pass through MOIRAI only
            _, flattened_forecasts = self.model(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad
            )

            # Collect features and labels for SVM
            all_features.append(flattened_forecasts.cpu().detach().numpy())
            all_labels.extend(emotion_labels.cpu().numpy())

        # Concatenate all features and fit SVM
        features_array = np.concatenate(all_features, axis=0)
        labels_array = np.array(all_labels)
        self.model.fit_svm(features_array, labels_array)
        print("SVM fitted.")

    def predict(
            self,
            features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions from both MOIRAI and SVM
        """
        self.model.eval()

        with torch.no_grad():
            # Prepare input tensors
            past_target, past_observed_target, past_is_pad = self.prepare_input_tensors(features)

            # Get predictions
            forecasts, flattened_forecasts = self.model(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad
            )

            # Get SVM predictions
            _, probabilities = self.model.predict_svm(
                flattened_forecasts.cpu().numpy()
            )

        return forecasts.cpu(), torch.from_numpy(probabilities)