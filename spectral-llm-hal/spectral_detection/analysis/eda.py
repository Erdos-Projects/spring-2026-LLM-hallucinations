# spectral_detection/pca_pipeline.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


@dataclass
class pipeline:
    """
    Spectral dimensionality reduction pipeline.

    Steps:
        1. Average across heads
        2. Log-transform eigenvalues
        3. Standardize
        4. PCA (retain 95% variance)

    Expected feature:
        eig_top10 with shape [L * H * K]
    """

    L: int          # number of layers
    H: int          # number of attention heads
    K: int = 10     # eigenvalues per head
    pca_variance: float = 0.95

    scaler: Optional[StandardScaler] = None
    pca: Optional[PCA] = None

    # ---------------------------------------------------
    # Helpers
    # ---------------------------------------------------

    @staticmethod
    def signed_log1p(x: np.ndarray) -> np.ndarray:
        """
        Stable log transform preserving sign.
        """
        return np.sign(x) * np.log1p(np.abs(x))

    def average_across_heads(self, eig_flat: np.ndarray) -> np.ndarray:
        """
        Convert [L*H*K] -> [L*K]
        """
        expected = self.L * self.H * self.K
        if eig_flat.size != expected:
            raise ValueError(
                f"Expected {expected} features but got {eig_flat.size}"
            )

        eig = eig_flat.reshape(self.L, self.H, self.K)
        eig_mean = eig.mean(axis=1)      # average over heads

        return eig_mean.reshape(self.L * self.K)

    # ---------------------------------------------------
    # Feature construction
    # ---------------------------------------------------

    def featurize(self, eig_flat: np.ndarray) -> np.ndarray:
        """
        Apply:
            head averaging
            log transform
        """
        x = self.average_across_heads(eig_flat.astype(np.float32))
        x = self.signed_log1p(x)
        return x

    # ---------------------------------------------------
    # Load .pt dataset
    # ---------------------------------------------------

    @staticmethod
    def label_to_int(lbl: Any) -> Optional[int]:
        if lbl is None:
            return None

        s = str(lbl).lower()
        if s in {"correct", "true", "yes"}:
            return 1
        if s in {"incorrect", "false", "no"}:
            return 0
        return None

    def load_pt(
        self,
        pt_path: str,
        feature_key: str = "eig_top10"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Returns feature matrix X and sample ids.
        """

        payload = torch.load(pt_path, map_location="cpu")
        data: Dict[str, Dict[str, Any]] = payload["data"]

        X_list = []
        ids = []

        for sample_id, item in data.items():

            eig = item.get(feature_key)
            if eig is None:
                continue

            eig_flat = eig.cpu().numpy()
            x = self.featurize(eig_flat)

            X_list.append(x)
            ids.append(sample_id)

        X = np.stack(X_list)

        return X, ids

    # ---------------------------------------------------
    # PCA pipeline
    # ---------------------------------------------------

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize + PCA
        """

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.pca = PCA(n_components=self.pca_variance)
        X_pca = self.pca.fit_transform(X_scaled)

        return X_pca

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply trained PCA to new data.
        """
        if self.scaler is None or self.pca is None:
            raise RuntimeError("Pipeline not fitted yet.")

        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)