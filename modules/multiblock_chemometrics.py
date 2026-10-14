"""
modules/multiblock_chemometrics.py — Multi-Block Chemometrics & Data Fusion Engine for ChemoSuite Pro.

Provides advanced multi-block data fusion algorithms for heterogeneous analytical modalities:
1. Block-Scaling & Normalization (Frobenius norm, unit variance per block).
2. Multi-Block Partial Least Squares Discriminant Analysis (MB-PLS-DA).
3. Sequential and Orthogonalized Partial Least Squares (SO-PLS).
4. Block Contribution & Variable Importance in Projection (B-VIP).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


def block_scale(
    blocks: Dict[str, np.ndarray],
    method: str = "frobenius"
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Scale multiple data blocks so that each analytical modality has equal initial leverage.

    Parameters:
    -----------
    blocks: Dict[str, np.ndarray]
        Dictionary mapping modality names to feature matrices (N samples x P_b features).
    method: str
        'frobenius': scale by 1 / sqrt(sum(X_b^2)) so each block has Frobenius norm = 1.
        'sqrt_p': scale by 1 / sqrt(P_b) so variance is normalized by feature count.
        'unit_variance': scale by 1 / trace(cov(X_b)).

    Returns:
    --------
    scaled_blocks: Dict[str, np.ndarray]
    scaling_factors: Dict[str, float]
    """
    scaled_blocks = {}
    scaling_factors = {}

    for name, X in blocks.items():
        X_arr = np.array(X, dtype=np.float64, copy=True)
        # Center each column
        X_centered = X_arr - np.mean(X_arr, axis=0, keepdims=True)

        if method == "frobenius":
            f_norm = np.linalg.norm(X_centered, "fro")
            factor = 1.0 / f_norm if f_norm > 1e-12 else 1.0
        elif method == "sqrt_p":
            p = max(1, X_arr.shape[1])
            factor = 1.0 / np.sqrt(p)
        elif method == "unit_variance":
            var_sum = np.sum(np.var(X_centered, axis=0))
            factor = 1.0 / np.sqrt(var_sum) if var_sum > 1e-12 else 1.0
        else:
            factor = 1.0

        scaled_blocks[name] = X_centered * factor
        scaling_factors[name] = float(factor)

    return scaled_blocks, scaling_factors


class MultiBlockPLSDA:
    """
    Multi-Block Partial Least Squares Discriminant Analysis (MB-PLS-DA).
    Fuses multiple spectral blocks (e.g., FT-ICR, NMR, FTIR, SEC, CHNS-O)
    into a unified predictive latent space with block weights and block contributions.
    """

    def __init__(self, n_components: int = 2, scale_blocks: bool = True):
        self.n_components = n_components
        self.scale_blocks = scale_blocks

        # Fitted parameters
        self.block_names: List[str] = []
        self.block_features: Dict[str, List[str]] = {}
        self.scaling_factors: Dict[str, float] = {}
        self.block_weights_: Dict[str, np.ndarray] = {}  # P_b x A
        self.block_loadings_: Dict[str, np.ndarray] = {}  # P_b x A
        self.block_scores_: Dict[str, np.ndarray] = {}  # N x A
        self.super_scores_: Optional[np.ndarray] = None  # N x A
        self.super_weights_: Optional[np.ndarray] = None  # B x A
        self.y_loadings_: Optional[np.ndarray] = None  # K x A
        self.block_contributions_: Dict[str, np.ndarray] = {}  # Modality % per component
        self.r2x_: float = 0.0
        self.r2y_: float = 0.0
        self.q2_: float = 0.0
        self.b_vip_: Dict[str, np.ndarray] = {}  # Block-VIP scores
        self.classes_: np.ndarray = np.array([])
        self.mean_y_: Optional[np.ndarray] = None

    def fit(
        self,
        blocks: Dict[str, Any],
        y: Any,
        feature_names: Optional[Dict[str, List[str]]] = None
    ) -> "MultiBlockPLSDA":
        """
        Fit MB-PLS-DA model on multi-modal block dictionary.

        Parameters:
        -----------
        blocks: Dict[str, np.ndarray or pd.DataFrame]
            Key is modality name, value is (N x P_b) feature matrix.
        y: np.ndarray, pd.Series or list
            Class labels for N samples.
        feature_names: Optional dict mapping modality name to list of feature column names.
        """
        self.block_names = list(blocks.keys())
        if not self.block_names:
            raise ValueError("At least one analytical modality block must be provided.")

        # Convert to numpy arrays & record feature names
        clean_blocks = {}
        n_samples = None
        for name in self.block_names:
            data = blocks[name]
            if isinstance(data, pd.DataFrame):
                clean_blocks[name] = data.values.astype(np.float64)
                if feature_names and name in feature_names:
                    self.block_features[name] = feature_names[name]
                else:
                    self.block_features[name] = [str(c) for c in data.columns]
            else:
                arr = np.asarray(data, dtype=np.float64)
                clean_blocks[name] = arr
                if feature_names and name in feature_names:
                    self.block_features[name] = feature_names[name]
                else:
                    self.block_features[name] = [f"{name}_feat_{j+1}" for j in range(arr.shape[1])]

            if n_samples is None:
                n_samples = clean_blocks[name].shape[0]
            elif clean_blocks[name].shape[0] != n_samples:
                raise ValueError(
                    f"Sample count mismatch: block '{name}' has {clean_blocks[name].shape[0]} samples, "
                    f"expected {n_samples}."
                )

        if n_samples < 3:
            raise ValueError(f"Insufficient samples ({n_samples}) for MB-PLS-DA. Minimum is 3.")

        # Format target Y (one-hot dummy matrix for multi-class classification)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) < 2:
            raise ValueError("Target y must contain at least 2 distinct classes.")

        Y_dummy = np.zeros((n_samples, len(self.classes_)), dtype=np.float64)
        for idx, cls in enumerate(self.classes_):
            Y_dummy[y_arr == cls, idx] = 1.0

        self.mean_y_ = np.mean(Y_dummy, axis=0, keepdims=True)
        Y_centered = Y_dummy - self.mean_y_

        # Block scaling
        if self.scale_blocks:
            scaled_blocks, self.scaling_factors = block_scale(clean_blocks, method="frobenius")
        else:
            scaled_blocks = {k: v - np.mean(v, axis=0, keepdims=True) for k, v in clean_blocks.items()}
            self.scaling_factors = {k: 1.0 for k in clean_blocks.keys()}

        # Multi-Block NIPALS algorithm
        n_comp = min(self.n_components, n_samples - 1)
        B = len(self.block_names)

        # Deflation workspaces
        X_def = {name: np.array(scaled_blocks[name], copy=True) for name in self.block_names}
        Y_def = np.array(Y_centered, copy=True)

        super_scores = np.zeros((n_samples, n_comp), dtype=np.float64)
        super_weights = np.zeros((B, n_comp), dtype=np.float64)
        y_loadings = np.zeros((len(self.classes_), n_comp), dtype=np.float64)

        for b_name in self.block_names:
            p_b = X_def[b_name].shape[1]
            self.block_weights_[b_name] = np.zeros((p_b, n_comp), dtype=np.float64)
            self.block_loadings_[b_name] = np.zeros((p_b, n_comp), dtype=np.float64)
            self.block_scores_[b_name] = np.zeros((n_samples, n_comp), dtype=np.float64)
            self.block_contributions_[b_name] = np.zeros(n_comp, dtype=np.float64)

        total_ss_x = sum(np.sum(scaled_blocks[b] ** 2) for b in self.block_names)
        total_ss_y = np.sum(Y_centered ** 2)

        for a in range(n_comp):
            # Initialize u with 1st column of Y
            u = Y_def[:, 0:1].copy()
            if np.linalg.norm(u) < 1e-12:
                u = Y_def[:, [np.argmax(np.var(Y_def, axis=0))]].copy()

            t_super = np.zeros((n_samples, 1))

            # Iterative NIPALS convergence
            for _iter in range(100):
                # Calculate block weights w_b = X_b^T * u
                t_blocks_iter = []
                for b_idx, b_name in enumerate(self.block_names):
                    X_b = X_def[b_name]
                    w_b = X_b.T @ u
                    norm_w = np.linalg.norm(w_b)
                    if norm_w > 1e-12:
                        w_b = w_b / norm_w
                    t_b = X_b @ w_b
                    t_blocks_iter.append(t_b)

                # Form super block T_concat (N x B)
                T_concat = np.hstack(t_blocks_iter)  # N x B

                # Super weights w_T = T_concat^T * u
                w_T = T_concat.T @ u
                norm_wT = np.linalg.norm(w_T)
                if norm_wT > 1e-12:
                    w_T = w_T / norm_wT

                # Super score t = T_concat * w_T
                t_new = T_concat @ w_T
                norm_t = np.linalg.norm(t_new)
                if norm_t > 1e-12:
                    t_new = t_new / norm_t

                # Calculate Y loading q = Y^T * t
                q = Y_def.T @ t_new
                norm_q = np.linalg.norm(q)
                if norm_q > 1e-12:
                    q = q / norm_q

                # Update u = Y * q / (q^T * q)
                u_new = Y_def @ q

                if np.linalg.norm(t_new - t_super) < 1e-7:
                    t_super = t_new
                    break
                t_super = t_new

            # Store component 'a' results
            super_scores[:, a] = t_super.ravel()
            super_weights[:, a] = w_T.ravel()
            y_loadings[:, a] = q.ravel()

            # Calculate individual block weights & loadings
            w_T_sq = (w_T.ravel() ** 2)
            sum_w_T_sq = np.sum(w_T_sq)
            if sum_w_T_sq > 1e-12:
                contribs = (w_T_sq / sum_w_T_sq) * 100.0
            else:
                contribs = np.ones(B) * (100.0 / B)

            for b_idx, b_name in enumerate(self.block_names):
                X_b = X_def[b_name]
                w_b = X_b.T @ u
                norm_w = np.linalg.norm(w_b)
                if norm_w > 1e-12:
                    w_b = w_b / norm_w
                t_b = X_b @ w_b

                # Block loading p_b = X_b^T * t_super / (t_super^T * t_super)
                p_b = (X_b.T @ t_super) / (t_super.T @ t_super)

                self.block_weights_[b_name][:, a] = w_b.ravel()
                self.block_loadings_[b_name][:, a] = p_b.ravel()
                self.block_scores_[b_name][:, a] = t_b.ravel()
                self.block_contributions_[b_name][a] = float(contribs[b_idx])

                # Deflate block: X_b = X_b - t_super * p_b^T
                X_def[b_name] = X_b - t_super @ p_b.T

            # Deflate Y: Y = Y - t_super * (t_super^T * Y / (t_super^T * t_super))
            Y_def = Y_def - t_super @ ((t_super.T @ Y_def) / (t_super.T @ t_super))

        self.super_scores_ = super_scores
        self.super_weights_ = super_weights
        self.y_loadings_ = y_loadings

        # Calculate R2X and R2Y
        residual_ss_x = sum(np.sum(X_def[b] ** 2) for b in self.block_names)
        self.r2x_ = float(max(0.0, 1.0 - residual_ss_x / total_ss_x)) if total_ss_x > 1e-12 else 0.0

        residual_ss_y = np.sum(Y_def ** 2)
        self.r2y_ = float(max(0.0, 1.0 - residual_ss_y / total_ss_y)) if total_ss_y > 1e-12 else 0.0

        # Calculate Block VIP scores
        self._compute_block_vip(scaled_blocks, Y_centered)

        # Cross-validation Q2 estimation
        self._compute_q2_cv(clean_blocks, y_arr, n_comp)

        return self

    def _compute_block_vip(self, scaled_blocks: Dict[str, np.ndarray], Y: np.ndarray):
        """Compute Variable Importance in Projection (VIP) for each modality block."""
        if self.super_scores_ is None or self.y_loadings_ is None:
            return

        n_comp = self.super_scores_.shape[1]
        # Sum of squares of Y explained per component
        ss_y_comp = np.zeros(n_comp)
        for a in range(n_comp):
            t_a = self.super_scores_[:, a:a+1]
            q_a = self.y_loadings_[:, a:a+1]
            ss_y_comp[a] = np.sum((t_a @ q_a.T) ** 2)

        total_ss_y_comp = np.sum(ss_y_comp)
        if total_ss_y_comp < 1e-12:
            total_ss_y_comp = 1.0

        for b_name in self.block_names:
            w_b = self.block_weights_[b_name]  # P_b x A
            p_b = w_b.shape[0]
            vip_sq = np.zeros(p_b)
            for a in range(n_comp):
                norm_w = np.linalg.norm(w_b[:, a])
                w_normed = w_b[:, a] / norm_w if norm_w > 1e-12 else w_b[:, a]
                vip_sq += (w_normed ** 2) * ss_y_comp[a]

            vip = np.sqrt(p_b * (vip_sq / total_ss_y_comp))
            self.b_vip_[b_name] = vip

    def _compute_q2_cv(self, clean_blocks: Dict[str, np.ndarray], y_arr: np.ndarray, n_comp: int):
        """Perform 5-fold cross-validation to estimate Q2."""
        n_samples = len(y_arr)
        if n_samples < 5:
            self.q2_ = float(self.r2y_ * 0.85)  # conservative heuristic for micro datasets
            return

        k_folds = min(5, n_samples)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        press = 0.0
        ss_y = 0.0

        Y_dummy = np.zeros((n_samples, len(self.classes_)), dtype=np.float64)
        for idx, cls in enumerate(self.classes_):
            Y_dummy[y_arr == cls, idx] = 1.0

        for train_idx, test_idx in kf.split(y_arr):
            # Train model on fold
            train_blocks = {k: v[train_idx] for k, v in clean_blocks.items()}
            test_blocks = {k: v[test_idx] for k, v in clean_blocks.items()}

            y_train = y_arr[train_idx]
            y_test = Y_dummy[test_idx]
            mean_fold_y = np.mean(Y_dummy[train_idx], axis=0, keepdims=True)

            try:
                sub_model = MultiBlockPLSDA(n_components=n_comp, scale_blocks=self.scale_blocks)
                sub_model.fit(train_blocks, y_train)
                y_pred = sub_model.predict_proba(test_blocks)
                press += np.sum((y_test - y_pred) ** 2)
            except Exception:
                press += np.sum((y_test - mean_fold_y) ** 2)

            ss_y += np.sum((y_test - mean_fold_y) ** 2)

        self.q2_ = float(max(0.0, 1.0 - press / ss_y)) if ss_y > 1e-12 else 0.0

    def predict_proba(self, blocks: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict class probabilities for new multi-modal blocks."""
        if self.super_scores_ is None or self.super_weights_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # Center and scale test blocks
        scaled_test = {}
        for b_name in self.block_names:
            X_raw = np.asarray(blocks[b_name], dtype=np.float64)
            factor = self.scaling_factors.get(b_name, 1.0)
            scaled_test[b_name] = (X_raw - np.mean(X_raw, axis=0, keepdims=True)) * factor

        n_samples = next(iter(scaled_test.values())).shape[0]
        n_comp = self.super_scores_.shape[1]

        # Project into super scores
        t_blocks = []
        for b_name in self.block_names:
            X_b = scaled_test[b_name]
            w_b = self.block_weights_[b_name]
            t_b = X_b @ w_b
            t_blocks.append(t_b)

        # Approximate prediction via linear score mapping
        y_pred = np.tile(self.mean_y_, (n_samples, 1))
        for a in range(n_comp):
            T_cat = np.hstack([t[:, a:a+1] for t in t_blocks])  # N x B
            w_T = self.super_weights_[:, a:a+1]
            t_sup = T_cat @ w_T
            q = self.y_loadings_[:, a:a+1]
            y_pred += t_sup @ q.T

        # Softmax / clip probabilities
        y_pred = np.clip(y_pred, 1e-5, None)
        sum_p = np.sum(y_pred, axis=1, keepdims=True)
        return y_pred / sum_p

    def predict(self, blocks: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict class labels for new multi-modal blocks."""
        probs = self.predict_proba(blocks)
        max_idx = np.argmax(probs, axis=1)
        return self.classes_[max_idx]

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Generate structured analytical summary of multi-block model."""
        rows = []
        n_comp = self.super_scores_.shape[1] if self.super_scores_ is not None else 0
        for b_name in self.block_names:
            p_b = len(self.block_features[b_name])
            contrib_mean = np.mean(self.block_contributions_[b_name]) if n_comp > 0 else 0.0
            top_vip_idx = np.argsort(self.b_vip_[b_name])[::-1][:3] if b_name in self.b_vip_ else []
            top_features = [self.block_features[b_name][i] for i in top_vip_idx]

            rows.append({
                "Modality": b_name,
                "Features Count": p_b,
                "Mean Contribution (%)": round(contrib_mean, 2),
                "Top Biomarkers (VIP)": ", ".join(top_features),
            })

        df = pd.DataFrame(rows)
        return df


class SequentialOrthogonalizedPLS:
    """
    Sequential and Orthogonalized Partial Least Squares (SO-PLS).
    Assesses incremental predictive value of sequential modalities:
    e.g. Block 1 (FT-ICR) -> Block 2 (NMR) -> Block 3 (FTIR).
    Orthogonalizes subsequent modalities against earlier latent spaces to answer:
    'What NEW information does modality k add that was NOT captured by previous modalities?'
    """

    def __init__(self, n_components_per_block: Optional[Dict[str, int]] = None):
        self.n_components = n_components_per_block or {}
        self.block_order: List[str] = []
        self.r2_cumulative: Dict[str, float] = {}
        self.r2_incremental: Dict[str, float] = {}
        self.block_scores: Dict[str, np.ndarray] = {}

    def fit(self, blocks: Dict[str, np.ndarray], y: np.ndarray, block_order: Optional[List[str]] = None):
        """Fit SO-PLS along specified sequence of modalities."""
        self.block_order = block_order or list(blocks.keys())
        if len(self.block_order) < 2:
            raise ValueError("SO-PLS requires at least 2 modalities in sequence.")

        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        n_samples = len(y_arr)

        Y_dummy = np.zeros((n_samples, len(classes)), dtype=np.float64)
        for idx, cls in enumerate(classes):
            Y_dummy[y_arr == cls, idx] = 1.0

        Y_res = Y_dummy - np.mean(Y_dummy, axis=0, keepdims=True)
        total_ss_y = np.sum(Y_res ** 2)
        if total_ss_y < 1e-12:
            total_ss_y = 1.0

        accumulated_scores = []
        cum_explained = 0.0

        for b_idx, b_name in enumerate(self.block_order):
            X_raw = np.array(blocks[b_name], dtype=np.float64)
            X_centered = X_raw - np.mean(X_raw, axis=0, keepdims=True)

            # Orthogonalize against prior accumulated scores
            X_orth = np.array(X_centered, copy=True)
            for T_prior in accumulated_scores:
                inv_tt = np.linalg.pinv(T_prior.T @ T_prior)
                X_orth = X_orth - T_prior @ (inv_tt @ (T_prior.T @ X_orth))

            # Number of components for this block
            max_comp = min(X_orth.shape[1], n_samples - 1, 3)
            n_comp = self.n_components.get(b_name, max(1, max_comp))

            # Fit PLS via NIPALS on current orthogonal X against Y residuals
            X_cur = np.array(X_orth, copy=True)
            T_b_list = []
            ss_before = np.sum(Y_res ** 2)

            for a in range(n_comp):
                # NIPALS on X_cur and Y_res
                u = Y_res[:, [0]].copy()
                t_comp = np.zeros((n_samples, 1))
                for _ in range(50):
                    w = X_cur.T @ u
                    norm_w = np.linalg.norm(w)
                    if norm_w > 1e-12:
                        w = w / norm_w
                    t = X_cur @ w
                    norm_t = np.linalg.norm(t)
                    if norm_t > 1e-12:
                        t = t / norm_t
                    q = Y_res.T @ t
                    u_new = Y_res @ q
                    if np.linalg.norm(t - t_comp) < 1e-6:
                        t_comp = t
                        break
                    t_comp = t

                p = (X_cur.T @ t_comp) / (t_comp.T @ t_comp)
                q = (Y_res.T @ t_comp) / (t_comp.T @ t_comp)

                # Deflate
                X_cur = X_cur - t_comp @ p.T
                Y_res = Y_res - t_comp @ q.T
                T_b_list.append(t_comp)

            T_b = np.hstack(T_b_list) if T_b_list else np.zeros((n_samples, 1))
            self.block_scores[b_name] = T_b
            accumulated_scores.append(T_b)

            ss_after = np.sum(Y_res ** 2)
            inc_r2 = float(max(0.0, (ss_before - ss_after) / total_ss_y))
            cum_explained += inc_r2

            self.r2_incremental[b_name] = inc_r2
            self.r2_cumulative[b_name] = min(1.0, cum_explained)

        return self
