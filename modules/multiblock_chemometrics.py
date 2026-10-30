"""
modules/multiblock_chemometrics.py — Multi-Block Chemometrics & Data Fusion Engine for ChemoSuite Pro.

Provides advanced multi-block data fusion algorithms for heterogeneous analytical modalities:
1. Block-Scaling & Normalization (Frobenius norm, unit variance per block).
2. Multi-Block Partial Least Squares Discriminant Analysis (MB-PLS-DA, super-score NIPALS
   after Westerhuis, Kourti & MacGregor 1998 / Wangen & Kowalski 1989).
3. Sequential and Orthogonalized Partial Least Squares (SO-PLS, Jørgensen & Næs 2007).
4. Block Contribution & Variable Importance in Projection (B-VIP).

Принцип: никаких «правдоподобных» констант. Если метрику невозможно вычислить
(слишком мало образцов, сбой фолда CV), возвращается NaN и предупреждение в `warnings_`.
"""

import warnings
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
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


# ==============================================================================
# Общие утилиты: предобработка с параметрами обучающей выборки, NIPALS PLS
# ==============================================================================

def _to_matrix(data: Any) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        arr = data.apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _fit_block_preprocessing(
    blocks: Dict[str, np.ndarray],
    autoscale: bool,
    scale_blocks: bool,
) -> Dict[str, Dict[str, Any]]:
    """
    Параметры предобработки каждого блока по ОБУЧАЮЩИМ данным: медианы (импутация NaN),
    средние, СКО (если autoscale) и блочный множитель Фробениуса (если scale_blocks).
    """
    params: Dict[str, Dict[str, Any]] = {}
    for name, X in blocks.items():
        X = np.asarray(X, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # колонка целиком из NaN
            med = np.nanmedian(X, axis=0) if X.size else np.zeros(X.shape[1])
        med = np.where(np.isnan(med), 0.0, med)
        Xf = np.where(np.isnan(X), med, X)
        mean = Xf.mean(axis=0)
        if autoscale:
            std = Xf.std(axis=0, ddof=0)
            std = np.where(std < 1e-12, 1.0, std)
        else:
            std = np.ones(X.shape[1])
        Z = (Xf - mean) / std
        if scale_blocks:
            f_norm = np.linalg.norm(Z, "fro")
            factor = 1.0 / f_norm if f_norm > 1e-12 else 1.0
        else:
            factor = 1.0
        params[name] = {"median": med, "mean": mean, "std": std, "factor": float(factor)}
    return params


def _apply_block_preprocessing(X: Any, p: Dict[str, Any]) -> np.ndarray:
    X = _to_matrix(X)
    Xf = np.where(np.isnan(X), p["median"], X)
    return ((Xf - p["mean"]) / p["std"]) * p["factor"]


def _nipals_pls(
    X: np.ndarray,
    Y: np.ndarray,
    n_comp: int,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """
    Классический NIPALS PLS2 (X и Y центрированы). На каждой итерации u обновляется:
    w = X'u/||X'u||, t = Xw, q = Y't/(t't), u = Yq/(q'q) — до сходимости t.
    Компоненты, для которых не осталось дисперсии X или Y, не извлекаются (усечение).
    """
    X_cur = np.array(X, dtype=np.float64, copy=True)
    Y_cur = np.array(Y, dtype=np.float64, copy=True)
    n, p = X_cur.shape
    W, P, Q, T = [], [], [], []
    n_iters, converged = [], []
    for _a in range(int(max(0, n_comp))):
        if np.sum(Y_cur ** 2) < 1e-20 or np.sum(X_cur ** 2) < 1e-20:
            break
        u = Y_cur[:, [int(np.argmax(np.var(Y_cur, axis=0)))]].copy()
        t_old = None
        ok = False
        degenerate = False
        it = 0
        for it in range(1, max_iter + 1):
            w = X_cur.T @ u
            nw = np.linalg.norm(w)
            if nw < 1e-14:
                degenerate = True
                break
            w = w / nw
            t = X_cur @ w
            tt = (t.T @ t).item()
            if tt < 1e-20:
                degenerate = True
                break
            q = (Y_cur.T @ t) / tt
            qq = (q.T @ q).item()
            if qq < 1e-20:
                degenerate = True
                break
            u = (Y_cur @ q) / qq
            if t_old is not None and np.linalg.norm(t - t_old) <= tol * max(1.0, np.linalg.norm(t)):
                ok = True
                break
            t_old = t
        if degenerate:
            break
        tt = (t.T @ t).item()
        p_load = (X_cur.T @ t) / tt
        q = (Y_cur.T @ t) / tt
        X_cur = X_cur - t @ p_load.T
        Y_cur = Y_cur - t @ q.T
        W.append(w.ravel()); P.append(p_load.ravel()); Q.append(q.ravel()); T.append(t.ravel())
        n_iters.append(it); converged.append(bool(ok))

    k = Y.shape[1]
    if W:
        W_m, P_m, Q_m, T_m = (np.column_stack(W), np.column_stack(P), np.column_stack(Q), np.column_stack(T))
    else:
        W_m, P_m, Q_m, T_m = np.zeros((p, 0)), np.zeros((p, 0)), np.zeros((k, 0)), np.zeros((n, 0))
    return {"W": W_m, "P": P_m, "Q": Q_m, "T": T_m, "X_res": X_cur, "Y_res": Y_cur,
            "n_iter": n_iters, "converged": converged}


def _w_star(W: np.ndarray, P: np.ndarray) -> np.ndarray:
    """W* = W (P'W)^-1 — веса для проекции НЕдефлированных X: T = X W*."""
    if W.shape[1] == 0:
        return W
    return W @ np.linalg.pinv(P.T @ W)


def _dummy_y(y_arr: np.ndarray, classes: np.ndarray) -> np.ndarray:
    Y = np.zeros((len(y_arr), len(classes)), dtype=np.float64)
    for idx, cls in enumerate(classes):
        Y[y_arr == cls, idx] = 1.0
    return Y


def _cv_q2(Y: np.ndarray, Y_cv: np.ndarray) -> float:
    """Q2 = 1 - PRESS/SS_tot (SS_tot относительно общего среднего). Без обрезки снизу."""
    ss = float(np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2))
    if ss < 1e-12 or np.any(np.isnan(Y_cv)):
        return float("nan")
    return float(1.0 - np.sum((Y - Y_cv) ** 2) / ss)


def _vip_from_weights(W_unit: np.ndarray, ssy: np.ndarray) -> np.ndarray:
    """VIP_j = sqrt(P * sum_a SSY_a w_ja^2 / sum_a SSY_a), столбцы W_unit — единичной нормы."""
    p = W_unit.shape[0]
    tot = float(np.sum(ssy))
    if p == 0:
        return np.zeros(0)
    if tot < 1e-20:
        return np.full(p, np.nan)
    return np.sqrt(p * ((W_unit ** 2) @ ssy) / tot)


class MultiBlockPLSDA:
    """
    Multi-Block Partial Least Squares Discriminant Analysis (MB-PLS-DA).
    Fuses multiple spectral blocks (e.g., FT-ICR, NMR, FTIR, SEC, CHNS-O)
    into a unified predictive latent space with block weights and block contributions.

    Алгоритм (super-score NIPALS, Westerhuis et al. 1998):
        w_b = X_b'u/||.||, t_b = X_b w_b; T = [t_1..t_B]; w_T = T'u/||.||; t = T w_T;
        q = Y't/(t't); u = Yq/(q'q) — до сходимости t; дефляция всех блоков по супер-скору t.
    Эквивалентен PLS по конкатенированным (масштабированным) блокам; прогноз через
    B = W* Q' с W* = W (P'W)^-1 в пространстве предобработанных переменных.

    Parameters
    ----------
    n_components : число латентных компонент.
    scale_blocks : нормировка каждого блока на единичную норму Фробениуса (равный стартовый вес).
    autoscale : автошкалирование переменных внутри блоков перед блочной нормировкой.
    cv_folds : число фолдов KFold для Q2 (при N < 5 Q2 не вычисляется -> NaN).
    """

    def __init__(
        self,
        n_components: int = 2,
        scale_blocks: bool = True,
        autoscale: bool = False,
        cv_folds: int = 5,
        max_iter: int = 500,
        tol: float = 1e-10,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.scale_blocks = scale_blocks
        self.autoscale = autoscale
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

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
        self.block_contributions_: Dict[str, np.ndarray] = {}  # Modality % per component (w_T^2)
        self.block_importance_: Dict[str, float] = {}  # доля объясненной SS(Y), сумма = 1
        self.block_r2x_: Dict[str, float] = {}
        self.r2x_: float = float("nan")
        self.r2y_: float = float("nan")
        self.q2_: float = float("nan")
        self.b_vip_: Dict[str, np.ndarray] = {}  # Block-VIP scores
        self.ssy_per_component_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None  # P_total x K
        self.classes_: np.ndarray = np.array([])
        self.mean_y_: Optional[np.ndarray] = None
        self.y_fitted_: Optional[np.ndarray] = None
        self.y_cv_pred_: Optional[np.ndarray] = None
        self.n_iter_: List[int] = []
        self.converged_: List[bool] = []
        self.warnings_: List[str] = []
        self._model: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    def _warn(self, msg: str):
        self.warnings_.append(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def _fit_core(self, clean_blocks: Dict[str, np.ndarray], Y_dummy: np.ndarray, n_comp: int) -> Dict[str, Any]:
        """Обучение MB-PLS на переданных данных (без CV). Возвращает все параметры модели."""
        names = list(clean_blocks.keys())
        prep = _fit_block_preprocessing(clean_blocks, self.autoscale, self.scale_blocks)
        Xs = {b: _apply_block_preprocessing(clean_blocks[b], prep[b]) for b in names}
        mean_y = Y_dummy.mean(axis=0, keepdims=True)
        Y_c = Y_dummy - mean_y

        X_def = {b: Xs[b].copy() for b in names}
        Y_def = Y_c.copy()

        T_sup, W_T, Q = [], [], []
        Wb = {b: [] for b in names}
        Pb = {b: [] for b in names}
        Tb = {b: [] for b in names}
        n_iters, conv = [], []

        for _a in range(int(max(0, n_comp))):
            if np.sum(Y_def ** 2) < 1e-20 or sum(np.sum(X_def[b] ** 2) for b in names) < 1e-20:
                break
            u = Y_def[:, [int(np.argmax(np.var(Y_def, axis=0)))]].copy()
            t_old = None
            ok = False
            degenerate = False
            it = 0
            for it in range(1, self.max_iter + 1):
                w_bs, t_bs = [], []
                for b in names:
                    w_b = X_def[b].T @ u
                    nw = np.linalg.norm(w_b)
                    w_b = w_b / nw if nw > 1e-14 else np.zeros_like(w_b)
                    w_bs.append(w_b)
                    t_bs.append(X_def[b] @ w_b)
                T_cat = np.hstack(t_bs)  # N x B
                w_T = T_cat.T @ u
                nwT = np.linalg.norm(w_T)
                if nwT < 1e-14:
                    degenerate = True
                    break
                w_T = w_T / nwT
                t = T_cat @ w_T
                tt = (t.T @ t).item()
                if tt < 1e-20:
                    degenerate = True
                    break
                q = (Y_def.T @ t) / tt
                qq = (q.T @ q).item()
                if qq < 1e-20:
                    degenerate = True
                    break
                u = (Y_def @ q) / qq
                if t_old is not None and np.linalg.norm(t - t_old) <= self.tol * max(1.0, np.linalg.norm(t)):
                    ok = True
                    break
                t_old = t
            if degenerate:
                break

            tt = (t.T @ t).item()
            q = (Y_def.T @ t) / tt
            for b_idx, b in enumerate(names):
                p_b = (X_def[b].T @ t) / tt
                Wb[b].append(w_bs[b_idx].ravel())
                Pb[b].append(p_b.ravel())
                Tb[b].append(t_bs[b_idx].ravel())
                X_def[b] = X_def[b] - t @ p_b.T
            Y_def = Y_def - t @ q.T
            T_sup.append(t.ravel()); W_T.append(w_T.ravel()); Q.append(q.ravel())
            n_iters.append(it); conv.append(bool(ok))

        if len(T_sup) == 0:
            raise ValueError("MB-PLS: не удалось извлечь ни одной латентной компоненты (нет дисперсии X или Y).")

        T_sup = np.column_stack(T_sup)
        W_T = np.column_stack(W_T)  # B x A
        Q = np.column_stack(Q)      # K x A
        Wb = {b: np.column_stack(Wb[b]) for b in names}
        Pb = {b: np.column_stack(Pb[b]) for b in names}
        Tb = {b: np.column_stack(Tb[b]) for b in names}

        # Веса в пространстве конкатенированных переменных: w = [w_T,b * w_b], ||w|| = 1
        W_cat = np.vstack([Wb[b] * W_T[i, :][None, :] for i, b in enumerate(names)])
        P_cat = np.vstack([Pb[b] for b in names])
        coef = _w_star(W_cat, P_cat) @ Q.T  # P_total x K

        return {
            "names": names, "prep": prep, "mean_y": mean_y, "coef": coef,
            "T": T_sup, "W_T": W_T, "Q": Q, "Wb": Wb, "Pb": Pb, "Tb": Tb,
            "W_cat": W_cat, "X_scaled": Xs, "X_res": X_def, "Y_res": Y_def, "Y_c": Y_c,
            "n_iter": n_iters, "converged": conv,
        }

    @staticmethod
    def _predict_core(model: Dict[str, Any], blocks: Dict[str, Any]) -> np.ndarray:
        X_cat = np.hstack([_apply_block_preprocessing(blocks[b], model["prep"][b]) for b in model["names"]])
        return model["mean_y"] + X_cat @ model["coef"]

    # ------------------------------------------------------------------
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
        if not isinstance(blocks, dict):
            raise TypeError("blocks must be a dict {modality_name: (N x P_b) matrix}.")
        self.block_names = list(blocks.keys())
        if not self.block_names:
            raise ValueError("At least one analytical modality block must be provided.")
        self.warnings_ = []

        # Convert to numpy arrays & record feature names
        clean_blocks = {}
        n_samples = None
        for name in self.block_names:
            data = blocks[name]
            arr = _to_matrix(data)
            clean_blocks[name] = arr
            if feature_names and name in feature_names:
                self.block_features[name] = list(feature_names[name])
            elif isinstance(data, pd.DataFrame):
                self.block_features[name] = [str(c) for c in data.columns]
            else:
                self.block_features[name] = [f"{name}_feat_{j+1}" for j in range(arr.shape[1])]

            if n_samples is None:
                n_samples = arr.shape[0]
            elif arr.shape[0] != n_samples:
                raise ValueError(
                    f"Sample count mismatch: block '{name}' has {arr.shape[0]} samples, "
                    f"expected {n_samples}."
                )

        if n_samples < 3:
            raise ValueError(f"Insufficient samples ({n_samples}) for MB-PLS-DA. Minimum is 3.")

        # Format target Y (one-hot dummy matrix for multi-class classification)
        y_arr = np.asarray(y)
        if len(y_arr) != n_samples:
            raise ValueError(f"y has {len(y_arr)} labels, expected {n_samples}.")
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) < 2:
            raise ValueError("Target y must contain at least 2 distinct classes.")
        Y_dummy = _dummy_y(y_arr, self.classes_)

        n_comp = int(max(1, min(self.n_components, n_samples - 1)))
        model = self._fit_core(clean_blocks, Y_dummy, n_comp)
        self._model = model
        A = model["T"].shape[1]
        if A < self.n_components:
            self._warn(
                f"MB-PLS: extracted {A} of {self.n_components} requested components "
                f"(limited by N-1 or exhausted X/Y variance)."
            )

        self.scaling_factors = {b: model["prep"][b]["factor"] for b in self.block_names}
        self.mean_y_ = model["mean_y"]
        self.super_scores_ = model["T"]
        self.super_weights_ = model["W_T"]
        self.y_loadings_ = model["Q"]
        self.coef_ = model["coef"]
        self.n_iter_ = model["n_iter"]
        self.converged_ = model["converged"]
        if not all(self.converged_):
            self._warn("MB-PLS NIPALS did not converge for some components within max_iter.")
        for i, b in enumerate(self.block_names):
            self.block_weights_[b] = model["Wb"][b]
            self.block_loadings_[b] = model["Pb"][b]
            self.block_scores_[b] = model["Tb"][b]
            # Доля блока в супер-скоре компоненты: w_T,b^2 (сумма по блокам = 100%)
            self.block_contributions_[b] = (model["W_T"][i, :] ** 2) * 100.0

        # R2X (общий и по блокам) и R2Y в пространстве предобработанных данных
        total_ss_x = sum(float(np.sum(model["X_scaled"][b] ** 2)) for b in self.block_names)
        resid_ss_x = sum(float(np.sum(model["X_res"][b] ** 2)) for b in self.block_names)
        self.r2x_ = float(1.0 - resid_ss_x / total_ss_x) if total_ss_x > 1e-20 else float("nan")
        for b in self.block_names:
            ss_b = float(np.sum(model["X_scaled"][b] ** 2))
            self.block_r2x_[b] = float(1.0 - np.sum(model["X_res"][b] ** 2) / ss_b) if ss_b > 1e-20 else float("nan")
        total_ss_y = float(np.sum(model["Y_c"] ** 2))
        self.r2y_ = float(1.0 - np.sum(model["Y_res"] ** 2) / total_ss_y) if total_ss_y > 1e-20 else float("nan")
        self.y_fitted_ = self._predict_core(model, clean_blocks)

        # Объясненная SS(Y) по компонентам: ||t_a||^2 ||q_a||^2
        self.ssy_per_component_ = np.sum(model["T"] ** 2, axis=0) * np.sum(model["Q"] ** 2, axis=0)

        # Calculate Block VIP scores
        self._compute_block_vip(model["X_scaled"], model["Y_c"])

        # Cross-validation Q2 estimation
        self._compute_q2_cv(clean_blocks, y_arr, A)

        return self

    def _compute_block_vip(self, scaled_blocks: Dict[str, np.ndarray], Y: np.ndarray):
        """
        VIP переменных всех блоков по конкатенированным весам w = [w_T,b * w_b] с весом компонент
        SSY_a = ||t_a||^2 ||q_a||^2 (объясненная SS(Y)). sum_j VIP_j^2 = P_total, т.е. VIP > 1 —
        выше среднего по ВСЕЙ модели. block_importance_[b] = sum_a SSY_a w_T,ab^2 / sum_a SSY_a.
        """
        if self.super_scores_ is None or self.y_loadings_ is None or self._model is None:
            return
        ssy = self.ssy_per_component_
        W_cat = self._model["W_cat"]
        norms = np.linalg.norm(W_cat, axis=0)
        W_unit = np.divide(W_cat, norms, out=np.zeros_like(W_cat), where=norms > 1e-14)
        vip_all = _vip_from_weights(W_unit, ssy)
        start = 0
        tot = float(np.sum(ssy))
        for i, b_name in enumerate(self.block_names):
            p_b = self.block_weights_[b_name].shape[0]
            self.b_vip_[b_name] = vip_all[start:start + p_b]
            start += p_b
            w_T_sq = self.super_weights_[i, :] ** 2
            self.block_importance_[b_name] = float(np.sum(ssy * w_T_sq) / tot) if tot > 1e-20 else float("nan")

    def _compute_q2_cv(self, clean_blocks: Dict[str, np.ndarray], y_arr: np.ndarray, n_comp: int):
        """K-fold cross-validation Q2 (test folds preprocessed with TRAINING medians/means/scales)."""
        n_samples = len(y_arr)
        self.y_cv_pred_ = None
        if n_samples < 5:
            self.q2_ = float("nan")
            self._warn(f"Q2 not computed: N={n_samples} < 5 samples is too few for cross-validation.")
            return

        Y_dummy = _dummy_y(y_arr, self.classes_)
        k_folds = int(min(max(2, self.cv_folds), n_samples))
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
        y_cv = np.full(Y_dummy.shape, np.nan)
        failures = []
        for f_idx, (train_idx, test_idx) in enumerate(kf.split(np.zeros(n_samples))):
            train_blocks = {k: v[train_idx] for k, v in clean_blocks.items()}
            test_blocks = {k: v[test_idx] for k, v in clean_blocks.items()}
            try:
                n_c = int(max(1, min(n_comp, len(train_idx) - 1)))
                sub = self._fit_core(train_blocks, Y_dummy[train_idx], n_c)
                y_cv[test_idx] = self._predict_core(sub, test_blocks)
            except Exception as exc:  # не маскируем: фиксируем и делаем Q2 неопределенным
                failures.append(f"fold {f_idx + 1}: {type(exc).__name__}: {exc}")

        self.y_cv_pred_ = y_cv
        if failures:
            self.q2_ = float("nan")
            self._warn("Q2 not computed, cross-validation failed in " + "; ".join(failures))
            return
        self.q2_ = _cv_q2(Y_dummy, y_cv)

    def predict_y(self, blocks: Dict[str, Any]) -> np.ndarray:
        """Raw PLS-DA predictions of the dummy Y matrix (N x K): mean_y + X_scaled @ B."""
        if self._model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self._predict_core(self._model, blocks)

    def predict_proba(self, blocks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Backward-compatible alias of predict_y(). Returns the UNCLIPPED PLS-DA regression output
        for each class (rows sum to 1, values may lie outside [0, 1]) — these are not calibrated
        probabilities.
        """
        return self.predict_y(blocks)

    def predict(self, blocks: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict class labels for new multi-modal blocks (argmax of predicted dummy Y)."""
        y_hat = self.predict_y(blocks)
        return self.classes_[np.argmax(y_hat, axis=1)]

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Generate structured analytical summary of multi-block model."""
        rows = []
        n_comp = self.super_scores_.shape[1] if self.super_scores_ is not None else 0
        for b_name in self.block_names:
            p_b = len(self.block_features[b_name])
            contrib_mean = float(np.mean(self.block_contributions_[b_name])) if n_comp > 0 else float("nan")
            top_vip_idx = np.argsort(self.b_vip_[b_name])[::-1][:3] if b_name in self.b_vip_ else []
            top_features = [self.block_features[b_name][i] for i in top_vip_idx]

            rows.append({
                "Modality": b_name,
                "Features Count": p_b,
                "Mean Contribution (%)": round(contrib_mean, 2),
                "Explained-Y Importance (%)": round(100.0 * self.block_importance_.get(b_name, float("nan")), 2),
                "R2X block (%)": round(100.0 * self.block_r2x_.get(b_name, float("nan")), 2),
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

    Для каждого блока: X_k,orth = X_k - T_prev D_k, D_k = (T_prev'T_prev)^-1 T_prev'X_k
    (D_k сохраняется для прогноза), затем NIPALS PLS2 X_k,orth -> остатки Y с итерациями u до сходимости.
    Прогноз: T_k,new = (X_k,new - T_prev,new D_k) W*_k, Y_hat = mean_y + sum_k T_k,new Q_k'.
    """

    def __init__(
        self,
        n_components_per_block: Optional[Dict[str, int]] = None,
        autoscale: bool = False,
        scale_blocks: bool = False,
        cv_folds: int = 5,
        max_iter: int = 500,
        tol: float = 1e-10,
        random_state: int = 42,
    ):
        if n_components_per_block is not None and not isinstance(n_components_per_block, dict):
            raise TypeError("n_components_per_block must be a dict {block_name: n_components}.")
        self.n_components = n_components_per_block or {}
        self.autoscale = autoscale
        self.scale_blocks = scale_blocks
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.block_order: List[str] = []
        self.r2_cumulative: Dict[str, float] = {}
        self.r2_incremental: Dict[str, float] = {}
        self.q2_cumulative: Dict[str, float] = {}
        self.q2_incremental: Dict[str, float] = {}
        self.block_scores: Dict[str, np.ndarray] = {}
        self.n_components_used_: Dict[str, int] = {}
        self.vip_: Dict[str, np.ndarray] = {}
        self.block_features: Dict[str, List[str]] = {}
        self.classes_: np.ndarray = np.array([])
        self.r2y_: float = float("nan")
        self.q2_: float = float("nan")
        self.y_fitted_: Optional[np.ndarray] = None
        self.y_cv_pred_: Optional[np.ndarray] = None
        self.converged_: Dict[str, List[bool]] = {}
        self.warnings_: List[str] = []
        self._model: Optional[Dict[str, Any]] = None

    def _warn(self, msg: str):
        self.warnings_.append(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def _fit_core(self, clean_blocks: Dict[str, np.ndarray], Y_dummy: np.ndarray) -> Dict[str, Any]:
        n_samples = Y_dummy.shape[0]
        prep = _fit_block_preprocessing({b: clean_blocks[b] for b in self.block_order}, self.autoscale, self.scale_blocks)
        mean_y = Y_dummy.mean(axis=0, keepdims=True)
        Y_res = Y_dummy - mean_y
        total_ss_y = float(np.sum(Y_res ** 2))

        T_acc = np.zeros((n_samples, 0))
        stages = []
        for b in self.block_order:
            X = _apply_block_preprocessing(clean_blocks[b], prep[b])
            if T_acc.shape[1] > 0:
                D = np.linalg.pinv(T_acc.T @ T_acc) @ (T_acc.T @ X)
                X_orth = X - T_acc @ D
            else:
                D = None
                X_orth = X
            default_comp = max(1, min(X.shape[1], n_samples - 1, 3))
            a_req = int(self.n_components.get(b, default_comp))
            a_max = max(0, min(a_req, X.shape[1], n_samples - 1 - T_acc.shape[1]))
            ss_before = float(np.sum(Y_res ** 2))
            res = _nipals_pls(X_orth, Y_res, a_max, max_iter=self.max_iter, tol=self.tol)
            W_star = _w_star(res["W"], res["P"])
            T_b = res["T"]
            Y_res = res["Y_res"]
            ss_after = float(np.sum(Y_res ** 2))
            stages.append({
                "block": b, "D": D, "W_star": W_star, "W": res["W"], "Q": res["Q"], "T": T_b,
                "n_comp": T_b.shape[1], "a_req": a_req, "converged": res["converged"],
                "ss_before": ss_before, "ss_after": ss_after,
            })
            T_acc = np.hstack([T_acc, T_b])
        return {"prep": prep, "mean_y": mean_y, "stages": stages, "total_ss_y": total_ss_y}

    def _predict_stages(self, model: Dict[str, Any], blocks: Dict[str, Any]) -> List[np.ndarray]:
        """Кумулятивные прогнозы Y после каждого блока (список длиной B)."""
        first = _to_matrix(blocks[self.block_order[0]])
        n = first.shape[0]
        y_hat = np.tile(model["mean_y"], (n, 1))
        T_acc = np.zeros((n, 0))
        out = []
        for st in model["stages"]:
            X = _apply_block_preprocessing(blocks[st["block"]], model["prep"][st["block"]])
            X_orth = X - T_acc @ st["D"] if st["D"] is not None else X
            if st["n_comp"] > 0:
                T_new = X_orth @ st["W_star"]
                y_hat = y_hat + T_new @ st["Q"].T
            else:
                T_new = np.zeros((n, 0))
            T_acc = np.hstack([T_acc, T_new])
            out.append(y_hat.copy())
        return out

    def fit(self, blocks: Dict[str, Any], y: np.ndarray, block_order: Optional[List[str]] = None):
        """Fit SO-PLS along specified sequence of modalities."""
        if not isinstance(blocks, dict):
            raise TypeError("blocks must be a dict {modality_name: (N x P_b) matrix}.")
        self.block_order = list(block_order or list(blocks.keys()))
        if len(self.block_order) < 2:
            raise ValueError("SO-PLS requires at least 2 modalities in sequence.")
        missing = [b for b in self.block_order if b not in blocks]
        if missing:
            raise ValueError(f"Blocks missing from input: {missing}")
        self.warnings_ = []

        clean_blocks = {}
        n_samples = None
        for b in self.block_order:
            arr = _to_matrix(blocks[b])
            clean_blocks[b] = arr
            self.block_features[b] = ([str(c) for c in blocks[b].columns] if isinstance(blocks[b], pd.DataFrame)
                                      else [f"{b}_feat_{j+1}" for j in range(arr.shape[1])])
            if n_samples is None:
                n_samples = arr.shape[0]
            elif arr.shape[0] != n_samples:
                raise ValueError(f"Sample count mismatch in block '{b}': {arr.shape[0]} vs {n_samples}.")

        y_arr = np.asarray(y)
        if len(y_arr) != n_samples:
            raise ValueError(f"y has {len(y_arr)} labels, expected {n_samples}.")
        if n_samples < 3:
            raise ValueError(f"Insufficient samples ({n_samples}) for SO-PLS. Minimum is 3.")
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) < 2:
            raise ValueError("Target y must contain at least 2 distinct classes.")
        Y_dummy = _dummy_y(y_arr, self.classes_)

        model = self._fit_core(clean_blocks, Y_dummy)
        self._model = model
        total_ss_y = model["total_ss_y"] if model["total_ss_y"] > 1e-20 else float("nan")

        cum = 0.0
        ssy_all, w_blocks = [], {}
        for st in model["stages"]:
            b = st["block"]
            inc = (st["ss_before"] - st["ss_after"]) / total_ss_y
            cum += inc
            self.r2_incremental[b] = float(inc)
            self.r2_cumulative[b] = float(cum)
            self.block_scores[b] = st["T"]
            self.n_components_used_[b] = st["n_comp"]
            self.converged_[b] = st["converged"]
            if st["n_comp"] < st["a_req"]:
                self._warn(f"SO-PLS: block '{b}' contributed {st['n_comp']} of {st['a_req']} requested components "
                           f"(rank exhausted by previous blocks or N-1 limit).")
            if st["n_comp"] > 0 and not all(st["converged"]):
                self._warn(f"SO-PLS: NIPALS did not converge for some components of block '{b}'.")
            ssy_b = np.sum(st["T"] ** 2, axis=0) * np.sum(st["Q"] ** 2, axis=0) if st["n_comp"] > 0 else np.zeros(0)
            ssy_all.append(ssy_b)
            w_blocks[b] = (st["W"], ssy_b)
        self.r2y_ = float(cum)

        # VIP: каждая компонента — PLS-компонента своего (ортогонализованного) блока;
        # вес компоненты = объясненная SS(Y), нормировка по ВСЕЙ модели (sum VIP^2 = P_total)
        tot_ssy = float(np.sum(np.concatenate(ssy_all))) if ssy_all else 0.0
        p_total = sum(clean_blocks[b].shape[1] for b in self.block_order)
        for b in self.block_order:
            W_b, ssy_b = w_blocks[b]
            p_b = clean_blocks[b].shape[1]
            if tot_ssy < 1e-20:
                self.vip_[b] = np.full(p_b, np.nan)
            elif W_b.shape[1] == 0:
                self.vip_[b] = np.zeros(p_b)
            else:
                self.vip_[b] = np.sqrt(p_total * ((W_b ** 2) @ ssy_b) / tot_ssy)

        self.y_fitted_ = self._predict_stages(model, clean_blocks)[-1]
        self._compute_q2_cv(clean_blocks, Y_dummy)
        return self

    def _compute_q2_cv(self, clean_blocks: Dict[str, np.ndarray], Y_dummy: np.ndarray):
        n_samples = Y_dummy.shape[0]
        self.y_cv_pred_ = None
        nan_dict = {b: float("nan") for b in self.block_order}
        if n_samples < 5:
            self.q2_ = float("nan")
            self.q2_cumulative, self.q2_incremental = dict(nan_dict), dict(nan_dict)
            self._warn(f"Q2 not computed: N={n_samples} < 5 samples is too few for cross-validation.")
            return
        k_folds = int(min(max(2, self.cv_folds), n_samples))
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
        B = len(self.block_order)
        y_cv = [np.full(Y_dummy.shape, np.nan) for _ in range(B)]
        failures = []
        for f_idx, (tr, te) in enumerate(kf.split(np.zeros(n_samples))):
            try:
                sub = self._fit_core({b: v[tr] for b, v in clean_blocks.items()}, Y_dummy[tr])
                preds = self._predict_stages(sub, {b: v[te] for b, v in clean_blocks.items()})
                for i in range(B):
                    y_cv[i][te] = preds[i]
            except Exception as exc:  # не маскируем сбой фолда
                failures.append(f"fold {f_idx + 1}: {type(exc).__name__}: {exc}")
        self.y_cv_pred_ = y_cv[-1]
        if failures:
            self.q2_ = float("nan")
            self.q2_cumulative, self.q2_incremental = dict(nan_dict), dict(nan_dict)
            self._warn("Q2 not computed, cross-validation failed in " + "; ".join(failures))
            return
        prev = 0.0
        for i, b in enumerate(self.block_order):
            q2_b = _cv_q2(Y_dummy, y_cv[i])
            self.q2_cumulative[b] = q2_b
            self.q2_incremental[b] = q2_b - prev
            prev = q2_b
        self.q2_ = self.q2_cumulative[self.block_order[-1]]

    def predict_y(self, blocks: Dict[str, Any]) -> np.ndarray:
        """Raw predictions of the dummy Y matrix after all blocks."""
        if self._model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self._predict_stages(self._model, blocks)[-1]

    def predict(self, blocks: Dict[str, Any]) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_y(blocks), axis=1)]

    def get_summary_dataframe(self) -> pd.DataFrame:
        rows = []
        for b in self.block_order:
            rows.append({
                "Modality": b,
                "Components": self.n_components_used_.get(b, 0),
                "Incremental R2Y (%)": round(100.0 * self.r2_incremental.get(b, float("nan")), 2),
                "Cumulative R2Y (%)": round(100.0 * self.r2_cumulative.get(b, float("nan")), 2),
                "Incremental Q2 (%)": round(100.0 * self.q2_incremental.get(b, float("nan")), 2),
                "Cumulative Q2 (%)": round(100.0 * self.q2_cumulative.get(b, float("nan")), 2),
            })
        return pd.DataFrame(rows)
