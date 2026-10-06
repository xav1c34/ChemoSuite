"""
chemo_ml.py — Вычислительное ядро хемометрики (Data Fusion, PLS-DA, VIP, Permutation Test).
Методология кафедры аналитической химии и лаборатории природных гуминовых систем химфака МГУ.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import f
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler


def classify_feature_blocks(columns: List[str]) -> Dict[str, str]:
    """Классифицирует названия колонок по принадлежности к аналитическому блоку."""
    mapping = {}
    for c in columns:
        c_low = c.lower()
        if any(k in c_low for k in ["a254", "a280", "a365", "a465", "a665", "e2_e3", "e4_e6", "s_r", "sr", "s_275", "s_350", "d2a", "mw_est", "uv_mw", "absorbance"]):
            mapping[c] = "UV-Vis"
        elif c.startswith("VK_") or any(k in c_low for k in ["mn", "mw", "dbe", "ai", "nosc", "cho_", "chon_", "chos_", "chons_"]):
            mapping[c] = "FT-ICR MS"
        elif any(k in c_low for k in ["fi", "hix", "suva", "c1", "c2", "c3", "c4", "fulvic", "lignin", "protein", "humic"]):
            mapping[c] = "EEM-PARAFAC"
        else:
            mapping[c] = "Other"
    return mapping


def extract_fticr_descriptors(assigned_df: pd.DataFrame, sample_id: str, basis: str = "count") -> Dict[str, float]:
    """
    Извлекает вектор дескрипторов FT-ICR MS из идентифицированных формул:
    20 ячеек Ван-Кревелена Перминовой, Mn, Mw, средние индексы и распределение гетероатомов.
    """
    if assigned_df is None or assigned_df.empty or "C" not in assigned_df.columns:
        return {}

    sub = assigned_df[
        (assigned_df["O/C"] >= 0.0) & (assigned_df["O/C"] <= 1.0) &
        (assigned_df["H/C"] >= 0.2) & (assigned_df["H/C"] <= 2.2)
    ].copy()

    res = {"Sample_ID": sample_id}

    oc_bins = [0.00, 0.25, 0.50, 0.75, 1.000001]
    hc_bins = [0.20, 0.60, 1.00, 1.40, 1.80, 2.200001]

    total_count = len(sub)
    total_int = sub["intensity"].sum() if "intensity" in sub.columns else float(total_count)

    grid_count = np.zeros((5, 4), dtype=float)
    grid_weight = np.zeros((5, 4), dtype=float)

    if total_count > 0:
        c_idx = np.clip(np.digitize(sub["O/C"], oc_bins) - 1, 0, 3)
        r_idx = np.clip(np.digitize(sub["H/C"], hc_bins) - 1, 0, 4)
        np.add.at(grid_count, (r_idx, c_idx), 1.0)
        if "intensity" in sub.columns:
            np.add.at(grid_weight, (r_idx, c_idx), sub["intensity"].values)
        else:
            grid_weight = grid_count.copy()

    pct_count = (grid_count / total_count * 100.0) if total_count > 0 else grid_count
    pct_weight = (grid_weight / total_int * 100.0) if total_int > 0 else grid_weight

    active_mat = pct_count if basis == "count" else pct_weight
    for r in range(5):
        for c in range(4):
            res[f"VK_{1 + r * 4 + c}"] = round(float(active_mat[r, c]), 3)

    res["Mn"] = round(float(assigned_df["mass"].mean()), 2)
    if "intensity" in assigned_df.columns and assigned_df["intensity"].sum() > 0:
        res["Mw"] = round(float(np.average(assigned_df["mass"], weights=assigned_df["intensity"])), 2)
    else:
        res["Mw"] = res["Mn"]

    res["H/C"] = round(float(assigned_df["H/C"].mean()), 3)
    res["O/C"] = round(float(assigned_df["O/C"].mean()), 3)
    res["DBE"] = round(float(assigned_df["DBE"].mean()), 2)
    res["AI"] = round(float(assigned_df["AI"].mean()), 3)
    if "NOSC" in assigned_df.columns:
        res["NOSC"] = round(float(assigned_df["NOSC"].mean()), 3)

    if "Hetero_Class" in assigned_df.columns:
        counts = assigned_df["Hetero_Class"].value_counts(normalize=True) * 100.0
        res["CHO_pct"] = round(float(counts.get("CHO", 0.0)), 2)
        res["CHON_pct"] = round(float(counts.get("CHON", 0.0)), 2)
        res["CHOS_pct"] = round(float(counts.get("CHOS", 0.0)), 2)
        res["CHONS_pct"] = round(float(counts.get("CHONS", 0.0)), 2)

    return res


def merge_feature_blocks(
    df_fticr: Optional[pd.DataFrame] = None,
    df_eem: Optional[pd.DataFrame] = None,
    df_uv: Optional[pd.DataFrame] = None,
    on_col: str = "Sample_ID",
) -> pd.DataFrame:
    """Универсальное соединение матриц FT-ICR, EEM и UV-Vis по идентификаторам образцов."""
    dfs = [df for df in [df_fticr, df_eem, df_uv] if df is not None and not df.empty]
    if not dfs:
        return pd.DataFrame()
    if len(dfs) == 1:
        return dfs[0].copy()

    def _merge_pair(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
        if d1.empty:
            return d2.copy()
        if d2.empty:
            return d1.copy()
        merged = pd.merge(d1, d2, on=on_col, how="inner")
        if not merged.empty:
            return merged

        # Слияние с очисткой от расширений файлов и регистра
        c1 = d1.copy()
        c2 = d2.copy()
        c1["_id_clean"] = c1[on_col].astype(str).str.strip().str.lower().str.replace(r"\.(csv|tsv|txt|dat)$", "", regex=True)
        c2["_id_clean"] = c2[on_col].astype(str).str.strip().str.lower().str.replace(r"\.(csv|tsv|txt|dat)$", "", regex=True)
        cols_to_drop = [c for c in [on_col] if c in c2.columns]
        merged = pd.merge(c1, c2.drop(columns=cols_to_drop), on="_id_clean", how="inner").drop(columns=["_id_clean"])
        return merged

    result = dfs[0]
    for nxt in dfs[1:]:
        result = _merge_pair(result, nxt)
    return result


def calculate_vip(model: PLSRegression, X: np.ndarray) -> np.ndarray:
    """Расчет индекса значимости переменных в проекции (Variable Importance in Projection, VIP)."""
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape

    ssy_components = np.zeros(h)
    for a in range(h):
        ssy_components[a] = np.dot(t[:, a], t[:, a]) * np.dot(q[:, a], q[:, a])

    ssy_total = np.sum(ssy_components)
    if ssy_total < 1e-12:
        return np.ones(p)

    vip = np.zeros(p)
    for j in range(p):
        weight_sum = 0.0
        for a in range(h):
            norm_w = np.linalg.norm(w[:, a])
            w_normed = w[j, a] / norm_w if norm_w > 0 else 0.0
            weight_sum += ssy_components[a] * (w_normed ** 2)
        vip[j] = np.sqrt(p * weight_sum / ssy_total)

    return vip


def compute_hotelling_ellipse(
    scores_x: np.ndarray, scores_y: np.ndarray, n_points: int = 100, confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """Расчет координат 95% эллипса Хотеллинга T^2."""
    n = len(scores_x)
    if n < 4:
        return np.array([]), np.array([])

    s1 = np.std(scores_x, ddof=1)
    s2 = np.std(scores_y, ddof=1)

    f_crit = f.ppf(confidence, 2, n - 2)
    radius_scale = np.sqrt((2 * (n - 1) / (n - 2)) * f_crit)

    r1 = s1 * radius_scale
    r2 = s2 * radius_scale

    theta = np.linspace(0, 2 * np.pi, n_points)
    x_ell = np.mean(scores_x) + r1 * np.cos(theta)
    y_ell = np.mean(scores_y) + r2 * np.sin(theta)
    return x_ell, y_ell


def prepare_fused_features(
    X_df: pd.DataFrame,
    strategy: str = "low_level",
    block_scaling: bool = True,
    n_mid_components: int = 3,
) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    """Подготовка и слияние матриц признаков по стратегиям Low-Level или Mid-Level."""
    feature_names = list(X_df.columns)
    block_map = classify_feature_blocks(feature_names)

    fticr_cols = [c for c in feature_names if block_map[c] == "FT-ICR MS"]
    eem_cols = [c for c in feature_names if block_map[c] == "EEM-PARAFAC"]
    uv_cols = [c for c in feature_names if block_map[c] == "UV-Vis"]
    other_cols = [c for c in feature_names if block_map[c] == "Other"]

    if strategy == "mid_level" and len(fticr_cols) >= 3:
        X_ft = StandardScaler().fit_transform(X_df[fticr_cols].values.astype(float))
        k_comp = min(n_mid_components, len(X_df) - 2, len(fticr_cols))
        pca = PCA(n_components=k_comp, random_state=42)
        ft_scores = pca.fit_transform(X_ft)

        pca_feat_names = [f"FTICR_PC{i+1}" for i in range(k_comp)]
        new_block_map = {name: "FT-ICR MS" for name in pca_feat_names}

        rem_cols = eem_cols + uv_cols + other_cols
        if rem_cols:
            X_rem = StandardScaler().fit_transform(X_df[rem_cols].values.astype(float))
            for c in rem_cols:
                new_block_map[c] = block_map[c]
            X_fused = np.hstack([ft_scores, X_rem])
            all_names = pca_feat_names + rem_cols
        else:
            X_fused = ft_scores
            all_names = pca_feat_names

        return X_fused, all_names, new_block_map
    else:
        X = X_df.values.astype(float).copy()
        if block_scaling:
            for b_name in ["FT-ICR MS", "EEM-PARAFAC", "UV-Vis", "Other"]:
                b_cols = [c for c in feature_names if block_map[c] == b_name]
                if len(b_cols) > 0:
                    scale_val = 1.0 / np.sqrt(len(b_cols))
                    for i, col in enumerate(feature_names):
                        if col in b_cols:
                            X[:, i] *= scale_val

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, feature_names, block_map


def train_plsda_model(
    X_df: pd.DataFrame,
    y_labels: np.ndarray,
    n_components: int = 2,
    fusion_strategy: str = "low_level",
    block_scaling: bool = True,
    n_mid_components: int = 3,
) -> Dict:
    """
    Обучение, валидация PLS-DA модели и расчет хемометрических метрик.
    """
    X_scaled, feature_names, block_map = prepare_fused_features(
        X_df, strategy=fusion_strategy, block_scaling=block_scaling, n_mid_components=n_mid_components
    )
    y = np.asarray(y_labels, dtype=float)
    n_samples, n_features = X_scaled.shape

    max_comp = max(1, min(n_components, n_samples - 1, n_features))
    pls = PLSRegression(n_components=max_comp, scale=False)
    pls.fit(X_scaled, y)

    scores = pls.x_scores_
    y_pred = pls.predict(X_scaled).flatten()
    y_class_pred = (y_pred >= 0.5).astype(int)

    # Матрица ошибок и диагностика
    cm = confusion_matrix((y >= 0.5).astype(int), y_class_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = float((sens + spec) / 2.0 * 100.0)
    acc = float((tp + tn) / n_samples * 100.0)

    # R2X и R2Y
    r2x = float(np.sum(np.var(scores, axis=0)) / np.sum(np.var(X_scaled, axis=0)) * 100.0)
    ss_total_y = np.sum((y - np.mean(y)) ** 2)
    ss_res_y = np.sum((y - y_pred) ** 2)
    r2y = float((1.0 - (ss_res_y / ss_total_y)) * 100.0 if ss_total_y > 0 else 0.0)

    # Leave-One-Out / K-Fold кросс-валидация для Q2
    cv = LeaveOneOut() if n_samples <= 30 else KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv_pred = np.zeros(n_samples)

    for train_idx, test_idx in cv.split(X_scaled):
        pls_cv = PLSRegression(n_components=max_comp, scale=False)
        pls_cv.fit(X_scaled[train_idx], y[train_idx])
        y_cv_pred[test_idx] = pls_cv.predict(X_scaled[test_idx]).flatten()

    press = np.sum((y - y_cv_pred) ** 2)
    q2 = float((1.0 - (press / ss_total_y)) * 100.0 if ss_total_y > 0 else 0.0)

    # Расчет VIP и привязка к блокам
    vip_values = calculate_vip(pls, X_scaled)
    blocks_col = [block_map.get(col, "Other") for col in feature_names]
    vip_df = pd.DataFrame({
        "Descriptor": feature_names,
        "VIP": np.round(vip_values, 3),
        "Block": blocks_col
    }).sort_values("VIP", ascending=False).reset_index(drop=True)

    # Вклад аналитических блоков в модель (по VIP^2)
    vip_sq = vip_df.copy()
    vip_sq["VIP_sq"] = vip_sq["VIP"] ** 2
    block_sums = vip_sq.groupby("Block")["VIP_sq"].sum()
    tot_sq = block_sums.sum()
    block_contribs = {}
    if tot_sq > 0:
        for b, s in block_sums.items():
            block_contribs[b] = round(float(s / tot_sq * 100.0), 1)

    t1 = scores[:, 0]
    t2 = scores[:, 1] if max_comp > 1 else np.zeros(n_samples)
    ell_x, ell_y = compute_hotelling_ellipse(t1, t2)

    return {
        "scores_t1": t1,
        "t1": t1,
        "scores_t2": t2,
        "t2": t2,
        "ellipse_x": ell_x,
        "ell_x": ell_x,
        "ellipse_y": ell_y,
        "ell_y": ell_y,
        "R2X": max(0.0, r2x),
        "R2Y": max(0.0, r2y),
        "Q2": q2,
        "Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "Sensitivity": sens * 100.0,
        "Specificity": spec * 100.0,
        "Confusion_Matrix": cm,
        "VIP_df": vip_df,
        "vip_df": vip_df,
        "Block_Contributions": block_contribs,
        "y_pred": y_pred,
        "y_cv_pred": y_cv_pred,
        "feature_names": feature_names,
    }


def run_permutation_test(
    X_df: pd.DataFrame,
    y_labels: np.ndarray,
    n_components: int = 2,
    n_permutations: int = 50,
    fusion_strategy: str = "low_level",
    block_scaling: bool = True,
    n_mid_components: int = 3,
    random_state: int = 42,
) -> Dict:
    """Пермутационный тест PLS-DA модели для подтверждения статистической значимости Q2."""
    rng = np.random.RandomState(random_state)
    X_scaled, _, _ = prepare_fused_features(
        X_df, strategy=fusion_strategy, block_scaling=block_scaling, n_mid_components=n_mid_components
    )
    y = np.asarray(y_labels, dtype=float)
    n_samples, n_features = X_scaled.shape
    max_comp = max(1, min(n_components, n_samples - 1, n_features))

    ss_tot = np.sum((y - np.mean(y)) ** 2)

    cv = LeaveOneOut() if n_samples <= 30 else KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(X_scaled))

    y_cv_orig = np.zeros(n_samples)
    for tr, te in splits:
        pls = PLSRegression(n_components=max_comp, scale=False)
        pls.fit(X_scaled[tr], y[tr])
        y_cv_orig[te] = pls.predict(X_scaled[te]).flatten()
    q2_orig = float((1.0 - np.sum((y - y_cv_orig) ** 2) / ss_tot) * 100.0)

    perm_q2 = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        ss_perm = np.sum((y_perm - np.mean(y_perm)) ** 2)

        y_cv_p = np.zeros(n_samples)
        for tr, te in splits:
            pls_p_cv = PLSRegression(n_components=max_comp, scale=False)
            pls_p_cv.fit(X_scaled[tr], y_perm[tr])
            y_cv_p[te] = pls_p_cv.predict(X_scaled[te]).flatten()
        q2_p = (1.0 - np.sum((y_perm - y_cv_p) ** 2) / ss_perm) * 100.0
        perm_q2.append(float(q2_p))

    perm_q2_arr = np.array(perm_q2)
    p_val_q2 = float((np.sum(perm_q2_arr >= q2_orig) + 1.0) / (n_permutations + 1.0))

    return {
        "q2_orig": q2_orig,
        "perm_q2": perm_q2_arr,
        "p_val_q2": p_val_q2,
    }


def generate_multimodal_benchmark() -> Tuple[pd.DataFrame, pd.Series]:
    """Генератор согласованного эталонного датасета (Фоновый Байкал vs Шлам-лигнин БЦБК, 14 проб)."""
    np.random.seed(42)
    n = 14
    sample_ids = [f"Lignin_Impact_{i + 1}" if i < 7 else f"Baikal_Control_{i - 6}" for i in range(n)]
    y_labels = np.array([1 if i < 7 else 0 for i in range(n)])

    records = []
    for i, s_id in enumerate(sample_ids):
        is_lignin = (i < 7)
        row = {"Sample_ID": s_id}

        for k in range(1, 21):
            if is_lignin and k in [5, 6, 7, 9]:
                row[f"VK_{k}"] = round(np.random.uniform(9.0, 18.0), 2)
            elif not is_lignin and k in [13, 14, 15]:
                row[f"VK_{k}"] = round(np.random.uniform(10.0, 20.0), 2)
            else:
                row[f"VK_{k}"] = round(np.random.uniform(0.5, 4.5), 2)

        row["AI"] = round(np.random.uniform(0.38, 0.58) if is_lignin else np.random.uniform(0.05, 0.22), 3)
        row["DBE"] = round(np.random.uniform(14.0, 22.0) if is_lignin else np.random.uniform(6.0, 11.0), 2)
        row["FI"] = round(np.random.uniform(1.15, 1.35) if is_lignin else np.random.uniform(1.65, 1.95), 2)
        row["HIX"] = round(np.random.uniform(9.5, 18.0) if is_lignin else np.random.uniform(1.5, 4.2), 2)
        row["SUVA254"] = round(np.random.uniform(3.8, 6.2) if is_lignin else np.random.uniform(1.2, 2.4), 2)
        row["C1_Fulvic"] = round(np.random.uniform(1.2, 2.5), 2)
        row["C2_Lignin_Humic"] = round(np.random.uniform(4.5, 8.5) if is_lignin else np.random.uniform(0.4, 1.2), 2)
        row["C3_Protein"] = round(np.random.uniform(0.1, 0.4) if is_lignin else np.random.uniform(1.5, 3.2), 2)

        # 3. Признаки спектрофотометрии УФ-Вид (UV-Vis)
        row["A254"] = round(np.random.uniform(0.45, 0.85) if is_lignin else np.random.uniform(0.08, 0.22), 4)
        row["E2_E3"] = round(np.random.uniform(2.4, 3.8) if is_lignin else np.random.uniform(4.5, 6.8), 2)
        row["S_R"] = round(np.random.uniform(0.65, 0.92) if is_lignin else np.random.uniform(1.05, 1.45), 2)
        row["d2A_280"] = round(np.random.uniform(-1.9, -0.9) if is_lignin else np.random.uniform(-0.35, 0.05), 4)
        row["Mw_est"] = round(np.random.uniform(2100.0, 3100.0) if is_lignin else np.random.uniform(850.0, 1600.0), 0)

        records.append(row)

    df = pd.DataFrame(records)
    return df, pd.Series(y_labels, name="Class_Target")