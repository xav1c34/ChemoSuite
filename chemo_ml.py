"""
chemo_ml.py — Вычислительное ядро хемометрики (Low-Level Data Fusion, PLS-DA, VIP).
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import f
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler


def calculate_vip(model: PLSRegression, X: np.ndarray) -> np.ndarray:
    """
    Расчет индекса значимости переменных в проекции (Variable Importance in Projection, VIP).
    """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape

    # Взвешенная сумма квадратов объясненной дисперсии Y для каждого LV
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

    # Квантиль распределения Фишера F(2, N-2)
    f_crit = f.ppf(confidence, 2, n - 2)
    radius_scale = np.sqrt((2 * (n - 1) / (n - 2)) * f_crit)

    r1 = s1 * radius_scale
    r2 = s2 * radius_scale

    theta = np.linspace(0, 2 * np.pi, n_points)
    x_ell = np.mean(scores_x) + r1 * np.cos(theta)
    y_ell = np.mean(scores_y) + r2 * np.sin(theta)
    return x_ell, y_ell


def train_plsda_model(
        X_df: pd.DataFrame, y_labels: np.ndarray, n_components: int = 2, block_scaling: bool = True
) -> Dict:
    """
    Обучение и валидация модели PLS-DA с кросс-валидацией Q^2.
    """
    feature_names = list(X_df.columns)
    X = X_df.values.astype(float)
    y = y_labels.astype(float)
    n_samples, n_features = X.shape

    # Блочное шкалирование (1 / sqrt(P_block)) для баланса FT-ICR и EEM
    if block_scaling:
        fticr_cols = [c for c in feature_names if c.startswith("VK_") or c in ["Mn", "Mw", "AI", "DBE"]]
        eem_cols = [c for c in feature_names if
                    c in ["FI", "HIX", "SUVA254", "C1_Fulvic", "C2_Lignin_Humic", "C3_Protein"]]

        if fticr_cols and eem_cols:
            scale_fticr = 1.0 / np.sqrt(len(fticr_cols))
            scale_eem = 1.0 / np.sqrt(len(eem_cols))
            for i, col in enumerate(feature_names):
                if col in fticr_cols:
                    X[:, i] *= scale_fticr
                elif col in eem_cols:
                    X[:, i] *= scale_eem

    # Стандартизация (Z-Score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Обучение базовой модели
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X_scaled, y)

    scores = pls.x_scores_
    loadings = pls.x_loadings_
    y_pred = pls.predict(X_scaled).flatten()
    y_class_pred = (y_pred >= 0.5).astype(int)
    acc = np.mean(y_class_pred == y)

    # Расчет объясненной дисперсии R2X и R2Y
    r2x = np.sum(np.var(scores, axis=0)) / np.sum(np.var(X_scaled, axis=0)) * 100.0
    ss_total_y = np.sum((y - np.mean(y)) ** 2)
    ss_res_y = np.sum((y - y_pred) ** 2)
    r2y = (1.0 - (ss_res_y / ss_total_y)) * 100.0 if ss_total_y > 0 else 0.0

    # Кросс-валидация Leave-One-Out для оценки Q^2
    cv = LeaveOneOut() if n_samples <= 30 else KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv_pred = np.zeros(n_samples)

    for train_idx, test_idx in cv.split(X_scaled):
        pls_cv = PLSRegression(n_components=n_components, scale=False)
        pls_cv.fit(X_scaled[train_idx], y[train_idx])
        y_cv_pred[test_idx] = pls_cv.predict(X_scaled[test_idx]).flatten()

    press = np.sum((y - y_cv_pred) ** 2)
    q2 = (1.0 - (press / ss_total_y)) * 100.0 if ss_total_y > 0 else 0.0

    # Расчет значимости признаков (VIP)
    vip_values = calculate_vip(pls, X_scaled)
    vip_df = pd.DataFrame({"Descriptor": feature_names, "VIP": vip_values}).sort_values("VIP",
                                                                                        ascending=False).reset_index(
        drop=True)

    # Эллипс 95%
    ell_x, ell_y = compute_hotelling_ellipse(scores[:, 0], scores[:, 1])

    return {
        "scores_t1": scores[:, 0],
        "scores_t2": scores[:, 1] if n_components > 1 else np.zeros(n_samples),
        "ellipse_x": ell_x,
        "ellipse_y": ell_y,
        "R2X": float(max(0.0, r2x)),
        "R2Y": float(max(0.0, r2y)),
        "Q2": float(q2),
        "Accuracy": float(acc * 100.0),
        "VIP_df": vip_df,
        "y_pred": y_pred,
    }


def generate_multimodal_benchmark() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Генерирует мультимодальный датасет (FT-ICR MS 20 ячеек + EEM индексы/вклады)
    для проб: Фоновый Байкал (0) vs Зона шлам-лигнина (1).
    """
    np.random.seed(42)
    n = 14
    sample_ids = [f"Lignin_Impact_{i + 1}" if i < 7 else f"Baikal_Control_{i - 6}" for i in range(n)]
    y_labels = np.array([1 if i < 7 else 0 for i in range(n)])

    records = []
    for i, s_id in enumerate(sample_ids):
        is_lignin = (i < 7)
        row = {"Sample_ID": s_id}

        # 1. Признаки FT-ICR MS (Заселенность 20 ячеек Ван-Кревелена Перминовой)
        # Лигнин концентрируется в зонах высокой ароматичности и среднего H/C (VK_5, VK_6, VK_7, VK_9)
        for k in range(1, 21):
            if is_lignin and k in [5, 6, 7, 9]:
                row[f"VK_{k}"] = round(np.random.uniform(9.0, 18.0), 2)
            elif not is_lignin and k in [13, 14, 15]:  # Природное алифатическое РОВ
                row[f"VK_{k}"] = round(np.random.uniform(10.0, 20.0), 2)
            else:
                row[f"VK_{k}"] = round(np.random.uniform(0.5, 4.5), 2)

        # Молекулярные индексы
        row["AI"] = round(np.random.uniform(0.38, 0.58) if is_lignin else np.random.uniform(0.05, 0.22), 3)
        row["DBE"] = round(np.random.uniform(14.0, 22.0) if is_lignin else np.random.uniform(6.0, 11.0), 2)

        # 2. Оптические признаки EEM-PARAFAC
        row["FI"] = round(np.random.uniform(1.15, 1.35) if is_lignin else np.random.uniform(1.65, 1.95), 2)
        row["HIX"] = round(np.random.uniform(9.5, 18.0) if is_lignin else np.random.uniform(1.5, 4.2), 2)
        row["SUVA254"] = round(np.random.uniform(3.8, 6.2) if is_lignin else np.random.uniform(1.2, 2.4), 2)
        row["C1_Fulvic"] = round(np.random.uniform(1.2, 2.5), 2)
        row["C2_Lignin_Humic"] = round(np.random.uniform(4.5, 8.5) if is_lignin else np.random.uniform(0.4, 1.2), 2)
        row["C3_Protein"] = round(np.random.uniform(0.1, 0.4) if is_lignin else np.random.uniform(1.5, 3.2), 2)

        records.append(row)

    df = pd.DataFrame(records)
    return df, pd.Series(y_labels, name="Class_Target")