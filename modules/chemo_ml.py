"""
chemo_ml.py — Вычислительное ядро хемометрики (Data Fusion, PLS-DA, VIP, Permutation Test).
Методология кафедры аналитической химии и лаборатории природных гуминовых систем химфака МГУ.
"""
import re
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import f
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler


def _name_tokens(name: str) -> List[str]:
    """Разбивает имя колонки на токены: буквенные и числовые группы (разделители _, -, /, пробел и т.д.).
    'd2A_280' -> ['d', '2', 'a', '280'];  'SUVA254' -> ['suva', '254'];  'C1_Fulvic' -> ['c', '1', 'fulvic']."""
    return re.findall(r"[a-z]+|\d+", str(name).lower())


def _phrase(keyword: str) -> Tuple[str, ...]:
    return tuple(_name_tokens(keyword))


# Префиксы модальностей: проверяется ПЕРВЫЙ токен (или пара токенов) имени, без поиска подстрок.
_BLOCK_PREFIXES: List[Tuple[Tuple[str, ...], str]] = [
    (("sec",), "SEC/GPC"), (("gpc",), "SEC/GPC"),
    (("uv",), "UV-Vis"), (("uvvis",), "UV-Vis"),
    (("eem",), "EEM-PARAFAC"), (("parafac",), "EEM-PARAFAC"),
    (("vk",), "FT-ICR MS"), (("fticr",), "FT-ICR MS"), (("ft", "icr"), "FT-ICR MS"),
    (("raman",), "Raman"),
    (("irms",), "IRMS"),
    (("ic",), "IC"),
    (("ftir",), "FTIR"),
    (("nmr",), "NMR"),
    (("py",), "Thermal"), (("tga",), "Thermal"), (("dtg",), "Thermal"), (("thermal",), "Thermal"),
    (("elem",), "Elemental"), (("elemental",), "Elemental"), (("chns",), "Elemental"), (("chnso",), "Elemental"),
]

# Ключевые слова для имён без префикса. Совпадение — только по целым токенам (непрерывная
# последовательность токенов), побеждает самое длинное совпадение; при равенстве — порядок блоков.
_BLOCK_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("SEC/GPC", ["polydispersity", "pdi", "hmw_lmw", "fraction_i", "fraction_ii", "fraction_iii", "fraction_iv"]),
    ("UV-Vis", ["a254", "a280", "a365", "a465", "a665", "e2_e3", "e4_e6", "s_r", "sr", "s_275", "s_350",
                "d2a", "mw_est", "uv_mw", "absorbance", "suva", "doc"]),
    ("FT-ICR MS", ["mn", "mw", "dbe", "ai", "ai_mod", "nosc", "cho", "chon", "chos", "chons", "kmd"]),
    ("Raman", ["raman", "id_ig", "ad_ag", "la_crystallite", "d_band", "g_band", "carbon_order", "fwhm_d", "fwhm_g",
               "d_pos", "g_pos"]),
    ("IRMS", ["d13c", "d15n", "d34s", "d2h", "d18o", "d_excess", "kraft_lignin_prob", "c_n_atomic"]),
    ("IC", ["so4_cl", "so4", "no3", "cl_mg", "acetate", "formate", "oxalate", "cbe", "charge_balance",
            "ionic_strength", "kraft_effluent", "lmwoa", "mineral_anions", "total_cations"]),
    ("FTIR", ["ftir", "wavenumber", "branching_idx", "lignin_idx", "gs_ratio", "polysaccharide_idx", "phenolic_idx"]),
    ("NMR", ["nmr", "cp_mas", "c_aliph", "c_o_alkyl", "c_aromatic", "c_carboxyl", "c_carbonyl", "fa_aromaticity",
             "hb_hi"]),
    ("Thermal", ["tga", "dtg", "thermostability", "char_yield", "moisture_loss", "aliphatic_loss", "aromatic_loss"]),
    ("Elemental", ["c_pct", "h_pct", "n_pct", "s_pct", "o_pct", "ash_pct", "h_c", "o_c", "n_c", "s_c", "ai_bulk",
                   "q_combustion"]),
    ("EEM-PARAFAC", ["fi", "hix", "bix", "c1", "c2", "c3", "c4", "c5", "c6", "fmax", "fulvic", "lignin", "protein",
                     "humic"]),
]
_BLOCK_KEYWORD_PHRASES: List[Tuple[str, List[Tuple[str, ...]]]] = [
    (b, [_phrase(k) for k in kws]) for b, kws in _BLOCK_KEYWORDS
]

# Точные имена дескрипторов, которые выдаёт extract_fticr_descriptors (средние по формулам FT-ICR MS)
_EXACT_BLOCK_NAMES: Dict[str, str] = {"h/c": "FT-ICR MS", "o/c": "FT-ICR MS"}


def _contains_phrase(tokens: List[str], phrase: Tuple[str, ...]) -> bool:
    n, k = len(tokens), len(phrase)
    if k == 0 or k > n:
        return False
    return any(tuple(tokens[i:i + k]) == phrase for i in range(n - k + 1))


def classify_feature_name(name: str) -> str:
    """Определяет аналитический блок одной колонки (токенно/префиксно, без поиска подстрок)."""
    exact = _EXACT_BLOCK_NAMES.get(str(name).strip().lower())
    if exact:
        return exact
    tokens = _name_tokens(name)
    if not tokens:
        return "Other"
    for prefix, block in _BLOCK_PREFIXES:
        # Префикс должен сопровождаться хотя бы одним токеном (VK_1, FTIR_Lignin_Idx, SEC_Mw)
        if len(tokens) > len(prefix) and tuple(tokens[:len(prefix)]) == prefix:
            return block
    best_block, best_len = "Other", 0
    for block, phrases in _BLOCK_KEYWORD_PHRASES:
        for ph in phrases:
            if len(ph) > best_len and _contains_phrase(tokens, ph):
                best_block, best_len = block, len(ph)
    return best_block


def classify_feature_blocks(columns: List[str]) -> Dict[str, str]:
    """Классифицирует названия колонок по принадлежности к аналитическому блоку.

    Сопоставление выполняется по префиксу модальности (первый токен: VK_, FTIR_, NMR_, SEC_, Elem_ ...),
    а для имён без префикса — по целым токенам ключевых слов (FI, HIX, C1, A254, E2_E3, AI, DBE ...).
    Поиск подстрок не используется: 'Profile' не попадает в EEM из-за 'fi', 'Column' — в FT-ICR из-за 'mn'.
    """
    return {c: classify_feature_name(c) for c in columns}


def extract_fticr_descriptors(assigned_df: pd.DataFrame, sample_id: str, basis: str = "count") -> Dict[str, float]:
    """
    Извлекает вектор дескрипторов FT-ICR MS из идентифицированных формул:
    20 ячеек Ван-Кревелена Перминовой, Mn, Mw, средние индексы и распределение гетероатомов.
    """
    if assigned_df is None or assigned_df.empty or "C" not in assigned_df.columns:
        return {}

    df = assigned_df.copy()
    c_safe = np.where(df["C"] > 0, df["C"], 1e-6)
    if "O/C" not in df.columns and "O" in df.columns:
        df["O/C"] = np.round(df["O"] / c_safe, 4)
    if "H/C" not in df.columns and "H" in df.columns:
        df["H/C"] = np.round(df["H"] / c_safe, 4)
    if "DBE" not in df.columns and "H" in df.columns:
        n_val = df["N"] if "N" in df.columns else 0.0
        df["DBE"] = np.round(1.0 + df["C"] - 0.5 * df["H"] + 0.5 * n_val, 2)
    if "AI" not in df.columns and "O" in df.columns and "H" in df.columns:
        # AI по Koch & Dittmar (2006) с эрратой (2016) — единая реализация fticr_core.calculate_descriptors:
        # AI = (1 + C - O - S - 0.5*(H + N + P)) / (C - O - S - N - P); AI = 0 при числителе или знаменателе <= 0.
        try:
            from modules import fticr_core as _fticr_core
        except ImportError:  # запуск с modules/ в sys.path без пакета modules
            import fticr_core as _fticr_core
        elem_cols = [col for col in ("C", "H", "O", "N", "S", "P") if col in df.columns]
        ai_desc = _fticr_core.calculate_descriptors(df[elem_cols].reset_index(drop=True))
        df["AI"] = np.round(ai_desc["AI"].to_numpy(dtype=float), 4)
    if "NOSC" not in df.columns and "O" in df.columns:
        n_val = df["N"] if "N" in df.columns else 0.0
        s_val = df["S"] if "S" in df.columns else 0.0
        df["NOSC"] = np.round(4.0 - ((4.0 * df["C"] + df["H"] - 3.0 * n_val - 2.0 * df["O"] - 2.0 * s_val) / c_safe), 3)

    sub = df[
        (df["O/C"] >= 0.0) & (df["O/C"] <= 1.0) &
        (df["H/C"] >= 0.2) & (df["H/C"] <= 2.2)
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

    res["Mn"] = round(float(df["mass"].mean()), 2)
    if "intensity" in df.columns and df["intensity"].sum() > 0:
        res["Mw"] = round(float(np.average(df["mass"], weights=df["intensity"])), 2)
    else:
        res["Mw"] = res["Mn"]

    res["H/C"] = round(float(df["H/C"].mean()), 3)
    res["O/C"] = round(float(df["O/C"].mean()), 3)
    res["DBE"] = round(float(df["DBE"].mean()), 2)
    res["AI"] = round(float(df["AI"].mean()), 3)
    if "NOSC" in df.columns:
        res["NOSC"] = round(float(df["NOSC"].mean()), 3)

    if "Hetero_Class" in df.columns:
        counts = df["Hetero_Class"].value_counts(normalize=True) * 100.0
        res["CHO_pct"] = round(float(counts.get("CHO", 0.0)), 2)
        res["CHON_pct"] = round(float(counts.get("CHON", 0.0)), 2)
        res["CHOS_pct"] = round(float(counts.get("CHOS", 0.0)), 2)
        res["CHONS_pct"] = round(float(counts.get("CHONS", 0.0)), 2)

    return res


def merge_feature_blocks(
    df_fticr: Optional[pd.DataFrame] = None,
    df_eem: Optional[pd.DataFrame] = None,
    df_uv: Optional[pd.DataFrame] = None,
    df_ftir: Optional[pd.DataFrame] = None,
    df_nmr: Optional[pd.DataFrame] = None,
    df_thermal: Optional[pd.DataFrame] = None,
    df_sec: Optional[pd.DataFrame] = None,
    df_elem: Optional[pd.DataFrame] = None,
    on_col: str = "Sample_ID",
) -> pd.DataFrame:
    """Универсальное соединение матриц всех 8 аналитических блоков (FT-ICR, EEM, UV-Vis, FTIR, NMR, Thermal, SEC/GPC, CHNS-O) по идентификаторам образцов."""
    dfs = [df for df in [df_fticr, df_eem, df_uv, df_ftir, df_nmr, df_thermal, df_sec, df_elem] if df is not None and not df.empty]
    if not dfs:
        return pd.DataFrame()
    for d in dfs:
        if on_col not in d.columns:
            cols_preview = ", ".join(list(d.columns[:5]))
            raise KeyError(
                f"В таблице отсутствует колонка '{on_col}' (найдены колонки: {cols_preview}). "
                f"Для Модуля 3 требуются таблицы дескрипторов образцов (Sample_ID, признаки), а не сырые спектры."
            )
    if len(dfs) == 1:
        return dfs[0].copy()

    def _merge_pair(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
        if d1.empty:
            return d2.copy()
        if d2.empty:
            return d1.copy()

        # Защита от дублирования метаколонок (Class_Target, Group, Label и т.д.)
        dup_meta = [c for c in d2.columns if c in d1.columns and c != on_col and c.lower() in ["class_target", "class", "target", "group", "label"]]
        d2_use = d2.drop(columns=dup_meta) if dup_meta else d2

        merged = pd.merge(d1, d2_use, on=on_col, how="inner")
        if not merged.empty:
            return merged

        # Слияние с очисткой от расширений файлов и регистра
        c1 = d1.copy()
        c2 = d2_use.copy()
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
    """
    Подготовка и слияние матриц признаков по стратегиям Low-Level или Mid-Level.

    Low-level: медианная импутация -> автошкалирование каждой переменной -> блочное взвешивание
    1/sqrt(P_b) для КАЖДОГО аналитического блока (порядок важен: после автошкалирования вес не
    «съедается» последующей стандартизацией).
    Mid-level: каждый блок с числом признаков > n_mid_components сжимается PCA (скоры PC),
    небольшие блоки передаются как есть (автошкалированные). Ни один блок не отбрасывается.

    Предобработка параметризуется только обучающими данными — см. FusionPreprocessor, который
    используется внутри фолдов кросс-валидации.
    """
    prep = FusionPreprocessor(strategy=strategy, block_scaling=block_scaling, n_mid_components=n_mid_components)
    X_fused = prep.fit_transform(X_df)
    return X_fused, list(prep.feature_names_out_), dict(prep.block_map_)


# Короткие префиксы для имён PCA-скоров mid-level (FTICR_PC1 сохраняется для совместимости)
_MID_LEVEL_PREFIX = {
    "FT-ICR MS": "FTICR", "EEM-PARAFAC": "EEM", "UV-Vis": "UV", "FTIR": "FTIR", "NMR": "NMR",
    "Thermal": "Thermal", "SEC/GPC": "SEC", "Elemental": "Elem", "Raman": "Raman", "IRMS": "IRMS",
    "IC": "IC", "Other": "Other",
}


def _coerce_numeric_df(X_df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    """Приводит матрицу признаков к числовому DataFrame (нечисловые значения -> NaN)."""
    if not isinstance(X_df, pd.DataFrame):
        arr = np.asarray(X_df)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        X_df = pd.DataFrame(arr, columns=[f"X{j + 1}" for j in range(arr.shape[1])])
    return X_df.apply(pd.to_numeric, errors="coerce").astype(float)


class FusionPreprocessor:
    """
    Предобработка мультимодальной матрицы с параметрами, оцененными ТОЛЬКО на обучающей выборке
    (медианы, средние/СКО, PCA-нагрузки, блочные веса). Используется и для финальной модели,
    и внутри каждого фолда кросс-валидации — без утечки информации из тестовых образцов.
    """

    def __init__(self, strategy: str = "low_level", block_scaling: bool = True, n_mid_components: int = 3):
        self.strategy = strategy
        self.block_scaling = block_scaling
        self.n_mid_components = n_mid_components

    def fit(self, X_df: Union[pd.DataFrame, np.ndarray]) -> "FusionPreprocessor":
        X = _coerce_numeric_df(X_df)
        self.columns_ = list(X.columns)
        medians = X.median()
        self.medians_ = medians.fillna(0.0)
        Xf = X.fillna(self.medians_).values.astype(float)
        n_samples = Xf.shape[0]

        self.mean_ = Xf.mean(axis=0)
        std = Xf.std(axis=0, ddof=0)
        std[std < 1e-12] = 1.0  # как StandardScaler: константные колонки -> 0
        self.std_ = std
        Z = (Xf - self.mean_) / self.std_

        in_map = classify_feature_blocks(self.columns_)
        self.input_block_map_ = in_map
        blocks_order: List[str] = []
        for c in self.columns_:
            if in_map[c] not in blocks_order:
                blocks_order.append(in_map[c])

        self.blocks_ = []  # список (block, idx, pca|None, out_names, weight)
        for b in blocks_order:
            idx = np.array([j for j, c in enumerate(self.columns_) if in_map[c] == b], dtype=int)
            p_b = len(idx)
            pca = None
            if self.strategy == "mid_level" and p_b > self.n_mid_components and n_samples >= 3:
                k_comp = int(max(1, min(self.n_mid_components, n_samples - 2, p_b)))
                pca = PCA(n_components=k_comp, random_state=42)
                out = pca.fit_transform(Z[:, idx])
                out_names = [f"{_MID_LEVEL_PREFIX.get(b, 'Block')}_PC{i + 1}" for i in range(k_comp)]
            else:
                out = Z[:, idx]
                out_names = [self.columns_[j] for j in idx]

            weight = 1.0
            if self.block_scaling:
                if pca is None and self.strategy != "mid_level":
                    # Low-level: классическое 1/sqrt(P_b) после автошкалирования
                    weight = 1.0 / np.sqrt(max(1, p_b))
                else:
                    # Mid-level: суммарная дисперсия блока (PC-скоры или автошкалированные колонки) -> 1
                    tot_var = float(np.sum(np.var(out, axis=0)))
                    weight = 1.0 / np.sqrt(tot_var) if tot_var > 1e-12 else 1.0
            self.blocks_.append((b, idx, pca, out_names, weight))

        self.feature_names_out_ = [n for (_, _, _, names, _) in self.blocks_ for n in names]
        self.block_map_ = {n: b for (b, _, _, names, _) in self.blocks_ for n in names}
        self.block_weights_ = {b: w for (b, _, _, _, w) in self.blocks_}
        return self

    def transform(self, X_df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = _coerce_numeric_df(X_df)
        if not isinstance(X_df, pd.DataFrame):
            X.columns = self.columns_[:X.shape[1]]
        X = X.reindex(columns=self.columns_)
        Xf = X.fillna(self.medians_).values.astype(float)
        Z = (Xf - self.mean_) / self.std_
        parts = []
        for (_, idx, pca, _, weight) in self.blocks_:
            out = pca.transform(Z[:, idx]) if pca is not None else Z[:, idx]
            parts.append(out * weight)
        if not parts:
            return np.zeros((Xf.shape[0], 0))
        return np.hstack(parts)

    def fit_transform(self, X_df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.fit(X_df).transform(X_df)


def _cv_splits(n_samples: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    """Схема CV, общая для всех моделей: LOO при N <= 30, иначе 5-fold (shuffle, seed 42)."""
    if n_samples <= 30:
        return list(LeaveOneOut().split(np.zeros(n_samples))), "LOO"
    return list(KFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(n_samples))), "5-fold"


def _fold_matrices(
    X_df: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    strategy: str,
    block_scaling: bool,
    n_mid_components: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Для каждого фолда: предобработка обучается на train и применяется к test (без утечки)."""
    X_df = _coerce_numeric_df(X_df)
    out = []
    for tr, te in splits:
        prep = FusionPreprocessor(strategy=strategy, block_scaling=block_scaling, n_mid_components=n_mid_components)
        X_tr = prep.fit_transform(X_df.iloc[tr])
        X_te = prep.transform(X_df.iloc[te])
        out.append((tr, te, X_tr, X_te))
    return out


def _encode_targets(y_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
    """Кодирование меток: one-hot (мультикласс) или 0/1 (бинарный). Возвращает (Y, classes, is_multi, y_eval)."""
    unique_classes = np.unique(y_arr)
    n_samples = len(y_arr)
    if len(unique_classes) > 2:
        Y_mat = np.zeros((n_samples, len(unique_classes)), dtype=float)
        for idx, c in enumerate(unique_classes):
            Y_mat[y_arr == c, idx] = 1.0
        return Y_mat, unique_classes, True, y_arr
    if set(unique_classes).issubset({0, 1, 0.0, 1.0, True, False, "0", "1"}):
        y_numeric = np.array([1.0 if str(v).strip() in ["1", "1.0", "True"] else 0.0 for v in y_arr])
    else:
        y_numeric = np.array([1.0 if v == unique_classes[1] else 0.0 for v in y_arr])
    return y_numeric, unique_classes, False, (y_numeric >= 0.5).astype(int)


def _class_predictions(y_pred_raw: np.ndarray, unique_classes: np.ndarray, is_multiclass: bool) -> np.ndarray:
    y_pred_raw = np.asarray(y_pred_raw, dtype=float)
    if is_multiclass:
        return unique_classes[np.argmax(y_pred_raw, axis=1)]
    return (y_pred_raw.ravel() >= 0.5).astype(int)


def _classification_metrics(y_true, y_pred, unique_classes, is_multiclass) -> Dict[str, object]:
    labels = unique_classes if is_multiclass else [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)) * 100.0)
    if is_multiclass:
        recalls = []
        for c_i in range(len(labels)):
            c_sum = np.sum(cm[c_i, :])
            recalls.append(float(cm[c_i, c_i] / c_sum) if c_sum > 0 else np.nan)
        bal_acc = float(np.nanmean(recalls) * 100.0) if np.any(~np.isnan(recalls)) else np.nan
        sens = spec = bal_acc
    else:
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / (tp + fn) * 100.0) if (tp + fn) > 0 else np.nan
        spec = float(tn / (tn + fp) * 100.0) if (tn + fp) > 0 else np.nan
        bal_acc = float(np.nanmean([sens, spec])) if not (np.isnan(sens) and np.isnan(spec)) else np.nan
    return {"cm": cm, "acc": acc, "bal_acc": bal_acc, "sens": sens, "spec": spec}


def _q2_from_predictions(Y_true: np.ndarray, Y_cv: np.ndarray) -> float:
    """Q2 = 1 - PRESS / SS_tot (в %); отрицательные значения сохраняются (модель хуже среднего)."""
    Y_true = np.asarray(Y_true, dtype=float)
    Y_cv = np.asarray(Y_cv, dtype=float).reshape(Y_true.shape)
    ss_tot = float(np.sum((Y_true - np.mean(Y_true, axis=0)) ** 2))
    if ss_tot <= 0 or np.any(np.isnan(Y_cv)):
        return float("nan")
    press = float(np.sum((Y_true - Y_cv) ** 2))
    return float((1.0 - press / ss_tot) * 100.0)


def _explained_x_pct(X: np.ndarray, T: np.ndarray, P: np.ndarray) -> float:
    """R2X = sum_a ||t_a p_a^T||^2 / ||X_c||^2 (в %)."""
    Xc = X - np.mean(X, axis=0, keepdims=True)
    ss_x = float(np.sum(Xc ** 2))
    if ss_x <= 1e-300:
        return float("nan")
    T = np.asarray(T, dtype=float).reshape(X.shape[0], -1)
    P = np.asarray(P, dtype=float).reshape(X.shape[1], -1)
    ss_comp = np.sum(T ** 2, axis=0) * np.sum(P ** 2, axis=0)
    return float(np.sum(ss_comp) / ss_x * 100.0)


def _pls_fold_predict(X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray, n_comp: int) -> np.ndarray:
    n_c = int(max(1, min(n_comp, X_tr.shape[0] - 1, X_tr.shape[1])))
    pls_cv = PLSRegression(n_components=n_c, scale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "y residual is constant" в вырожденных фолдах
        pls_cv.fit(X_tr, Y_tr)
    return np.asarray(pls_cv.predict(X_te), dtype=float)


def _plsda_cv_predictions(fold_mats, Y_mat: np.ndarray, n_comp: int) -> np.ndarray:
    y_cv = np.zeros_like(Y_mat, dtype=float)
    for tr, te, X_tr, X_te in fold_mats:
        pred = _pls_fold_predict(X_tr, Y_mat[tr], X_te, n_comp)
        y_cv[te] = pred.reshape(y_cv[te].shape)
    return y_cv


def compute_hotelling_ellipsoid_3d(
    scores_x: np.ndarray, scores_y: np.ndarray, scores_z: np.ndarray,
    n_theta: int = 30, n_phi: int = 30, confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Расчет сетки координат 95% эллипсоида Хотеллинга T^2 в 3D пространстве."""
    n = len(scores_x)
    if n < 5:
        return np.array([[]]), np.array([[]]), np.array([[]])

    s1 = np.std(scores_x, ddof=1)
    s2 = np.std(scores_y, ddof=1)
    s3 = np.std(scores_z, ddof=1)

    f_crit = f.ppf(confidence, 3, n - 3)
    radius_scale = np.sqrt((3 * (n - 1) / (n - 3)) * f_crit)

    r1, r2, r3 = s1 * radius_scale, s2 * radius_scale, s3 * radius_scale

    u = np.linspace(0, 2 * np.pi, n_theta)
    v = np.linspace(0, np.pi, n_phi)
    U, V = np.meshgrid(u, v)

    x_ell = np.mean(scores_x) + r1 * np.cos(U) * np.sin(V)
    y_ell = np.mean(scores_y) + r2 * np.sin(U) * np.sin(V)
    z_ell = np.mean(scores_z) + r3 * np.cos(V)

    return x_ell, y_ell, z_ell


def train_plsda_model(
    X_df: pd.DataFrame,
    y_labels: Union[np.ndarray, pd.Series, List],
    n_components: int = 2,
    fusion_strategy: str = "low_level",
    block_scaling: bool = True,
    n_mid_components: int = 3,
) -> Dict:
    """
    Обучение и валидация PLS-DA модели (бинарная и мультиклассовая классификация).
    Возвращает 2D и 3D проекции Scores, VIP-маркеры, долю блоков и диагностику.

    R2X/R2Y/Accuracy — показатели на обучающей выборке; Q2 и *_CV — по кросс-валидации,
    в которой импутация, шкалирование, блочные веса и PCA (mid-level) обучаются внутри каждого фолда.
    """
    X_df = _coerce_numeric_df(X_df)
    prep = FusionPreprocessor(strategy=fusion_strategy, block_scaling=block_scaling, n_mid_components=n_mid_components)
    X_scaled = prep.fit_transform(X_df)
    feature_names = list(prep.feature_names_out_)
    block_map = dict(prep.block_map_)

    y_arr = np.asarray(y_labels)
    unique_classes = np.unique(y_arr)
    k_classes = len(unique_classes)
    if k_classes < 2:
        raise ValueError("Для классификации требуется как минимум 2 различных класса!")

    n_samples, n_features = X_scaled.shape
    Y_mat, unique_classes, is_multiclass, y_eval_true = _encode_targets(y_arr)

    max_comp = max(1, min(n_components, n_samples - 1, n_features))
    pls = PLSRegression(n_components=max_comp, scale=False)
    pls.fit(X_scaled, Y_mat)

    scores = pls.x_scores_
    t1 = scores[:, 0]
    t2 = scores[:, 1] if max_comp > 1 else np.zeros(n_samples)
    t3 = scores[:, 2] if max_comp > 2 else np.zeros(n_samples)

    # Прогнозы (обучающая выборка)
    y_pred_raw = pls.predict(X_scaled)
    y_class_pred = _class_predictions(y_pred_raw, unique_classes, is_multiclass)
    m_train = _classification_metrics(y_eval_true, y_class_pred, unique_classes, is_multiclass)

    # R2X = sum ||t_a p_a'||^2 / ||X||^2 и R2Y
    r2x = _explained_x_pct(X_scaled, pls.x_scores_, pls.x_loadings_)
    Y_fit = np.asarray(y_pred_raw, dtype=float).reshape(np.shape(Y_mat))
    ss_tot_y = float(np.sum((Y_mat - np.mean(Y_mat, axis=0)) ** 2))
    ss_res_y = float(np.sum((Y_mat - Y_fit) ** 2))
    r2y = float((1.0 - (ss_res_y / ss_tot_y)) * 100.0) if ss_tot_y > 0 else float("nan")

    # Кросс-валидация для Q2 (LOO / 5-Fold) с предобработкой внутри фолдов
    splits, cv_scheme = _cv_splits(n_samples)
    fold_mats = _fold_matrices(X_df, splits, fusion_strategy, block_scaling, n_mid_components)
    y_cv_pred = _plsda_cv_predictions(fold_mats, Y_mat, max_comp)
    q2 = _q2_from_predictions(Y_mat, y_cv_pred)
    m_cv = _classification_metrics(
        y_eval_true, _class_predictions(y_cv_pred, unique_classes, is_multiclass), unique_classes, is_multiclass
    )

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

    # 2D и 3D эллипсы Хотеллинга
    ell_x, ell_y = compute_hotelling_ellipse(t1, t2)
    ell_3d_x, ell_3d_y, ell_3d_z = compute_hotelling_ellipsoid_3d(t1, t2, t3) if max_comp >= 3 else (None, None, None)

    return {
        "model_type": "PLS-DA",
        "scores_t1": t1,
        "t1": t1,
        "scores_t2": t2,
        "t2": t2,
        "scores_t3": t3,
        "t3": t3,
        "ellipse_x": ell_x,
        "ell_x": ell_x,
        "ellipse_y": ell_y,
        "ell_3d_x": ell_3d_x,
        "ell_3d_y": ell_3d_y,
        "ell_3d_z": ell_3d_z,
        "ell_x_3d": ell_3d_x,
        "ell_y_3d": ell_3d_y,
        "ell_z_3d": ell_3d_z,
        "R2X": r2x,
        "R2Y": r2y,
        "Q2": q2,
        "Accuracy": m_train["acc"],
        "Balanced_Accuracy": m_train["bal_acc"],
        "Sensitivity": m_train["sens"],
        "Specificity": m_train["spec"],
        "Confusion_Matrix": m_train["cm"],
        "Accuracy_CV": m_cv["acc"],
        "Balanced_Accuracy_CV": m_cv["bal_acc"],
        "Confusion_Matrix_CV": m_cv["cm"],
        "CV_Scheme": cv_scheme,
        "VIP_df": vip_df,
        "vip_df": vip_df,
        "Block_Contributions": block_contribs,
        "y_pred": y_pred_raw,
        "y_cv_pred": y_cv_pred,
        "classes": unique_classes,
        "is_multiclass": is_multiclass,
        "feature_names": feature_names,
    }


def _fit_opls_core(X: np.ndarray, y: np.ndarray, n_ortho: int) -> Dict[str, object]:
    """
    Ядро OPLS (Trygg & Wold, 2002) для одного y: n_ortho ортогональных компонент + 1 предиктивная.
    Все параметры (средние, веса, нагрузки) оцениваются только на переданных данных.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_samples, n_features = X.shape
    x_mean = X.mean(axis=0)
    y_mean = float(np.mean(y))
    Xc = X - x_mean
    y_cent = y - y_mean

    X_work = Xc.copy()
    T_ortho, P_ortho, W_ortho = [], [], []
    n_ortho_eff = int(max(0, min(n_ortho, n_samples - 2)))
    for _ in range(n_ortho_eff):
        w = X_work.T @ y_cent
        w_norm = np.linalg.norm(w)
        if w_norm <= 1e-12:
            break  # y не коррелирует с X — ортогональная компонента не определена
        w = w / w_norm
        t = X_work @ w
        tt = float(t @ t)
        if tt <= 1e-12:
            break
        p_load = (X_work.T @ t) / tt
        w_o = p_load - float(w @ p_load) * w
        w_o_norm = np.linalg.norm(w_o)
        if w_o_norm <= 1e-12:
            break
        w_o = w_o / w_o_norm
        t_o = X_work @ w_o
        too = float(t_o @ t_o)
        if too <= 1e-12:
            break
        p_o = (X_work.T @ t_o) / too
        X_work = X_work - np.outer(t_o, p_o)
        T_ortho.append(t_o)
        P_ortho.append(p_o)
        W_ortho.append(w_o)

    # Предиктивная компонента
    w_p = X_work.T @ y_cent
    w_p_norm = np.linalg.norm(w_p)
    w_p = w_p / w_p_norm if w_p_norm > 1e-12 else np.zeros(n_features)
    t_p = X_work @ w_p
    tpp = float(t_p @ t_p)
    p_p = (X_work.T @ t_p) / tpp if tpp > 1e-12 else np.zeros(n_features)
    b_p = float(t_p @ y_cent) / tpp if tpp > 1e-12 else 0.0

    return {
        "x_mean": x_mean, "y_mean": y_mean,
        "W_ortho": W_ortho, "P_ortho": P_ortho, "T_ortho": T_ortho,
        "w_p": w_p, "p_p": p_p, "t_p": t_p, "b_p": b_p,
    }


def _predict_opls_core(model: Dict[str, object], X_new: np.ndarray) -> np.ndarray:
    """Прогноз OPLS для новых образцов: удаление ортогональных компонент обученной модели -> t_p -> y."""
    Xc = np.asarray(X_new, dtype=float) - model["x_mean"]
    for w_o, p_o in zip(model["W_ortho"], model["P_ortho"]):
        t_o = Xc @ w_o
        Xc = Xc - np.outer(t_o, p_o)
    t_p = Xc @ model["w_p"]
    return model["y_mean"] + t_p * model["b_p"]


def train_oplsda_model(
    X_df: pd.DataFrame,
    y_labels: Union[np.ndarray, pd.Series, List],
    n_ortho: int = 1,
    fusion_strategy: str = "low_level",
    block_scaling: bool = True,
    n_mid_components: int = 3,
) -> Dict:
    """
    Обучение OPLS-DA модели (Ортогональный PLS-DA) и расчет S-plot (ковариация vs корреляция).
    Разделяет дисперсию на предиктивную (t_pred, строго связанную с Y)
    и ортогональную (t_ortho, систематический шум/дрейф, не связанный с Y).

    Поддерживается только бинарная задача (для > 2 классов — ValueError). n_ortho=0 допустим
    (OPLS с 0 ортогональных компонент эквивалентен 1-компонентному PLS-DA).
    Q2 считается той же OPLS-моделью (1 + n_ortho), переобучаемой в каждом фолде вместе с предобработкой.
    Колонка "VIP" — предиктивный VIP OPLS: VIP_pred_j = sqrt(P) * |w_p,j| / ||w_p||.
    """
    X_df = _coerce_numeric_df(X_df)
    y_raw = np.asarray(y_labels)
    unique_classes = np.unique(y_raw)
    if len(unique_classes) < 2:
        raise ValueError("Для OPLS-DA требуется как минимум 2 класса!")
    if len(unique_classes) > 2:
        raise ValueError(
            f"OPLS-DA реализован только для 2 классов, получено {len(unique_classes)} "
            f"({', '.join(map(str, unique_classes))}). Используйте PLS-DA / sPLS-DA для мультиклассовой задачи "
            f"или объедините классы явно."
        )
    if n_ortho < 0:
        raise ValueError("Число ортогональных компонент n_ortho не может быть отрицательным.")

    prep = FusionPreprocessor(strategy=fusion_strategy, block_scaling=block_scaling, n_mid_components=n_mid_components)
    X_scaled = prep.fit_transform(X_df)
    feature_names = list(prep.feature_names_out_)
    block_map = dict(prep.block_map_)

    y, unique_classes, _, y_true_int = _encode_targets(y_raw)
    n_samples, n_features = X_scaled.shape

    model = _fit_opls_core(X_scaled, y, n_ortho)
    t_p = model["t_p"]
    p_p = model["p_p"]
    w_p = model["w_p"]
    T_ortho = model["T_ortho"]
    P_ortho = model["P_ortho"]
    t_o_main = T_ortho[0] if T_ortho else np.zeros(n_samples)
    n_ortho_used = len(T_ortho)

    # Прогноз Y (обучение) и метрики
    y_pred = model["y_mean"] + t_p * model["b_p"]
    y_class_pred = (y_pred >= 0.5).astype(int)
    m_train = _classification_metrics(y_true_int, y_class_pred, unique_classes, False)

    # Дисперсии: R2X_pred = ||t_p p_p'||^2/||X||^2, R2X_ortho = sum_o ||t_o p_o'||^2/||X||^2
    r2x_pred = _explained_x_pct(X_scaled, t_p, p_p)
    if T_ortho:
        r2x_ortho = _explained_x_pct(X_scaled, np.column_stack(T_ortho), np.column_stack(P_ortho))
    else:
        r2x_ortho = 0.0
    ss_tot_y = float(np.sum((y - np.mean(y)) ** 2))
    ss_res_y = float(np.sum((y - y_pred) ** 2))
    r2y = float((1.0 - ss_res_y / ss_tot_y) * 100.0) if ss_tot_y > 0 else float("nan")

    # Cross-validation для Q2: та же OPLS-модель (1 + n_ortho), предобработка внутри фолда
    splits, cv_scheme = _cv_splits(n_samples)
    fold_mats = _fold_matrices(X_df, splits, fusion_strategy, block_scaling, n_mid_components)
    y_cv = np.zeros(n_samples)
    for tr, te, X_tr, X_te in fold_mats:
        m_fold = _fit_opls_core(X_tr, y[tr], n_ortho)
        y_cv[te] = _predict_opls_core(m_fold, X_te)
    q2 = _q2_from_predictions(y, y_cv)
    m_cv = _classification_metrics(y_true_int, (y_cv >= 0.5).astype(int), unique_classes, False)

    # Расчет S-plot: ковариация p[1] vs корреляция p(corr)[1]
    Xc = X_scaled - X_scaled.mean(axis=0)
    t_c = t_p - t_p.mean()
    denom_n = max(1, n_samples - 1)
    cov_p = (Xc.T @ t_c) / denom_n
    s_x = np.std(X_scaled, axis=0, ddof=1) if n_samples > 1 else np.zeros(n_features)
    s_tp = float(np.std(t_p, ddof=1)) if n_samples > 1 else 0.0
    denom = s_x * s_tp
    corr_p = np.divide(cov_p, denom, out=np.full_like(cov_p, np.nan), where=(denom > 1e-12))

    blocks_col = [block_map.get(col, "Other") for col in feature_names]
    # Предиктивный VIP OPLS (одна предиктивная компонента): sqrt(P) * |w_p| / ||w_p||
    w_p_norm = float(np.linalg.norm(w_p))
    vip_vals = np.sqrt(n_features) * np.abs(w_p) / w_p_norm if w_p_norm > 1e-12 else np.full(n_features, np.nan)

    s_plot_df = pd.DataFrame({
        "Descriptor": feature_names,
        "p1_cov": np.round(cov_p, 4),
        "p_corr": np.round(corr_p, 4),
        "VIP": np.round(vip_vals, 3),
        "Block": blocks_col,
    }).sort_values("VIP", ascending=False).reset_index(drop=True)

    # Вклад блоков (по VIP_pred^2 = доля ||w_p||^2 в блоке)
    vip_sq = s_plot_df.copy()
    vip_sq["VIP_sq"] = vip_sq["VIP"] ** 2
    b_sums = vip_sq.groupby("Block")["VIP_sq"].sum()
    tot_sq = b_sums.sum()
    b_contribs = {}
    if tot_sq > 0:
        for b, s in b_sums.items():
            b_contribs[b] = round(float(s / tot_sq * 100.0), 1)

    ell_x, ell_y = compute_hotelling_ellipse(t_p, t_o_main)

    return {
        "model_type": "OPLS-DA",
        "t_pred": t_p,
        "t_ortho": t_o_main,
        "scores_t1": t_p,
        "scores_t2": t_o_main,
        "t1": t_p,
        "t2": t_o_main,
        "ell_x": ell_x,
        "ell_y": ell_y,
        "ellipse_x": ell_x,
        "ellipse_y": ell_y,
        "R2X": r2x_pred + r2x_ortho,
        "R2X_pred": r2x_pred,
        "R2X_ortho": r2x_ortho,
        "R2Y": r2y,
        "Q2": q2,
        "n_ortho": n_ortho_used,
        "Accuracy": m_train["acc"],
        "Balanced_Accuracy": m_train["bal_acc"],
        "Sensitivity": m_train["sens"],
        "Specificity": m_train["spec"],
        "Confusion_Matrix": m_train["cm"],
        "Accuracy_CV": m_cv["acc"],
        "Balanced_Accuracy_CV": m_cv["bal_acc"],
        "Confusion_Matrix_CV": m_cv["cm"],
        "CV_Scheme": cv_scheme,
        "S_Plot_df": s_plot_df,
        "s_plot_df": s_plot_df,
        "VIP_df": s_plot_df[["Descriptor", "VIP", "Block"]],
        "vip_df": s_plot_df[["Descriptor", "VIP", "Block"]],
        "Block_Contributions": b_contribs,
        "y_pred": y_pred,
        "y_cv_pred": y_cv,
        "classes": unique_classes,
        "is_multiclass": False,
        "feature_names": feature_names,
    }


def _soft_threshold_keepx(w: np.ndarray, keep: int) -> np.ndarray:
    """
    Мягкое пороговое преобразование (L1, как keepX в mixOmics): порог lambda = (keep+1)-е по величине |w|,
    w_j <- sign(w_j) * max(|w_j| - lambda, 0). Ненулевыми остаются не более keep коэффициентов,
    их величины уменьшаются на lambda (а не копируются как при жестком top-k отборе).
    """
    w = np.asarray(w, dtype=float).ravel()
    p = len(w)
    if keep >= p:
        return w.copy()
    abs_w = np.abs(w)
    lam = np.sort(abs_w)[::-1][keep]
    return np.sign(w) * np.maximum(abs_w - lam, 0.0)


def _fit_spls_core(X_in: np.ndarray, Y_in: np.ndarray, n_c: int, k_keep: int,
                   max_iter: int = 500, tol: float = 1e-10):
    """
    sPLS (Lê Cao et al., 2008; режим regression): для каждой компоненты чередуются
    u = soft(M v, keepX)/||.||,  v = M^T u/||.||, M = X^T Y (с итерациями до сходимости при многомерном Y).
    Возвращает W, P, Q, T и B = W (P^T W)^+ Q^T.
    """
    X_curr = np.array(X_in, dtype=float, copy=True)
    Y_curr = np.array(Y_in, dtype=float, copy=True)
    n_p = X_in.shape[1]
    W = np.zeros((n_p, n_c), dtype=float)
    P = np.zeros((n_p, n_c), dtype=float)
    Q = np.zeros((Y_in.shape[1], n_c), dtype=float)
    T = np.zeros((X_in.shape[0], n_c), dtype=float)

    for h in range(n_c):
        M = X_curr.T @ Y_curr  # P x K
        if np.linalg.norm(M) < 1e-12:
            break
        if M.shape[1] == 1:
            v = np.ones(1)
        else:
            _, _, vt = np.linalg.svd(M, full_matrices=False)
            v = vt[0]
        u = None
        for _ in range(max_iter):
            u_new = _soft_threshold_keepx(M @ v, k_keep)
            nu = np.linalg.norm(u_new)
            if nu < 1e-12:
                u = None
                break
            u_new = u_new / nu
            if M.shape[1] == 1:
                u = u_new
                break
            v_new = M.T @ u_new
            nv = np.linalg.norm(v_new)
            if nv < 1e-12:
                u = u_new
                break
            v = v_new / nv
            if u is not None and np.linalg.norm(u_new - u) < tol:
                u = u_new
                break
            u = u_new
        if u is None:
            break

        t_h = X_curr @ u
        denom_t = float(t_h @ t_h)
        if denom_t < 1e-12:
            break
        p_h = (X_curr.T @ t_h) / denom_t
        q_h = (Y_curr.T @ t_h) / denom_t

        # Дефляция
        X_curr = X_curr - np.outer(t_h, p_h)
        Y_curr = Y_curr - np.outer(t_h, q_h)

        W[:, h] = u
        P[:, h] = p_h
        Q[:, h] = q_h
        T[:, h] = t_h

    # Матрица коэффициентов регрессии B = W (P^T W)^(-1) Q^T
    B_coeff = W @ np.linalg.pinv(P.T @ W) @ Q.T
    return W, P, Q, T, B_coeff


def train_splsda_model(
    X_df: pd.DataFrame,
    y_labels: Union[np.ndarray, pd.Series, List],
    n_components: int = 2,
    keep_x: int = 15,
    fusion_strategy: str = "low_level",
    block_scaling: bool = True,
    n_mid_components: int = 3,
) -> Dict:
    """
    Разреженный дискриминантный анализ (Sparse PLS-DA / sPLS-DA).
    Вектор весов каждой компоненты подвергается мягкому пороговому преобразованию (L1, soft-thresholding):
    порог = (keep_x+1)-е по величине |w|, из |w| вычитается порог, затем вектор перенормируется —
    шум обнуляется, остаётся компактная панель ключевых биомаркеров.

    Parameters:
    -----------
    X_df : pd.DataFrame
        Мультимодальная матрица дескрипторов (FT-ICR MS, EEM-PARAFAC, UV-Vis).
    y_labels : Union[np.ndarray, pd.Series, List]
        Вектор меток классов.
    n_components : int
        Число латентных компонент (по умолчанию 2).
    keep_x : int
        Число отбираемых дескрипторов на компоненту (параметр разреженности).
    fusion_strategy : str
        Стратегия слияния ('low_level' или 'mid_level').
    block_scaling : bool
        Флаг блочного масштабирования 1/sqrt(P_k).
    n_mid_components : int
        Число компонент PCA для сжатия крупных блоков в mid-level стратегии.

    Returns:
    --------
    Dict с результатами: проекции scores, sparse loadings, отобранные биомаркеры, R2X, R2Y, Q2 и точность.
    Колонка "VIP" — VIP по разреженным весам: sqrt(P * sum_a SSY_a w_ja^2 / sum_a SSY_a).
    """
    X_df = _coerce_numeric_df(X_df)
    prep = FusionPreprocessor(strategy=fusion_strategy, block_scaling=block_scaling, n_mid_components=n_mid_components)
    X_scaled = prep.fit_transform(X_df)
    feature_names = list(prep.feature_names_out_)
    block_map = dict(prep.block_map_)

    y_arr = np.asarray(y_labels)
    unique_classes = np.unique(y_arr)
    k_classes = len(unique_classes)
    if k_classes < 2:
        raise ValueError("Для классификации требуется как минимум 2 различных класса!")

    n_samples, n_features = X_scaled.shape
    Y_enc, unique_classes, is_multiclass, y_eval_true = _encode_targets(y_arr)
    Y_mat = Y_enc if is_multiclass else Y_enc.reshape(-1, 1)

    max_comp = max(1, min(n_components, n_samples - 1, n_features))
    k_features_to_keep = max(1, min(keep_x, n_features))

    Y_mean = np.mean(Y_mat, axis=0)
    X_mean = np.mean(X_scaled, axis=0)
    Y_centered = Y_mat - Y_mean

    W_mat, P_mat, Q_mat, T_mat, B_mat = _fit_spls_core(X_scaled - X_mean, Y_centered, max_comp, k_features_to_keep)

    t1 = T_mat[:, 0]
    t2 = T_mat[:, 1] if max_comp > 1 else np.zeros(n_samples)
    t3 = T_mat[:, 2] if max_comp > 2 else np.zeros(n_samples)

    # Прогнозы модели (с учетом интерцепта)
    y_pred_raw = Y_mean + (X_scaled - X_mean) @ B_mat
    y_class_pred = _class_predictions(y_pred_raw, unique_classes, is_multiclass)
    m_train = _classification_metrics(y_eval_true, y_class_pred, unique_classes, is_multiclass)

    # Дисперсия R2X и R2Y
    r2x = _explained_x_pct(X_scaled, T_mat, P_mat)
    ss_tot_y = float(np.sum((Y_mat - np.mean(Y_mat, axis=0)) ** 2))
    ss_res_y = float(np.sum((Y_mat - y_pred_raw) ** 2))
    r2y = float((1.0 - (ss_res_y / ss_tot_y)) * 100.0) if ss_tot_y > 0 else float("nan")

    # Кросс-валидация Q2 (предобработка и sPLS переобучаются в каждом фолде)
    splits, cv_scheme = _cv_splits(n_samples)
    fold_mats = _fold_matrices(X_df, splits, fusion_strategy, block_scaling, n_mid_components)
    y_cv_pred = np.zeros_like(Y_mat)
    for train_idx, test_idx, X_tr, X_te in fold_mats:
        y_tr_m = np.mean(Y_mat[train_idx], axis=0)
        x_tr_m = np.mean(X_tr, axis=0)
        n_c_fold = max(1, min(max_comp, len(train_idx) - 1, X_tr.shape[1]))
        k_fold = max(1, min(k_features_to_keep, X_tr.shape[1]))
        _, _, _, _, b_cv = _fit_spls_core(X_tr - x_tr_m, Y_mat[train_idx] - y_tr_m, n_c_fold, k_fold)
        y_cv_pred[test_idx] = y_tr_m + (X_te - x_tr_m) @ b_cv
    q2 = _q2_from_predictions(Y_mat, y_cv_pred)
    m_cv = _classification_metrics(
        y_eval_true, _class_predictions(y_cv_pred, unique_classes, is_multiclass), unique_classes, is_multiclass
    )

    # Отобранные разреженные признаки
    abs_weights = np.sqrt(np.sum(W_mat ** 2, axis=1))
    blocks_col = [block_map.get(col, "Other") for col in feature_names]
    selected_mask = (abs_weights > 1e-12)

    # VIP по разреженным весам (веса компонент нормированы, SSY_a = ||t_a||^2 ||q_a||^2)
    ssy = np.sum(T_mat ** 2, axis=0) * np.sum(Q_mat ** 2, axis=0)
    if float(np.sum(ssy)) > 1e-12:
        w_norms = np.linalg.norm(W_mat, axis=0)
        W_n = np.divide(W_mat, w_norms, out=np.zeros_like(W_mat), where=(w_norms > 1e-12))
        vip_vals = np.sqrt(n_features * (W_n ** 2 @ ssy) / float(np.sum(ssy)))
    else:
        vip_vals = np.full(n_features, np.nan)

    sparse_df = pd.DataFrame({
        "Descriptor": feature_names,
        "Weight_Comp1": np.round(W_mat[:, 0], 4),
        "Weight_Comp2": np.round(W_mat[:, 1] if max_comp > 1 else np.zeros(n_features), 4),
        "Absolute_Weight": np.round(abs_weights, 4),
        "Block": blocks_col,
        "Is_Selected": selected_mask,
        "VIP": np.round(vip_vals, 3),
    }).sort_values("Absolute_Weight", ascending=False).reset_index(drop=True)

    selected_features = sparse_df[sparse_df["Is_Selected"]]["Descriptor"].tolist()

    # Вклад блоков среди отобранных признаков
    sel_df = sparse_df[sparse_df["Is_Selected"]].copy()
    b_contribs = {}
    if not sel_df.empty:
        sel_df["W_sq"] = sel_df["Absolute_Weight"] ** 2
        b_sums = sel_df.groupby("Block")["W_sq"].sum()
        tot_sq = b_sums.sum()
        if tot_sq > 0:
            for b, s in b_sums.items():
                b_contribs[b] = round(float(s / tot_sq * 100.0), 1)

    ell_x, ell_y = compute_hotelling_ellipse(t1, t2)
    ell_3d_x, ell_3d_y, ell_3d_z = compute_hotelling_ellipsoid_3d(t1, t2, t3) if max_comp >= 3 else (None, None, None)

    vip_sorted = sparse_df[["Descriptor", "VIP", "Block"]].sort_values("VIP", ascending=False).reset_index(drop=True)

    return {
        "model_type": "Sparse PLS-DA (sPLS-DA)",
        "scores_t1": t1,
        "t1": t1,
        "scores_t2": t2,
        "t2": t2,
        "scores_t3": t3,
        "t3": t3,
        "ellipse_x": ell_x,
        "ell_x": ell_x,
        "ellipse_y": ell_y,
        "ell_y": ell_y,
        "ell_3d_x": ell_3d_x,
        "ell_3d_y": ell_3d_y,
        "ell_3d_z": ell_3d_z,
        "ell_x_3d": ell_3d_x,
        "ell_y_3d": ell_3d_y,
        "ell_z_3d": ell_3d_z,
        "R2X": r2x,
        "R2Y": r2y,
        "Q2": q2,
        "Accuracy": m_train["acc"],
        "Balanced_Accuracy": m_train["bal_acc"],
        "Sensitivity": m_train["sens"],
        "Specificity": m_train["spec"],
        "Confusion_Matrix": m_train["cm"],
        "Accuracy_CV": m_cv["acc"],
        "Balanced_Accuracy_CV": m_cv["bal_acc"],
        "Confusion_Matrix_CV": m_cv["cm"],
        "CV_Scheme": cv_scheme,
        "VIP_df": vip_sorted,
        "vip_df": vip_sorted,
        "Sparse_Loadings_df": sparse_df,
        "sparse_df": sparse_df,
        "Selected_Features": selected_features,
        "n_selected": len(selected_features),
        "Block_Contributions": b_contribs,
        "y_pred": y_pred_raw,
        "y_cv_pred": y_cv_pred,
        "classes": unique_classes,
        "is_multiclass": is_multiclass,
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
    """
    Пермутационный тест PLS-DA модели для подтверждения статистической значимости Q2.

    Используется та же схема CV и тот же конвейер, что и в train_plsda_model: импутация,
    шкалирование, блочные веса и PCA обучаются на обучающей части каждого фолда. Поскольку
    предобработка не зависит от y, матрицы фолдов вычисляются один раз и переиспользуются
    для всех перестановок (результат идентичен переобучению конвейера на каждой перестановке).
    """
    rng = np.random.RandomState(random_state)
    X_df = _coerce_numeric_df(X_df)
    y_arr = np.asarray(y_labels)
    unique_classes = np.unique(y_arr)
    k_classes = len(unique_classes)
    if k_classes < 2:
        raise ValueError("Для пермутационного теста требуется как минимум 2 класса!")

    n_samples = X_df.shape[0]
    Y_mat, unique_classes, is_multiclass, _ = _encode_targets(y_arr)

    splits, cv_scheme = _cv_splits(n_samples)
    fold_mats = _fold_matrices(X_df, splits, fusion_strategy, block_scaling, n_mid_components)
    n_features_full = FusionPreprocessor(
        strategy=fusion_strategy, block_scaling=block_scaling, n_mid_components=n_mid_components
    ).fit_transform(X_df).shape[1]
    max_comp = max(1, min(n_components, n_samples - 1, n_features_full))

    y_cv_orig = _plsda_cv_predictions(fold_mats, Y_mat, max_comp)
    q2_orig = _q2_from_predictions(Y_mat, y_cv_orig)

    perm_q2 = []
    for _ in range(n_permutations):
        perm_idx = rng.permutation(n_samples)
        Y_perm = Y_mat[perm_idx]
        y_cv_p = _plsda_cv_predictions(fold_mats, Y_perm, max_comp)
        perm_q2.append(_q2_from_predictions(Y_perm, y_cv_p))

    perm_q2_arr = np.array(perm_q2, dtype=float)
    valid = ~np.isnan(perm_q2_arr)
    if np.isnan(q2_orig) or not np.any(valid):
        p_val_q2 = float("nan")
    else:
        p_val_q2 = float((np.sum(perm_q2_arr[valid] >= q2_orig) + 1.0) / (np.sum(valid) + 1.0))

    return {
        "q2_orig": q2_orig,
        "perm_q2": perm_q2_arr,
        "p_val_q2": p_val_q2,
        "CV_Scheme": cv_scheme,
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

        # 4. Признаки колебательной спектроскопии (FTIR)
        row["FTIR_Aromaticity_Idx"] = round(np.random.uniform(2.8, 4.5) if is_lignin else np.random.uniform(0.6, 1.4), 3)
        row["FTIR_Oxidation_Idx"] = round(np.random.uniform(0.85, 1.35) if is_lignin else np.random.uniform(0.35, 0.65), 3)
        row["FTIR_Lignin_Idx"] = round(np.random.uniform(0.75, 1.05) if is_lignin else np.random.uniform(0.12, 0.28), 3)
        row["FTIR_GS_Ratio"] = round(np.random.uniform(1.8, 3.2) if is_lignin else np.random.uniform(0.8, 1.4), 3)
        row["FTIR_Polysaccharide_Idx"] = round(np.random.uniform(0.35, 0.65) if is_lignin else np.random.uniform(1.2, 2.2), 3)

        records.append(row)

    df = pd.DataFrame(records)
    return df, pd.Series(y_labels, name="Class_Target")