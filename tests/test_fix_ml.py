"""
tests/test_fix_ml.py — Регрессионные тесты исправлений хемометрического ядра:
chemo_ml (классификация блоков, блочное шкалирование, mid-level, утечка в CV, R2X, OPLS-DA, sPLS-DA),
multiblock_chemometrics (MB-PLS / SO-PLS), cross_correlation (ddof, BH-FDR, 2D-COS Нода)
и views/chemo_ml_view (метрики MB/SO-PLS из модели, ограничения для elucidation).

Каждый тест сверяет результат с независимой эталонной реализацией (sklearn / scipy / формула),
а не с «правдоподобной» константой.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modules import chemo_ml, cross_correlation, multiblock_chemometrics


# ---------------------------------------------------------------------------
# Синтетические данные
# ---------------------------------------------------------------------------
_BLOCK_COLS = {
    "FT-ICR MS": [f"VK_{k}" for k in range(1, 9)],
    "EEM-PARAFAC": ["EEM_C1", "EEM_C2", "EEM_C3"],
    "FTIR": ["FTIR_1720", "FTIR_1600", "FTIR_1510", "FTIR_1030"],
    "NMR": ["NMR_Aromatic", "NMR_Aliphatic", "NMR_O_Alkyl", "NMR_Carboxyl"],
    "SEC/GPC": ["SEC_Mw", "SEC_Mn"],
}


def _fusion_df(n=16, seed=0, nan_frac=0.0):
    rng = np.random.RandomState(seed)
    y = np.array([0, 1] * (n // 2))
    cols = [c for cs in _BLOCK_COLS.values() for c in cs]
    X = rng.randn(n, len(cols)) * rng.uniform(0.5, 20.0, size=len(cols)) + rng.uniform(-5, 50, size=len(cols))
    X[y == 1, :6] += 4.0   # FT-ICR сигнал
    X[y == 1, 11:13] += 3.0  # FTIR сигнал
    if nan_frac > 0:
        mask = rng.rand(*X.shape) < nan_frac
        mask[:, 0] = False
        X[mask] = np.nan
    return pd.DataFrame(X, columns=cols), y


def _q2_pct(Y, Y_cv):
    Y = np.asarray(Y, dtype=float)
    Y_cv = np.asarray(Y_cv, dtype=float).reshape(Y.shape)
    return 100.0 * (1.0 - np.sum((Y - Y_cv) ** 2) / np.sum((Y - Y.mean(axis=0)) ** 2))


# ---------------------------------------------------------------------------
# chemo_ml: классификация признаков по блокам (без подстрочных коллизий)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("formula, ai_expected", [
    ({"C": 10, "H": 11, "N": 1, "O": 3}, 1.0 / 3.0),  # (1+10-3-0.5*(11+1)) / (10-3-1) = 2/6
    ({"C": 60, "H": 0, "N": 0, "O": 0}, 61.0 / 60.0),  # без искусственной обрезки до 1
    ({"C": 6, "H": 14, "N": 0, "O": 6}, 0.0),          # числитель/знаменатель <= 0 -> AI = 0
])
def test_fticr_descriptor_ai_follows_koch_dittmar_with_erratum(formula, ai_expected):
    assigned = pd.DataFrame([{**formula, "mass": 200.0, "intensity": 1.0}])
    desc = chemo_ml.extract_fticr_descriptors(assigned, "S1")
    assert desc["AI"] == pytest.approx(round(ai_expected, 3), abs=1e-9)


@pytest.mark.parametrize("name, expected", [
    ("Elem_NOSC", "Elemental"),
    ("Elem_AI_bulk", "Elemental"),
    ("NMR_C_acetal_pct", "NMR"),
    ("FTIR_GS_Ratio", "FTIR"),
    ("DOC", "UV-Vis"),                     # DOC приходит из UV-блока вместе с SUVA254
    ("lmwoa_fraction", "IC"),
    ("Raman_Mw", "Raman"),
    ("Profile_Width", "Other"),
    ("Main_Peak_Height", "Other"),
    ("Column_Temp", "Other"),
    ("SSR_Index", "Other"),
    ("PC1_Score", "Other"),
])
def test_block_classification_has_no_substring_collisions(name, expected):
    assert chemo_ml.classify_feature_blocks([name])[name] == expected


# ---------------------------------------------------------------------------
# chemo_ml: блочное шкалирование и mid-level слияние
# ---------------------------------------------------------------------------
def test_low_level_block_scaling_survives_standardisation_for_every_block():
    df, _ = _fusion_df()
    X_bs, names, bmap = chemo_ml.prepare_fused_features(df, "low_level", block_scaling=True)
    X_raw, _, _ = chemo_ml.prepare_fused_features(df, "low_level", block_scaling=False)
    assert not np.allclose(X_bs, X_raw)
    for block, cols in _BLOCK_COLS.items():
        idx = [names.index(c) for c in cols]
        assert all(bmap[c] == block for c in cols)
        # 1/sqrt(P_b) после автошкалирования -> суммарная дисперсия каждого блока = 1
        assert np.isclose(np.sum(np.var(X_bs[:, idx], axis=0)), 1.0, atol=1e-10), block


def test_mid_level_fusion_keeps_every_block():
    df, _ = _fusion_df()
    X, names, bmap = chemo_ml.prepare_fused_features(df, "mid_level", block_scaling=True, n_mid_components=3)
    assert set(bmap.values()) == set(_BLOCK_COLS)
    assert X.shape[1] == len(names)
    assert {"FTIR_PC1", "NMR_PC1", "FTICR_PC1"} <= set(names)  # блоки > 3 признаков сжаты PCA
    assert {"SEC_Mw", "SEC_Mn"} <= set(names)  # малые блоки сохранены как есть


@pytest.mark.parametrize("strategy", ["low_level", "mid_level"])
def test_fusion_and_models_tolerate_all_nan_descriptor_columns(strategy):
    # Напр. FTIR_GS_Ratio = NaN, когда полосы отсутствуют: колонка не должна ломать слияние/CV
    df, y = _fusion_df(n=16, seed=9, nan_frac=0.1)
    df["FTIR_GS_Ratio"] = np.nan
    X, names, _ = chemo_ml.prepare_fused_features(df, strategy, block_scaling=True)
    assert np.all(np.isfinite(X))
    for res in (
        chemo_ml.train_plsda_model(df, y, n_components=2, fusion_strategy=strategy),
        chemo_ml.train_oplsda_model(df, y, n_ortho=1, fusion_strategy=strategy),
        chemo_ml.train_splsda_model(df, y, n_components=2, keep_x=5, fusion_strategy=strategy),
    ):
        assert np.isfinite(res["Q2"]) and np.isfinite(res["R2Y"])
    view_res = _view()._fit_multiblock_model("mbpls", df, y, 2, True)
    assert np.isfinite(view_res["Q2"])


# ---------------------------------------------------------------------------
# chemo_ml: PLS-DA — без утечки предобработки в CV, корректный R2X
# ---------------------------------------------------------------------------
def _ref_infold_plsda_q2(df, y, n_comp):
    X = df.values.astype(float)
    y = y.astype(float)
    y_cv = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             PLSRegression(n_components=n_comp, scale=False))
        pipe.fit(X[tr], y[tr])
        y_cv[te] = pipe.predict(X[te]).ravel()
    return _q2_pct(y, y_cv)


def test_plsda_q2_has_no_preprocessing_leakage():
    df, y = _fusion_df(n=16, seed=3, nan_frac=0.15)
    ref = _ref_infold_plsda_q2(df, y, 2)
    res = chemo_ml.train_plsda_model(df, y, n_components=2, block_scaling=False)
    assert np.isclose(res["Q2"], ref, atol=1e-6)


def test_permutation_test_q2_orig_has_no_preprocessing_leakage():
    df, y = _fusion_df(n=16, seed=3, nan_frac=0.15)
    ref = _ref_infold_plsda_q2(df, y, 2)
    perm = chemo_ml.run_permutation_test(df, y, n_components=2, n_permutations=3, block_scaling=False)
    assert np.isclose(perm["q2_orig"], ref, atol=1e-6)


def test_plsda_r2x_is_explained_sum_of_squares():
    df, y = _fusion_df(n=16, seed=5)
    X = StandardScaler().fit_transform(df.values)
    pls = PLSRegression(n_components=2, scale=False).fit(X, y.astype(float))
    ss_comp = np.sum(pls.x_scores_ ** 2, axis=0) * np.sum(pls.x_loadings_ ** 2, axis=0)
    r2x_ref = 100.0 * np.sum(ss_comp) / np.sum(X ** 2)
    res = chemo_ml.train_plsda_model(df, y, n_components=2, block_scaling=False)
    assert np.isclose(res["R2X"], r2x_ref, atol=1e-6)


# ---------------------------------------------------------------------------
# chemo_ml: OPLS-DA
# ---------------------------------------------------------------------------
def _ref_opls_fit(X, y, n_ortho):
    """Эталонный OPLS (Trygg & Wold, 2002) для одного y."""
    xm, ym = X.mean(axis=0), float(y.mean())
    E, yc = X - xm, y - ym
    Wo, Po = [], []
    for _ in range(n_ortho):
        w = E.T @ yc
        w /= np.linalg.norm(w)
        t = E @ w
        p = E.T @ t / (t @ t)
        wo = p - (w @ p) * w
        wo /= np.linalg.norm(wo)
        to = E @ wo
        po = E.T @ to / (to @ to)
        E = E - np.outer(to, po)
        Wo.append(wo)
        Po.append(po)
    w = E.T @ yc
    w /= np.linalg.norm(w)
    t = E @ w
    return {"xm": xm, "ym": ym, "Wo": Wo, "Po": Po, "w": w, "b": (t @ yc) / (t @ t)}


def _ref_opls_predict(m, X_new):
    E = X_new - m["xm"]
    for wo, po in zip(m["Wo"], m["Po"]):
        E = E - np.outer(E @ wo, po)
    return m["ym"] + (E @ m["w"]) * m["b"]


def test_oplsda_rejects_more_than_two_classes():
    df, _ = _fusion_df(n=18)
    y3 = np.array([0, 1, 2] * 6)
    with pytest.raises(ValueError):
        chemo_ml.train_oplsda_model(df, y3, n_ortho=1)


def test_oplsda_respects_zero_orthogonal_components():
    df, y = _fusion_df(n=16, seed=7)
    res = chemo_ml.train_oplsda_model(df, y, n_ortho=0, block_scaling=False)
    assert res["R2X_ortho"] == 0.0
    assert np.allclose(res["t_ortho"], 0.0)
    # OPLS с 0 ортогональных компонент = 1-компонентный PLS
    X = StandardScaler().fit_transform(df.values)
    t1 = PLSRegression(n_components=1, scale=False).fit(X, y.astype(float)).x_scores_[:, 0]
    assert abs(np.corrcoef(res["t_pred"], t1)[0, 1]) > 1 - 1e-8


def test_oplsda_q2_is_computed_with_the_same_opls_model():
    df, y = _fusion_df(n=16, seed=11)
    X = df.values.astype(float)
    yf = y.astype(float)
    y_cv = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler().fit(X[tr])
        m = _ref_opls_fit(sc.transform(X[tr]), yf[tr], 1)
        y_cv[te] = _ref_opls_predict(m, sc.transform(X[te]))
    res = chemo_ml.train_oplsda_model(df, y, n_ortho=1, block_scaling=False)
    assert np.isclose(res["Q2"], _q2_pct(yf, y_cv), atol=1e-6)


def test_oplsda_vip_is_predictive_vip():
    df, y = _fusion_df(n=16, seed=11)
    X = StandardScaler().fit_transform(df.values)
    m = _ref_opls_fit(X, y.astype(float), 1)
    vip_ref = pd.Series(np.sqrt(X.shape[1]) * np.abs(m["w"]), index=df.columns)
    res = chemo_ml.train_oplsda_model(df, y, n_ortho=1, block_scaling=False)
    got = res["VIP_df"].set_index("Descriptor")["VIP"].reindex(df.columns)
    assert np.allclose(got.values, vip_ref.values, atol=1e-3)


# ---------------------------------------------------------------------------
# chemo_ml: sPLS-DA
# ---------------------------------------------------------------------------
def test_splsda_uses_soft_thresholding():
    df, y = _fusion_df(n=16, seed=13)
    keep = 5
    X = StandardScaler().fit_transform(df.values)
    w = X.T @ (y - y.mean())
    lam = np.sort(np.abs(w))[::-1][keep]
    w_ref = np.sign(w) * np.maximum(np.abs(w) - lam, 0.0)
    w_ref /= np.linalg.norm(w_ref)

    res = chemo_ml.train_splsda_model(df, y, n_components=2, keep_x=keep, block_scaling=False)
    got = res["Sparse_Loadings_df"].set_index("Descriptor")["Weight_Comp1"].reindex(df.columns).values
    sign = np.sign(got @ w_ref)
    assert np.count_nonzero(w_ref) == keep
    assert np.allclose(sign * got, w_ref, atol=1e-4)


def test_splsda_vip_is_derived_from_sparse_weights():
    df, y = _fusion_df(n=16, seed=13)
    res = chemo_ml.train_splsda_model(df, y, n_components=2, keep_x=5, block_scaling=False)
    sp = res["Sparse_Loadings_df"]
    vip = res["VIP_df"].set_index("Descriptor")["VIP"]
    # sum VIP^2 = P; признаки с нулевыми весами во всех компонентах имеют VIP = 0
    assert np.isclose(np.sum(vip.values ** 2), len(df.columns), rtol=1e-2)
    unselected = sp.loc[sp["Absolute_Weight"] == 0, "Descriptor"]
    assert len(unselected) > 0
    assert np.allclose(vip.reindex(unselected).values, 0.0)


# ---------------------------------------------------------------------------
# multiblock_chemometrics: MB-PLS
# ---------------------------------------------------------------------------
def _mb_data(n=20, seed=21):
    rng = np.random.RandomState(seed)
    y = np.array(["A", "B"] * (n // 2))
    X1 = rng.randn(n, 6) * 3.0 + 10.0
    X2 = rng.randn(n, 4) * 0.2
    X1[y == "B", :3] += 2.5
    X2[y == "B", 0] += 0.3
    return {"MS": X1, "NMR": X2}, y


def _frobenius_blocks(blocks, ref_blocks=None):
    ref_blocks = ref_blocks or blocks
    out = []
    for b in blocks:
        mean = ref_blocks[b].mean(axis=0)
        fro = np.linalg.norm(ref_blocks[b] - mean, "fro")
        out.append((blocks[b] - mean) / fro)
    return np.hstack(out)


def _dummy(y, classes):
    return np.column_stack([(y == c).astype(float) for c in classes])


def test_mbpls_equals_pls_on_concatenated_scaled_blocks():
    blocks, y = _mb_data()
    model = multiblock_chemometrics.MultiBlockPLSDA(n_components=2, scale_blocks=True).fit(blocks, y)
    X_cat = _frobenius_blocks(blocks)
    Y = _dummy(y, np.unique(y))
    pls = PLSRegression(n_components=2, scale=False).fit(X_cat, Y)
    for a in range(2):
        assert abs(np.corrcoef(model.super_scores_[:, a], pls.x_scores_[:, a])[0, 1]) > 1 - 1e-8
    # Прогноз через W* (а не W) и без обрезки/перенормировки «вероятностей»
    assert np.allclose(model.predict_proba(blocks), pls.predict(X_cat), atol=1e-8)


def test_mbpls_predicts_new_samples_with_training_preprocessing():
    blocks, y = _mb_data()
    model = multiblock_chemometrics.MultiBlockPLSDA(n_components=2, scale_blocks=True).fit(blocks, y)
    pls = PLSRegression(n_components=2, scale=False).fit(_frobenius_blocks(blocks), _dummy(y, np.unique(y)))
    new = {"MS": blocks["MS"][[0, 5]] * 3.0, "NMR": blocks["NMR"][[0, 5]] * 3.0}
    expected = pls.predict(_frobenius_blocks(new, ref_blocks=blocks))
    pred_pair = model.predict_proba(new)
    pred_single = model.predict_proba({b: v[[0]] for b, v in new.items()})
    assert np.allclose(pred_pair, expected, atol=1e-8)
    assert np.allclose(pred_single[0], pred_pair[0], atol=1e-10)  # не зависит от других образцов
    assert np.any((pred_pair < 0) | (pred_pair > 1))  # регрессионный выход не обрезается до [0, 1]


def test_mbpls_q2_uses_training_fold_preprocessing():
    blocks, y = _mb_data()
    Y = _dummy(y, np.unique(y))
    y_cv = np.zeros_like(Y)
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=42).split(Y):
        tr_blocks = {b: v[tr] for b, v in blocks.items()}
        te_blocks = {b: v[te] for b, v in blocks.items()}
        pls = PLSRegression(n_components=2, scale=False).fit(_frobenius_blocks(tr_blocks), Y[tr])
        y_cv[te] = pls.predict(_frobenius_blocks(te_blocks, ref_blocks=tr_blocks))
    model = multiblock_chemometrics.MultiBlockPLSDA(n_components=2, scale_blocks=True).fit(blocks, y)
    assert np.isclose(100.0 * model.q2_, _q2_pct(Y, y_cv), atol=1e-6)


def test_mbpls_negative_q2_is_not_clipped():
    rng = np.random.RandomState(4)
    n = 20
    blocks = {"MS": rng.randn(n, 15), "NMR": rng.randn(n, 10)}
    y = np.array(["A", "B"] * (n // 2))
    model = multiblock_chemometrics.MultiBlockPLSDA(n_components=3, scale_blocks=True).fit(blocks, y)
    assert model.q2_ < 0.0


def test_mbpls_small_n_q2_is_nan_with_warning():
    blocks, y = _mb_data(n=4)
    with pytest.warns(RuntimeWarning, match="Q2"):
        model = multiblock_chemometrics.MultiBlockPLSDA(n_components=2).fit(blocks, y)
    assert np.isnan(model.q2_)
    assert model.r2y_ > 0  # обучение есть, но Q2 не подменяется на r2y * const
    assert any("Q2" in w for w in model.warnings_)


def test_mbpls_vip_weights_components_by_explained_y():
    blocks, y = _mb_data()
    model = multiblock_chemometrics.MultiBlockPLSDA(n_components=2, scale_blocks=True).fit(blocks, y)
    X_cat = _frobenius_blocks(blocks)
    pls = PLSRegression(n_components=2, scale=False).fit(X_cat, _dummy(y, np.unique(y)))
    ssy = np.sum(pls.x_scores_ ** 2, axis=0) * np.sum(pls.y_loadings_ ** 2, axis=0)
    vip_ref = np.sqrt(X_cat.shape[1] * (pls.x_weights_ ** 2 @ ssy) / ssy.sum())
    vip_got = np.concatenate([model.b_vip_["MS"], model.b_vip_["NMR"]])
    assert np.allclose(vip_got, vip_ref, atol=1e-6)
    assert np.isclose(sum(model.block_importance_.values()), 1.0)


# ---------------------------------------------------------------------------
# multiblock_chemometrics: SO-PLS
# ---------------------------------------------------------------------------
def test_sopls_first_block_is_plain_pls_and_q2_is_reported():
    blocks, y = _mb_data()
    Y = _dummy(y, np.unique(y))
    model = multiblock_chemometrics.SequentialOrthogonalizedPLS(
        n_components_per_block={"MS": 2, "NMR": 1}
    ).fit(blocks, y, block_order=["MS", "NMR"])

    Z1 = blocks["MS"] - blocks["MS"].mean(axis=0)
    pls = PLSRegression(n_components=2, scale=False).fit(Z1, Y)
    Yc = Y - Y.mean(axis=0)
    r2_ref = 1.0 - np.sum((Y - pls.predict(Z1)) ** 2) / np.sum(Yc ** 2)
    assert np.isclose(model.r2_cumulative["MS"], r2_ref, atol=1e-8)

    # Прогноз (через W* и сохранённые D) воспроизводит R2Y, полученный дефляцией
    y_hat = model.predict_y(blocks)
    assert np.isclose(1.0 - np.sum((Y - y_hat) ** 2) / np.sum(Yc ** 2), model.r2y_, atol=1e-8)
    assert set(model.q2_cumulative) == {"MS", "NMR"}
    assert np.isfinite(model.q2_)


# ---------------------------------------------------------------------------
# cross_correlation
# ---------------------------------------------------------------------------
def test_heterospectral_correlation_matches_pearson_and_flags_constants():
    rng = np.random.RandomState(0)
    a = rng.randn(8, 3)
    b = np.column_stack([a[:, 0] * 2.0 + 0.1 * rng.randn(8), rng.randn(8), np.full(8, 5.0)])
    r = cross_correlation.compute_heterospectral_2d_correlation(a, b, scale=True)
    r_ref = np.corrcoef(a.T, b[:, :2].T)[:3, 3:]
    assert np.allclose(r[:, :2], r_ref, atol=1e-12)
    assert np.all(np.isnan(r[:, 2]))  # корреляция с константой не определена


def test_multiblock_cross_correlation_uses_bh_fdr():
    rng = np.random.RandomState(1)
    n = 12
    base = rng.randn(n)
    df = pd.DataFrame({
        "FTIR_1720": base + 0.3 * rng.randn(n),
        "FTIR_1600": rng.randn(n),
        "NMR_Aromatic": base + 0.6 * rng.randn(n),
        "NMR_Aliphatic": rng.randn(n),
        "SEC_Mw": -base + 0.9 * rng.randn(n),
        "SEC_Mn": rng.randn(n),
    })
    res = cross_correlation.compute_multiblock_cross_correlation(df, target_col=None)
    p = res["p_values"].values
    iu = np.triu_indices(p.shape[0], k=1)
    q_ref = stats.false_discovery_control(p[iu], method="bh")
    assert np.allclose(res["q_values"].values[iu], q_ref, atol=1e-12)
    assert res["n_tests"] == len(iu[0])
    top = res["top_cross_correlations"]
    assert not top.empty
    assert (top["Significant"] == (top["q_value"] < 0.05)).all()


def test_multiblock_cross_correlation_undefined_pairs_are_nan():
    df = pd.DataFrame({
        "FTIR_1720": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "NMR_Aromatic": [2.0, np.nan, np.nan, np.nan, 9.0, np.nan],
        "SEC_Mw": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
    })
    res = cross_correlation.compute_multiblock_cross_correlation(df, target_col=None)
    assert np.isnan(res["correlation_matrix"].loc["FTIR_1720", "NMR_Aromatic"])
    assert np.isnan(res["p_values"].loc["FTIR_1720", "NMR_Aromatic"])
    assert np.isfinite(res["correlation_matrix"].loc["FTIR_1720", "SEC_Mw"])


def test_noda_2dcos_synchronous_and_asynchronous_spectra():
    N3 = cross_correlation.hilbert_noda_matrix(3)
    pi = np.pi
    assert np.allclose(N3, [[0, 1 / pi, 1 / (2 * pi)], [-1 / pi, 0, 1 / pi], [-1 / (2 * pi), -1 / pi, 0]])

    m = 9
    theta = np.linspace(0, np.pi, m)
    spectra = np.column_stack([np.sin(theta), np.sin(theta - 0.6), theta ** 2])
    out = cross_correlation.compute_2d_correlation_spectra(spectra)
    Y = spectra - spectra.mean(axis=0)
    N = np.array([[0.0 if j == k else 1.0 / (pi * (k - j)) for k in range(m)] for j in range(m)])
    assert np.allclose(out["synchronous"], np.cov(spectra.T), atol=1e-12)
    assert np.allclose(out["asynchronous"], Y.T @ N @ Y / (m - 1), atol=1e-12)
    assert np.allclose(out["asynchronous"], -out["asynchronous"].T, atol=1e-12)
    assert abs(out["asynchronous"][0, 1]) > 1e-3  # фазовый сдвиг -> ненулевой асинхронный пик


# ---------------------------------------------------------------------------
# views/chemo_ml_view
# ---------------------------------------------------------------------------
def _view():
    from views import chemo_ml_view
    return chemo_ml_view


def _view_df(n=16, seed=31):
    df, y = _fusion_df(n=n, seed=seed)
    return df[_BLOCK_COLS["FT-ICR MS"] + _BLOCK_COLS["FTIR"] + _BLOCK_COLS["NMR"]], y


def test_view_mbpls_metrics_come_from_fitted_model():
    view = _view()
    df, y = _view_df()
    res = view._fit_multiblock_model("mbpls", df, y, 2, True)
    blocks = {b: df[cols] for b, cols in _BLOCK_COLS.items() if cols[0] in df.columns}
    model = multiblock_chemometrics.MultiBlockPLSDA(n_components=2, scale_blocks=True, autoscale=True).fit(blocks, y)
    assert np.isclose(res["R2X"], 100.0 * model.r2x_)
    assert np.isclose(res["R2Y"], 100.0 * model.r2y_)
    assert np.isclose(res["Q2"], 100.0 * model.q2_)
    assert np.isclose(res["Accuracy"], 100.0 * np.mean(model.predict(blocks) == y))
    assert set(res["Block_Contributions"]) == {"FT-ICR MS", "FTIR", "NMR"}


def test_view_sopls_metrics_come_from_fitted_model():
    view = _view()
    df, y = _view_df()
    res = view._fit_multiblock_model("sopls", df, y, 1, False)
    order = ["FT-ICR MS", "FTIR", "NMR"]
    blocks = {b: df[_BLOCK_COLS[b]] for b in order}
    model = multiblock_chemometrics.SequentialOrthogonalizedPLS(
        n_components_per_block={b: 1 for b in order}, autoscale=True, scale_blocks=False
    ).fit(blocks, y, block_order=order)
    assert np.isclose(res["R2Y"], 100.0 * model.r2y_)
    assert np.isclose(res["Q2"], 100.0 * model.q2_)
    assert np.isnan(res["R2X"])  # единый R2X для SO-PLS не определён — не выдумываем число
    with pytest.raises(ValueError):
        view._fit_multiblock_model("sopls", df[_BLOCK_COLS["FTIR"]], y, 1, False)


def test_view_elucidation_constraints_follow_structure_elucidation_api():
    view = _view()
    c = view._build_elucidation_constraints(
        "C10H12O3", ["1720 (C=O carbonyl)", "1510 (Guaiacyl G-ring)"], exact_mass=180.0786,
        nmr_pct={"aliphatic": 20.0, "o_alkyl": 10.0, "aromatic": 60.0, "carboxyl": 10.0},
    )
    assert c.formula == "C10H12O3"
    assert c.ftir_carbonyl == 1 and c.ftir_aromatic_c_c == 1
    assert c.ftir_phenol_oh == -1 and c.ftir_ether_c_o == -1  # не выбрано = не учитывать (а не «отсутствует»)
    assert np.isclose(c.nmr_aromatic_pct, 0.6) and np.isclose(c.nmr_aliphatic_pct, 0.2)
    c2 = view._build_elucidation_constraints("C8H8O3", [])
    assert c2.nmr_aromatic_pct is None and c2.ftir_carbonyl == -1


def test_view_elucidation_rows_tolerate_missing_optional_fields():
    view = _view()

    class _Cand:
        name = "vanillin"
        formula = "C8H8O3"
        exact_mass = 152.04734
        concordance_score = float("nan")

    rows = view._elucidation_rows([_Cand()])
    assert rows[0]["Compound"] == "vanillin"
    assert rows[0]["MMCS Score"] == "—"
    assert rows[0]["CID"] is None and rows[0]["SMILES"] == ""
