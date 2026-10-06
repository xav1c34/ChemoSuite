"""
tests/test_chemo_ml.py — Модульные тесты хемометрического ядра (chemo_ml.py).
"""
import numpy as np
import pandas as pd
import pytest

import chemo_ml


def test_classify_feature_blocks():
    """Тест автоматической классификации признаков по аналитическим блокам."""
    cols = ["VK_1", "VK_20", "AI", "DBE", "FI", "HIX", "C1_Fulvic", "A254", "E2_E3", "Mw_est", "Random_Col"]
    mapping = chemo_ml.classify_feature_blocks(cols)

    assert mapping["VK_1"] == "FT-ICR MS"
    assert mapping["AI"] == "FT-ICR MS"
    assert mapping["FI"] == "EEM-PARAFAC"
    assert mapping["C1_Fulvic"] == "EEM-PARAFAC"
    assert mapping["A254"] == "UV-Vis"
    assert mapping["Mw_est"] == "UV-Vis"
    assert mapping["Random_Col"] == "Other"


def test_merge_feature_blocks():
    """Тест слияния таблиц признаков с нечетким сопоставлением Sample_ID."""
    df_ft = pd.DataFrame({"Sample_ID": ["Sample_1.csv", "Sample_2.csv"], "VK_5": [12.5, 3.2]})
    df_eem = pd.DataFrame({"Sample_ID": ["sample_1", "sample_2"], "HIX": [15.2, 2.1]})
    df_uv = pd.DataFrame({"Sample_ID": ["Sample_1", "Sample_2"], "E2_E3": [2.8, 6.1]})

    merged = chemo_ml.merge_feature_blocks(df_ft, df_eem, df_uv)

    assert len(merged) == 2
    assert "VK_5" in merged.columns
    assert "HIX" in merged.columns
    assert "E2_E3" in merged.columns


def test_prepare_fused_features_low_and_mid():
    """Тест стратегий Low-Level и Mid-Level Data Fusion с обработкой пропусков."""
    df, _ = chemo_ml.generate_multimodal_benchmark()
    X_df = df.drop(columns=["Sample_ID"])

    # Добавим искусственный NaN, чтобы проверить устойчивость к пропускам
    X_df.iloc[0, 2] = np.nan

    # 1. Low-Level Fusion
    X_low, names_low, blocks_low = chemo_ml.prepare_fused_features(X_df, strategy="low_level", block_scaling=True)
    assert not np.isnan(X_low).any()
    assert X_low.shape[1] == len(names_low)

    # 2. Mid-Level Fusion (PCA сжатие FT-ICR)
    X_mid, names_mid, blocks_mid = chemo_ml.prepare_fused_features(X_df, strategy="mid_level", n_mid_components=3)
    assert not np.isnan(X_mid).any()
    assert "FTICR_PC1" in names_mid
    assert "FTICR_PC2" in names_mid
    assert "FTICR_PC3" in names_mid


def test_train_plsda_model_metrics():
    """Тест обучения модели PLS-DA и расчета диагностических метрик."""
    df, y = chemo_ml.generate_multimodal_benchmark()
    X = df.drop(columns=["Sample_ID"])

    res = chemo_ml.train_plsda_model(X, y.values, n_components=2, fusion_strategy="low_level")

    # Проверка ключевых метрик
    assert "R2X" in res and res["R2X"] > 0
    assert "R2Y" in res and res["R2Y"] > 0
    assert "Q2" in res and res["Q2"] > 50.0  # Для четко разделенного бенчмарка Q2 высокий
    assert "Accuracy" in res and res["Accuracy"] >= 90.0
    assert "Confusion_Matrix" in res
    assert res["Confusion_Matrix"].shape == (2, 2)
    assert "Block_Contributions" in res

    # Проверка вклада блоков
    blocks = res["Block_Contributions"]
    assert "FT-ICR MS" in blocks
    assert "EEM-PARAFAC" in blocks
    assert "UV-Vis" in blocks
    assert np.isclose(sum(blocks.values()), 100.0, atol=1.0)

    # Проверка VIP
    vip_df = res["VIP_df"]
    assert not vip_df.empty
    assert "Descriptor" in vip_df.columns
    assert "VIP" in vip_df.columns
    assert "Block" in vip_df.columns


def test_hotelling_ellipse():
    """Тест вычисления эллипса Хотеллинга T^2."""
    x = np.random.normal(0, 1, 30)
    y = np.random.normal(0, 1, 30)

    ell_x, ell_y = chemo_ml.compute_hotelling_ellipse(x, y, n_points=50, confidence=0.95)
    assert len(ell_x) == 50
    assert len(ell_y) == 50
    # Центр эллипса должен быть близок к среднему значению координат
    assert np.isclose(np.mean(ell_x), np.mean(x), atol=0.2)
    assert np.isclose(np.mean(ell_y), np.mean(y), atol=0.2)


def test_permutation_test():
    """Тест пермутационного теста валидации Q2."""
    df, y = chemo_ml.generate_multimodal_benchmark()
    X = df.drop(columns=["Sample_ID"])

    perm_res = chemo_ml.run_permutation_test(X, y.values, n_components=2, n_permutations=15, random_state=42)

    assert "q2_orig" in perm_res
    assert "perm_q2" in perm_res
    assert "p_val_q2" in perm_res
    assert len(perm_res["perm_q2"]) == 15
    assert 0.0 <= perm_res["p_val_q2"] <= 1.0


def test_hotelling_ellipsoid_3d():
    """Тест вычисления координат 3D эллипсоида Хотеллинга T^2."""
    x = np.random.normal(0, 1, 25)
    y = np.random.normal(0, 1, 25)
    z = np.random.normal(0, 1, 25)

    x_ell, y_ell, z_ell = chemo_ml.compute_hotelling_ellipsoid_3d(x, y, z, n_theta=20, n_phi=20)
    assert x_ell.shape == (20, 20)
    assert y_ell.shape == (20, 20)
    assert z_ell.shape == (20, 20)
    assert np.isclose(np.mean(x_ell), np.mean(x), atol=0.2)
    assert np.isclose(np.mean(y_ell), np.mean(y), atol=0.2)
    assert np.isclose(np.mean(z_ell), np.mean(z), atol=0.2)


def test_multiclass_plsda():
    """Тест мультиклассовой PLS-DA классификации (3 класса, 3D Scores, confusion matrix)."""
    df, _ = chemo_ml.generate_multimodal_benchmark()
    X = df.drop(columns=["Sample_ID"])
    # 3 класса
    y_multi = np.array(["Baikal", "Baikal", "Baikal", "Baikal", "Baikal",
                        "Lignin", "Lignin", "Lignin", "Lignin", "Lignin",
                        "Sediment", "Sediment", "Sediment", "Sediment"])

    res = chemo_ml.train_plsda_model(X, y_multi, n_components=3, fusion_strategy="low_level")
    assert res["is_multiclass"] is True
    assert len(res["classes"]) == 3
    assert res["Confusion_Matrix"].shape == (3, 3)
    assert "scores_t3" in res and len(res["scores_t3"]) == len(y_multi)
    assert res["ell_x_3d"].shape == (30, 30)

    # Проверка мультиклассового пермутационного теста
    perm_res = chemo_ml.run_permutation_test(X, y_multi, n_components=3, n_permutations=10, random_state=42)
    assert "q2_orig" in perm_res
    assert len(perm_res["perm_q2"]) == 10
    assert 0.0 <= perm_res["p_val_q2"] <= 1.0


def test_oplsda_model_and_splot():
    """Тест OPLS-DA модели: ортогональное расщепление вариаций и S-Plot."""
    df, y = chemo_ml.generate_multimodal_benchmark()
    X = df.drop(columns=["Sample_ID"])

    res = chemo_ml.train_oplsda_model(X, y.values, n_ortho=1, fusion_strategy="low_level")
    assert res["model_type"] == "OPLS-DA"
    assert "t_pred" in res and "t_ortho" in res
    assert len(res["t_pred"]) == len(df)
    assert res["R2X_pred"] > 0
    assert res["R2X_ortho"] > 0
    assert res["R2Y"] > 0

    s_plot = res["S_Plot_df"]
    assert isinstance(s_plot, pd.DataFrame)
    assert not s_plot.empty
    assert "p1_cov" in s_plot.columns
    assert "p_corr" in s_plot.columns
    assert "VIP" in s_plot.columns
    assert "Block" in s_plot.columns
    # p_corr должен лежать в интервале [-1, 1]
    assert (s_plot["p_corr"] >= -1.01).all() and (s_plot["p_corr"] <= 1.01).all()

