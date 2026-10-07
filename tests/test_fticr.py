"""
tests/test_fticr.py — Модульные тесты ядра FT-ICR MS (fticr_core.py).
"""
import numpy as np
import pandas as pd
import pytest

import fticr_core


def test_fast_formula_assigner():
    """Тест приписки брутто-формул по точной массе в ESI(-)."""
    # Ванилиновая кислота C8H8O4: M_neut = 168.042259, ESI(-) [M-H]- = 167.034983
    # Кофейная кислота C9H8O4: M_neut = 180.042259, ESI(-) [M-H]- = 179.034983
    m1 = 8 * 12.0 + 8 * 1.007825 + 4 * 15.994915 - 1.007276  # 167.034984
    m2 = 9 * 12.0 + 8 * 1.007825 + 4 * 15.994915 - 1.007276  # 179.034984

    peaks_df = pd.DataFrame({
        "mass": [m1, m2, 300.12345],  # 3-я масса заведомо без формулы при жестких границах
        "intensity": [50000.0, 80000.0, 1000.0],
    })

    bounds = {"C": (4, 15), "H": (4, 25), "O": (1, 10), "N": (0, 0), "S": (0, 0)}
    res = fticr_core.fast_formula_assigner(
        peaks_df=peaks_df,
        bounds=bounds,
        max_hc=2.5,
        max_oc=1.2,
        ppm_tolerance=2.0,
        ion_mode="ESI(-)",
        max_charge=1,
    )

    assert not res.empty
    assert len(res) >= 2
    formulas = list(res["Formula"].values)
    assert "C8H8O4" in formulas
    assert "C9H8O4" in formulas
    assert all(abs(err) <= 2.0 for err in res["error_ppm"])


def test_calculate_descriptors():
    """Тест расчета химических индексов (H/C, O/C, DBE, AI, NOSC)."""
    df = pd.DataFrame({
        "C": [10, 20],
        "H": [10, 30],
        "O": [4, 6],
        "N": [1, 0],
        "S": [0, 1],
    })

    res = fticr_core.calculate_descriptors(df, lang="ru")

    assert "H/C" in res.columns
    assert "O/C" in res.columns
    assert "DBE" in res.columns
    assert "AI" in res.columns
    assert "NOSC" in res.columns
    assert "Hetero_Class" in res.columns
    assert "Compound_Class" in res.columns

    # C10H10O4N1: H/C = 1.0, O/C = 0.4, DBE = 1 + 10 - 5 + 0.5 = 6.5
    assert np.isclose(res.loc[0, "H/C"], 1.0)
    assert np.isclose(res.loc[0, "O/C"], 0.4)
    assert np.isclose(res.loc[0, "DBE"], 6.5)
    assert res.loc[0, "Hetero_Class"] == "CHON"
    assert res.loc[1, "Hetero_Class"] == "CHOS"


def test_compute_kmd():
    """Тест вычисления дефекта массы Кендрика (KMD)."""
    masses = np.array([200.05, 214.06565, 228.0813])  # гомологическая серия CH2
    km, kmd, nkm = fticr_core.compute_kmd(masses, base_key="CH2")

    assert len(km) == 3
    assert len(kmd) == 3
    assert len(nkm) == 3
    # Разница в KMD между членами истинной серии CH2 должна быть близка к нулю
    diff_kmd = np.diff(kmd)
    assert np.all(np.abs(diff_kmd) < 0.005)


def test_compute_vk20_grid():
    """Тест сетки хемотипирования 20 ячеек Ван-Кревелена Перминовой."""
    # 4 точки в разных ячейках
    df = pd.DataFrame({
        "O/C": [0.1, 0.4, 0.6, 0.9],
        "H/C": [0.4, 0.8, 1.2, 1.6],
        "intensity": [100.0, 200.0, 300.0, 400.0],
    })

    pct_count, pct_weight, grid_df = fticr_core.compute_vk20_grid(df)

    assert pct_count.shape == (5, 4)
    assert pct_weight.shape == (5, 4)
    assert len(grid_df) == 20
    assert np.isclose(np.sum(pct_count), 100.0)
    assert np.isclose(np.sum(pct_weight), 100.0)


def test_spectral_algebra():
    """Тест спектральной алгебры (A-B, A+B, A ∩ B)."""
    df_a = pd.DataFrame({"mass": [100.0, 200.0, 300.0], "intensity": [10, 20, 30]})
    df_b = pd.DataFrame({"mass": [200.0, 300.0, 400.0], "intensity": [25, 35, 45]})

    diff_ab = fticr_core.perform_spectral_algebra(df_a, df_b, operation="A - B", ppm_tol=5.0)
    assert len(diff_ab) == 1
    assert np.isclose(diff_ab.iloc[0]["mass"], 100.0)

    inter = fticr_core.perform_spectral_algebra(df_a, df_b, operation="A ∩ B", ppm_tol=5.0)
    assert len(inter) == 2
    assert set(np.round(inter["mass"].values, 1)) == {200.0, 300.0}

    union = fticr_core.perform_spectral_algebra(df_a, df_b, operation="A + B", ppm_tol=5.0)
    assert len(union) == 4


def test_tmds_screening():
    """Тест скрининга массовых разностей TMDS."""
    # Создаем пару с разницей массы в CH2 (14.01565 Да)
    m1 = 200.00000
    m2 = m1 + 14.01565
    peaks_df = pd.DataFrame({"mass": [m1, m2], "intensity": [100.0, 200.0]})

    summary, pairs = fticr_core.run_tmds_screening(peaks_df, top_n=10, tol_mda=2.0)
    assert not summary.empty
    ch2_hit = summary[summary["Transformation"].str.contains("CH2")]
    assert not ch2_hit.empty
    assert ch2_hit.iloc[0]["Count"] == 1
    assert len(pairs) == 1


def test_biomolecular_regions_and_distribution():
    """Тест классификации биомолекулярных полигонов Ван-Кревелена и расчета распределения."""
    df = pd.DataFrame({
        "C": [10, 20, 15, 8],
        "H": [10, 36, 12, 16],
        "O": [4, 2, 7, 1],
        "N": [0, 0, 0, 0],
        "S": [0, 0, 0, 0],
    })
    desc = fticr_core.calculate_descriptors(df, lang="ru")
    assert "Bio_Class" in desc.columns
    # C20H36O2: H/C = 1.8, O/C = 0.1 -> Lipids
    assert desc.loc[1, "Bio_Class"] == "Липиды"
    # C10H10O4: H/C = 1.0, O/C = 0.4 -> Lignins / Polyphenols
    assert "Лигнины" in desc.loc[0, "Bio_Class"]

    dist = fticr_core.get_biomolecular_distribution(desc, lang="ru")
    assert isinstance(dist, dict)
    assert np.isclose(sum(dist.values()), 100.0)


def test_batch_process_fticr_spectra():
    """Тест пакетной обработки коллекции масс-спектров."""
    m_vanillin = 8 * 12.0 + 8 * 1.007825 + 4 * 15.994915 - 1.007276
    m_caffeic = 9 * 12.0 + 8 * 1.007825 + 4 * 15.994915 - 1.007276
    content_a = f"mass,intensity\n{m_vanillin},50000\n{m_caffeic},80000\n".encode("utf-8")
    content_b = f"mass,intensity\n{m_vanillin},30000\n".encode("utf-8")

    files_dict = {
        "Sample_A.csv": content_a,
        "Sample_B.csv": content_b,
    }

    summary_df, spectra_dict = fticr_core.batch_process_fticr_spectra(files_dict, ppm_tolerance=3.0)

    assert not summary_df.empty
    assert len(summary_df) == 2
    assert "Sample_ID" in summary_df.columns
    assert set(summary_df["Sample_ID"].values) == {"Sample_A", "Sample_B"}
    assert "VK_1" in summary_df.columns
    assert "VK_20" in summary_df.columns
    assert "AI" in summary_df.columns
    assert "DBE" in summary_df.columns
    assert len(spectra_dict) == 2
    assert "Sample_A.csv" in spectra_dict

    # Тест загрузки через zip-архив (байты)
    import io
    import zipfile
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("Zip_Sample_01.csv", content_a)
        zf.writestr("Zip_Sample_02.csv", content_b)
    zip_bytes = zip_buffer.getvalue()

    summary_zip_df, zip_spectra = fticr_core.batch_process_fticr_spectra(zip_bytes, ppm_tolerance=3.0)
    assert len(summary_zip_df) == 2
    assert set(summary_zip_df["Sample_ID"].values) == {"Zip_Sample_01", "Zip_Sample_02"}


def test_tmds_network_graph():
    """Тест построения графа сети реакций TMDS и топологических хабов."""
    # Создаем 4 пика, связанные превращениями CH2 и O
    m0 = 200.00000
    m_ch2 = m0 + 14.01565
    m_o = m0 + 15.994915
    m_ch2_o = m_ch2 + 15.994915

    peaks_df = pd.DataFrame({
        "mass": [m0, m_ch2, m_o, m_ch2_o],
        "intensity": [1000.0, 800.0, 600.0, 400.0],
        "Formula": ["C10H16O4", "C11H18O4", "C10H16O5", "C11H18O5"],
        "Hetero_Class": ["CHO", "CHO", "CHO", "CHO"],
    })

    net_res = fticr_core.build_tmds_network_graph(peaks_df, top_n=10, tol_mda=2.0, max_edges=50, layout="spring")

    assert "nodes_df" in net_res
    assert "edges_df" in net_res
    assert "hubs_df" in net_res
    assert not net_res["nodes_df"].empty
    assert not net_res["edges_df"].empty
    assert not net_res["hubs_df"].empty
    assert "Degree" in net_res["nodes_df"].columns
    assert "x" in net_res["nodes_df"].columns
    assert "y" in net_res["nodes_df"].columns
    assert len(net_res["edges_df"]) >= 2
    # Проверка, что хаб имеет степень >= 2
    assert net_res["hubs_df"].iloc[0]["Degree"] >= 2


def test_compute_geochemical_vector_fluxes():
    """Тест расчета геохимических векторных потоков реакций."""
    m0 = 200.00000
    m_ch2 = m0 + 14.015650
    m_o = m0 + 15.994915
    m_co2 = m0 + 43.989829

    peaks_df = pd.DataFrame({
        "mass": [m0, m_ch2, m_o, m_co2],
        "intensity": [1000.0, 900.0, 800.0, 700.0],
        "Formula": ["C10H16O4", "C11H18O4", "C10H16O5", "C11H16O6"],
    })

    flux_res = fticr_core.compute_geochemical_vector_fluxes(peaks_df, tol_mda=2.0)
    assert "flux_counts" in flux_res
    assert "indices" in flux_res
    assert "summary_df" in flux_res
    assert flux_res["total_reactions"] >= 3
    assert flux_res["flux_counts"]["CH2"] >= 1
    assert flux_res["flux_counts"]["O"] >= 1
    assert flux_res["flux_counts"]["CO2"] >= 1
    assert "ox_decarb_ratio" in flux_res["indices"]
    assert "alkylation_share_pct" in flux_res["indices"]


def test_find_transformation_pathways():
    """Тест поиска многостадийных путей биогеохимической трансформации."""
    # Цепочка: M0 -> (+O) -> M1 -> (+SO3) -> M2 -> (+CH2) -> M3
    m0 = 300.000000
    m1 = m0 + 15.994915  # +O
    m2 = m1 + 79.956815  # +SO3 -> 395.951730
    m3 = m2 + 14.015650  # +CH2 -> 409.967380

    peaks_df = pd.DataFrame({
        "mass": [m0, m1, m2, m3],
        "intensity": [1000.0, 800.0, 600.0, 400.0],
        "Formula": ["Precursor", "Int_O", "Int_Sulf", "Product"],
    })

    # Поиск пути от m0 к m2 (двухстадийный: +O, +SO3)
    paths = fticr_core.find_transformation_pathways(peaks_df, source_mass=m0, target_mass=m2, max_depth=3, tol_mda=2.0)
    assert len(paths) >= 1
    best_path = paths[0]
    assert best_path["depth"] == 2
    assert len(best_path["steps"]) == 2
    assert best_path["steps"][0]["code"] == "O"
    assert best_path["steps"][0]["direction"] == "+"
    assert best_path["steps"][1]["code"] == "SO3"
    assert best_path["steps"][1]["direction"] == "+"
    assert best_path["cumulative_error_mda"] < 0.1

    # Поиск пути от m0 к m3 (трехстадийный)
    paths_long = fticr_core.find_transformation_pathways(peaks_df, source_mass=m0, target_mass=m3, max_depth=4, tol_mda=2.0)
    assert len(paths_long) >= 1
    assert paths_long[0]["depth"] == 3
    assert paths_long[0]["steps"][2]["code"] == "CH2"

    # Несуществующий путь (изолированная масса)
    no_path = fticr_core.find_transformation_pathways(peaks_df, source_mass=m0, target_mass=999.0, max_depth=4)
    assert no_path == []



