"""
tests/test_async_worker.py — Unit tests for background job manager and vectorized acceleration kernels.
"""
import time
import numpy as np
import pytest
from modules.async_worker import (
    JobManager,
    vectorized_als_baseline,
    fast_mass_filter,
    batch_kendrick_transform,
)


def test_job_manager_execution_and_progress():
    jm = JobManager(max_workers=2)

    def sample_heavy_calc(n_steps=5, _report_progress=None, _cancel_event=None):
        total = 0
        for i in range(n_steps):
            if _cancel_event and _cancel_event.is_set():
                break
            time.sleep(0.05)
            total += i
            if _report_progress:
                _report_progress((i + 1) / n_steps, f"Step {i+1}/{n_steps}")
        return total

    jid = jm.submit_job(sample_heavy_calc, n_steps=4)
    assert jid is not None

    # Poll briefly for completion
    for _ in range(50):
        job = jm.get_job(jid)
        if job and job.status == "completed":
            break
        time.sleep(0.05)

    assert job is not None
    assert job.status == "completed"
    assert job.result == 6  # 0 + 1 + 2 + 3
    assert job.progress == 1.0


def test_vectorized_als_baseline():
    x = np.linspace(0, 10, 200)
    # Baseline + peak
    baseline = 0.5 * x + 2.0
    peak = 5.0 * np.exp(-((x - 5.0) ** 2) / 0.5)
    signal = baseline + peak

    est_baseline = vectorized_als_baseline(signal, lam=1e4, p=0.01, niter=10)
    assert len(est_baseline) == len(signal)
    # At non-peak regions, baseline should match closely
    assert np.allclose(est_baseline[:50], baseline[:50], atol=0.5)
    # Under the peak (idx 100), estimated baseline must NOT follow the full height of the peak
    assert est_baseline[100] < signal[100] - 2.0


def test_fast_mass_filter_binary_search():
    # Sorted mass array
    mz_sorted = np.array([100.0, 150.002, 150.004, 200.0, 300.0])
    target = 150.003

    # With 20 ppm tolerance: 150.003 * 20e-6 = 0.003 -> [150.000, 150.006] -> indices 1 and 2
    matched_idx = fast_mass_filter(mz_sorted, target, ppm_tol=20.0)
    assert len(matched_idx) == 2
    assert 1 in matched_idx and 2 in matched_idx

    # Tight tolerance (1 ppm): should match 0
    tight_idx = fast_mass_filter(mz_sorted, target, ppm_tol=1.0)
    assert len(tight_idx) == 0


def test_batch_kendrick_transform():
    mz_peaks = np.array([280.150, 294.165, 308.181])  # CH2 homologous series delta ~14.01565
    km, nkm, kmd = batch_kendrick_transform(mz_peaks)

    assert len(km) == 3
    assert len(nkm) == 3
    assert len(kmd) == 3
    # Check that KMD of homologous series members are virtually identical
    assert np.allclose(kmd[1], kmd[0], atol=1e-3)
    assert np.allclose(kmd[2], kmd[1], atol=1e-3)
