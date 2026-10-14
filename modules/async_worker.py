"""
modules/async_worker.py — High-Performance Vectorization & Asynchronous Background Worker.

Provides:
1. ThreadPool/ProcessPool JobManager for non-blocking execution of long chemometric tasks
   (OPLS-DA permutations, PARAFAC split-half, Monte Carlo validation).
2. Accelerated vectorized baseline correction (Sparse Cholesky ALS).
3. Ultra-fast binary search mass filtering & vectorized Kendrick Mass Defect (KMD) transforms.
"""

import uuid
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Tuple, List
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class AsyncJob:
    """Represents a background computational task."""
    job_id: str
    func_name: str
    status: str = "pending"  # "pending", "running", "completed", "failed", "cancelled"
    progress: float = 0.0    # 0.0 to 1.0
    status_message: str = "Initialized"
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    cancel_flag: threading.Event = field(default_factory=threading.Event)


class JobManager:
    """
    Thread-safe background task orchestrator for Streamlit interactive scientific apps.
    Prevents Streamlit session freezes during multi-minute chemometric calculations.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, max_workers: int = 4):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(JobManager, cls).__new__(cls)
                cls._instance._init_manager(max_workers)
            return cls._instance

    def _init_manager(self, max_workers: int):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ChemoJob")
        self.jobs: Dict[str, AsyncJob] = {}
        self.futures: Dict[str, Future] = {}
        self._job_lock = threading.Lock()

    def submit_job(
        self,
        func: Callable,
        *args,
        job_id: Optional[str] = None,
        func_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Submit a computational callable to the background pool."""
        with self._job_lock:
            jid = job_id or str(uuid.uuid4())[:8]
            name = func_name or func.__name__
            job = AsyncJob(job_id=jid, func_name=name)
            self.jobs[jid] = job

            def _wrapped_task():
                job.status = "running"
                job.status_message = "Calculation in progress..."

                def _progress_reporter(val: float, msg: str = ""):
                    job.progress = max(0.0, min(1.0, float(val)))
                    if msg:
                        job.status_message = msg

                kwargs["_report_progress"] = _progress_reporter
                kwargs["_cancel_event"] = job.cancel_flag

                try:
                    res = func(*args, **kwargs)
                    if job.cancel_flag.is_set():
                        job.status = "cancelled"
                        job.status_message = "Computation was cancelled by user."
                    else:
                        job.result = res
                        job.progress = 1.0
                        job.status = "completed"
                        job.status_message = "Done."
                except Exception as ex:
                    job.status = "failed"
                    job.error = str(ex)
                    job.status_message = f"Error: {ex}"
                finally:
                    job.finished_at = time.time()

            future = self.executor.submit(_wrapped_task)
            self.futures[jid] = future
            return jid

    def get_job(self, job_id: str) -> Optional[AsyncJob]:
        """Fetch current status and progress of job."""
        with self._job_lock:
            return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Signal job cancellation."""
        with self._job_lock:
            job = self.jobs.get(job_id)
            if job and job.status in ("pending", "running"):
                job.cancel_flag.set()
                job.status_message = "Cancelling..."
                return True
            return False

    def cleanup_old_jobs(self, max_age_seconds: float = 3600.0):
        """Remove finished jobs older than max age."""
        now = time.time()
        with self._job_lock:
            to_remove = [
                jid for jid, j in self.jobs.items()
                if j.finished_at and (now - j.finished_at) > max_age_seconds
            ]
            for jid in to_remove:
                del self.jobs[jid]
                if jid in self.futures:
                    del self.futures[jid]


# Global singleton instance
job_manager = JobManager()


# ==============================================================================
# VECTORIZED NUMERICAL KERNELS
# ==============================================================================

def vectorized_als_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10
) -> np.ndarray:
    """
    Accelerated Asymmetric Least Squares (ALS) baseline correction.
    Uses sparse banded linear equation solving for 10x-50x speedup.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    L = len(y_arr)
    if L < 3:
        return np.zeros_like(y_arr)

    # Second order difference operator
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L), format="csc", dtype=np.float64)
    DTD = D.T @ D
    H = lam * DTD

    w = np.ones(L, dtype=np.float64)
    z = np.zeros(L, dtype=np.float64)

    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L), format="csc")
        A = W + H
        B = w * y_arr
        z = spsolve(A, B)
        # Update weights: p if y > z (peaks), 1 - p if baseline
        w = p * (y_arr > z) + (1.0 - p) * (y_arr <= z)

    return z


def fast_mass_filter(
    mz_sorted: np.ndarray,
    target_mz: float,
    ppm_tol: float = 2.0
) -> np.ndarray:
    """
    Ultra-fast O(log N) binary search mass filter using np.searchsorted.
    Returns boolean mask or matching index array within +/- ppm_tol.
    """
    mz_arr = np.asarray(mz_sorted, dtype=np.float64)
    delta_mz = target_mz * (ppm_tol * 1e-6)
    lower = target_mz - delta_mz
    upper = target_mz + delta_mz

    idx_start = np.searchsorted(mz_arr, lower, side="left")
    idx_end = np.searchsorted(mz_arr, upper, side="right")

    if idx_start < idx_end:
        return np.arange(idx_start, idx_end, dtype=np.int64)
    return np.empty(0, dtype=np.int64)


def batch_kendrick_transform(
    mz_array: np.ndarray,
    base_exact_mass: float = 14.0156500642,  # CH2 default
    base_nominal_mass: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized computation of Kendrick Mass (KM), Nominal Kendrick Mass (NKM),
    and Kendrick Mass Defect (KMD) for entire mass spectrometry peak lists.

    Returns:
    --------
    km: np.ndarray
    nkm: np.ndarray (integer)
    kmd: np.ndarray (NKM - KM)
    """
    mz = np.asarray(mz_array, dtype=np.float64)
    km = mz * (base_nominal_mass / base_exact_mass)
    nkm = np.ceil(km).astype(np.int64)
    kmd = nkm - km
    return km, nkm, kmd
