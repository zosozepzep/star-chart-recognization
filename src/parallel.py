"""Bounded, ordered frame execution for CPU inference.

Only pixels (or FITS paths) and inference parameters cross the worker boundary.
Sequence objects, including validation truth, are never sent to workers.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import nullcontext
import multiprocessing
import os
from pathlib import Path


def execution_options(config):
    options = config.get('execution', {})
    if not isinstance(options, dict):
        raise ValueError('execution 必须为配置映射')
    workers = options.get('workers', 1)  # Old complete configurations stay serial.
    executor = options.get('executor', 'process')
    if type(workers) is not int or workers < 0:
        raise ValueError('workers 必须为非负整数：0 自动，1 串行，2 及以上并行')
    if executor not in ('process', 'thread'):
        raise ValueError('executor 必须为 process 或 thread')
    return workers, executor


def cpu_capacity():
    """Respect CPU affinity and Docker's cgroup quota when available."""
    count = os.cpu_count() or 1
    if hasattr(os, 'sched_getaffinity'):
        count = min(count, len(os.sched_getaffinity(0)))
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            count = min(count, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        try:
            quota = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
            period = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
            if quota > 0 and period > 0:
                count = min(count, max(1, quota // period))
        except (OSError, ValueError):
            pass
    return max(1, count)


def execution_plan(config, frame_count):
    requested, executor = execution_options(config)
    # 4096² frames have large temporary arrays. Auto deliberately caps at four;
    # an explicit count lets the operator choose a different memory/CPU tradeoff.
    workers = min(requested or min(4, cpu_capacity()), frame_count)
    return dict(requested_workers=requested, workers=max(1, workers),
                executor=executor if workers > 1 else 'serial',
                native_threads_per_worker=1 if workers > 1 else None)


def frame_input(sequence, frame):
    from src.dataio.fits_loader import FrameSequence
    if type(sequence) is FrameSequence:
        return sequence.headers[frame].path
    return sequence.image(frame)


def read_frame(source):
    if isinstance(source, (str, Path)):
        from src.dataio.fits_loader import load_image
        return load_image(source)
    return source


def _process_call(function, job):
    from threadpoolctl import threadpool_limits
    # The callable's modules have been imported when this job is unpickled.
    with threadpool_limits(limits=1):
        return function(job)


def map_frames(function, jobs, plan):
    """Yield results in input order; keep at most `workers` jobs outstanding.

    Bounded submission also bounds images rendered by in-memory sequences.
    Exceptions cancel queued jobs and join active workers before returning.
    """
    workers = plan['workers']
    if workers == 1:
        for job in jobs:
            yield function(job)
        return

    from threadpoolctl import threadpool_limits
    process = plan['executor'] == 'process'
    pool_class = ProcessPoolExecutor if process else ThreadPoolExecutor
    kwargs = {'mp_context': multiprocessing.get_context('spawn')} if process else {}
    # Native-library thread limits are process-wide: set once around the whole
    # thread pool, never enter competing limit contexts in individual threads.
    limits = nullcontext() if process else threadpool_limits(limits=1)
    with limits:
        pool = pool_class(max_workers=workers, **kwargs)
        pending, ready = {}, {}
        indexed = iter(enumerate(jobs))
        next_result = 0
        try:
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) + len(ready) < workers:
                    try:
                        index, job = next(indexed)
                    except StopIteration:
                        exhausted = True
                        break
                    future = (pool.submit(_process_call, function, job) if process
                              else pool.submit(function, job))
                    pending[future] = index
                if pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        ready[pending.pop(future)] = future.result()
                while next_result in ready:
                    yield ready.pop(next_result)
                    next_result += 1
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
