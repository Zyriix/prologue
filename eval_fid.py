import sys
import os
import argparse
from typing import Tuple
import numpy as np
import tensorflow.compat.v1 as tf

from evaluator import Evaluator, FIDStatistics

REF_FID_STATS_SIDECAR_SUFFIX = ".fid_stats_cache.npz"


def _has_ref_fid_stats_keys(keys: set) -> bool:
    return (
        "mu" in keys
        and "sigma" in keys
        and "mu_s" in keys
        and "sigma_s" in keys
    )


def _fid_stats_pair_from_z(z) -> Tuple[FIDStatistics, FIDStatistics]:
    """Build (pool, spatial) stats from an open NpzFile; arrays copied to host memory."""
    return (
        FIDStatistics(np.asarray(z["mu"]), np.asarray(z["sigma"])),
        FIDStatistics(np.asarray(z["mu_s"]), np.asarray(z["sigma_s"])),
    )


def _try_write_ref_sidecar(ref_batch: str, ref_stats: FIDStatistics, ref_spatial: FIDStatistics) -> None:
    sidecar = ref_batch + REF_FID_STATS_SIDECAR_SUFFIX
    d = os.path.dirname(os.path.abspath(sidecar)) or "."
    if not os.access(d, os.W_OK):
        print(f"[ref] cache dir not writable, skip: {sidecar}", flush=True)
        return
    np.savez_compressed(
        sidecar,
        mu=ref_stats.mu,
        sigma=ref_stats.sigma,
        mu_s=ref_spatial.mu,
        sigma_s=ref_spatial.sigma,
    )
    print(f"[ref] wrote FID stats cache: {sidecar}", flush=True)


def _load_ref_fid_statistics(evaluator: Evaluator, ref_batch: str) -> tuple[FIDStatistics, FIDStatistics]:
    """Sidecar cache if available; else embedded mu/sigma; else run Inception on ``arr_0``."""
    sidecar = ref_batch + REF_FID_STATS_SIDECAR_SUFFIX

    if os.path.isfile(sidecar):
        try:
            with np.load(sidecar) as z:
                keys = set(z.files)
                if _has_ref_fid_stats_keys(keys):
                    print(f"[ref] (1) FID stats from cache: {sidecar}", flush=True)
                    return _fid_stats_pair_from_z(z)
        except Exception as e:
            print(f"[ref] (1) cache unreadable ({e}), fall through", flush=True)

    with np.load(ref_batch) as z:
        main_keys = set(z.files)
        if _has_ref_fid_stats_keys(main_keys):
            print("[ref] (2) embedded stats in main ref -> load + refresh cache", flush=True)
            out = _fid_stats_pair_from_z(z)
            _try_write_ref_sidecar(ref_batch, out[0], out[1])
            return out
        if "arr_0" not in main_keys:
            raise ValueError(
                f"ref_batch has no FID stats (mu/sigma/mu_s/sigma_s) and no arr_0: {ref_batch}"
            )

    print("[ref] (3) compute ref stats from main arr_0 (Inception), then write cache", flush=True)
    ref_acts = evaluator.read_activations(ref_batch)
    ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_batch, ref_acts)
    _try_write_ref_sidecar(ref_batch, ref_stats, ref_stats_spatial)
    return ref_stats, ref_stats_spatial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_batch", type=str, required=True)
    parser.add_argument("--sample_batch", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--compute_is", action="store_true",
                        help="Also compute Inception Score (slower)")
    args = parser.parse_args()

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    config.gpu_options.allow_growth = True

    try:
        evaluator = Evaluator(tf.Session(config=config), batch_size=args.batch_size)
        evaluator.warmup()

        ref_stats, _ = _load_ref_fid_statistics(evaluator, args.ref_batch)

        sample_acts = evaluator.read_activations(args.sample_batch)
        sample_stats, _ = evaluator.read_statistics(args.sample_batch, sample_acts)

        fid = sample_stats.frechet_distance(ref_stats)
        print(f"FID_RESULT:{fid}")

        if args.compute_is:
            inception_score = evaluator.compute_inception_score(sample_acts[0])
            print(f"IS_RESULT:{inception_score}")
    except Exception as e:
        print(f"Error computing FID: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
