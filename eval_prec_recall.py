#!/usr/bin/env python3
"""
Standalone Precision & Recall evaluation.

Requires per-sample image npz files (not stats-only), since P/R needs
per-sample Inception pool_3 features for kNN manifold estimation.

Usage:
    python eval_prec_recall.py \
        --ref_batch /path/to/ref_images.npz \
        --sample_batch /path/to/sample_images.npz \
        [--batch_size 64]

Output (parseable):
    PREC_RESULT:<precision>
    REC_RESULT:<recall>
"""
import sys
import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf

from evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Compute Improved Precision & Recall")
    parser.add_argument("--ref_batch", type=str, required=True,
                        help="Path to reference image npz (must contain 'arr_0' with uint8 NHWC images)")
    parser.add_argument("--sample_batch", type=str, required=True,
                        help="Path to sample image npz (must contain 'arr_0' with uint8 NHWC images)")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    for path, name in [(args.ref_batch, "ref_batch"), (args.sample_batch, "sample_batch")]:
        if not os.path.exists(path):
            print(f"Error: {name} not found: {path}", file=sys.stderr)
            sys.exit(1)
        with np.load(path) as obj:
            if "arr_0" not in obj.files:
                print(f"Error: {name} must contain 'arr_0' (raw images). "
                      f"Got keys: {obj.files}. Stats-only npz not supported for P/R.",
                      file=sys.stderr)
                sys.exit(1)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    try:
        evaluator = Evaluator(tf.Session(config=config), batch_size=args.batch_size)
        evaluator.warmup()
        evaluator.manifold_estimator.warmup()

        print("Computing reference activations...")
        ref_acts = evaluator.read_activations(args.ref_batch)
        print(f"  ref pool features: {ref_acts[0].shape}")

        print("Computing sample activations...")
        sample_acts = evaluator.read_activations(args.sample_batch)
        print(f"  sample pool features: {sample_acts[0].shape}")

        print("Computing Precision & Recall (kNN manifold)...")
        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])

        print(f"\nPrecision: {prec:.6f}")
        print(f"Recall:    {recall:.6f}")
        print(f"PREC_RESULT:{prec}")
        print(f"REC_RESULT:{recall}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
