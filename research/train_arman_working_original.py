#!/usr/bin/env python3
"""
ArMan Fine-tuning Pipeline — v2
================================
Same checkpoint/metrics/plotting machinery as the original script, plus the
fine-tuning fixes recommended by kraken's own docs for adapting a pretrained
model (Muharaf) onto a new, fairly dissimilar dataset (ArMan):

  - --resize defaults to "new" instead of "union"
      kraken docs: "When fine-tuning, it is recommended to use new mode not
      union, as the network will rapidly unlearn missing labels in the new
      dataset."
  - --warmup added (was completely missing before)
      kraken docs: "It is necessary to use learning rate warmup in addition
      to freezing the backbone ... to have the model converge during
      fine-tuning." Recommended to match freeze-backbone duration for 1-2
      epochs when the new dataset is "fairly dissimilar" from the base model.
  - --freeze-backbone / --warmup now auto-default to exactly one epoch's
      worth of steps (computed from the train manifest + batch size) if you
      don't explicitly pass them, instead of an arbitrary step count that
      may land mid-epoch.
  - --augment added (default ON, can disable with --no-augment).
  - is_best detection fixed: now also looks for the early_stopping counter
      resetting to 0/N, which is how kraken actually signals "this epoch
      was a new best" in its progress output (the old "saving best" /
      "new best" text regex never matched real kraken output, so every
      logged row showed best=False even when checkpoints WERE being saved).
"""

import subprocess, os, sys, re, json, csv, time, math, argparse, random, statistics
from pathlib import Path
from datetime import datetime

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        required=True)
    p.add_argument("--dataset",      required=True)
    p.add_argument("--output",       default="./arman_training")
    p.add_argument("--device",       default="cpu")
    p.add_argument("--epochs",       type=int,   default=50,
                   help="Passed through to ketos, but note: with the default "
                        "--quit early, ketos ignores this as a hard cap and "
                        "trains until --lag epochs pass with no improvement.")
    p.add_argument("--batch",        type=int,   default=8)
    p.add_argument("--lrate",        type=float, default=5e-5)
    p.add_argument("--lag",          type=int,   default=10)
    p.add_argument("--freeze",       type=int,   default=None,
                   help="Steps to freeze the backbone for. If omitted, "
                        "auto-set to exactly 1 epoch's worth of steps "
                        "(ceil(num_train_lines / batch)).")
    p.add_argument("--warmup",       type=int,   default=None,
                   help="LR warmup steps. If omitted, defaults to the same "
                        "value as --freeze (recommended: warm up over the "
                        "same span you unfreeze into).")
    p.add_argument("--augment",      action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Enable kraken's data augmentation (default: on). "
                        "Disable with --no-augment.")
    p.add_argument("--schedule",     default="cosine",
                   choices=["cosine","1cycle","exponential","reduceonplateau"])
    p.add_argument("--resize",       default="new", choices=["union","new","both","fail"],
                   help="kraken recommends 'new' (or 'both') over 'union' "
                        "when fine-tuning onto a different dataset, so the "
                        "model isn't carrying dead output classes from the "
                        "base model's codec. Default changed to 'new'.")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--runs",         type=int,   default=1)
    p.add_argument("--skip_compile", action="store_true")
    return p.parse_args()

def setup_dirs(output):
    dirs = {
        "root"        : output,
        "arrows"      : os.path.join(output, "arrows"),
        "checkpoints" : os.path.join(output, "checkpoints"),
        "logs"        : os.path.join(output, "logs"),
        "plots"       : os.path.join(output, "plots"),
        "metrics"     : os.path.join(output, "metrics"),
        "test_results": os.path.join(output, "test_results"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    print(f"  Random seed set to {seed}")

def count_manifest_lines(manifest_path):
    """Count non-empty lines in a manifest file (= number of training images)."""
    with open(manifest_path) as f:
        return sum(1 for l in f if l.strip())

def resolve_freeze_warmup(args, train_manifest):
    """Auto-default --freeze and --warmup to exactly 1 epoch's worth of steps
    if the user didn't explicitly set them."""
    num_lines = count_manifest_lines(train_manifest)
    steps_per_epoch = math.ceil(num_lines / args.batch)

    if args.freeze is None:
        args.freeze = steps_per_epoch
        print(f"  [AUTO] --freeze not set -> using 1 epoch ({steps_per_epoch} steps, "
              f"{num_lines} lines / batch {args.batch})")
    if args.warmup is None:
        args.warmup = args.freeze
        print(f"  [AUTO] --warmup not set -> matching --freeze ({args.warmup} steps)")

    return args

def compile_arrow(manifest_path, arrow_path, label):
    if os.path.exists(arrow_path):
        print(f"  [SKIP] {label} arrow exists: {arrow_path}")
        return True
    with open(manifest_path) as f:
        image_paths = [l.strip() for l in f if l.strip()]
    if not image_paths:
        print(f"  [ERROR] manifest empty: {manifest_path}")
        return False
    print(f"  Compiling {label} ({len(image_paths)} images) -> {arrow_path}")
    cmd = ["ketos", "compile", "-f", "path", "-o", arrow_path] + image_paths
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [ERROR] compile failed for {label}")
        return False
    print(f"  Done: {arrow_path}")
    return True

def create_arrow_manifest(arrow_path, manifest_path):
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(arrow_path + "\n")
    return manifest_path

# ===========================================================================
# Metric parsing – now stores accuracy values separately and computes error rates
# ===========================================================================
PATTERNS = {
    "train_loss" : [
        re.compile(r'train_loss_epoch[\s:]+([\d.]+)'),
        re.compile(r'train_loss_step[\s:]+([\d.]+)'),
        re.compile(r'[Tt]rain(?:ing)?\s+loss[\s:]+([\d.]+)'),
        re.compile(r'loss[\s:]+([\d.]+)'),
    ],
    "val_loss"   : [
        re.compile(r'val_loss[\s:]+([\d.]+)'),
        re.compile(r'[Vv]al(?:idation)?\s+loss[\s:]+([\d.]+)'),
    ],
    "val_accuracy" : [
        re.compile(r'val_accuracy[\s:]+([\d.]+)'),
    ],
    "val_word_accuracy" : [
        re.compile(r'val_word_accuracy[\s:]+([\d.]+)'),
    ],
    # Fallback for older Kraken versions that directly print CER/WER
    "cer_raw" : [
        re.compile(r'[Cc][Ee][Rr][\s:]+([\d.]+)'),
    ],
    "wer_raw" : [
        re.compile(r'[Ww][Ee][Rr][\s:]+([\d.]+)'),
    ],
    "is_best"    : [
        # Old text-based markers (kept as a fallback, in case a kraken
        # version actually prints them).
        re.compile(r'[Bb]est\s+model|saving\s+best|new\s+best', re.I),
        # Real signal: kraken's early_stopping counter resets to 0/N exactly
        # when the current epoch is a new best (i.e. it just saved
        # *_best.mlmodel). This is what actually appears in the log.
        re.compile(r'early_stopping:\s*0/\d+'),
    ],
    "checkpoint" : [
        re.compile(r'[Ss]av(?:ing|ed)\s+(?:checkpoint|model)[\s:]+(\S+\.(?:mlmodel|ckpt))'),
    ],
}

def parse_line(line, current):
    """Parse metrics from a line."""
    for key, patterns in PATTERNS.items():
        if current.get(key) is not None and key not in ("is_best","checkpoint"):
            continue
        for pat in patterns:
            m = pat.search(line)
            if m:
                if key == "is_best":
                    current["is_best"] = True
                elif key == "checkpoint":
                    current["checkpoint"] = m.group(1)
                else:
                    try:
                        current[key] = float(m.group(1))
                    except ValueError:
                        pass
                break

# ===========================================================================
# Training
# ===========================================================================
def run_training(args, dirs, train_arrow_manifest, val_arrow_manifest, run_idx, seed):
    run_id      = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run{run_idx}"
    ckpt_prefix = os.path.join(dirs["checkpoints"], f"arman_run{run_idx}")
    log_txt     = os.path.join(dirs["logs"], f"training_{run_id}.log")

    cmd = [
        "ketos", "--device", args.device,
        "train",
        "--load",            args.model,
        "--resize",          args.resize,
        "-f",                "binary",
        "-t",                train_arrow_manifest,
        "-e",                val_arrow_manifest,
        "--logger",          "tensorboard",
        "--log-dir",         dirs["logs"],
        "-o",                ckpt_prefix,
        "--schedule",        args.schedule,
        "--epochs",          str(args.epochs),
        "--lag",             str(args.lag),
        "--lrate",           str(args.lrate),
        "--freeze-backbone", str(args.freeze),
        "--warmup",          str(args.warmup),
        "--batch-size",      str(args.batch),
    ]
    if args.augment:
        cmd.append("--augment")

    print(f"\n{'='*60}")
    print(f"  Run {run_idx}  [{run_id}]  seed={seed}")
    print(f"  " + " \\\n    ".join(cmd))
    print(f"{'='*60}\n")

    cfg = {
        "run_id": run_id, "run_idx": run_idx, "seed": seed,
        "model": args.model, "dataset": args.dataset,
        "device": args.device, "epochs": args.epochs,
        "batch_size": args.batch, "lrate": args.lrate,
        "schedule": args.schedule, "freeze": args.freeze,
        "warmup": args.warmup, "augment": args.augment,
        "lag": args.lag, "resize": args.resize,
        "train_manifest": train_arrow_manifest,
        "val_manifest": val_arrow_manifest,
        "command": cmd,
    }
    with open(os.path.join(dirs["metrics"], f"config_{run_id}.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    all_epochs   = []
    current      = {"epoch": None, "train_loss": None, "val_loss": None,
                    "val_accuracy": None, "val_word_accuracy": None,
                    "cer_raw": None, "wer_raw": None,
                    "is_best": False, "checkpoint": None}
    start_time   = time.time()
    epoch_counter = 0

    metrics_csv  = os.path.join(dirs["metrics"], f"metrics_{run_id}.csv")
    metrics_json = os.path.join(dirs["metrics"], f"metrics_{run_id}.json")
    csv_fields   = ["epoch","train_loss","val_loss","cer","wer",
                    "is_best","checkpoint","elapsed_sec"]

    with open(log_txt, "w", encoding="utf-8") as logf, \
         open(metrics_csv, "w", newline="", encoding="utf-8") as csvf:

        writer = csv.DictWriter(csvf, fieldnames=csv_fields)
        writer.writeheader()

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            print(line)
            logf.write(raw_line); logf.flush()

            parse_line(line, current)

            if "train_loss_epoch" in line and current.get("train_loss") is not None:
                current["epoch"] = epoch_counter
                epoch_counter += 1

                # Compute error rates from accuracy if available, else fallback to raw
                cer_val = None
                wer_val = None
                if current.get("val_accuracy") is not None:
                    cer_val = round(1.0 - current["val_accuracy"], 6)
                elif current.get("cer_raw") is not None:
                    cer_val = current["cer_raw"]  # already an error rate

                if current.get("val_word_accuracy") is not None:
                    wer_val = round(1.0 - current["val_word_accuracy"], 6)
                elif current.get("wer_raw") is not None:
                    wer_val = current["wer_raw"]

                row = {
                    "epoch"      : current["epoch"],
                    "train_loss" : current["train_loss"],
                    "val_loss"   : current["val_loss"],
                    "cer"        : cer_val,
                    "wer"        : wer_val,
                    "is_best"    : current["is_best"],
                    "checkpoint" : current["checkpoint"],
                    "elapsed_sec": round(time.time() - start_time, 1),
                }
                all_epochs.append(row)
                writer.writerow(row)
                csvf.flush()

                print(f"  [SAVED] Epoch {row['epoch']} -> "
                      f"train_loss={row['train_loss']:.3f}, "
                      f"cer={row['cer']}, wer={row['wer']}, "
                      f"best={row['is_best']}")

                # Reset accumulator
                current = {"epoch": None, "train_loss": None, "val_loss": None,
                           "val_accuracy": None, "val_word_accuracy": None,
                           "cer_raw": None, "wer_raw": None,
                           "is_best": False, "checkpoint": None}

        proc.wait()

    if current.get("train_loss") is not None and current.get("epoch") is None:
        current["epoch"] = epoch_counter
        # compute cer/wer as above
        cer_val = None
        wer_val = None
        if current.get("val_accuracy") is not None:
            cer_val = round(1.0 - current["val_accuracy"], 6)
        elif current.get("cer_raw") is not None:
            cer_val = current["cer_raw"]
        if current.get("val_word_accuracy") is not None:
            wer_val = round(1.0 - current["val_word_accuracy"], 6)
        elif current.get("wer_raw") is not None:
            wer_val = current["wer_raw"]
        row = {
            "epoch"      : current["epoch"],
            "train_loss" : current["train_loss"],
            "val_loss"   : current["val_loss"],
            "cer"        : cer_val,
            "wer"        : wer_val,
            "is_best"    : current["is_best"],
            "checkpoint" : current["checkpoint"],
            "elapsed_sec": round(time.time() - start_time, 1),
        }
        all_epochs.append(row)
        writer.writerow(row)

    if not all_epochs:
        print(f"\n[ERROR] No epoch data captured for {run_id}!")
        print("        Check that 'train_loss_epoch' appears in Kraken output.")
        sys.exit(1)

    with open(metrics_json, "w") as f:
        json.dump({"run_id": run_id, "config": cfg, "epochs": all_epochs}, f, indent=2)

    print(f"\n  ✅ Metrics CSV  -> {metrics_csv}  ({len(all_epochs)} epochs)")
    print(f"  ✅ Metrics JSON -> {metrics_json}")
    print(f"  ✅ Full log     -> {log_txt}")

    return run_id, all_epochs, cfg

def find_best_model(ckpt_dir, run_idx):
    prefix   = os.path.join(ckpt_dir, f"arman_run{run_idx}")
    best_path = f"{prefix}_best.mlmodel"
    if os.path.exists(best_path):
        return best_path
    mlmodels = list(Path(ckpt_dir).glob("*.mlmodel"))
    if mlmodels:
        newest = max(mlmodels, key=lambda p: p.stat().st_mtime)
        print(f"  [WARN] Expected {best_path}, using newest: {newest}")
        return str(newest)
    print(f"  [WARN] No .mlmodel found in {ckpt_dir}")
    return None

def run_test_evaluation(best_model_path, test_arrow_manifest, dirs, run_id, device):
    print(f"\n{'='*60}")
    print(f"  TEST SET EVALUATION  (paper results)")
    print(f"  Model : {best_model_path}")
    print(f"  Test manifest : {test_arrow_manifest}")
    print(f"{'='*60}")

    if not os.path.exists(best_model_path):
        print(f"  [ERROR] best model not found: {best_model_path}")
        return None

    # Check if test arrow exists
    if not os.path.exists(test_arrow_manifest):
        print(f"  [ERROR] Test manifest not found: {test_arrow_manifest}")
        return None
    with open(test_arrow_manifest, 'r') as f:
        test_arrow_path = f.read().strip()
    if not os.path.exists(test_arrow_path):
        print(f"  [ERROR] Test arrow file not found: {test_arrow_path}")
        print("          Please run compilation without --skip_compile or copy test.arrow manually.")
        return None

    cmd = [
        "ketos", "--device", device,
        "test",
        "-m", best_model_path,
        "-f", "binary",
        "-e", test_arrow_manifest,
    ]

    print(f"  Running: {' '.join(cmd)}")

    out_path = os.path.join(dirs["test_results"], f"test_results_{run_id}.txt")
    result = subprocess.run(cmd, capture_output=True, text=True)

    with open(out_path, "w", encoding="utf-8") as outf:
        outf.write(result.stdout)
        outf.write(result.stderr)
    print(result.stdout)

    test_metrics = {
        "model": best_model_path, "test_manifest": test_arrow_manifest,
        "run_id": run_id, "raw_output": result.stdout,
    }

    # Parse test output – look for val_accuracy or val_word_accuracy, else fallback
    acc, word_acc = None, None
    cer_raw, wer_raw = None, None
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        m = re.search(r'val_accuracy[\s:]+([\d.]+)', line)
        if m:
            acc = float(m.group(1))
        m = re.search(r'val_word_accuracy[\s:]+([\d.]+)', line)
        if m:
            word_acc = float(m.group(1))
        # fallback
        m = re.search(r'[Cc][Ee][Rr][\s:]+([\d.]+)', line)
        if m:
            cer_raw = float(m.group(1))
        m = re.search(r'[Ww][Ee][Rr][\s:]+([\d.]+)', line)
        if m:
            wer_raw = float(m.group(1))

    if acc is not None:
        test_metrics["cer"] = round(1.0 - acc, 6)
    elif cer_raw is not None:
        test_metrics["cer"] = cer_raw

    if word_acc is not None:
        test_metrics["wer"] = round(1.0 - word_acc, 6)
    elif wer_raw is not None:
        test_metrics["wer"] = wer_raw

    json_path = os.path.join(dirs["test_results"], f"test_results_{run_id}.json")
    with open(json_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    csv_path = os.path.join(dirs["test_results"], "test_results_all.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id","cer","wer","checkpoint"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "run_id": run_id,
            "cer": test_metrics.get("cer"),
            "wer": test_metrics.get("wer"),
            "checkpoint": best_model_path,
        })

    print(f"\n  ✅ TEST RESULTS (report these in the paper):")
    print(f"     CER : {test_metrics.get('cer', '--')}")
    print(f"     WER : {test_metrics.get('wer', '--')}")
    print(f"  JSON  -> {json_path}")
    print(f"  CSV   -> {csv_path}")
    print(f"  Raw   -> {out_path}")

    return test_metrics

# ----------------------------------------------------------------------------
# The rest (make_plots, save_multirun_summary, main) remain unchanged except
# for minor improvements to arrow-checking, plus auto freeze/warmup resolution.
# ----------------------------------------------------------------------------

def make_plots(all_epochs, dirs, run_id, test_result=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  [WARN] matplotlib not installed — pip install matplotlib")
        return

    if not all_epochs:
        print("  [WARN] No epoch data — skipping plots")
        return

    epochs     = [r["epoch"]      for r in all_epochs]
    train_loss = [r["train_loss"] for r in all_epochs]
    val_loss   = [r["val_loss"]   for r in all_epochs]
    cer        = [r["cer"]        for r in all_epochs]
    wer        = [r["wer"]        for r in all_epochs]
    best_eps   = [r["epoch"] for r in all_epochs if r.get("is_best")]

    has_data = any(v is not None for v in cer + wer + train_loss + val_loss)
    if not has_data:
        print("  [WARN] All metric values are null — plots skipped.")
        return

    def _ax(ax, series_list, labels, colors, title, ylabel):
        for y, label, color in zip(series_list, labels, colors):
            eps  = [e for e,v in zip(epochs, y) if v is not None and e is not None]
            vals = [v for v in y if v is not None]
            if vals:
                ax.plot(eps, vals, label=label, color=color, linewidth=1.8)
                mi = vals.index(min(vals))
                ax.scatter([eps[mi]], [vals[mi]], color="#ffd040", zorder=5, s=40)
        for be in best_eps:
            ax.axvline(be, color="#ffd040", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"ArMan Training Curves — {run_id}", fontsize=13, fontweight="bold")

    _ax(axes[0,0], [train_loss, val_loss], ["Train","Val"],
        ["#5b9bd5","#cf6679"], "Loss", "Loss")
    _ax(axes[0,1], [cer], ["CER"], ["#4caf7d"], "Character Error Rate", "CER %")
    _ax(axes[1,0], [wer], ["WER"], ["#d4a843"], "Word Error Rate", "WER %")
    _ax(axes[1,1], [cer, wer], ["CER","WER"],
        ["#4caf7d","#d4a843"], "CER & WER", "%")

    if test_result and test_result.get("cer") is not None:
        fig.text(0.5, 0.02,
                 f"Test CER: {test_result['cer']:.2f}%  |  Test WER: {test_result.get('wer','--'):.2f}%",
                 ha="center", fontsize=11, fontweight="bold", color="#333333",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff9c4", edgecolor="#ffd040"))
        fig.subplots_adjust(bottom=0.08)

    plt.tight_layout()
    out = os.path.join(dirs["plots"], f"curves_{run_id}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {out}")

    for key, vals, color, ylabel in [
        ("train_loss", train_loss, "#5b9bd5", "Loss"),
        ("val_loss",   val_loss,   "#cf6679", "Loss"),
        ("cer",        cer,        "#4caf7d", "CER %"),
        ("wer",        wer,        "#d4a843", "WER %"),
    ]:
        eps  = [e for e,v in zip(epochs, vals) if v is not None and e is not None]
        vv   = [v for v in vals if v is not None]
        if not vv: continue
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.plot(eps, vv, color=color, linewidth=2)
        for be in best_eps:
            ax2.axvline(be, color="#ffd040", linewidth=1, linestyle="--", alpha=0.7)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel(ylabel)
        ax2.set_title(key.upper(), fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.25)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["plots"], f"{key}_{run_id}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()

def save_multirun_summary(all_test_results, dirs):
    if len(all_test_results) < 2:
        return
    cers = [r["cer"] for r in all_test_results if r and r.get("cer") is not None]
    wers = [r["wer"] for r in all_test_results if r and r.get("wer") is not None]
    if not cers:
        return
    summary = {
        "n_runs"   : len(all_test_results),
        "cer_mean" : round(statistics.mean(cers), 4),
        "cer_std"  : round(statistics.stdev(cers), 4) if len(cers) > 1 else 0,
        "cer_min"  : round(min(cers), 4),
        "cer_max"  : round(max(cers), 4),
        "wer_mean" : round(statistics.mean(wers), 4) if wers else None,
        "wer_std"  : round(statistics.stdev(wers), 4) if len(wers) > 1 else 0,
        "per_run"  : all_test_results,
    }
    path = os.path.join(dirs["test_results"], "multirun_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"  MULTI-RUN SUMMARY ({summary['n_runs']} runs)")
    print(f"  CER: {summary['cer_mean']} ± {summary['cer_std']}")
    print(f"  WER: {summary['wer_mean']} ± {summary['wer_std']}")
    print(f"  Saved -> {path}")
    print(f"{'='*60}")

def main():
    args = get_args()
    dirs = setup_dirs(args.output)

    dataset         = args.dataset
    train_manifest  = os.path.join(dataset, "train_manifest.txt")
    val_manifest    = os.path.join(dataset, "val_manifest.txt")
    test_manifest   = os.path.join(dataset, "test_manifest.txt")

    for p in [train_manifest, val_manifest]:
        if not os.path.exists(p):
            print(f"[ERROR] missing: {p}"); sys.exit(1)
    if not os.path.exists(args.model):
        print(f"[ERROR] model not found: {args.model}"); sys.exit(1)

    has_test = os.path.exists(test_manifest)
    if not has_test:
        print(f"  [WARN] No test_manifest.txt found — test evaluation will be skipped")

    # Auto-resolve --freeze / --warmup to 1 epoch's worth of steps if the
    # user didn't explicitly set them (uses the raw image manifest, since
    # that 1:1 maps to training examples regardless of arrow compilation).
    args = resolve_freeze_warmup(args, train_manifest)

    train_arrow = os.path.join(dirs["arrows"], "train.arrow")
    val_arrow   = os.path.join(dirs["arrows"], "val.arrow")
    test_arrow  = os.path.join(dirs["arrows"], "test.arrow")

    if not args.skip_compile:
        if not compile_arrow(train_manifest, train_arrow, "train"): sys.exit(1)
        if not compile_arrow(val_manifest,   val_arrow,   "val"):   sys.exit(1)
        if has_test:
            if not compile_arrow(test_manifest, test_arrow, "test"): sys.exit(1)
    else:
        print("  [SKIP] Arrow compilation skipped")
        # Check that required arrow files exist
        for f in [train_arrow, val_arrow]:
            if not os.path.exists(f):
                print(f"  [ERROR] Arrow file missing: {f}. Cannot run with --skip_compile.")
                sys.exit(1)
        if has_test and not os.path.exists(test_arrow):
            print(f"  [WARN] Test arrow missing: {test_arrow}. Test evaluation will be skipped.")

    train_arrow_manifest = os.path.join(dirs["arrows"], "train_arrow_manifest.txt")
    val_arrow_manifest   = os.path.join(dirs["arrows"], "val_arrow_manifest.txt")
    test_arrow_manifest  = os.path.join(dirs["arrows"], "test_arrow_manifest.txt")

    create_arrow_manifest(train_arrow, train_arrow_manifest)
    create_arrow_manifest(val_arrow,   val_arrow_manifest)
    if has_test and os.path.exists(test_arrow):
        create_arrow_manifest(test_arrow, test_arrow_manifest)
    elif has_test:
        # create manifest anyway but it will be missing; run_test_evaluation will catch it
        create_arrow_manifest(test_arrow, test_arrow_manifest)

    print(f"  Arrow manifests created:")
    print(f"    Train: {train_arrow_manifest}")
    print(f"    Val:   {val_arrow_manifest}")
    if has_test:
        print(f"    Test:  {test_arrow_manifest}")

    print(f"\n  Fine-tuning config:")
    print(f"    resize          : {args.resize}")
    print(f"    freeze-backbone : {args.freeze} steps")
    print(f"    warmup          : {args.warmup} steps")
    print(f"    augment         : {args.augment}")
    print(f"    lrate           : {args.lrate}")

    all_test_results = []

    for run_idx in range(1, args.runs + 1):
        seed = args.seed + run_idx - 1
        set_seed(seed)

        print(f"\n{'#'*60}")
        print(f"#  RUN {run_idx} of {args.runs}   seed={seed}")
        print(f"{'#'*60}")

        run_id, all_epochs, cfg = run_training(
            args, dirs, train_arrow_manifest, val_arrow_manifest, run_idx, seed
        )

        make_plots(all_epochs, dirs, run_id)

        if has_test and os.path.exists(test_arrow_manifest):
            best_model = find_best_model(dirs["checkpoints"], run_idx)
            if best_model:
                test_result = run_test_evaluation(
                    best_model, test_arrow_manifest, dirs, run_id, args.device
                )
                if test_result:
                    all_test_results.append(test_result)
                    make_plots(all_epochs, dirs, run_id, test_result)
        else:
            print("\n  [SKIP] Test evaluation skipped (no test manifest or missing test.arrow)")

    if args.runs > 1:
        save_multirun_summary(all_test_results, dirs)

    print(f"\n{'='*60}")
    print(f"  ALL DONE")
    print(f"  {args.output}/")
    print(f"  ├── arrows/           compiled .arrow + manifests")
    print(f"  ├── checkpoints/      best + all epoch checkpoints")
    print(f"  ├── logs/             stdout logs + TensorBoard events")
    print(f"  ├── metrics/          config JSON + per-epoch CSV + JSON")
    print(f"  ├── plots/            training curves PNG (paper-ready)")
    print(f"  └── test_results/     <- REPORT THESE IN THE PAPER")
    print(f"\n  TensorBoard (ground truth metrics):")
    print(f"    tensorboard --logdir {dirs['logs']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()