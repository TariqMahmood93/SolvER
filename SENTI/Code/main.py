# main.py
import argparse
import sys
from injecting_null import run_null_injection
from SentI import run_imputation

def parse_arg() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Incremental null injection + FAISS-based imputation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--path", default="/content/drive/MyDrive")
    p.add_argument("--datasets", nargs="+", default=["adultsample"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1234])
    p.add_argument(
        "--cum_pcts",
        nargs="+",
        type=float,
        default=[0.05, 0.05, 0.10, 0.20],
        help="Incremental null‐percentages (e.g. 0.05 0.05 → 5% then +5%)"
    )
    p.add_argument("--initial", type=int, default=1000)
    p.add_argument("--step", type=int, default=100)
    p.add_argument(
        "--mode",
        choices=["inject", "SENT-I", "all"],
        default="all",
        help="inject = only create *_nonimputed.csv\n"
             "SENT-I = only run FAISS-based imputation\n"
             "all    = full pipeline",
    )

    args, unknown = p.parse_known_args()
    if unknown:
        print(
            "[info] Ignoring unrecognised args from the host environment:",
            unknown,
            file=sys.stderr,
        )
    return args

def main():
    args = parse_arg()

    # compute true cumulative percentages:
    cumulative = [sum(args.cum_pcts[: i + 1]) for i in range(len(args.cum_pcts))]

    for ds in args.datasets:
        if args.mode in ("inject", "all"):
            run_null_injection(
                startcsv_path=args.path,
                dataset=ds,
                seeds=args.seeds,
                cum_pcts=cumulative,
                initial_size=args.initial,
                step=args.step,
            )
        if args.mode in ("SENT-I", "all"):
            run_imputation(
                startcsv_path=args.path,
                dataset=ds,
                seeds=args.seeds,
                cum_pcts=cumulative,
                initial_size=args.initial,
                step=args.step,
            )

    print("\nAll done.")

if __name__ == "__main__":
    main()
