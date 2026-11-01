import argparse
import glob
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine all evaluation CSV files into a single CSV"
    )
    parser.add_argument(
        "--path", required=True,
        help="Directory containing evaluation CSV files"
    )
    parser.add_argument(
        "--output", default="final_evaluation_all.csv",
        help="Filename for the combined output CSV"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.path
    output_file = args.output

    # Pattern matches all *_evaluation.csv excluding final_evaluation_ prefixed
    pattern = os.path.join(path, "*_evaluation.csv")
    files = [f for f in glob.glob(pattern)
             if not os.path.basename(f).startswith("final_evaluation_")]

    if not files:
        print(f"No evaluation CSV files found in directory: {path}")
        return

    print(f"Found {len(files)} files. Combining...", flush=True)
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Warning: could not read {f}: {e}", flush=True)

    if not df_list:
        print("No valid CSV files to combine.")
        return

    combined_df = pd.concat(df_list, ignore_index=True)
    output_path = os.path.join(path, output_file)
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined evaluation CSV: {output_path}", flush=True)

if __name__ == "__main__":
    main()
