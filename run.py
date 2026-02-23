import sys, os
base = os.path.dirname(__file__)
site = os.path.join(base, "site-packages")
if os.path.isdir(site):
    sys.path.insert(0, site)
rlt = os.path.join(base, "rlutilities")
if os.path.isdir(rlt):
    sys.path.insert(0, rlt)
rlt = os.path.join(base, "replayprocessing")
if os.path.isdir(rlt):
    sys.path.insert(0, rlt)
from pathlib import Path
from replay_processing.replaytopreds import replay_to_preds

def main():
    # Expect: run.py <replay_filepath> <output_csv_filepath>
    if len(sys.argv) != 3:
        print("Usage: run.py <replay_filepath> <output_csv_filepath>", file=sys.stderr)
        sys.exit(1)

    replay_filepath = sys.argv[1]
    output_csv = sys.argv[2]

    # Normalize paths (optional but nice)
    replay_filepath = str(Path(replay_filepath).resolve())
    output_csv = str(Path(output_csv).resolve())

    try:
        preds_filepath = replay_to_preds(replay_filepath, output_csv)
    except Exception as e:
        # If something explodes, exit non-zero so C++ sees it failed
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Optional: print the path so you can debug from the C++ side
    print(preds_filepath)
    sys.exit(0)


if __name__ == "__main__":
    main()

