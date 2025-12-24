import subprocess
import os

EXE_PATH = r"D:\python\replay_processing\dist\neurlcar_applet\neurlcar_applet.exe"
REPLAY_PATH = r"D:\python\replay_processing\tests\testreplays\dec2025.replay"
OUTPUT_CSV = r"D:\python\replay_processing\tests\testreplays\dec2025_output.csv"

def main():
    if not os.path.exists(EXE_PATH):
        print(f"EXE not found at: {EXE_PATH}")
        return

    if not os.path.exists(REPLAY_PATH):
        print(f"Replay not found at: {REPLAY_PATH}")
        return

    print(f"Running:\n  {EXE_PATH}\n  with replay:\n  {REPLAY_PATH}\n  output:\n  {OUTPUT_CSV}\n")

    cmd = [EXE_PATH, REPLAY_PATH, OUTPUT_CSV]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("=== Return code ===")
    print(result.returncode)

    print("\n=== STDOUT ===")
    print(result.stdout)

    print("\n=== STDERR ===")
    print(result.stderr)

    if os.path.exists(OUTPUT_CSV):
        print(f"\nOutput CSV created: {OUTPUT_CSV}")
    else:
        print("\nNo output CSV was created.")

if __name__ == "__main__":
    main()
