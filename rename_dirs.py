import re
from pathlib import Path
import argparse
import sys

def main():
    ap = argparse.ArgumentParser(description="Rename species folders by removing the leading number and dot.")
    ap.add_argument("--root", default="cropped_data", help="Root directory containing numbered species folders")
    ap.add_argument("--dry-run", action="store_true", help="Print what would change, but do not rename")
    ap.add_argument("--force", action="store_true", help="Overwrite existing target dirs if they exist (use with care)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"[ERR] Root '{root}' not found or not a directory.")
        sys.exit(1)

    # Matches: "001.Black_footed_Albatross" -> "Black_footed_Albatross"
    pat = re.compile(r"^(\d+)\.(.+)$")

    changed = 0
    skipped = 0

    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        m = pat.match(d.name)
        if not m:
            # Not a numbered dir; skip
            skipped += 1
            continue

        new_name = m.group(2)  # keep underscores; just drop the "###."
        target = d.parent / new_name

        if target.exists():
            if args.force:
                if not args.dry_run:
                    # Remove existing empty dir or raise if not empty
                    try:
                        target.rmdir()
                    except OSError:
                        print(f"[ERR] Target exists and is not empty: {target}. Use a different root or resolve manually.")
                        continue
                print(f"[WARN] Target already existed: {target} (will overwrite due to --force)")

            else:
                print(f"[SKIP] Target already exists: {target} (use --force to overwrite)")
                skipped += 1
                continue

        print(f"{d.name}  -->  {new_name}")
        if not args.dry_run:
            d.rename(target)
        changed += 1

    print(f"\nDone. Renamed: {changed}, skipped: {skipped}. {'(dry-run)' if args.dry_run else ''}")

if __name__ == "__main__":
    main()
