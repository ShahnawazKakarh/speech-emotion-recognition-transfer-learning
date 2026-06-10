"""
Inspect the Urdu-Sindhi Speech Emotion Corpus feature ZIP files
(Syed et al. 2020, Zenodo DOI: 10.5281/zenodo.3685274).

The Zenodo release contains .mat (MATLAB) feature files only — raw audio
is held back for ethical reasons. This script reveals:
  1. how many files per language
  2. what variables each .mat contains (features + labels?)
  3. shape and dtype of each variable

Usage:
    python scripts/inspect_urdu_sindhi.py /path/to/Feats_-_Urdu.zip
    python scripts/inspect_urdu_sindhi.py /path/to/Feats_-_Sindhi.zip

If files are extracted to a folder, pass the folder instead:
    python scripts/inspect_urdu_sindhi.py /path/to/Feats_-_Urdu/

Author: Muhammad Shahnawaz Khan
"""

from __future__ import annotations

import io
import sys
import zipfile
from collections import Counter
from pathlib import Path

try:
    import scipy.io
except ImportError:
    print("ERROR: scipy is required. Install with:")
    print("    pip install scipy")
    sys.exit(1)


def inspect_mat_data(data: dict, source: str) -> None:
    """Print a structured summary of a loaded .mat dict."""
    user_keys = [k for k in data.keys() if not k.startswith("__")]
    print(f"  Variables: {user_keys}")
    for k in user_keys:
        arr = data[k]
        try:
            shape = arr.shape
            dtype = arr.dtype
            print(f"    {k:20s} shape={str(shape):20s} dtype={dtype}")
            # Show first row / first element for context
            if arr.size > 0 and arr.ndim >= 1:
                first = arr.flat[0]
                if isinstance(first, (int, float)) or hasattr(first, "item"):
                    print(f"      first values: {arr.flat[0]} ... {arr.flat[min(arr.size-1, 4)]}")
                else:
                    print(f"      first value (truncated): {str(first)[:80]}")
        except AttributeError:
            print(f"    {k:20s} (non-array): {str(arr)[:80]}")


def inspect_zip(zip_path: Path) -> None:
    """Inspect a .zip containing .mat files."""
    print(f"\n{'='*72}")
    print(f"ZIP: {zip_path}")
    print(f"{'='*72}")

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

        # File-type breakdown
        suffixes = Counter(Path(n).suffix.lower() for n in names if n.strip())
        print(f"Total entries: {len(names)}")
        print(f"By extension:  {dict(suffixes)}")

        # Filename pattern inspection (look for emotion / speaker hints)
        mats = [n for n in names if n.lower().endswith(".mat")]
        print(f".mat files:    {len(mats)}")
        if mats:
            print(f"First 5 names: {mats[:5]}")
            print(f"Last 5 names:  {mats[-5:]}")

            # Inspect the first .mat
            print(f"\n--- Loading {mats[0]} ---")
            with z.open(mats[0]) as f:
                buf = io.BytesIO(f.read())
            data = scipy.io.loadmat(buf)
            inspect_mat_data(data, mats[0])

            # If there are multiple .mat files, peek at the second too
            if len(mats) > 1:
                print(f"\n--- Loading {mats[1]} ---")
                with z.open(mats[1]) as f:
                    buf = io.BytesIO(f.read())
                data = scipy.io.loadmat(buf)
                inspect_mat_data(data, mats[1])


def inspect_dir(dir_path: Path) -> None:
    """Inspect a directory of already-extracted .mat files."""
    print(f"\n{'='*72}")
    print(f"DIR: {dir_path}")
    print(f"{'='*72}")
    mats = sorted(dir_path.rglob("*.mat"))
    print(f"Total .mat files: {len(mats)}")
    if not mats:
        print("No .mat files found.")
        return
    print(f"First 5 names: {[m.name for m in mats[:5]]}")
    print(f"\n--- Loading {mats[0].name} ---")
    inspect_mat_data(scipy.io.loadmat(mats[0]), mats[0].name)
    if len(mats) > 1:
        print(f"\n--- Loading {mats[1].name} ---")
        inspect_mat_data(scipy.io.loadmat(mats[1]), mats[1].name)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Path not found: {path}")
        return 1
    if path.is_dir():
        inspect_dir(path)
    elif path.suffix.lower() == ".zip":
        inspect_zip(path)
    else:
        print(f"Unsupported input: {path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
