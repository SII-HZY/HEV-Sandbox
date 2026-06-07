#!/usr/bin/env python
"""Compatibility wrapper for rebuilding normalized HEV benchmark data.

The maintained implementation is ``scripts/prepare_dataset.py``. This wrapper
keeps the older script name available for users and documentation.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_dataset import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
