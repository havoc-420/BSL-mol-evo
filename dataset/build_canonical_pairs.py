#!/usr/bin/env python3
"""
Generic entrypoint for canonical primitive pair construction.

当前实际逻辑复用 `build_canonical_qm9_pairs.py`，但 CLI 入口名改成更贴近
计划文档中的 `build_canonical_pairs.py`，便于后续直接用于任意 item 表。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mol_evo.dataset.build_canonical_qm9_pairs import main


if __name__ == "__main__":
    main()
