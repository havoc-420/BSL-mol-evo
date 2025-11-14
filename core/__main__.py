#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
允许通过 python -m mol_evo.core 命令行方式运行 evolver 模块
"""

from .evolver import main
import sys

if __name__ == "__main__":
    main()