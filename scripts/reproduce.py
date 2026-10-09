#!/usr/bin/env python3
"""Run the curve-model study."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compute_market.curves.cli import main
if __name__=='__main__':raise SystemExit(main(['study',*sys.argv[1:]]))
