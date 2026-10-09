#!/usr/bin/env python3
"""Reproduce the matched futures-only versus mixed-instrument holdout."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from compute_market.curves.system_study import run_instrument_ablation
if __name__=='__main__':run_instrument_ablation(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/study')
