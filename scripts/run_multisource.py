"""Reproduce the source expansion without changing the original ChEMBL snapshot."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['prepare','train','package','publish','all'],default='all')
    args=parser.parse_args()
    from multisource_data import prepare
    if args.stage in ['prepare','all']:
        print(json.dumps(prepare(),ensure_ascii=False,indent=2),flush=True)
        from fetch_pubchem_metadata import main as fetch_metadata
        from audit_sources import main as audit
        fetch_metadata()
        audit()
    if args.stage in ['train','all']:
        from multisource_models import train
        print(json.dumps(train(),ensure_ascii=False,indent=2),flush=True)
    if args.stage in ['package','all']:
        from multisource_hub import package
        print(json.dumps(package(),ensure_ascii=False,indent=2),flush=True)
    if args.stage=='publish':
        from multisource_hub import publish
        print(json.dumps(publish(),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
