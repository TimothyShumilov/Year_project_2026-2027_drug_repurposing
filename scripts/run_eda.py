"""Run the EDA and actual RF experiments with deterministic reusable artifacts."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from affinity_eda import prepare_data,get_fingerprints,fingerprint_audit,fingerprint_sensitivity,approved_coverage,train_benchmark,augmentation_invariance,explain_baseline,RESULTS,CFG,_train_target


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['prepare','timing','baseline','figures','all'],default='all')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    pairs,molecules,summary=prepare_data(force=args.force and args.stage=='prepare')
    X=get_fingerprints(molecules)
    fingerprint_audit(molecules,X)
    approved_coverage(pairs)
    if args.stage=='prepare':
        print(json.dumps({k:summary[k] for k in ['curated_pairs','curated_molecules','benchmark_targets','benchmark_pairs']},indent=2));return
    if args.stage=='timing':
        dataset=pairs.loc[pairs.benchmark_target]
        counts=dataset.groupby('target_chembl_id').size().sort_values(ascending=False)
        selected=[counts.index[0],counts.index[len(counts)//2],counts.index[-1]]
        times=[]
        for target in selected:
            block=dataset.loc[dataset.target_chembl_id.eq(target)]
            start=time.perf_counter()
            result=_train_target(target,block,X,CFG['rf_candidates']['RF_leaf1'],'scaffold_split',42)
            record={'target':target,'pairs':len(block),'seconds':time.perf_counter()-start}
            times.append(record);print(json.dumps(record),flush=True)
        print(json.dumps({'measurements':times,'benchmark_targets':len(counts)},indent=2));return
    baseline=train_benchmark(pairs,molecules,X,force=args.force)
    fingerprint_sensitivity(pairs,molecules,X,baseline)
    augment=augmentation_invariance(molecules)
    explain_baseline(pairs,molecules,X,baseline)
    if args.stage in ['figures','all']:
        from eda_figures import generate_figures
        print(generate_figures(pairs,molecules,summary,baseline))
    print(json.dumps(baseline,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
