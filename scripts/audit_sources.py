"""Record source applicability, actual use, access limits and label-scale checks."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from affinity_eda import save_json
from multisource_data import DATA,RESULTS


def main():
    from pki_sources import build_report
    report=build_report()
    licenses={'ChEMBL':'CC BY-SA 3.0; integrated adaptation BY-SA 4.0',
              'BindingDB':'Native data: CC BY; upstream notices retained',
              'PubChem':'Source/depositor terms retained; factual observations only',
              'DrugCentral':'CC BY-SA 4.0'}
    entries=[]
    for row in report['sources']:
        entries.append({**row, 'status':'used_in_pKi_baseline', 'license':licenses[row['source']],
                        'prepared_rows_by_endpoint':{'pKi':row['prepared_observations']}})
    save_json(RESULTS/'source_registry.json',entries)
    estimates=pd.read_parquet(DATA/'source_estimates.parquet')
    keys=['compound_id','target_id','endpoint']
    dc=estimates.loc[estimates.source.eq('DrugCentral'),keys+['label']]
    chembl=estimates.loc[estimates.source.eq('ChEMBL'),keys+['label']]
    overlap=dc.merge(chembl,on=keys,suffixes=('_dc','_chembl'),validate='one_to_one')
    correct=np.abs(overlap.label_dc-overlap.label_chembl)
    incorrect=np.abs((9-np.log10(overlap.label_dc))-overlap.label_chembl)
    audit={'overlapping_exact_structure_target_pairs':len(overlap),
           'median_absolute_difference_if_ACT_VALUE_is_pKi':float(correct.median()),
           'median_absolute_difference_if_misread_as_nM':float(incorrect.median()),
           'fraction_within_0_1_pKi':float(correct.le(.1).mean()),
           'interpretation':'DrugCentral is already on the p-scale; disagreement between experiments still remains.'}
    if not audit['median_absolute_difference_if_ACT_VALUE_is_pKi'] < audit['median_absolute_difference_if_misread_as_nM']:
        raise ValueError('DrugCentral scale sanity check failed')
    save_json(RESULTS/'drugcentral_scale_audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
