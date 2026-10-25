"""Validate the prepared dataset, splits, trained results and final deliverables."""
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from affinity_eda import CFG,DATA,RESULTS,sha256


def main():
    checks=[]
    def check(label,condition):
        if not condition:raise AssertionError(label)
        checks.append(label)
    pairs=pd.read_parquet(DATA/'pairs.parquet')
    molecules=pd.read_parquet(DATA/'molecules.parquet')
    eda=json.loads((RESULTS/'eda_summary.json').read_text())
    baseline=json.loads((RESULTS/'baseline_summary.json').read_text())
    fingerprints=np.load(DATA/'fingerprints.npz')
    X=fingerprints['X']
    check('unique curated molecule-target pairs',not pairs.duplicated(['compound_id','target_chembl_id']).any())
    check('finite regression targets',np.isfinite(pairs.pki).all())
    check('repeated-measurement policy applied',pairs.repeat_range.le(CFG['maximum_repeat_range_pki']).all())
    check('unique molecule identifiers and indices',molecules.compound_id.is_unique and molecules.molecule_index.is_unique)
    check('all pairs refer to prepared structures',set(pairs.compound_id)<=set(molecules.compound_id))
    check('summary counts match tables',len(pairs)==eda['curated_pairs'] and len(molecules)==eda['curated_molecules'])
    check('fingerprint matrix has expected shape',X.shape==(len(molecules),CFG['fingerprint']['n_bits']))
    check('fingerprints contain only binary values',np.isin(X,[0,1]).all())
    check('fingerprint rows have verified molecule identities',np.array_equal(fingerprints['compound_ids'],molecules.compound_id.to_numpy()))
    for split,column in [('random_split','connectivity'),('scaffold_split','scaffold')]:
        blocks={part:set(molecules.loc[molecules[split].eq(part),column]) for part in ['train','validation','test']}
        check(split+' groups never cross partitions',not(blocks['train']&blocks['validation'] or blocks['train']&blocks['test'] or blocks['validation']&blocks['test']))
        mapping=molecules.set_index('compound_id')[split]
        check(split+' pair labels agree with molecule labels',pairs[split].eq(pairs.compound_id.map(mapping)).all())
    check('every supported target satisfies declared size threshold',
          pairs.loc[pairs.benchmark_target].groupby('target_chembl_id').size().ge(CFG['minimum_pairs_per_target']).all())
    for name,filename in [('pairs','pairs.parquet'),('molecules','molecules.parquet'),('fingerprints','fingerprints.npz')]:
        check('baseline used current '+name,baseline['input_sha256'][name]==sha256(DATA/filename))
    detail=pd.read_csv(RESULTS/'metrics_by_target.csv')
    macro=pd.read_csv(RESULTS/'metrics_macro.csv')
    selection=json.loads((RESULTS/'selection.json').read_text())
    validation=detail.loc[detail.part.eq('validation')&detail.model.eq('RF')&detail.experiment.isin(CFG['rf_candidates'])]
    ranking=validation.groupby('experiment').MAE.mean().sort_values()
    check('selection is the lowest validation macro MAE',selection['selected']==ranking.index[0])
    check('no test-based hyperparameter selection claimed',selection['test_used_for_selection'] is False)
    predicted=pd.read_parquet(RESULTS/'test_predictions.parquet')
    for experiment,block in predicted.groupby('experiment'):
        split=block['split'].iloc[0]
        check(experiment+' contains only test molecules',
              molecules.set_index('compound_id').loc[block.compound_id,split].eq('test').all())
    main=predicted.loc[predicted.experiment.eq(baseline['selected'])]
    gold=pairs.set_index(['compound_id','target_chembl_id']).pki
    keys=pd.MultiIndex.from_frame(main[['compound_id','target_chembl_id']])
    check('saved prediction labels exactly match curated pKi',np.allclose(main.pki,gold.loc[keys]))
    by_target=main.groupby('target_chembl_id').apply(lambda b:np.abs(b.pki-b.prediction).mean(),include_groups=False)
    check('published macro MAE recomputed from actual predictions',np.isclose(by_target.mean(),baseline['test_macro_MAE']))
    check('all eligible heads reported',len(by_target)==eda['benchmark_targets'])
    notebook=nbformat.read(ROOT/'EDA.ipynb',as_version=4)
    nbformat.validate(notebook)
    code_cells=[c for c in notebook.cells if c.cell_type=='code']
    check('all notebook code cells were executed',all(c.execution_count is not None for c in code_cells))
    check('notebook has no execution errors',not any(o.output_type=='error' for c in code_cells for o in c.outputs))
    check('notebook embeds generated scientific figures',sum(o.output_type in {'display_data','execute_result'} and 'image/png' in o.get('data',{}) for c in code_cells for o in c.outputs)>=5)
    check('practical-value PDF has exactly one A4 page',len(PdfReader(ROOT/'output/pdf/practical_value.pdf').pages)==1)
    figures=list((RESULTS/'figures').glob('*.png'))
    check('scientific plots are present',len(figures)>=7)
    artifacts=sorted({ROOT/'EDA.ipynb',ROOT/'requirements.txt',ROOT/'config/eda.json',
                      ROOT/'output/pdf/practical_value.pdf',ROOT/'readme.md',ROOT/'data/README.md',
                      *ROOT.glob('scripts/*.py'),*ROOT.glob('src/*.py'),*ROOT.glob('tests/*.py'),
                      *DATA.glob('*'),*RESULTS.glob('*.csv'),*RESULTS.glob('*.json'),*figures})
    artifacts=[p for p in artifacts if p.is_file() and p.name!='manifest.json']
    manifest={'validated_at_utc':datetime.now(timezone.utc).isoformat(),'checks_passed':checks,
              'source':eda['source'],'data_license':'CC-BY-SA-3.0; see data/README.md',
              'artifacts':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in artifacts]}
    (RESULTS/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(f'PASS: {len(checks)} checks; {len(code_cells)} executed cells; {len(artifacts)} checksummed artifacts.')


if __name__=='__main__':main()
