"""Verify scientific invariants, actual predictions and the publication package."""
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from huggingface_hub import DatasetCard

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from affinity_eda import sha256,save_json
from multisource_data import DATA,RESULTS
from multisource_models import original_keys,CONFIG
from multisource_hub import STAGING


def main():
    passed=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        passed.append(name)
    pairs=pd.read_parquet(DATA/'pairs.parquet')
    mols=pd.read_parquet(DATA/'molecules.parquet')
    observations=pd.read_parquet(DATA/'observations.parquet')
    check('only three allowed sources contribute observations',
          set(observations.source)=={'ChEMBL','BindingDB','DrugCentral'})
    check('PubChem labels and provenance are absent from curated data',
          set(pairs.source)==set(CONFIG['included_sources']) and
          all(set(sources)<=set(CONFIG['included_sources']) for sources in pairs.sources) and
          not observations.upstream_source.fillna('').str.contains('PubChem',case=False).any())
    estimates=pd.read_parquet(DATA/'source_estimates.parquet')
    check('source estimates cannot reintroduce PubChem',set(estimates.source)==set(CONFIG['included_sources']))
    check('one final pair per structure target endpoint',not pairs.duplicated(['compound_id','target_id','endpoint']).any())
    check('finite labels',np.isfinite(pairs.label).all())
    check('all pairs have prepared structures',set(pairs.compound_id)<=set(mols.compound_id))
    check('molecule index is a unique contiguous fingerprint index',mols.molecule_index.tolist()==list(range(len(mols))))
    check('unsupported auxiliary source conflicts are excluded',pairs.loc[pairs.source.ne('ChEMBL'),'cross_source_range'].le(2).all())
    for split,column in [('scaffold_split','scaffold'),('random_split','connectivity')]:
        groups={p:set(mols.loc[mols[split].eq(p),column]) for p in ['train','validation','test']}
        check(split+' groups do not cross partitions',not(groups['train']&groups['test'] or groups['train']&groups['validation'] or groups['validation']&groups['test']))
        check(split+' pair assignments agree with structure assignments',pairs[split].eq(pairs.compound_id.map(mols.set_index('compound_id')[split])).all())
    old=original_keys()
    joined=old.merge(pairs.loc[pairs.endpoint.eq('pKi')],on=['compound_id','target_id'],how='left',suffixes=('_old','_new'),validate='one_to_one')
    check('all original benchmark pairs remain',joined.label.notna().all())
    check('original benchmark labels are unchanged',np.allclose(joined.pki,joined.label))
    check('original benchmark partitions are unchanged',joined.scaffold_split_old.eq(joined.scaffold_split_new).all())
    sequences=pd.read_parquet(DATA/'proteins.parquet').set_index('Entry')
    bdb=observations.loc[observations.source.eq('BindingDB')]
    unique=bdb[['target_id','protein_sequence']].drop_duplicates()
    check('added sequence-annotated records use human canonical proteins',sequences.loc[unique.target_id,'Organism (ID)'].eq(9606).all())
    check('experimental sequences match canonical contiguous segments',all(
        len(r.protein_sequence)>=50 and r.protein_sequence in sequences.loc[r.target_id,'Sequence'] for r in unique.itertuples()))
    fp=np.load(DATA/'fingerprints.npz')
    check('fingerprint rows match molecule identities',np.array_equal(fp['compound_ids'],mols.compound_id.to_numpy()))
    check('fingerprint matrix is binary with the declared dimension',fp['X'].shape==(len(mols),2048) and np.isin(fp['X'],[0,1]).all())
    scale=json.loads((RESULTS/'drugcentral_scale_audit.json').read_text())
    check('DrugCentral scale was checked against overlapping ChEMBL pairs',scale['overlapping_exact_structure_target_pairs']>0 and scale['median_absolute_difference_if_ACT_VALUE_is_pKi']<scale['median_absolute_difference_if_misread_as_nM'])
    baseline=json.loads((RESULTS/'baseline_summary.json').read_text())
    for result in baseline['datasets']:
        name=result['dataset']
        if not result['eligible_targets']:
            check(name+' honestly reports unsupported coverage',name=='pKd')
            continue
        metrics=pd.read_csv(RESULTS/name/'metrics_macro.csv')
        selection=metrics.loc[metrics.part.eq(result['validation_selection_part'])&metrics.model.eq('RF')].set_index('candidate').MAE
        check(name+' parameters selected only on validation',selection.idxmin()==result['selected'] and not result['test_used_for_selection'])
        pred=pd.read_parquet(RESULTS/name/'test_predictions.parquet')
        check(name+' predictions contain only test examples',pred.scaffold_split.eq('test').all())
        by_target=pred.groupby('target_id').apply(lambda g:np.abs(g.label-g.prediction).mean(),include_groups=False)
        published=next(r for r in result['test_results'] if r['model']=='RF' and r['part']=='test')
        check(name+' MAE recomputed from saved predictions',np.isclose(by_target.mean(),published['MAE']))
        check(name+' reports all supported heads',len(by_target)==result['eligible_targets'])
        if name in ['davis','kiba']:
            data=pd.read_parquet(DATA/(name+'_pairs.parquet'))
            m=pd.read_parquet(DATA/(name+'_molecules.parquet'))
            a=set(m.loc[m.scaffold_split.eq('train'),'scaffold']);b=set(m.loc[m.scaffold_split.eq('test'),'scaffold'])
            check(name+' standalone scaffold groups are disjoint',a.isdisjoint(b))
        else:data=pairs.loc[pairs.endpoint.eq(name)]
        gold=data.set_index(['compound_id','target_id']).label
        keys=pd.MultiIndex.from_frame(pred[['compound_id','target_id']])
        check(name+' prediction labels agree with prepared data',np.allclose(pred.label,gold.loc[keys]))
    primary=pd.read_parquet(RESULTS/'pKi/test_predictions.parquet')
    anchor=primary.loc[primary.original_part.eq('test')]
    check('paired comparison retains exactly the original test',len(anchor)==16665 and anchor.target_id.nunique()==131)
    usage=json.loads((RESULTS/'pki_source_usage.json').read_text())
    check('only the pKi model is in the current benchmark', [r['dataset'] for r in baseline['datasets']]==['pKi'])
    check('source contribution table covers the benchmark exactly',
          sum(r['total_pairs'] for r in usage['sources'])==usage['benchmark_pairs'] and
          sum(r['test_pairs'] for r in usage['sources'])==len(primary))
    check('source proportions use retained source pairs as denominator',all(
        sum(r[s+'_pairs'] for s in ['train','validation','test'])==r['total_pairs'] and
        all(np.isclose(r[s+'_percent'],100*r[s+'_pairs']/r['total_pairs']) for s in ['train','validation','test']) for r in usage['sources']))
    check('validation selection uses only original ChEMBL pairs',sum(r['validation_selection_pairs'] for r in usage['sources'])==15439 and
          all(r['validation_selection_pairs']==0 for r in usage['sources'] if r['source']!='ChEMBL'))
    check('all three direct sources have explicit preparation rules',len(usage['sources'])==3 and all(r['preparation'] for r in usage['sources']))
    check('source criteria and exclusion reasons are recorded',len(usage['selection_criteria'])>=5 and
          {r['source'] for r in usage['excluded_datasets']}=={'PubChem BioAssay','DAVIS','KIBA','PDBbind','DrugBank','Open Targets','ClinicalTrials.gov'})
    check('new data version invalidates earlier model checkpoints',CONFIG['version']==3 and 'PubChem' in CONFIG['excluded_sources'])
    manifest=json.loads((STAGING/'manifest.json').read_text())
    check('all publication files match the manifest',all((STAGING/f['path']).stat().st_size==f['bytes'] and sha256(STAGING/f['path'])==f['sha256'] for f in manifest['files']))
    card=DatasetCard((STAGING/'README.md').read_text())
    check('Hugging Face card contains only the current pKi subsets',
          {c['config_name'] for c in card.data.configs}=={'affinity_pki','chembl_reference','molecules','source_observations'})
    for config in card.data.configs:
        for file in config['data_files']:
            check(config['config_name']+'/'+file['split']+' is readable Parquet',len(pd.read_parquet(STAGING/file['path']))>0)
    published=pd.concat([pd.read_parquet(STAGING/'data/affinity_pki'/(s+'.parquet')) for s in ['train','validation','test']])
    check('published labels are exclusively pKi',published.endpoint.eq('pKi').all())
    check('published labels and provenance exclude PubChem',set(published.source)==set(CONFIG['included_sources']) and
          all(set(sources)<=set(CONFIG['included_sources']) for sources in published.sources))
    check('published source observations exclude PubChem',set(pd.read_parquet(STAGING/'data/source_observations/train.parquet').source)==set(CONFIG['included_sources']))
    published_used=published.loc[published.benchmark_target]
    check('published benchmark count matches actual experiment',len(published_used)==usage['benchmark_pairs'])
    for row in usage['sources']:
        block=published_used.loc[published_used.source.eq(row['source'])]
        check(row['source']+' published source split counts match report',all(
            int(block.scaffold_split.eq(s).sum())==row[s+'_pairs'] for s in ['train','validation','test']))
    published_mols=pd.read_parquet(STAGING/'prepared/molecules.parquet')
    published_fp=np.load(STAGING/'features/fingerprints.npz')
    check('published fingerprints align with scoped molecules',
          np.array_equal(published_fp['compound_ids'],published_mols.compound_id.to_numpy()) and
          published_mols.molecule_index.tolist()==list(range(len(published_mols))))
    old_indices=mols.set_index('compound_id').loc[published_mols.compound_id,'molecule_index'].to_numpy()
    check('scoped publication preserves every molecular fingerprint',np.array_equal(published_fp['X'],fp['X'][old_indices]))
    book=nbformat.read(ROOT/'EDA.ipynb',as_version=4)
    nbformat.validate(book)
    code=[c for c in book.cells if c.cell_type=='code']
    check('the full updated notebook is executed without errors',all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in code))
    text='\n'.join(c.source for c in book.cells if c.cell_type=='markdown')
    check('notebook explains selection and exclusions', 'Критерии выбора датасетов' in text and 'Исключённые датасеты' in text)
    artifact_paths=[*ROOT.glob('src/multisource*.py'),ROOT/'src/pki_sources.py',ROOT/'config/multisource.json',
                    ROOT/'scripts/download_sources.py',ROOT/'scripts/run_multisource.py',ROOT/'scripts/verify_multisource.py',
                    ROOT/'scripts/audit_sources.py',ROOT/'scripts/verify_hf_publication.py',
                    *RESULTS.glob('*.json'),*RESULTS.glob('*.csv'),*RESULTS.glob('*.png'),*RESULTS.glob('*/*.csv')]
    artifact_paths=[p for p in artifact_paths if p.name!='verification.json']
    save_json(RESULTS/'verification.json',{'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'checks':passed,'artifacts':[{'path':str(p.relative_to(ROOT)),'sha256':sha256(p)} for p in sorted(artifact_paths)]})
    print(f'PASS: {len(passed)} multi-source checks; {len(code)} executed notebook cells; {len(manifest["files"])} publication files.')


if __name__=='__main__':main()
