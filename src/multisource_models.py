"""Morgan/RF baseline on the integrated pKi task."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from sklearn.ensemble import RandomForestRegressor

from affinity_eda import metrics, fingerprint_generator, save_json, sha256
from multisource_data import ROOT, DATA, RESULTS, fingerprints, benchmark, init

CONFIG=json.loads((ROOT/'config/multisource.json').read_text())


def original_keys():
    original=pd.read_parquet(ROOT/'data/processed/pairs.parquet')
    components=pd.read_csv(ROOT/'data/processed/target_components.csv')
    counts=components.groupby('target_chembl_id').size()
    mapping=components.loc[components.target_chembl_id.isin(counts[counts.eq(1)].index)].set_index('target_chembl_id').accession
    original=original.loc[original.benchmark_target].copy()
    original['target_id']=original.target_chembl_id.map(mapping)
    return original


def eligible(pairs,name):
    counts=pd.crosstab(pairs.target_id,pairs.scaffold_split).reindex(columns=['train','validation','test'],fill_value=0)
    enough=counts.sum(axis=1).ge(CONFIG['minimum_pairs'][name]) & counts.train.ge(CONFIG['minimum_train'][name]) &\
           counts.validation.ge(CONFIG['minimum_validation_test'][name]) & counts.test.ge(CONFIG['minimum_validation_test'][name])
    counts['eligible']=enough
    counts.to_csv(RESULTS/(name+'_target_counts.csv'))
    return pairs.loc[pairs.target_id.isin(counts.index[enough])].copy()


def fingerprint_matrix(molecules):
    gen=fingerprint_generator();X=np.zeros((len(molecules),2048),dtype=np.uint8)
    for i,mol in enumerate(Chem.MolFromSmiles(s) for s in molecules.smiles):
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol),X[i])
    return X


def fit_target(target,data,X,parameters):
    train=data.loc[data.scaffold_split.eq('train')]
    model=RandomForestRegressor(**parameters,n_jobs=1,random_state=CONFIG['seed'])
    model.fit(X[train.molecule_index.to_numpy()],train.label.to_numpy())
    rows=[];predictions=[]
    for part in ['validation','test']:
        block=data.loc[data.scaffold_split.eq(part)].copy()
        prediction=model.predict(X[block.molecule_index.to_numpy()])
        mean=np.full(len(block),train.label.mean())
        for kind,pred in [('RF',prediction),('Mean',mean)]:
            rows.append({'target_id':target,'part':part,'model':kind,**metrics(block.label.to_numpy(),pred)})
        if 'original_part' in block:
            mask=block.original_part.eq(part).to_numpy()
            if mask.any():
                for kind,pred in [('RF',prediction),('Mean',mean)]:
                    rows.append({'target_id':target,'part':part+'_original','model':kind,
                                 **metrics(block.loc[mask,'label'].to_numpy(),pred[mask])})
        if part=='test':
            frame=block.copy();frame['prediction']=prediction;frame['mean_prediction']=mean
            predictions.append(frame)
    return rows,pd.concat(predictions,ignore_index=True)


def run_dataset(name,pairs,molecules,X):
    directory=RESULTS/name;directory.mkdir(exist_ok=True)
    pairs=eligible(pairs,name)
    if pairs.empty:
        save_json(directory/'summary.json',{'dataset':name,'eligible_targets':0,'reason':'Insufficient train/validation/test coverage'})
        return json.loads((directory/'summary.json').read_text())
    spec={'config':CONFIG,'pairs_sha256':__import__('hashlib').sha256(pd.util.hash_pandas_object(
        pairs[['compound_id','target_id','label','scaffold_split']],index=False).to_numpy().tobytes()).hexdigest()}
    cache_key=__import__('hashlib').sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:20]
    cache=ROOT/'.cache/multisource_models'/name/cache_key
    cache.mkdir(parents=True,exist_ok=True)
    groups=list(pairs.groupby('target_id',sort=True))
    all_rows=[];times=[]
    for candidate,params in CONFIG['candidates'].items():
        marker=cache/(candidate+'.json');csv=cache/(candidate+'_metrics.csv');parquet=cache/(candidate+'_predictions.parquet')
        if marker.exists():
            prior=json.loads(marker.read_text())
            if prior['specification']!=spec:raise ValueError('Stale checkpoint '+name+'/'+candidate)
            rows=pd.read_csv(csv,float_precision='round_trip');times.append(prior['seconds'])
            print(name,candidate,'using verified checkpoint',flush=True)
        else:
            started=time.perf_counter();collected=[];predicted=[]
            with ThreadPoolExecutor(max_workers=CONFIG['parallel_targets']) as pool:
                jobs=[pool.submit(fit_target,t,g,X,params) for t,g in groups]
                for i,future in enumerate(as_completed(jobs),1):
                    r,p=future.result();collected.extend(r);predicted.append(p)
                    if i%50==0 or i==len(jobs):print(name,candidate,i,'/',len(jobs),f'{time.perf_counter()-started:.1f}s',flush=True)
            rows=pd.DataFrame(collected);rows.to_csv(csv,index=False)
            pd.concat(predicted,ignore_index=True).to_parquet(parquet,index=False)
            elapsed=time.perf_counter()-started;times.append(elapsed)
            save_json(marker,{'specification':spec,'seconds':elapsed})
        rows['candidate']=candidate;all_rows.append(rows)
    detail=pd.concat(all_rows,ignore_index=True)
    selection_part='validation_original' if name=='pKi' else 'validation'
    selection=detail.loc[detail.part.eq(selection_part)&detail.model.eq('RF')].groupby('candidate').MAE.mean().sort_values()
    chosen=selection.index[0]
    predictions=pd.read_parquet(cache/(chosen+'_predictions.parquet'))
    predictions.to_parquet(directory/'test_predictions.parquet',index=False)
    detail.to_csv(directory/'metrics_by_target.csv',index=False)
    macro=detail.groupby(['candidate','part','model']).agg(targets=('target_id','nunique'),pairs=('n','sum'),
        MAE=('MAE','mean'),RMSE=('RMSE','mean'),R2=('R2','mean'),Spearman=('Spearman','mean')).reset_index()
    macro.to_csv(directory/'metrics_macro.csv',index=False)
    result={'dataset':name,'endpoint':pairs.endpoint.iloc[0],'selected':chosen,
        'validation_selection_part':selection_part,'validation_MAE':selection.to_dict(),'test_used_for_selection':False,
        'eligible_targets':len(groups),'eligible_pairs':len(pairs),'test_pairs':len(predictions),
        'training_seconds':sum(times),'parameters':CONFIG['candidates'][chosen],
        'test_results':macro.loc[macro.candidate.eq(chosen)&macro.part.str.startswith('test')].replace({np.nan:None}).to_dict('records'),
        'selection_note':'All parameter selection uses validation. Test predictions were not used to select the source policy or parameters.'}
    if name=='davis':
        uncensored=predictions.loc[~predictions.ceiling_censored]
        values=uncensored.groupby('target_id').apply(lambda g:np.abs(g.label-g.prediction).mean(),include_groups=False)
        result['uncensored_test_pairs']=len(uncensored)
        result['uncensored_test_targets']=len(values)
        result['uncensored_macro_MAE']=float(values.mean())
    save_json(directory/'summary.json',result)
    return result


def source_slices():
    estimates=pd.read_parquet(DATA/'source_estimates.parquet')
    rows=[]
    for endpoint in ['pKi']:
        path=RESULTS/endpoint/'test_predictions.parquet'
        if not path.exists():continue
        predictions=pd.read_parquet(path)
        sliced=estimates.loc[estimates.endpoint.eq(endpoint)].merge(
            predictions[['compound_id','target_id','prediction']],on=['compound_id','target_id'],validate='many_to_one')
        for source,block in sliced.groupby('source'):
            maes=block.groupby('target_id').apply(lambda g:np.abs(g.label-g.prediction).mean(),include_groups=False)
            rows.append({'endpoint':endpoint,'source':source,'targets':len(maes),'pairs':len(block),
                         'macro_MAE':float(maes.mean()),'micro_MAE':float(np.abs(block.label-block.prediction).mean()),
                         'scope':'source-specific labels on globally held-out scaffolds; not an independent external cohort'})
    pd.DataFrame(rows).to_csv(RESULTS/'source_test_slices.csv',index=False)
    return rows


def train():
    init()
    merged=pd.read_parquet(DATA/'pairs.parquet')
    if set(merged.source)!=set(CONFIG['included_sources']) or not merged.endpoint.eq('pKi').all():
        raise ValueError('Prepared data do not match the configured source/endpoint policy')
    if any('PubChem' in sources for sources in merged.sources):
        raise ValueError('Excluded PubChem evidence remains in the prepared pairs')
    molecules=pd.read_parquet(DATA/'molecules.parquet')
    X=fingerprints(molecules)
    original=original_keys()
    summary=[]
    for endpoint in ['pKi']:
        pairs=merged.loc[merged.endpoint.eq(endpoint)].copy()
        if endpoint=='pKi':
            reference=original[['compound_id','target_id','pki','scaffold_split']].rename(
                columns={'pki':'original_label','scaffold_split':'original_part'})
            check=reference.merge(pairs,on=['compound_id','target_id'],how='left',validate='one_to_one')
            if check.label.isna().any() or not np.allclose(check.original_label,check.label) or not check.original_part.eq(check.scaffold_split).all():
                raise AssertionError('The original benchmark labels and partitions must be preserved exactly')
            pairs=pairs.merge(reference,on=['compound_id','target_id'],how='left',validate='one_to_one')
        summary.append(run_dataset(endpoint,pairs,molecules,X))
    del X
    comparison=[]
    original_result=json.loads((ROOT/'results/baseline_summary.json').read_text())
    primary=next(x for x in summary if x['dataset']=='pKi')
    expanded=next(r for r in primary['test_results'] if r['model']=='RF' and r['part']=='test_original')
    comparison=[{'training_sources':'ChEMBL','targets':131,'test_pairs':original_result['test_pairs'],
                 'MAE':original_result['test_macro_MAE'],'RMSE':original_result['test_macro_RMSE'],
                 'Spearman':original_result['test_macro_Spearman']},
                {'training_sources':' + '.join(CONFIG['included_sources']),'targets':expanded['targets'],
                 'test_pairs':expanded['pairs'],'MAE':expanded['MAE'],'RMSE':expanded['RMSE'],'Spearman':expanded['Spearman']}]
    pd.DataFrame(comparison).to_csv(RESULTS/'original_test_comparison.csv',index=False)
    result={'config':CONFIG,'datasets':summary,'source_slices':source_slices(),
            'original_test_comparison':comparison,'total_training_seconds':sum(r.get('training_seconds',0) for r in summary)}
    save_json(RESULTS/'baseline_summary.json',result)
    from pki_sources import build_report
    build_report()
    return result
