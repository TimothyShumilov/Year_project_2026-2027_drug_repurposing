"""Reproducible pKi curation, molecule-level splits and Morgan/RF experiments."""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
RAW, DATA, RESULTS = ROOT/'data/raw', ROOT/'data/processed', ROOT/'results'
CFG = json.loads((ROOT/'config/eda.json').read_text())
ALLOWED_ELEMENTS = {1,5,6,7,8,9,14,15,16,17,34,35,53}


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pki_from_nm(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Ki must contain finite positive values in nM')
    return 9.0 - np.log10(values)


def structure_features(smiles):
    """Minimal standardisation; preserve charge, tautomer and stereochemistry."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    if mol is None:
        return {'source_smiles':smiles, 'structure_status':'invalid_or_missing'}
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    organic = [m for m in fragments if any(a.GetAtomicNum()==6 for a in m.GetAtoms())]
    if not organic:
        return {'source_smiles':smiles, 'structure_status':'no_organic_fragment'}
    organic.sort(key=lambda m:m.GetNumHeavyAtoms(), reverse=True)
    # Two substantial organic components can be a mixture. Do not silently pick one.
    if len(organic)>1 and organic[1].GetNumHeavyAtoms()>=6:
        return {'source_smiles':smiles, 'structure_status':'multiple_organic_components'}
    mol = organic[0]
    if mol.GetNumHeavyAtoms()>CFG['maximum_heavy_atoms']:
        return {'source_smiles':smiles, 'structure_status':'over_100_heavy_atoms'}
    if any(a.GetAtomicNum() not in ALLOWED_ELEMENTS for a in mol.GetAtoms()):
        return {'source_smiles':smiles, 'structure_status':'unsupported_element'}
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    connectivity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=False) or 'ACYCLIC'
    return {'source_smiles':smiles, 'structure_status':'valid', 'smiles':canonical,
            'compound_id':'MOL_'+hashlib.sha256(canonical.encode()).hexdigest()[:20],
            'connectivity':connectivity, 'scaffold':scaffold,
            'molecular_weight':Descriptors.MolWt(mol), 'logp':Descriptors.MolLogP(mol),
            'tpsa':Descriptors.TPSA(mol), 'heavy_atoms':mol.GetNumHeavyAtoms(),
            'hbd':Lipinski.NumHDonors(mol), 'hba':Lipinski.NumHAcceptors(mol),
            'rotatable_bonds':Lipinski.NumRotatableBonds(mol),
            'rings':Lipinski.RingCount(mol), 'fragment_removed':len(fragments)>1}


def grouped_split(molecules, column, seed):
    """Group membership is determined without inspecting pKi values."""
    labels = pd.Series('train', index=molecules.index, dtype=object)
    splitter = GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed)
    train, rest = next(splitter.split(molecules,groups=molecules[column]))
    tmp = molecules.iloc[rest]
    validation, test = next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=seed+1).split(tmp,groups=tmp[column]))
    labels.iloc[rest[validation]] = 'validation'
    labels.iloc[rest[test]] = 'test'
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        assert set(molecules.loc[labels==left,column]).isdisjoint(molecules.loc[labels==right,column])
    return labels


def prepare_data(force=False):
    required=[DATA/'pairs.parquet',DATA/'molecules.parquet',RESULTS/'eda_summary.json']
    if not force and all(p.exists() for p in required):
        cached=json.loads(required[2].read_text())
        if cached['config'] == CFG:
            return pd.read_parquet(required[0]),pd.read_parquet(required[1]),cached
    DATA.mkdir(parents=True,exist_ok=True)
    RESULTS.mkdir(parents=True,exist_ok=True)
    raw = pd.read_parquet(RAW/'activities.parquet')
    targets = pd.DataFrame(json.loads((RAW/'targets.json').read_text()))
    assays = pd.DataFrame(json.loads((RAW/'assays.json').read_text()))
    source_manifest=json.loads((RAW/'manifest.json').read_text())
    raw.describe(include='all').T.to_csv(RESULTS/'raw_describe.csv')
    missing=pd.DataFrame({'missing':raw.isna().sum(),'missing_fraction':raw.isna().mean()})
    missing.to_csv(RESULTS/'raw_missingness.csv')
    pd.crosstab(raw.standard_relation.fillna('missing'),raw.pchembl_value.isna(),normalize='index').to_csv(RESULTS/'pchembl_missing_by_relation.csv')
    raw.groupby('standard_relation',dropna=False).size().to_csv(RESULTS/'measurement_relations.csv')
    raw.groupby('data_validity_comment',dropna=False).size().to_csv(RESULTS/'validity_comments.csv')
    targets.target_type.value_counts().to_csv(RESULTS/'raw_target_types.csv')
    assert raw.activity_id.is_unique
    work=raw.copy()
    funnel=[{'step':'Получено из API','remaining':len(work),'removed':0}]
    def keep(label,mask):
        nonlocal work
        before=len(work)
        work=work.loc[mask].copy()
        funnel.append({'step':label,'remaining':len(work),'removed':before-len(work)})
        print(f'{label}: {len(work)} measurements',flush=True)
    keep('Точное измерение Ki',work.standard_relation.eq('='))
    work['ki_nm']=pd.to_numeric(work.standard_value,errors='coerce')
    keep('Положительное конечное значение в нМ',work.standard_units.eq('nM') & np.isfinite(work.ki_nm) & work.ki_nm.gt(0))
    keep('Нет флага ошибки измерения',work.data_validity_comment.fillna('').isin(['','Manually validated']) & work.standard_flag.eq(1))
    keep('Нет флага возможного дубликата',work.potential_duplicate.eq(0))
    type_map=targets.set_index('target_chembl_id').target_type
    work['target_type']=work.target_chembl_id.map(type_map)
    keep('Отдельный человеческий белок',work.target_type.eq('SINGLE PROTEIN') & work.target_tax_id.astype(str).eq('9606'))
    confidence=assays.set_index('assay_chembl_id').confidence_score
    work['confidence_score']=work.assay_chembl_id.map(confidence)
    keep('Прямое сопоставление мишени с уверенностью 9',work.confidence_score.eq(9))
    variant_assays=set(assays.loc[assays.variant_sequence.notna(),'assay_chembl_id'])
    keep('Нет аннотации мутантного белка',work.assay_variant_mutation.isna() & ~work.assay_chembl_id.isin(variant_assays))
    work['pki']=pki_from_nm(work.ki_nm)
    provided=pd.to_numeric(work.pchembl_value,errors='coerce')
    pki_check={'with_provided_pchembl':int(provided.notna().sum()),
               'different_by_over_0_05':int((provided.notna() & (provided-work.pki).abs().gt(.05)).sum()),
               'recomputed_missing_pchembl':int(provided.isna().sum())}
    keep('pKi согласуется с указанным pChEMBL',provided.isna() | (provided-work.pki).abs().le(.05))
    source_smiles=work.canonical_smiles.drop_duplicates().tolist()
    print(f'Standardising {len(source_smiles)} source structures',flush=True)
    features=[]
    # Threads avoid serialising RDKit objects; output ordering remains deterministic.
    with ThreadPoolExecutor(max_workers=6) as executor:
        for n,record in enumerate(executor.map(structure_features,source_smiles),1):
            features.append(record)
            if n%20000==0: print(f'Structures: {n}/{len(source_smiles)}',flush=True)
    structure_df=pd.DataFrame(features)
    structure_status=structure_df.structure_status.value_counts().to_dict()
    structure_df.to_parquet(DATA/'structure_audit.parquet',index=False)
    valid=structure_df.loc[structure_df.structure_status.eq('valid')].copy()
    work=work.merge(valid,how='left',left_on='canonical_smiles',right_on='source_smiles',validate='many_to_one')
    keep('Пригодная органическая структура до 100 тяжёлых атомов',work.compound_id.notna())
    # Repeated measurements of the same standardised molecule-target pair stay together.
    grouped=work.groupby(['compound_id','target_chembl_id'],sort=True)
    pairs=grouped.agg(pki=('pki','median'),n_measurements=('pki','size'),
        pki_min=('pki','min'),pki_max=('pki','max'),
        first_year=('document_year','min'),last_year=('document_year','max'),
        n_assays=('assay_chembl_id','nunique'),n_documents=('document_chembl_id','nunique')).reset_index()
    pairs['repeat_range']=pairs.pki_max-pairs.pki_min
    repeated={'pairs_before_conflict_filter':len(pairs),'repeated_pairs':int(pairs.n_measurements.gt(1).sum()),
              'conflicting_pairs_over_2_pki':int(pairs.repeat_range.gt(CFG['maximum_repeat_range_pki']).sum())}
    pairs.loc[pairs.repeat_range.gt(CFG['maximum_repeat_range_pki'])].to_csv(RESULTS/'conflicting_pairs.csv',index=False)
    pairs=pairs.loc[pairs.repeat_range.le(CFG['maximum_repeat_range_pki'])].copy()
    molecules=valid.drop_duplicates('compound_id').drop(columns=['source_smiles','structure_status']).sort_values('compound_id').reset_index(drop=True)
    molecules=molecules.loc[molecules.compound_id.isin(pairs.compound_id)].reset_index(drop=True)
    molecules['molecule_index']=np.arange(len(molecules))
    molecules['random_split']=grouped_split(molecules,'connectivity',CFG['seed'])
    molecules['scaffold_split']=grouped_split(molecules,'scaffold',CFG['seed'])
    pairs=pairs.merge(molecules[['compound_id','molecule_index','random_split','scaffold_split']],on='compound_id',validate='many_to_one')
    names=targets.set_index('target_chembl_id').pref_name
    pairs['target_name']=pairs.target_chembl_id.map(names)
    counts=pairs.groupby('target_chembl_id').size()
    requested=counts[counts>=CFG['minimum_pairs_per_target']].index
    eligible=set(requested)
    for split in ['random_split','scaffold_split']:
        sizes=pd.crosstab(pairs.target_chembl_id,pairs[split]).reindex(columns=['train','validation','test'],fill_value=0)
        sizes.to_csv(RESULTS/(split+'_target_counts.csv'))
        enough=sizes.train.ge(CFG['minimum_train_pairs']) & sizes.validation.ge(CFG['minimum_validation_pairs']) & sizes.test.ge(CFG['minimum_test_pairs'])
        eligible &= set(sizes.index[enough])
    pairs['benchmark_target']=pairs.target_chembl_id.isin(eligible)
    counts.rename('n_pairs').to_csv(RESULTS/'target_pair_counts.csv')
    pairs.to_parquet(DATA/'pairs.parquet',index=False)
    molecules.to_parquet(DATA/'molecules.parquet',index=False)
    targets.drop(columns=['target_components']).to_csv(DATA/'targets.csv',index=False)
    accession_rows=[]
    for record in targets.to_dict('records'):
        for component in record.get('target_components') or []:
            accession_rows.append({'target_chembl_id':record['target_chembl_id'],'accession':component.get('accession'),
                                   'component_type':component.get('component_type')})
    pd.DataFrame(accession_rows).to_csv(DATA/'target_components.csv',index=False)
    pd.DataFrame(funnel).to_csv(RESULTS/'cleaning_funnel.csv',index=False)
    numerical=molecules.select_dtypes(include='number').drop(columns='molecule_index')
    numerical.describe().T.to_csv(RESULTS/'molecule_describe.csv')
    numerical.corr(method='spearman').to_csv(RESULTS/'descriptor_correlations.csv')
    pairs[['pki','n_measurements','repeat_range','first_year','n_assays']].describe().T.to_csv(RESULTS/'pair_describe.csv')
    q1,q3=pairs.pki.quantile([.25,.75]);iqr=q3-q1
    summary={'source':source_manifest,'raw_rows':len(raw),'raw_targets':int(raw.target_chembl_id.nunique()),
        'raw_unique_molecules':int(raw.molecule_chembl_id.nunique()),'raw_global_missing_fraction':float(raw.isna().to_numpy().mean()),
        'funnel':funnel,'pki_check':pki_check,'structure_status':structure_status,'repeats':repeated,
        'curated_pairs':len(pairs),'curated_molecules':len(molecules),'curated_targets':int(pairs.target_chembl_id.nunique()),
        'benchmark_targets':len(eligible),'benchmark_pairs':int(pairs.benchmark_target.sum()),
        'benchmark_molecules':int(pairs.loc[pairs.benchmark_target,'compound_id'].nunique()),
        'target_min_pairs':CFG['minimum_pairs_per_target'],'targets_dropped_for_split_coverage':len(requested)-len(eligible),
        'scaffolds':int(molecules.scaffold.nunique()),'acyclic_molecules':int(molecules.scaffold.eq('ACYCLIC').sum()),
        'pki_min':float(pairs.pki.min()),'pki_max':float(pairs.pki.max()),'pki_median':float(pairs.pki.median()),
        'iqr_outlier_pairs':int((pairs.pki.lt(q1-1.5*iqr)|pairs.pki.gt(q3+1.5*iqr)).sum()),
        'matrix_missing_fraction':1-len(pairs)/(len(molecules)*pairs.target_chembl_id.nunique()),
        'missing_publication_year_pairs':int(pairs.first_year.isna().sum()),
        'split_molecule_counts':{s:molecules[s].value_counts().to_dict() for s in ['random_split','scaffold_split']},
        'config':CFG}
    for split in ['random_split','scaffold_split']:
        summary[split+'_scaffold_overlap_train_test']=len(set(molecules.loc[molecules[split].eq('train'),'scaffold']) & set(molecules.loc[molecules[split].eq('test'),'scaffold']))
    save_json(RESULTS/'eda_summary.json',summary)
    print(f'Curated: {len(pairs)} pairs; {len(molecules)} molecules; {len(eligible)} benchmark targets',flush=True)
    return pairs,molecules,summary


def fingerprint_generator():
    c=CFG['fingerprint']
    return rdFingerprintGenerator.GetMorganGenerator(radius=c['radius'],fpSize=c['n_bits'],includeChirality=c['include_chirality'])


def approved_coverage(pairs):
    destination=RESULTS/'approved_coverage.json'
    if destination.exists():return json.loads(destination.read_text())
    approved=json.loads((RAW/'approved_molecules.json').read_text())
    records=[]
    for record in approved:
        if record['molecule_type']!='Small molecule' or record['withdrawn_flag']:
            continue
        features=structure_features((record.get('molecule_structures') or {}).get('canonical_smiles'))
        if features['structure_status']!='valid':continue
        records.append({'drug_id':record['molecule_chembl_id'],'drug_name':record.get('pref_name'),
                        'parent_id':(record.get('molecule_hierarchy') or {}).get('parent_chembl_id'),
                        'compound_id':features['compound_id'],'smiles':features['smiles']})
    library=pd.DataFrame(records).drop_duplicates('compound_id').sort_values('compound_id')
    total=pairs.groupby('compound_id').target_chembl_id.nunique()
    benchmark=pairs.loc[pairs.benchmark_target].groupby('compound_id').target_chembl_id.nunique()
    library['measured_targets']=library.compound_id.map(total).fillna(0).astype(int)
    library['benchmark_targets']=library.compound_id.map(benchmark).fillna(0).astype(int)
    library.to_parquet(DATA/'approved_library.parquet',index=False)
    result={'approved_molecule_records':len(approved),'eligible_unique_small_molecule_structures':len(library),
            'with_curated_Ki_measurement':int(library.measured_targets.gt(0).sum()),
            'with_benchmark_Ki_measurement':int(library.benchmark_targets.gt(0).sum()),
            'definition':'ChEMBL max_phase=4, nonwithdrawn Small molecule record, same structure filters; exact canonical structure matching'}
    save_json(destination,result);return result


def get_fingerprints(molecules,force=False):
    path=DATA/'fingerprints.npz'
    if path.exists() and not force:
        stored=np.load(path)
        if 'fingerprint_spec' in stored and str(stored['fingerprint_spec'])==json.dumps(CFG['fingerprint'],sort_keys=True) and np.array_equal(stored['compound_ids'],molecules.compound_id.to_numpy()):
            return stored['X']
    generator=fingerprint_generator()
    X=np.zeros((len(molecules),CFG['fingerprint']['n_bits']),dtype=np.uint8)
    for start in range(0,len(molecules),2000):
        mols=[Chem.MolFromSmiles(s) for s in molecules.smiles.iloc[start:start+2000]]
        for j,fp in enumerate(generator.GetFingerprints(mols,numThreads=4)):
            DataStructs.ConvertToNumpyArray(fp,X[start+j])
        if start%20000==0: print(f'Fingerprints: {start}/{len(molecules)}',flush=True)
    np.savez_compressed(path,X=X,compound_ids=molecules.compound_id.to_numpy(dtype=str),
                        fingerprint_spec=json.dumps(CFG['fingerprint'],sort_keys=True))
    return X


def fingerprint_audit(molecules,X):
    path=RESULTS/'fingerprint_audit.json'
    packed=np.packbits(X,axis=1)
    keys=np.asarray([hashlib.sha256(row.tobytes()).hexdigest() for row in packed])
    active=X.sum(axis=1)
    result={'molecules':len(X),'bits':X.shape[1],
            'active_bits_min':int(active.min()),'active_bits_median':float(np.median(active)),
            'active_bits_max':int(active.max()),'constant_bits':int((np.ptp(X,axis=0)==0).sum()),
            'unique_fingerprints':int(len(np.unique(keys))),
            'different_structures_with_same_fingerprint_excess':int(len(keys)-len(np.unique(keys)))}
    for split in ['random_split','scaffold_split']:
        a=set(keys[molecules[split].eq('train')]);b=set(keys[molecules[split].eq('test')])
        result[split+'_shared_fingerprints_train_test']=len(a&b)
    save_json(path,result)
    pd.DataFrame({'bit':np.arange(X.shape[1]),'fraction_on':X.mean(axis=0)}).to_csv(RESULTS/'fingerprint_bit_frequencies.csv',index=False)
    return result


def metrics(y,prediction):
    corr=float(spearmanr(y,prediction).statistic) if np.ptp(y)>0 and np.ptp(prediction)>0 else None
    return {'n':len(y),'MAE':float(mean_absolute_error(y,prediction)),
            'RMSE':float(np.sqrt(mean_squared_error(y,prediction))),
            'R2':float(r2_score(y,prediction)), 'Spearman':corr if corr is None or np.isfinite(corr) else None}


def _train_target(target,data,X,params,split,seed,dropout=0,return_model=False):
    train=data.loc[data[split].eq('train')]
    valid=data.loc[data[split].eq('validation')]
    test=data.loc[data[split].eq('test')]
    indices=train.molecule_index.to_numpy()
    xtrain=X[indices]
    if dropout:
        rng=np.random.default_rng(seed)
        xtrain=xtrain.copy()
        active=np.nonzero(xtrain)
        remove=rng.random(len(active[0]))<dropout
        xtrain[active[0][remove],active[1][remove]]=0
    start=time.perf_counter()
    model=RandomForestRegressor(**params,n_jobs=1,random_state=seed)
    model.fit(xtrain,train.pki.to_numpy())
    prediction={'validation':model.predict(X[valid.molecule_index]),'test':model.predict(X[test.molecule_index])}
    rows=[]
    for part,block in [('validation',valid),('test',test)]:
        for kind,pred in [('RF',prediction[part]),('Mean',np.full(len(block),train.pki.mean()))]:
            rows.append({'target_id':target,'target_name':data.target_name.iloc[0], 'part':part,'model':kind,
                         **metrics(block.pki.to_numpy(),pred)})
    test_prediction=test[['compound_id','target_chembl_id','molecule_index','pki','target_name']].copy()
    test_prediction['prediction']=prediction['test']
    test_prediction['mean_prediction']=train.pki.mean()
    result={'metrics':rows,'test_predictions':test_prediction,'fit_seconds':time.perf_counter()-start}
    if return_model: result['model']=model
    return result


def train_benchmark(pairs,molecules,X,force=False):
    destination=RESULTS/'baseline_summary.json'
    input_hashes={'pairs':sha256(DATA/'pairs.parquet'),'molecules':sha256(DATA/'molecules.parquet'),
                  'fingerprints':sha256(DATA/'fingerprints.npz')}
    if destination.exists() and not force:
        cached=json.loads(destination.read_text())
        if cached['config']==CFG and cached.get('input_sha256')==input_hashes:
            return cached
        raise ValueError('Cached baseline does not match data/configuration. Use force=True explicitly.')
    dataset=pairs.loc[pairs.benchmark_target].copy()
    groups={t:g for t,g in dataset.groupby('target_chembl_id',sort=True)}
    if not groups: raise ValueError('No targets meet the declared data and split coverage criteria')
    print(f'Training {len(groups)} targets; {len(dataset)} measured pairs',flush=True)
    measurements=[];predictions=[];timings=[]
    def run(name,split,params,seed=42,dropout=0,save_models=False):
        checkpoint=RESULTS/'experiments'/name
        specification={'inputs':input_hashes,'parameters':params,'split':split,'seed':seed,'dropout':dropout}
        if (checkpoint/'completed.json').exists():
            record=json.loads((checkpoint/'completed.json').read_text())
            if record['specification']!=specification:
                raise ValueError(f'Stale experiment {name}; remove only this results/experiments subfolder or use a new snapshot')
            measurements.extend(pd.read_csv(checkpoint/'metrics.csv').to_dict('records'))
            predictions.append(pd.read_parquet(checkpoint/'test_predictions.parquet'))
            timings.append({'experiment':name,'seconds':record['seconds'],'targets':len(groups)})
            print(f'{name}: using completed, input-validated experiment',flush=True)
            return record['seconds']
        started=time.perf_counter();results=[]
        experiment_rows=[];experiment_predictions=[]
        with ThreadPoolExecutor(max_workers=CFG['parallel_targets']) as executor:
            futures={executor.submit(_train_target,t,g,X,params,split,seed,dropout,save_models):t for t,g in groups.items()}
            for n,future in enumerate(as_completed(futures),1):
                target=futures[future];result=future.result();results.append(result)
                for row in result['metrics']:
                    entry={**row,'experiment':name,'split':split,'seed':seed,'dropout':dropout}
                    measurements.append(entry);experiment_rows.append(entry)
                frame=result['test_predictions'];frame['experiment']=name;frame['split']=split
                predictions.append(frame);experiment_predictions.append(frame)
                if save_models:
                    path=ROOT/'models'/name/(target+'.joblib');path.parent.mkdir(parents=True,exist_ok=True)
                    joblib.dump(result['model'],path,compress=3)
                if n%20==0 or n==len(groups):
                    print(f'{name}: {n}/{len(groups)} targets; {time.perf_counter()-started:.1f}s',flush=True)
        duration=time.perf_counter()-started
        timings.append({'experiment':name,'seconds':duration,'targets':len(groups)})
        checkpoint.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(experiment_rows).to_csv(checkpoint/'metrics.csv',index=False)
        pd.concat(experiment_predictions,ignore_index=True).to_parquet(checkpoint/'test_predictions.parquet',index=False)
        save_json(checkpoint/'completed.json',{'specification':specification,'seconds':duration})
        # Checkpoint after each real fit group, so a server handoff can resume.
        pd.DataFrame(measurements).to_csv(RESULTS/'metrics_by_target.csv',index=False)
        pd.DataFrame(timings).to_csv(RESULTS/'training_times.csv',index=False)
        return duration
    # Hyperparameters selected strictly on scaffold validation, before examining test quality.
    for name,params in CFG['rf_candidates'].items():
        run(name,'scaffold_split',params,seed=CFG['seed'])
    detail=pd.DataFrame(measurements)
    validation=detail.loc[detail.part.eq('validation') & detail.model.eq('RF')].groupby('experiment').MAE.mean().sort_values()
    selected=validation.index[0]
    save_json(RESULTS/'selection.json',{'criterion':'macro validation MAE on scaffold split',
              'selected':selected,'validation_macro_MAE':validation.to_dict(),'test_used_for_selection':False})
    params=CFG['rf_candidates'][selected]
    run('RF_random','random_split',params,seed=CFG['seed'])
    run('RF_dropout5','scaffold_split',params,seed=CFG['seed'],dropout=CFG['augmentation_dropout'])
    # Two additional initialisations quantify baseline training variability.
    for seed in [43,44]:
        run('RF_seed'+str(seed),'scaffold_split',params,seed=seed)
    # All head results are reproducible; save a representative fitted head below.
    (ROOT/'models').mkdir(exist_ok=True)
    model_index={'selected':selected,'parameters':params,'seed':CFG['seed'],
                 'targets':sorted(groups),'training_split':'scaffold_split/train',
                 'checkpoint_scope':'One representative target is saved by explain_baseline; all target results are in results.'}
    save_json(ROOT/'models/model_index.json',model_index)
    all_metrics=pd.DataFrame(measurements)
    combined=pd.concat(predictions,ignore_index=True)
    combined.to_parquet(RESULTS/'test_predictions.parquet',index=False)
    macro=all_metrics.groupby(['experiment','split','part','model'],sort=True).agg(
        targets=('target_id','nunique'),pairs=('n','sum'),MAE=('MAE','mean'),RMSE=('RMSE','mean'),
        R2=('R2','mean'),Spearman=('Spearman','mean')).reset_index()
    macro.to_csv(RESULTS/'metrics_macro.csv',index=False)
    baseline=macro.loc[macro.experiment.eq(selected)&macro.part.eq('test')&macro.model.eq('RF')].iloc[0]
    dummy=macro.loc[macro.experiment.eq(selected)&macro.part.eq('test')&macro.model.eq('Mean')].iloc[0]
    seeds=macro.loc[macro.experiment.isin([selected,'RF_seed43','RF_seed44'])&macro.part.eq('test')&macro.model.eq('RF')]
    result={'selected':selected,'parameters':params,'validation_macro_MAE':float(validation.iloc[0]),
            'test_macro_MAE':float(baseline.MAE),'test_macro_RMSE':float(baseline.RMSE),
            'test_macro_R2':float(baseline.R2),'test_macro_Spearman':float(baseline.Spearman),
            'dummy_test_macro_MAE':float(dummy.MAE),'targets':len(groups),'test_pairs':int(baseline.pairs),
            'three_seed_macro_MAE_mean':float(seeds.MAE.mean()),'three_seed_macro_MAE_sd':float(seeds.MAE.std(ddof=1)),
            'total_training_seconds':float(sum(t['seconds'] for t in timings)),
            'config':CFG,'input_sha256':input_hashes,'test_is_blind_for_model_selection':True}
    save_json(destination,result)
    return result


def augmentation_invariance(molecules,force=False):
    path=RESULTS/'augmentation_invariance.json'
    if path.exists() and not force: return json.loads(path.read_text())
    sample=molecules.sample(n=min(CFG['augmentation_invariance_molecules'],len(molecules)),random_state=CFG['seed'])
    gen=fingerprint_generator();rng=np.random.default_rng(CFG['seed'])
    trials=0;smiles_changes=0;smiles_equal=0;permutation_equal=0
    examples=[]
    for record in sample.itertuples():
        mol=Chem.MolFromSmiles(record.smiles);reference=gen.GetFingerprint(mol)
        variants=Chem.MolToRandomSmilesVect(mol,CFG['augmentation_repeats'],randomSeed=CFG['seed'],isomericSmiles=True)
        for variant in variants:
            trials+=1;smiles_changes+=variant!=record.smiles
            smiles_equal+=gen.GetFingerprint(Chem.MolFromSmiles(variant))==reference
            permuted=Chem.RenumberAtoms(mol,rng.permutation(mol.GetNumAtoms()).tolist())
            permutation_equal+=gen.GetFingerprint(permuted)==reference
        if len(examples)<3: examples.append({'canonical':record.smiles,'random_smiles':variants[0]})
    result={'molecules':len(sample),'trials_per_method':trials,'random_smiles_changed':smiles_changes,
            'random_smiles_identical_fingerprints':smiles_equal,'atom_permutation_identical_fingerprints':permutation_equal,
            'examples':examples}
    save_json(path,result);return result


def fingerprint_sensitivity(pairs,molecules,X,baseline):
    """Post-hoc diagnostic only; does not change splits, targets or model selection."""
    path=RESULTS/'fingerprint_sensitivity.json'
    if path.exists():return json.loads(path.read_text())
    keys=np.asarray([hashlib.sha256(row.tobytes()).hexdigest() for row in np.packbits(X,axis=1)])
    predictions=pd.read_parquet(RESULTS/'test_predictions.parquet')
    main=predictions.loc[predictions.experiment.eq(baseline['selected'])]
    rows=[]
    for target,test in main.groupby('target_chembl_id'):
        train=pairs.loc[pairs.target_chembl_id.eq(target)&pairs.scaffold_split.eq('train')]
        known=set(keys[train.molecule_index.to_numpy()])
        shared=np.asarray([key in known for key in keys[test.molecule_index.to_numpy()]])
        remaining=test.loc[~shared]
        rows.append({'target_id':target,'test_pairs':len(test),'shared_fingerprint_test_pairs':int(shared.sum()),
                     'MAE_all':float(np.abs(test.pki-test.prediction).mean()),
                     'MAE_without_shared_fingerprints':float(np.abs(remaining.pki-remaining.prediction).mean()) if len(remaining) else None})
    table=pd.DataFrame(rows)
    table.to_csv(RESULTS/'fingerprint_sensitivity_by_target.csv',index=False)
    result={'diagnostic_only':True,'targets':len(table),
            'test_pairs':int(table.test_pairs.sum()),
            'test_pairs_with_fingerprint_seen_for_same_target':int(table.shared_fingerprint_test_pairs.sum()),
            'macro_MAE_all':float(table.MAE_all.mean()),
            'macro_MAE_without_shared_fingerprints':float(table.MAE_without_shared_fingerprints.mean()),
            'targets_remaining':int(table.MAE_without_shared_fingerprints.notna().sum())}
    save_json(path,result);return result


def explain_baseline(pairs,molecules,X,baseline):
    path=RESULTS/'feature_interpretation.json'
    if path.exists(): return json.loads(path.read_text())
    target=pairs.loc[pairs.benchmark_target].groupby('target_chembl_id').size().sort_values(ascending=False).index[0]
    block=pairs.loc[pairs.target_chembl_id.eq(target)]
    result=_train_target(target,block,X,baseline['parameters'],'scaffold_split',CFG['seed'],return_model=True)
    model=result['model'];valid=block.loc[block.scaffold_split.eq('validation')]
    rng=np.random.default_rng(CFG['seed'])
    if len(valid)>1000: valid=valid.sample(1000,random_state=CFG['seed'])
    xval=X[valid.molecule_index].copy();yval=valid.pki.to_numpy()
    original=mean_absolute_error(yval,model.predict(xval))
    top=np.argsort(model.feature_importances_)[::-1][:15]
    rows=[]
    for bit in top:
        changes=[]
        for _ in range(3):
            permuted=xval.copy();permuted[:,bit]=rng.permutation(permuted[:,bit])
            changes.append(float(mean_absolute_error(yval,model.predict(permuted))-original))
        rows.append({'bit':int(bit),'impurity_importance':float(model.feature_importances_[bit]),
                     'permutation_MAE_increase':float(np.mean(changes))})
    pd.DataFrame(rows).to_csv(RESULTS/'feature_importances.csv',index=False)
    generator=fingerprint_generator();occurrences=[]
    relevant=molecules.loc[molecules.molecule_index.isin(block.molecule_index)]
    for record in relevant.head(3000).itertuples():
        mol=Chem.MolFromSmiles(record.smiles)
        output=rdFingerprintGenerator.AdditionalOutput();output.CollectBitInfoMap()
        generator.GetFingerprint(mol,additionalOutput=output)
        info=output.GetBitInfoMap()
        for bit in top[:5]:
            if int(bit) in info:
                atom,radius=info[int(bit)][0]
                env=Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atom)
                fragment=Chem.MolToSmiles(Chem.PathToSubmol(mol,env)) if radius else mol.GetAtomWithIdx(atom).GetSymbol()
                occurrences.append({'bit':int(bit),'compound_id':record.compound_id,'atom':atom,'radius':radius,'fragment':fragment})
    pd.DataFrame(occurrences).drop_duplicates(['bit','fragment']).to_csv(RESULTS/'fingerprint_bit_examples.csv',index=False)
    summary={'target_id':target,'target_name':block.target_name.iloc[0],'validation_examples':len(valid),
              'validation_MAE':float(original),'top_bits':rows,
              'note':'Permutation importance is measured on validation only; a hashed bit can represent multiple environments and is not a causal mechanism.'}
    save_json(path,summary)
    model_path=ROOT/'models'/f'{target}_example.joblib'
    joblib.dump(model,model_path,compress=3)
    return summary
