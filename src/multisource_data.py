"""Source-aware affinity tables. Different endpoints never share a label scale."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs

from affinity_eda import ROOT, CFG, structure_features, fingerprint_generator, grouped_split, save_json, sha256

RAW = ROOT / 'data/raw/multisource'
LEGACY_DATA = ROOT / 'data/multisource'
DATA = ROOT / 'data/multisource_v3'
RESULTS = ROOT / 'results/multisource'
PIPELINE_VERSION = 3
INCLUDED_SOURCES = ('ChEMBL', 'BindingDB', 'DrugCentral')
MUTATION = r'(?i:mutant|mutation)|\b[A-Z][0-9]{2,4}[A-Z]\b'
BASE_COLUMNS = ['source','provider','source_record_id','source_study_id','upstream_source',
                'source_smiles','target_id','target_name','endpoint','label','protein_sequence']


def init():
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def extract_bindingdb():
    """Stream the TSV, retaining native human single-chain Ki only."""
    init()
    path = DATA / 'bindingdb_observations_unchecked.parquet'
    audit_path=RESULTS/'bindingdb_extraction.json'
    if path.exists() and audit_path.exists() and json.loads(audit_path.read_text()).get('parser_version')==PIPELINE_VERSION:
        return pd.read_parquet(path)
    fields = ['BindingDB Reactant_set_id','Ligand SMILES','Target Name',
              'Target Source Organism According to Curator or DataSource','Ki (nM)','Kd (nM)','IC50 (nM)',
              'Curation/DataSource','Article DOI','PMID','PubChem AID',
              'Number of Protein Chains in Target (>1 implies a multichain complex)',
              'BindingDB Target Chain Sequence 1','UniProt (SwissProt) Primary ID of Target Chain 1',
              'UniProt (TrEMBL) Primary ID of Target Chain 1']
    rows=[];audit=Counter();origins=Counter()
    with zipfile.ZipFile(RAW/'bindingdb_202610.tsv.zip') as archive:
        member=next(n for n in archive.namelist() if n.endswith('.tsv'))
        with archive.open(member) as stream:
            chunks=pd.read_csv(stream,sep='\t',dtype=str,usecols=fields,chunksize=50000,
                               quoting=csv.QUOTE_NONE,keep_default_na=False)
            for i,chunk in enumerate(chunks,1):
                audit['raw_rows']+=len(chunk)
                origins.update(chunk['Curation/DataSource'])
                work=chunk.loc[chunk[fields[3]].str.strip().eq('Homo sapiens') & chunk[fields[11]].eq('1')].copy()
                audit['human_single_chain_rows']+=len(work)
                # ChEMBL is already imported with the stricter original assay-quality filters.
                work=work.loc[work['Curation/DataSource'].isin([
                    'Curated from the literature by BindingDB','US Patent','WIPO'])].copy()
                audit['native_bindingdb_rows']+=len(work)
                work['target_id']=work[fields[13]].where(work[fields[13]].ne(''),work[fields[14]])
                work=work.loc[work.target_id.str.fullmatch(r'[A-Z0-9]{6,10}(?:-\d+)?') &
                              ~work['Target Name'].str.contains(MUTATION,regex=True)].copy()
                audit['mapped_without_explicit_mutation']+=len(work)
                for measure,endpoint in [('Ki (nM)','pKi')]:
                    selection=work
                    values=pd.to_numeric(selection[measure].str.strip(),errors='coerce')
                    ok=np.isfinite(values)&values.gt(0)
                    selection=selection.loc[ok].copy()
                    if selection.empty: continue
                    frame=pd.DataFrame({
                        'source':'BindingDB', 'provider':'BindingDB/202610',
                        'source_record_id':'BDB_'+selection['BindingDB Reactant_set_id']+'_'+endpoint,
                        'source_study_id':selection['Article DOI'].where(selection['Article DOI'].ne(''),selection.PMID),
                        'upstream_source':selection['Curation/DataSource'],
                        'source_smiles':selection['Ligand SMILES'],'target_id':selection.target_id,
                        'target_name':selection['Target Name'],'endpoint':endpoint,
                        'label':9-np.log10(values.loc[ok]),'protein_sequence':selection['BindingDB Target Chain Sequence 1']})
                    rows.append(frame)
                if i%10==0: print(f'BindingDB scanned {audit["raw_rows"]:,} rows',flush=True)
    observations=pd.concat(rows,ignore_index=True)
    if not observations.source.eq('BindingDB').any():
        raise ValueError('No native BindingDB observations: review the upstream curation-source categories')
    observations.to_parquet(path,index=False)
    save_json(audit_path,{'parser_version':PIPELINE_VERSION,'funnel':dict(audit),'curation_sources':dict(origins),
              'retained_by_source_endpoint':observations.groupby(['source','endpoint']).size().rename('rows').reset_index().to_dict('records')})
    print('BindingDB retained',len(observations),flush=True)
    return observations


def chembl_observations():
    pairs=pd.read_parquet(ROOT/'data/processed/pairs.parquet')
    molecules=pd.read_parquet(ROOT/'data/processed/molecules.parquet')
    components=pd.read_csv(ROOT/'data/processed/target_components.csv')
    counts=components.groupby('target_chembl_id').size()
    mapping=components.loc[components.target_chembl_id.isin(counts[counts.eq(1)].index)].set_index('target_chembl_id').accession
    data=pairs.merge(molecules[['compound_id','smiles']],on='compound_id',validate='many_to_one')
    data['target_id']=data.target_chembl_id.map(mapping)
    data=data.loc[data.target_id.notna()].copy()
    frame=pd.DataFrame({'source':'ChEMBL','provider':'ChEMBL_37',
        'source_record_id':data.compound_id+'_'+data.target_chembl_id,
        'source_study_id':'','upstream_source':'ChEMBL', 'source_smiles':data.smiles,
        'target_id':data.target_id,'target_name':data.target_name,'endpoint':'pKi',
        'label':data.pki,'protein_sequence':''})
    frame['original_chembl_target_id']=data.target_chembl_id
    return frame


def drugcentral_observations():
    path=DATA/'drugcentral_observations_unchecked.parquet'
    if path.exists():return pd.read_parquet(path)
    structures={}
    with gzip.open(RAW/'drugcentral_structures.sdf.gz','rb') as stream:
        for mol in Chem.ForwardSDMolSupplier(stream):
            if mol is not None and mol.HasProp('ID'):
                structures[int(mol.GetProp('ID'))]=Chem.MolToSmiles(mol,isomericSmiles=True)
    raw=pd.read_csv(RAW/'drugcentral_interactions.tsv.gz',sep='\t')
    work=raw.loc[raw.ORGANISM.eq('Homo sapiens') & raw.RELATION.eq('=') &
                 raw.ACT_TYPE.eq('Ki') & np.isfinite(raw.ACT_VALUE) &
                 ~raw.ACT_COMMENT.fillna('').str.contains(MUTATION,regex=True)].copy()
    work['source_smiles']=work.STRUCT_ID.map(structures)
    work=work.loc[work.source_smiles.notna() & work.ACCESSION.str.fullmatch(r'[A-Z0-9]{6,10}(?:-\d+)?',na=False)].copy()
    frame=pd.DataFrame({'source':'DrugCentral','provider':'DrugCentral/2027',
        'source_record_id':'DC_'+work.index.astype(str),'source_study_id':work.ACT_SOURCE_URL.fillna(''),
        'upstream_source':work.ACT_SOURCE.fillna(''),'source_smiles':work.source_smiles,
        'target_id':work.ACCESSION,'target_name':work.TARGET_NAME,'endpoint':'p'+work.ACT_TYPE,
        # DrugCentral ACT_VALUE is already on the negative-log-molar scale.
        'label':work.ACT_VALUE,'protein_sequence':''})
    frame.to_parquet(path,index=False)
    save_json(RESULTS/'drugcentral_extraction.json',{'raw_rows':len(raw),'retained_rows':len(frame),
        'scale':'ACT_VALUE is negative log10(molar), not a concentration',
        'references':['https://zitniklab.hms.harvard.edu/ToolUniverse/tools/drugcentral_tools.html',
                      'https://academic.oup.com/nar/article/45/D1/D932/2333938'],
        'rows_by_endpoint':frame.endpoint.value_counts().to_dict(),
        'upstream_sources':frame.upstream_source.value_counts().to_dict()})
    return frame


def proteins(accessions):
    """Fetch canonical human protein sequences for identity/mutation checks and DL."""
    requested=set(accessions)
    for cached in [DATA/'proteins.parquet',LEGACY_DATA/'proteins.parquet']:
        if cached.exists():
            table=pd.read_parquet(cached)
            if requested<=set(table.Entry):
                result=table.loc[table.Entry.isin(requested)].copy()
                result.to_parquet(DATA/'proteins.parquet',index=False)
                return result
    directory=RAW/'uniprot';directory.mkdir(exist_ok=True)
    batches=[sorted(set(accessions))[i:i+75] for i in range(0,len(set(accessions)),75)]
    def one(batch):
        query=' OR '.join('accession:'+a for a in batch)
        key=hashlib.sha256(query.encode()).hexdigest()
        path=directory/(key+'.tsv')
        if not path.exists():
            response=requests.get('https://rest.uniprot.org/uniprotkb/stream',params={
                'query':query,'format':'tsv','fields':'accession,id,organism_id,protein_name,gene_primary,sequence'},timeout=120)
            response.raise_for_status();path.write_text(response.text)
        data=pd.read_csv(path,sep='\t')
        return data
    with ThreadPoolExecutor(max_workers=4) as pool:
        frames=list(pool.map(one,batches))
    result=pd.concat(frames,ignore_index=True).drop_duplicates('Entry')
    result.to_parquet(DATA/'proteins.parquet',index=False)
    return result


def normalise_structures(observations):
    path=DATA/'structure_lookup.parquet'
    known=pd.read_parquet(ROOT/'data/processed/structure_audit.parquet')
    # The original processed SMILES may differ from their source after salt removal.
    original=pd.read_parquet(ROOT/'data/processed/molecules.parquet')
    original=original.drop(columns=['molecule_index','random_split','scaffold_split']).copy()
    original['source_smiles']=original.smiles;original['structure_status']='valid'
    frames=[known,original]
    for cached in [LEGACY_DATA/'structure_lookup.parquet',path]:
        if cached.exists():frames.append(pd.read_parquet(cached))
    known=pd.concat(frames,ignore_index=True).drop_duplicates('source_smiles',keep='last')
    missing=sorted(set(observations.source_smiles)-set(known.source_smiles))
    if missing:
        print(f'Standardising {len(missing):,} additional structures',flush=True)
        features=[]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for i,record in enumerate(pool.map(structure_features,missing),1):
                features.append(record)
                if i%20000==0:print('Additional structures',i,'/',len(missing),flush=True)
        known=pd.concat([known,pd.DataFrame(features)],ignore_index=True)
        known.to_parquet(path,index=False)
    return known


def extend_split(molecules,column,original_split):
    original=pd.read_parquet(ROOT/'data/processed/molecules.parquet')
    assert original.groupby(column)[original_split].nunique().max()==1
    lookup=original.drop_duplicates(column).set_index(column)[original_split]
    def new_group(value):
        fraction=int(hashlib.sha256((str(CFG['seed'])+'|'+value).encode()).hexdigest()[:16],16)/16**16
        return 'train' if fraction<.8 else ('validation' if fraction<.9 else 'test')
    return molecules[column].map(lambda value:lookup[value] if value in lookup.index else new_group(value))


def prepare():
    init()
    complete=RESULTS/'data_summary.json'
    if complete.exists():
        prior=json.loads(complete.read_text())
        if prior.get('pipeline_version')==PIPELINE_VERSION and (DATA/'pairs.parquet').exists():
            return prior
    legacy=LEGACY_DATA/'observations.parquet'
    if legacy.exists():
        previous=pd.read_parquet(legacy)
        all_obs=previous.reindex(columns=BASE_COLUMNS+['original_chembl_target_id']).copy()
        input_scope='Previously curated observations; source/endpoint policy reapplied and labels reaggregated.'
    else:
        bdb=extract_bindingdb();chembl=chembl_observations();dc=drugcentral_observations()
        all_obs=pd.concat([chembl,bdb,dc],ignore_index=True)
        input_scope='Source measurements before canonical-sequence and structure checks.'
    all_obs=all_obs.loc[all_obs.source.isin(INCLUDED_SOURCES)&all_obs.endpoint.eq('pKi')].copy()
    all_obs=all_obs.loc[~all_obs.upstream_source.fillna('').str.contains('PubChem',case=False)].copy()
    protein_table=proteins(all_obs.target_id.dropna().unique().tolist())
    human=set(protein_table.loc[protein_table['Organism (ID)'].eq(9606),'Entry'])
    all_obs=all_obs.loc[all_obs.target_id.isin(human)].copy()
    canonical=protein_table.set_index('Entry').Sequence.to_dict()
    checked=all_obs.source.eq('BindingDB')
    def matches(row):
        sequence=row.protein_sequence
        return isinstance(sequence,str) and len(sequence)>=50 and sequence in canonical.get(row.target_id,'')
    match=all_obs.loc[checked,['target_id','protein_sequence']].apply(matches,axis=1)
    bad_index=match.index[~match]
    sequence_audit={'bindingdb_rows_checked':int(checked.sum()),'rows_without_canonical_sequence_match':len(bad_index),
                    'input_scope':input_scope,
                    'rule':'single-chain experimental sequence must be >=50 aa and an exact substring of canonical human UniProt sequence'}
    all_obs=all_obs.drop(index=bad_index).reset_index(drop=True)
    save_json(RESULTS/'sequence_audit.json',sequence_audit)
    structures=normalise_structures(all_obs)
    features=[c for c in structures if c not in ['source_smiles','structure_status']]
    valid=structures.loc[structures.structure_status.eq('valid'),['source_smiles',*features]].copy()
    before=len(all_obs)
    all_obs=all_obs.merge(valid,on='source_smiles',how='inner',validate='many_to_one')
    all_obs=all_obs.drop_duplicates(['source','source_record_id','endpoint'])
    all_obs.to_parquet(DATA/'observations.parquet',index=False)
    # Each source first contributes one label per pair; cross-source copies are
    # resolved by a declared priority rather than counted as independent evidence.
    keys=['compound_id','target_id','endpoint','source']
    estimates=all_obs.groupby(keys).agg(label=('label','median'),within_source_range=('label',lambda s:s.max()-s.min()),
        source_records=('source_record_id','nunique'),target_name=('target_name','first')).reset_index()
    conflicts=estimates.loc[estimates.within_source_range.gt(2)].copy()
    conflicts.to_parquet(DATA/'within_source_conflicts.parquet',index=False)
    estimates=estimates.loc[estimates.within_source_range.le(2)].copy()
    priority={'ChEMBL':0,'BindingDB':1,'DrugCentral':2}
    estimates['priority']=estimates.source.map(priority)
    pair_keys=['compound_id','target_id','endpoint']
    selected=estimates.sort_values('priority').drop_duplicates(pair_keys).copy()
    evidence=estimates.groupby(pair_keys).agg(sources=('source',lambda s:sorted(set(s))),
        source_count=('source','nunique'),between_source_min=('label','min'),between_source_max=('label','max')).reset_index()
    selected=selected.merge(evidence,on=pair_keys,validate='one_to_one')
    selected['cross_source_range']=selected.between_source_max-selected.between_source_min
    selected.loc[selected.cross_source_range.gt(2)].to_parquet(DATA/'cross_source_conflicts.parquet',index=False)
    # Preserve original curated ChEMBL labels, including their existing test set.
    # Conflicting auxiliary evidence is flagged, not allowed to change the old test.
    selected=selected.loc[selected.source.eq('ChEMBL')|selected.cross_source_range.le(2)].copy()
    molecule_columns=['compound_id','smiles','connectivity','scaffold','molecular_weight','logp','tpsa',
                      'heavy_atoms','hbd','hba','rotatable_bonds','rings','fragment_removed']
    molecules=all_obs[molecule_columns].drop_duplicates('compound_id').sort_values('compound_id').reset_index(drop=True)
    molecules=molecules.loc[molecules.compound_id.isin(selected.compound_id)].reset_index(drop=True)
    molecules['molecule_index']=np.arange(len(molecules))
    molecules['scaffold_split']=extend_split(molecules,'scaffold','scaffold_split')
    molecules['random_split']=extend_split(molecules,'connectivity','random_split')
    selected=selected.merge(molecules[['compound_id','molecule_index','random_split','scaffold_split']],on='compound_id',validate='many_to_one')
    molecules.to_parquet(DATA/'molecules.parquet',index=False)
    selected.to_parquet(DATA/'pairs.parquet',index=False)
    estimates.to_parquet(DATA/'source_estimates.parquet',index=False)
    summary={'pipeline_version':PIPELINE_VERSION,'included_sources':list(INCLUDED_SOURCES),'input_scope':input_scope,
        'observations':len(all_obs),'pairs':len(selected),'molecules':len(molecules),
        'structure_rejected_observations':before-len(all_obs),'sequence_audit':sequence_audit,
        'within_source_conflicting_pairs':len(conflicts),
        'rows_by_source_endpoint':all_obs.groupby(['source','endpoint']).size().rename('rows').reset_index().to_dict('records'),
        'pairs_by_endpoint':selected.endpoint.value_counts().to_dict(),
        'source_priority':priority,'split_policy':'Original ChEMBL group assignments are fixed; new groups use SHA256(seed|group).',
        'cross_source_policy':'Prefer ChEMBL > native BindingDB > DrugCentral; flag ranges >2 and preserve original ChEMBL labels.'}
    save_json(complete,summary)
    return summary


def fingerprints(molecules):
    path=DATA/'fingerprints.npz'
    if path.exists():
        saved=np.load(path)
        if np.array_equal(saved['compound_ids'],molecules.compound_id.to_numpy()):return saved['X']
        raise ValueError('Fingerprint cache does not match molecule ordering')
    previous=LEGACY_DATA/'fingerprints.npz'
    if previous.exists():
        saved=np.load(previous)
        indices=pd.Index(saved['compound_ids']).get_indexer(molecules.compound_id)
        if (indices>=0).all():
            X=saved['X'][indices]
            np.savez_compressed(path,X=X,compound_ids=molecules.compound_id.to_numpy(dtype=str))
            return X
    gen=fingerprint_generator();X=np.zeros((len(molecules),2048),dtype=np.uint8)
    for start in range(0,len(molecules),2000):
        mols=[Chem.MolFromSmiles(s) for s in molecules.smiles.iloc[start:start+2000]]
        for j,fp in enumerate(gen.GetFingerprints(mols,numThreads=4)):
            DataStructs.ConvertToNumpyArray(fp,X[start+j])
        if start%20000==0:print('Multisource fingerprints',start,'/',len(molecules),flush=True)
    np.savez_compressed(path,X=X,compound_ids=molecules.compound_id.to_numpy(dtype=str))
    return X


def benchmark(name):
    init()
    destination=DATA/(name+'_pairs.parquet')
    molecule_path=DATA/(name+'_molecules.parquet')
    if destination.exists():return pd.read_parquet(destination),pd.read_parquet(molecule_path)
    raw=pd.read_csv(RAW/(name+'.tab'),sep='\t')
    source_structures=pd.DataFrame([structure_features(s) for s in sorted(raw.X1.unique())])
    valid=source_structures.loc[source_structures.structure_status.eq('valid')].copy()
    work=raw.merge(valid,left_on='X1',right_on='source_smiles',validate='many_to_one')
    work['label']=9-np.log10(work.Y) if name=='davis' else work.Y
    work['endpoint']='pKd' if name=='davis' else 'KIBA_score'
    work['ceiling_censored']=work.Y.eq(10000) if name=='davis' else False
    work['target_id']=work.ID2.astype(str)
    work['target_name']=work.target_id
    molecules=valid.drop(columns=['source_smiles','structure_status']).drop_duplicates('compound_id').sort_values('compound_id').reset_index(drop=True)
    molecules['molecule_index']=np.arange(len(molecules))
    molecules['scaffold_split']=grouped_split(molecules,'scaffold',42)
    work=work.drop(columns='molecule_index',errors='ignore').merge(molecules[['compound_id','molecule_index','scaffold_split']],on='compound_id',validate='many_to_one')
    # Duplicate structural records are one labelled pair for each benchmark.
    pairs=work.groupby(['compound_id','target_id','endpoint']).agg(label=('label','median'),
        ceiling_censored=('ceiling_censored','max'),protein_sequence=('X2','first'),target_name=('target_name','first'),
        molecule_index=('molecule_index','first'),scaffold_split=('scaffold_split','first')).reset_index()
    pairs.to_parquet(destination,index=False);molecules.to_parquet(molecule_path,index=False)
    save_json(RESULTS/(name+'_summary.json'),{'raw_pairs':len(raw),'prepared_pairs':len(pairs),
        'molecules':len(molecules),'targets':pairs.target_id.nunique(),
        'endpoint':pairs.endpoint.iloc[0],'ceiling_censored_pairs':int(pairs.ceiling_censored.sum()),
        'split_molecules':molecules.scaffold_split.value_counts().to_dict(),
        'evaluation_policy':'Standalone benchmark, trained on its own train split; not an independent external validation of the pKi model.'})
    return pairs,molecules
