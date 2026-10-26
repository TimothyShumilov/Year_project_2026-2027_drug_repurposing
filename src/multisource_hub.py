"""Build a documented, checksummed Hugging Face dataset with explicit subsets."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import HfApi, get_token

from affinity_eda import save_json,sha256
from multisource_data import ROOT,DATA,RESULTS
from multisource_models import CONFIG,original_keys
from pki_sources import build_report, usage_markdown, preparation_markdown

STAGING=ROOT/'output/huggingface/drug-affinity-pki'


def package():
    STAGING.mkdir(parents=True,exist_ok=True)
    usage=build_report()
    pairs=pd.read_parquet(DATA/'pairs.parquet').query("endpoint == 'pKi'").copy()
    original_molecules=pd.read_parquet(DATA/'molecules.parquet')
    molecules=original_molecules.loc[original_molecules.compound_id.isin(pairs.compound_id)].copy().reset_index(drop=True)
    old_indices=molecules.molecule_index.to_numpy().copy()
    molecules['molecule_index']=np.arange(len(molecules))
    indices=molecules.set_index('compound_id').molecule_index
    pairs['molecule_index']=pairs.compound_id.map(indices)
    proteins=pd.read_parquet(DATA/'proteins.parquet')
    proteins=proteins.loc[proteins.Entry.isin(pairs.target_id)].copy()
    configurations=[];counts=[]
    def write(name,frame,split_column=None,notes=''):
        directory=STAGING/'data'/name;directory.mkdir(parents=True,exist_ok=True)
        paths=[]
        blocks=frame.groupby(split_column,sort=True) if split_column else [('train',frame)]
        for part,block in blocks:
            path=directory/(part+'.parquet')
            block.to_parquet(path,index=False)
            paths.append({'split':part,'path':str(path.relative_to(STAGING))})
            counts.append({'config':name,'split':part,'rows':len(block)})
        configurations.append({'config_name':name,'data_files':paths,**({'default':True} if name=='affinity_pki' else {})})
    merged=pairs.merge(molecules.drop(columns=['molecule_index','random_split','scaffold_split']),on='compound_id',validate='many_to_one')
    target_counts=pd.read_csv(RESULTS/'pKi_target_counts.csv',index_col=0)
    merged['benchmark_target']=merged.target_id.isin(target_counts.index[target_counts.eligible])
    write('affinity_pki',merged,'scaffold_split')
    reference=original_keys().merge(molecules[['compound_id','smiles']],on='compound_id',validate='many_to_one')
    reference['label']=reference.pki;reference['endpoint']='pKi';reference['source']='ChEMBL'
    write('chembl_reference',reference,'scaffold_split')
    write('molecules',molecules)
    observations=pd.read_parquet(DATA/'observations.parquet').query("endpoint == 'pKi'").copy()
    observation_columns=['source','provider','upstream_source','source_record_id','source_study_id',
        'compound_id','smiles','target_id','target_name','endpoint','label','pubchem_aid']
    write('source_observations',observations.reindex(columns=observation_columns))
    aids={int(a.removeprefix('aid')) for a in usage['pubchem_pki_assay_ids']}
    assays=pd.read_parquet(DATA/'pubchem_assays.parquet')
    assays=assays.loc[assays.aid.isin(aids)].copy()
    write('pubchem_assays',assays)
    npz=STAGING/'features';npz.mkdir(exist_ok=True)
    fingerprints=np.load(DATA/'fingerprints.npz')
    np.savez_compressed(npz/'fingerprints.npz',X=fingerprints['X'][old_indices],
                        compound_ids=molecules.compound_id.to_numpy(dtype=str))
    prepared=STAGING/'prepared';prepared.mkdir(exist_ok=True)
    for name,frame in [('pairs',pairs),('molecules',molecules),('observations',observations),
                       ('proteins',proteins),('pubchem_assays',assays)]:
        frame.to_parquet(prepared/(name+'.parquet'),index=False)
    for name in ['source_estimates','within_source_conflicts','cross_source_conflicts']:
        pd.read_parquet(DATA/(name+'.parquet')).query("endpoint == 'pKi'").to_parquet(prepared/(name+'.parquet'),index=False)
    reference_files=STAGING/'reference/chembl';reference_files.mkdir(parents=True,exist_ok=True)
    for path in (ROOT/'data/processed').glob('*'):
        if path.is_file():shutil.copy2(path,reference_files/path.name)
    (reference_files/'LICENSE.md').write_text(
        'Original prepared ChEMBL 37 reference snapshot: CC BY-SA 3.0 Unported.\n'
        'EMBL-EBI ChEMBL, https://www.ebi.ac.uk/chembl/ .\n'
        'https://creativecommons.org/licenses/by-sa/3.0/ .\n'
        'Changes from source: Ki filtering, structure checks, aggregation, descriptors and fixed splits.\n')
    for name in ['baseline_summary.json','original_test_comparison.csv','source_test_slices.csv',
                 'source_registry.json','drugcentral_scale_audit.json','pki_source_usage.json','pki_source_usage.csv']:
        path=RESULTS/name
        if path.exists():
            target=STAGING/'metadata'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
    evaluation=STAGING/'evaluation';evaluation.mkdir(exist_ok=True)
    predictions=pd.read_parquet(RESULTS/'pKi/test_predictions.parquet')
    predictions['molecule_index']=predictions.compound_id.map(indices)
    predictions.to_parquet(evaluation/'pKi_test_predictions.parquet',index=False)
    downloads=[]
    for path in (ROOT/'data/raw/multisource').glob('*.download.json'):
        if path.name.startswith(('bindingdb_','drugcentral_')):
            downloads.append(json.loads(path.read_text()))
    save_json(STAGING/'metadata/upstream_downloads.json',downloads)
    shutil.copy2(ROOT/'config/multisource.json',STAGING/'metadata/experiment_config.json')
    summary=json.loads((RESULTS/'data_summary.json').read_text())
    summary.update(observations=len(observations),pairs=len(pairs),molecules=len(molecules),
                   pairs_by_endpoint={'pKi':len(pairs)},
                   rows_by_source_endpoint=[r for r in summary['rows_by_source_endpoint'] if r['endpoint']=='pKi'])
    summary.pop('sequence_audit',None)
    summary.pop('within_source_conflicting_pairs',None)
    summary.pop('structure_rejected_observations',None)
    summary['scope']='pKi only; source usage in pki_source_usage.json'
    save_json(STAGING/'metadata/data_summary.json',summary)
    model=json.loads((RESULTS/'baseline_summary.json').read_text())
    metadata={'pretty_name':'pKi baseline: ChEMBL, BindingDB, PubChem and DrugCentral',
              'license':'cc-by-sa-4.0','language':['en'],'task_categories':['tabular-regression'],
              'tags':['chemistry','drug-discovery','molecular-property-prediction','random-forest'],
              'configs':configurations}
    counts_table=pd.DataFrame(counts).pivot(index='config',columns='split',values='rows').fillna(0).astype(int)
    table='| Subset | train | validation | test |\n|---|---:|---:|---:|\n'+'\n'.join(
        f'| {name} | {row.get("train",0):,} | {row.get("validation",0):,} | {row.get("test",0):,} |'
        for name,row in counts_table.iterrows())
    text=f'''# pKi data for the Morgan/RF baseline

Prepared public-source snapshots, molecular structures, labels, source provenance and fixed splits.
Main curated corpus: **{summary['pairs_by_endpoint']['pKi']:,} pKi pairs**; the reported baseline uses **{usage['benchmark_pairs']:,} pairs for {usage['benchmark_targets']} targets**. This revision contains only the pKi task.

```python
from datasets import load_dataset
data = load_dataset("{CONFIG['hf_repository']}", "affinity_pki")
train = data["train"].filter(lambda row: row["benchmark_target"])
```

{table}

`molecules`, `source_observations` and `pubchem_assays` are reference tables. Their `train` split is a storage convention. Use the affinity tables' `scaffold_split` labels when joining them.

## Labels and features

The model predicts **pKi**: `pKi = -log10(Ki in mol/L) = 9 - log10(Ki in nM)`. DrugCentral ACT_VALUE is already pKi. Larger pKi means stronger interaction.

`smiles` describes the molecular structure. `target_id` is a UniProt accession identifying which per-target RF to use. The RF input is only the radius-2, 2048-bit chiral Morgan fingerprint. Protein sequences in `prepared/proteins.parquet` support curation checks; they are not model inputs or separate affinity labels. `features/fingerprints.npz` stores `X` and matching `compound_ids`, aligned with `molecule_index`.

## Sources, actual proportions and modifications

The table counts each selected compound/target label once, after curation and baseline target selection. Percentages are within each source row. Headers are Russian to match the executed EDA.

{usage_markdown(usage)}

Only **15,439 original ChEMBL validation pairs** select hyperparameters. The other **3,201 validation pairs** are diagnostic. PubChem has no validation pairs in this baseline. All test pairs are reserved for final evaluation. These are globally held-out scaffolds, not independent external source cohorts.

{preparation_markdown(usage)}

The source-usage table and exact counts are also available in `metadata/pki_source_usage.csv` and `.json`.

## Curation and source overlap

Only exact, finite affinity measurements are used. Main ChEMBL annotations follow confidence 9, human single-protein and no annotated-variant rules. Native BindingDB and PubChem observations require a single human protein and an experimental sequence of at least 50 residues matching a contiguous segment of the canonical UniProt sequence. Organic fragments are standardised while retaining charge and stereochemistry; ambiguous mixtures and structures with more than 100 heavy atoms are excluded.

Repeated labels are reduced to a median within each source/pair/endpoint. Ranges above two p-units are quarantined. Cross-source copies become one pair using ChEMBL > native BindingDB > PubChem > DrugCentral priority. Discordant auxiliary evidence is flagged; original ChEMBL labels and splits are preserved. Thus extra copies do not count as independent training examples. `sources`, `source_count`, `cross_source_range`, and `source_observations` retain the provenance.

PubChem observations were obtained through the curated BindingDB import, with BioAssay identifiers checked through PubChem PUG REST. DrugCentral itself aggregates sources; its `upstream_source` is retained. Source slices of held-out predictions are not independent external cohorts.

## Evaluation

Original ChEMBL scaffold assignments are immutable. New structures sharing an old scaffold inherit its split; new scaffolds use SHA256(seed|scaffold) for nominal 80/10/10 allocation. All four sources share those molecule groups, so actual row proportions vary. Original `chembl_reference` validation/test labels are unchanged.

`benchmark_target` marks targets meeting the declared train/validation/test coverage criteria. Sparse unsupported targets remain available for research; filter this flag to reproduce the reported baseline.

All experiments used validation-only hyperparameter selection. On the same original 131-target test, RF MAE was {model['original_test_comparison'][0]['MAE']:.3f} pKi with ChEMBL alone and {model['original_test_comparison'][1]['MAE']:.3f} after source expansion. Additional data did not improve that test. Full metrics and counts are in `metadata/baseline_summary.json`.

## Limitations

Unknown activity is not a negative label. Missing variant metadata does not establish wild-type status. Laboratories, assay conditions and source selection differ. Four PubChem test pairs cannot support a reliable source-wide quality estimate. Predictions describe molecular interactions and require experimental validation for biological or therapeutic use.

## Provenance, licenses and attribution

The transformed integrated collection is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Unchanged original reference files in `reference/chembl/` retain CC BY-SA 3.0. Original source notices remain applicable; see [LICENSES.md](LICENSES.md). Changes include filtering, structure standardisation, sequence checks, label transformation/aggregation, source reconciliation and fixed splits.

Code and executed EDA: [annual project repository](https://github.com/TimothyShumilov/Year_project_2026-2027_drug_repurposing/tree/codex/checkpoint-2-eda-rf). SHA-256 file checksums and retrieval records accompany this release.

For exact local reproduction of the RF runs, download this repository and copy `prepared/*.parquet` to the project's `data/multisource/`, `features/fingerprints.npz` to that same directory, and `reference/chembl/*` to `data/processed/`. Run `scripts/run_multisource.py --stage train` and compare its metrics with the supplied metadata.
'''
    (STAGING/'README.md').write_text('---\n'+yaml.safe_dump(metadata,sort_keys=False,allow_unicode=True)+'---\n\n'+text)
    licenses='''# Source licenses and citations

Original grants and attributions are preserved; this collection does not claim ownership of underlying facts.

- **ChEMBL 37, EMBL-EBI:** CC BY-SA 3.0. Zdrazil et al., *The ChEMBL Database in 2023*, doi:10.1093/nar/gkad1004. https://www.ebi.ac.uk/chembl/ and https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions . The transformed integrated tables use the later BY-SA 4.0 version allowed by section 4(b) of https://creativecommons.org/licenses/by-sa/3.0/legalcode . Original BY-SA 3.0 notices remain intact.
- **BindingDB 202610:** native literature and patent observations; see https://www.bindingdb.org/rwd/bind/info.jsp (CC BY 3.0) and the authors' 2024 update doi:10.1093/nar/gkae1075 (CC BY 4.0). Imported ChEMBL rows were not independently re-added. Attribution, source flags, URLs and record identifiers are preserved. Publication: https://academic.oup.com/nar/article/53/D1/D1633/7906836 .
- **PubChem BioAssay:** quantitative observations imported through BindingDB, original assay IDs and source metadata retained. PubChem does not claim a new license over depositor content. See https://pubchem.ncbi.nlm.nih.gov/docs/use-of-pubchem-data and https://pubchem.ncbi.nlm.nih.gov/docs/bioassays . Only factual identifiers, measurements, result-field names and assay titles are included; depositor protocols and descriptions are not republished.
- **DrugCentral:** CC BY-SA 4.0, https://www.drugcentral.org/privacy . Ursu et al., *DrugCentral: online drug compendium*, doi:10.1093/nar/gkw993. Numeric source values are already on the negative-log-molar scale; original ACT_SOURCE is retained.
- **UniProt curation support:** CC BY 4.0, https://www.uniprot.org/help/license . Canonical human sequences are retained only for identity/sequence checks, not as an additional training source or RF feature.
'''
    (STAGING/'LICENSES.md').write_text(licenses)
    files=[{'path':str(p.relative_to(STAGING)),'bytes':p.stat().st_size,'sha256':sha256(p)}
           for p in sorted(STAGING.rglob('*')) if p.is_file() and p.name!='manifest.json']
    manifest={'prepared_at_utc':datetime.now(timezone.utc).isoformat(),'repository':CONFIG['hf_repository'],
              'configurations':counts,'files':files}
    save_json(STAGING/'manifest.json',manifest)
    save_json(RESULTS/'hf_package.json',{'repository':CONFIG['hf_repository'],'local_path':str(STAGING),
        'files':len(files)+1,'bytes':sum(f['bytes'] for f in files),'configurations':counts,
        'publication_status':'prepared; upload requires authenticated write access'})
    return json.loads((RESULTS/'hf_package.json').read_text())


def publish():
    if get_token() is None:
        raise RuntimeError('Hugging Face login is missing. Run .venv/bin/hf auth login; do not put the token in source files or chat.')
    api=HfApi();identity=api.whoami()
    if identity['name']!='tim-shu':
        raise RuntimeError('Authenticated account differs from the requested tim-shu namespace')
    manifest=json.loads((STAGING/'manifest.json').read_text())
    for file in manifest['files']:
        if sha256(STAGING/file['path'])!=file['sha256']:raise ValueError('Package changed after validation')
    repo=CONFIG['hf_repository']
    api.create_repo(repo_id=repo,repo_type='dataset',private=False,exist_ok=True)
    expected=set(f['path'] for f in manifest['files'])|{'manifest.json'}
    existing=set(api.list_repo_files(repo_id=repo,repo_type='dataset'))
    obsolete=sorted(existing-expected-{'.gitattributes'})
    commit=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=STAGING,delete_patterns=obsolete,
        commit_message='Focus dataset on pKi baseline; document actual source splits and transformations')
    info=api.dataset_info(repo_id=repo,revision=commit.oid,files_metadata=True)
    remote={s.rfilename:s for s in info.siblings}
    if not expected<=set(remote):raise RuntimeError('Remote verification: missing files')
    if set(remote)-expected-{'.gitattributes'}:raise RuntimeError('Remote verification: obsolete files remain')
    for item in manifest['files']:
        if remote[item['path']].size!=item['bytes']:raise RuntimeError('Remote file size mismatch')
    record={'repository':repo,'url':'https://huggingface.co/datasets/'+repo,'revision':commit.oid,
            'uploaded_files':len(expected),'removed_obsolete_files':obsolete,
            'verified_at_utc':datetime.now(timezone.utc).isoformat()}
    save_json(RESULTS/'hf_publication.json',record)
    package_path=RESULTS/'hf_package.json'
    state=json.loads(package_path.read_text())
    state.update(publication_status='published',url=record['url'],revision=record['revision'])
    save_json(package_path,state)
    return record
