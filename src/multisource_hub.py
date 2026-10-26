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
from multisource_data import ROOT,DATA,RESULTS,benchmark
from multisource_models import CONFIG,original_keys

STAGING=ROOT/'output/huggingface/drug-affinity-multisource'


def package():
    STAGING.mkdir(parents=True,exist_ok=True)
    pairs=pd.read_parquet(DATA/'pairs.parquet')
    molecules=pd.read_parquet(DATA/'molecules.parquet')
    proteins=pd.read_parquet(DATA/'proteins.parquet')
    sequence=proteins.set_index('Entry').Sequence
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
    merged['protein_sequence']=merged.target_id.map(sequence)
    for endpoint,name in [('pKi','affinity_pki'),('pIC50','affinity_pic50'),('pKd','affinity_pkd')]:
        frame=merged.loc[merged.endpoint.eq(endpoint)].copy()
        count_path=RESULTS/(endpoint+'_target_counts.csv')
        target_counts=pd.read_csv(count_path,index_col=0)
        frame['benchmark_target']=frame.target_id.isin(target_counts.index[target_counts.eligible])
        write(name,frame,'scaffold_split')
    reference=original_keys().merge(molecules[['compound_id','smiles']],on='compound_id',validate='many_to_one')
    reference['protein_sequence']=reference.target_id.map(sequence)
    reference['label']=reference.pki;reference['endpoint']='pKi';reference['source']='ChEMBL'
    write('chembl_reference',reference,'scaffold_split')
    for name in ['davis','kiba']:
        p,m=benchmark(name)
        frame=p.merge(m[['compound_id','smiles','scaffold']],on='compound_id',validate='many_to_one')
        target_counts=pd.read_csv(RESULTS/(name+'_target_counts.csv'),index_col=0)
        frame['benchmark_target']=frame.target_id.isin(target_counts.index[target_counts.eligible])
        write(name,frame,'scaffold_split')
    write('molecules',molecules)
    write('proteins',proteins.rename(columns={'Entry':'target_id','Sequence':'protein_sequence'}))
    observations=pd.read_parquet(DATA/'observations.parquet')
    observation_columns=['source','provider','upstream_source','source_record_id','source_study_id',
        'compound_id','target_id','target_name','endpoint','label','pubchem_aid']
    write('source_observations',observations.reindex(columns=observation_columns))
    if (DATA/'pubchem_assays.parquet').exists():write('pubchem_assays',pd.read_parquet(DATA/'pubchem_assays.parquet'))
    approved=pd.read_parquet(ROOT/'data/processed/approved_library.parquet')
    write('approved_library',approved)
    npz=STAGING/'features';npz.mkdir(exist_ok=True)
    shutil.copy2(DATA/'fingerprints.npz',npz/'fingerprints.npz')
    prepared=STAGING/'prepared';prepared.mkdir(exist_ok=True)
    for path in DATA.glob('*.parquet'):
        if not path.name.endswith('_unchecked.parquet'):
            shutil.copy2(path,prepared/path.name)
    reference_files=STAGING/'reference/chembl';reference_files.mkdir(parents=True,exist_ok=True)
    for path in (ROOT/'data/processed').glob('*'):
        if path.is_file():shutil.copy2(path,reference_files/path.name)
    (reference_files/'LICENSE.md').write_text(
        'Original prepared ChEMBL 37 reference snapshot: CC BY-SA 3.0 Unported.\n'
        'EMBL-EBI ChEMBL, https://www.ebi.ac.uk/chembl/ .\n'
        'https://creativecommons.org/licenses/by-sa/3.0/ .\n'
        'Changes from source: Ki filtering, structure checks, aggregation, descriptors and fixed splits.\n')
    for name in ['data_summary.json','baseline_summary.json','original_test_comparison.csv','source_test_slices.csv',
                 'source_registry.json','sequence_audit.json','drugcentral_scale_audit.json','pubchem_metadata_audit.json']:
        path=RESULTS/name
        if path.exists():
            target=STAGING/'metadata'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
    evaluation=STAGING/'evaluation';evaluation.mkdir(exist_ok=True)
    for name in ['pKi','pIC50','davis','kiba']:
        shutil.copy2(RESULTS/name/'test_predictions.parquet',evaluation/(name+'_test_predictions.parquet'))
    downloads=[]
    for path in (ROOT/'data/raw/multisource').glob('*.download.json'):
        downloads.append(json.loads(path.read_text()))
    save_json(STAGING/'metadata/upstream_downloads.json',downloads)
    shutil.copy2(ROOT/'data/raw/multisource/tdc_datacite.json',STAGING/'metadata/tdc_doi_license.json')
    shutil.copy2(ROOT/'config/multisource.json',STAGING/'metadata/experiment_config.json')
    summary=json.loads((RESULTS/'data_summary.json').read_text())
    model=json.loads((RESULTS/'baseline_summary.json').read_text())
    metadata={'pretty_name':'Drug affinity: ChEMBL, BindingDB, PubChem, DrugCentral, DAVIS and KIBA',
              'license':'cc-by-sa-4.0','language':['en'],'task_categories':['tabular-regression'],
              'tags':['chemistry','drug-discovery','molecular-property-prediction','random-forest'],
              'configs':configurations}
    counts_table=pd.DataFrame(counts).pivot(index='config',columns='split',values='rows').fillna(0).astype(int)
    table='| Subset | train | validation | test |\n|---|---:|---:|---:|\n'+'\n'.join(
        f'| {name} | {row.get("train",0):,} | {row.get("validation",0):,} | {row.get("test",0):,} |'
        for name,row in counts_table.iterrows())
    text=f'''# Drug affinity data for a molecular ML/DL project

Prepared public-source snapshots, molecular structures, labels, source provenance and fixed splits.
Main integrated corpus: **{summary['pairs_by_endpoint']['pKi']:,} pKi pairs**. Additional pIC50 and pKd labels are separate.

```python
from datasets import load_dataset
data = load_dataset("{CONFIG['hf_repository']}", "affinity_pki")
train = data["train"].filter(lambda row: row["benchmark_target"])
```

{table}

`molecules`, `proteins`, `source_observations`, `pubchem_assays` and `approved_library` are reference tables. Their `train` split is a storage convention, not a training permission. Use the affinity tables' `scaffold_split` labels when joining them.

## Labels and features

`label` is pKi, pIC50, pKd or KIBA score as recorded by `endpoint`. Concentrations in nM are transformed as `9 - log10(value)`; DrugCentral ACT_VALUE is already on the p-scale. Different endpoints are never pooled into one regression target. Larger pKi/pKd/pIC50 means stronger measured activity within that endpoint.

`smiles` describes each prepared molecular structure. `target_id` is UniProt accession in the integrated corpus, the original kinase identifier in DAVIS, and the original protein identifier in KIBA. The baseline trains one RF per target with radius-2, 2048-bit chiral Morgan fingerprints. Protein sequences and descriptors are provided for audit and future DL; they are not RF inputs. `features/fingerprints.npz` stores `X` and matching `compound_ids`; it follows the `molecules` table's `molecule_index`.

## Curation and source overlap

Only exact, finite affinity measurements are used. Main ChEMBL annotations follow confidence 9, human single-protein and no annotated-variant rules. Native BindingDB and PubChem observations require a single human protein and an experimental sequence of at least 50 residues matching a contiguous segment of the canonical UniProt sequence. Organic fragments are standardised while retaining charge and stereochemistry; ambiguous mixtures and structures with more than 100 heavy atoms are excluded.

Repeated labels are reduced to a median within each source/pair/endpoint. Ranges above two p-units are quarantined. Cross-source copies become one pair using ChEMBL > native BindingDB > PubChem > DrugCentral priority. Discordant auxiliary evidence is flagged; original ChEMBL labels and splits are preserved. Thus extra copies do not count as independent training examples. `sources`, `source_count`, `cross_source_range`, and `source_observations` retain the provenance.

PubChem observations were obtained through the curated BindingDB import, with BioAssay identifiers checked through PubChem PUG REST. DrugCentral itself aggregates sources; its `upstream_source` is retained. Source slices of held-out predictions are not independent external cohorts.

## Evaluation

Original ChEMBL scaffold assignments are immutable. New structures sharing an old scaffold inherit its split; new scaffolds use SHA256(seed|scaffold) for 80/10/10 allocation. Every source and endpoint in the integrated corpus shares those molecule groups. Original `chembl_reference` validation/test labels are unchanged.

DAVIS and KIBA are **standalone benchmarks**, each trained using its own train split and selected on its own validation split. They are not external validation of the pKi model. Their molecules can overlap the integrated corpus; do not pool these configurations for a leakage-free evaluation. DAVIS values at 10,000 nM are ceiling-censored; `ceiling_censored` marks them, and uncensored results are reported separately.

`benchmark_target` marks targets meeting the declared train/validation/test coverage criteria. Sparse unsupported targets remain available for research but were not scored as supported RF heads. The integrated pKd subset has no target satisfying the frozen coverage thresholds, so it has no fitted RF benchmark in this release.

All experiments used validation-only hyperparameter selection. On the same original 131-target test, RF MAE was {model['original_test_comparison'][0]['MAE']:.3f} pKi with ChEMBL alone and {model['original_test_comparison'][1]['MAE']:.3f} after source expansion. Additional data did not improve that test. Full metrics and counts are in `metadata/baseline_summary.json`.

## Limitations

Unknown activity is not a negative label. Missing variant metadata does not establish wild-type status. Laboratories, assay conditions and source selection differ. Source-specific and benchmark errors cannot be averaged across endpoint scales. Predictions describe molecular interactions and require experimental validation for biological or therapeutic use.

## Provenance, licenses and attribution

The transformed integrated collection is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Unchanged original reference files in `reference/chembl/` retain CC BY-SA 3.0. Original source notices remain applicable; see [LICENSES.md](LICENSES.md). Changes include filtering, structure standardisation, sequence checks, label transformation/aggregation, source reconciliation and fixed splits.

PDBbind and DrugBank are not included: no authorised publishable copy was available. Open Targets and ClinicalTrials.gov do not supply the molecular affinity labels evaluated here. Their status is documented in `metadata/source_registry.json`; absence is not a claim that they are scientifically unhelpful.

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
- **UniProt:** CC BY 4.0, https://www.uniprot.org/help/license . Canonical sequences were retrieved using the public REST API, then selected for the used human accessions.
- **DAVIS and KIBA via Therapeutics Data Commons:** Harvard Dataverse doi:10.7910/DVN/21LKWG, version 105.0. The official DOI metadata declares CC0 1.0 (https://api.datacite.org/dois/10.7910/DVN/21LKWG). TDC documentation links CC BY 4.0 while also saying “Not Specified”; both notices are retained here rather than silently inventing a source license. The prepared benchmark subsets are transformed and attributed. Davis et al., *Comprehensive analysis of kinase inhibitor selectivity*, doi:10.1038/nbt.1990; Tang et al., *Making sense of large-scale kinase inhibitor bioactivity data sets*, doi:10.1021/ci400709d; Huang et al., *Therapeutics Data Commons*, https://tdcommons.ai/multi_pred_tasks/dti/ .

PDBbind and DrugBank data and derivative records are excluded. Their access and redistribution restrictions have not been replaced by a Creative Commons label.
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
    commit=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=STAGING,
        commit_message='Publish curated multi-source affinity datasets and fixed benchmark splits')
    info=api.dataset_info(repo_id=repo,revision=commit.oid,files_metadata=True)
    remote={s.rfilename:s for s in info.siblings}
    expected=set(f['path'] for f in manifest['files'])|{'manifest.json'}
    if not expected<=set(remote):raise RuntimeError('Remote verification: missing files')
    for item in manifest['files']:
        if remote[item['path']].size!=item['bytes']:raise RuntimeError('Remote file size mismatch')
    record={'repository':repo,'url':'https://huggingface.co/datasets/'+repo,'revision':commit.oid,
            'uploaded_files':len(expected),'verified_at_utc':datetime.now(timezone.utc).isoformat()}
    save_json(RESULTS/'hf_publication.json',record)
    package_path=RESULTS/'hf_package.json'
    state=json.loads(package_path.read_text())
    state.update(publication_status='published',url=record['url'],revision=record['revision'])
    save_json(package_path,state)
    return record
