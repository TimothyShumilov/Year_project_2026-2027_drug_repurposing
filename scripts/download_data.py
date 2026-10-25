"""Public ChEMBL snapshot, immutable response cache and complete pagination."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
API = 'https://www.ebi.ac.uk/chembl/api/data/'
ACTIVITY_FIELDS = [
    'activity_id', 'molecule_chembl_id', 'parent_molecule_chembl_id', 'canonical_smiles',
    'target_chembl_id', 'target_pref_name', 'target_organism', 'target_tax_id',
    'assay_chembl_id', 'assay_type', 'assay_variant_accession', 'assay_variant_mutation',
    'standard_type', 'standard_relation', 'standard_value', 'standard_units',
    'pchembl_value', 'data_validity_comment', 'potential_duplicate', 'standard_flag',
    'document_chembl_id', 'document_year', 'src_id',
]


def fetch(resource, params=None, offline=False):
    url = API + resource + '.json'
    if params:
        url += '?' + urllib.parse.urlencode(params)
    key = hashlib.sha256(url.encode()).hexdigest()
    path = RAW / 'cache' / (key + '.json.gz')
    if path.exists():
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)['response']
    if offline:
        raise FileNotFoundError(f'No cached response: {url}')
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 AffinityEDA/1.0', 'Accept': 'application/json'})
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.load(response)
            record = {'url': url, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                      'response_sha256': hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
                      'response': data}
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            with gzip.open(temporary, 'wt') as stream:
                json.dump(record, stream, ensure_ascii=False)
            temporary.replace(path)
            return data
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == 4:
                raise
            print(f'Retrying {resource}, attempt {attempt+2}: {type(exc).__name__}', flush=True)
            time.sleep(min(2**attempt, 10))


def pages(resource, key, params, offline=False, workers=4):
    params = {'limit': 1000, **params}
    first = fetch(resource, {**params, 'offset': 0}, offline)
    total = first['page_meta']['total_count']
    print(f'{resource}: {len(first[key])}/{total}', flush=True)
    output = {0: first[key]}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, resource, {**params, 'offset': i}, offline): i
                   for i in range(1000, total, 1000)}
        for n, future in enumerate(as_completed(futures), 1):
            data = future.result()
            if data['page_meta']['total_count'] != total:
                raise ValueError('ChEMBL record count changed during pagination')
            output[futures[future]] = data[key]
            if n % 20 == 0 or len(output) == len(futures)+1:
                print(f'{resource}: {sum(map(len, output.values()))}/{total}', flush=True)
    rows = [row for offset in sorted(output) for row in output[offset]]
    if len(rows) != total:
        raise ValueError(f'Incomplete {resource} download: {len(rows)} != {total}')
    return rows, total


def batches(resource, key, id_field, ids, fields, offline=False, batch_size=100):
    identifiers = sorted(set(ids))
    groups = [identifiers[i:i+batch_size] for i in range(0, len(identifiers), batch_size)]
    def one(group):
        data = fetch(resource, {'limit': 1000, id_field+'__in': ','.join(group),
                               'only': ','.join(fields)}, offline)
        if len(data[key]) != data['page_meta']['total_count']:
            raise ValueError(f'Unexpected paginated batch in {resource}')
        return data[key]
    output = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for n, result in enumerate(executor.map(one, groups), 1):
            output.extend(result)
            if n % 20 == 0 or n == len(groups):
                print(f'{resource}: {len(output)} records, batch {n}/{len(groups)}', flush=True)
    return output


def download(offline=False):
    RAW.mkdir(parents=True, exist_ok=True)
    status = fetch('status', offline=offline)
    activities, count = pages('activity', 'activities', {
        'standard_type': 'Ki', 'assay_type': 'B', 'target_organism': 'Homo sapiens',
        'order_by': 'activity_id', 'only': ','.join(ACTIVITY_FIELDS)}, offline)
    df = pd.DataFrame(activities)
    assert df.activity_id.is_unique
    df.to_parquet(RAW / 'activities.parquet', index=False)
    target_ids = df.target_chembl_id.dropna().unique().tolist()
    targets = batches('target', 'targets', 'target_chembl_id', target_ids,
                      ['target_chembl_id', 'pref_name', 'target_type', 'tax_id', 'organism', 'target_components'], offline)
    (RAW/'targets.json').write_text(json.dumps(targets, ensure_ascii=False))
    assays = batches('assay', 'assays', 'assay_chembl_id', df.assay_chembl_id.dropna().unique(),
                     ['assay_chembl_id','confidence_score','relationship_type','variant_sequence',
                      'target_chembl_id','assay_type','assay_organism','bao_format'], offline)
    (RAW/'assays.json').write_text(json.dumps(assays, ensure_ascii=False))
    approved, approved_count = pages('molecule', 'molecules', {
        'max_phase':4, 'order_by':'molecule_chembl_id',
        'only':'molecule_chembl_id,pref_name,molecule_type,molecule_hierarchy,molecule_structures,withdrawn_flag'}, offline)
    (RAW/'approved_molecules.json').write_text(json.dumps(approved, ensure_ascii=False))
    manifest = {'source':'ChEMBL', 'release':status['chembl_db_version'],
                'release_date':status['chembl_release_date'],
                'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                'activity_query':{'standard_type':'Ki','assay_type':'B','target_organism':'Homo sapiens'},
                'activity_rows':count,'unique_activity_ids':int(df.activity_id.nunique()),
                'target_rows':len(targets),'assay_rows':len(assays),'approved_molecule_rows':approved_count,
                'cache_files':len(list((RAW/'cache').glob('*.json.gz')))}
    (RAW/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true')
    download(parser.parse_args().offline)
