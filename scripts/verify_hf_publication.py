"""Verify the published revision through unauthenticated public downloads."""
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = ROOT/'results/multisource'
    publication = json.loads((results/'hf_publication.json').read_text())
    package = json.loads((results/'hf_package.json').read_text())
    manifest = json.loads((Path(package['local_path'])/'manifest.json').read_text())
    repo, revision = publication['repository'], publication['revision']
    response = requests.get(f'https://huggingface.co/api/datasets/{repo}/revision/{revision}',
                            params={'blobs': 'true'}, timeout=30)
    response.raise_for_status()
    info = response.json()
    assert not info['private']
    remote = {f['rfilename']: f for f in info['siblings']}
    expected = {f['path'] for f in manifest['files']} | {'manifest.json'}
    assert set(remote)-{'.gitattributes'} == expected
    lfs_checks = 0
    for file in manifest['files']:
        actual = remote[file['path']]
        assert actual['size'] == file['bytes'], file['path']
        if actual.get('lfs'):
            assert actual['lfs']['sha256'] == file['sha256'], file['path']
            lfs_checks += 1
    downloads = []
    for name in ['README.md', 'metadata/pki_source_usage.json', 'data/affinity_pki/validation.parquet']:
        response = requests.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}', timeout=60)
        response.raise_for_status()
        blob = response.content
        file = next(f for f in manifest['files'] if f['path'] == name)
        assert hashlib.sha256(blob).hexdigest() == file['sha256']
        record = {'path': name, 'bytes': len(blob), 'sha256': file['sha256']}
        if name.endswith('.parquet'):
            table = pd.read_parquet(io.BytesIO(blob))
            assert table.endpoint.eq('pKi').all() and table.scaffold_split.eq('validation').all()
            record['rows'] = len(table)
        elif name.endswith('.md'):
            metadata = yaml.safe_load(blob.decode().split('---', 2)[1])
            assert {c['config_name'] for c in metadata['configs']} == {
                'affinity_pki', 'chembl_reference', 'molecules', 'source_observations', 'pubchem_assays'}
            record['configurations'] = len(metadata['configs'])
        else:
            usage = json.loads(blob)
            assert {r['source'] for r in usage['sources']} == {'ChEMBL', 'BindingDB', 'PubChem', 'DrugCentral'}
            assert usage['validation_selection_pairs'] == 15439
            record['direct_sources'] = len(usage['sources'])
        downloads.append(record)
    record = {'repository': repo, 'revision': revision, 'public': True, 'verified_files': len(expected),
              'verified_lfs_sha256': lfs_checks, 'download_checks': downloads,
              'verified_at_utc': datetime.now(timezone.utc).isoformat()}
    (results/'hf_remote_verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
