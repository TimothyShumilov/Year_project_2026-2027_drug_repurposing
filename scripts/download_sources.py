"""Download public upstream snapshots with resumable ranges and provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/raw/multisource'
FILES = {
    'bindingdb': ('https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202610_tsv.zip', 'bindingdb_202610.tsv.zip'),
    'davis': ('https://dataverse.harvard.edu/api/access/datafile/5219748', 'davis.tab'),
    'kiba': ('https://dataverse.harvard.edu/api/access/datafile/5255037', 'kiba.tab'),
    'drugcentral': ('https://unmtid-dbs.net/download/DrugCentral/2027/drug.target.interaction.tsv.gz', 'drugcentral_interactions.tsv.gz'),
    'drugcentral_structures': ('https://unmtid-dbs.net/download/DrugCentral/2027/structures.molV3.sdf.gz', 'drugcentral_structures.sdf.gz'),
}


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def download(name, workers=4):
    url, filename = FILES[name]
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / filename
    record_path = DEST / (name + '.download.json')
    if path.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        if record['url'] == url and record['sha256'] == digest(path):
            print(f'{name}: verified cached file ({path.stat().st_size/1e6:.1f} MB)', flush=True)
            return record
    headers = {'User-Agent': 'AffinityEDA/2.0', 'Accept-Encoding': 'identity'}
    with requests.get(url, headers={**headers, 'Range': 'bytes=0-0'}, stream=True, timeout=90) as probe:
        probe.raise_for_status()
        match = re.fullmatch(r'bytes 0-0/(\d+)', probe.headers.get('Content-Range', ''))
        size = int(match[1]) if match else None
        etag = probe.headers.get('ETag')
    started = time.monotonic()
    if size is not None:
        block_size = 8 * 1024**2
        parts = DEST / (filename + '.parts')
        parts.mkdir(exist_ok=True)
        spec = {'url': url, 'size': size, 'etag': etag, 'block_size': block_size}
        spec_path = parts / 'spec.json'
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise ValueError(f'{name}: upstream changed; preserve the old parts and choose a new snapshot')
        spec_path.write_text(json.dumps(spec))

        def one(start):
            end = min(start + block_size, size) - 1
            part = parts / f'{start:012d}'
            length = end - start + 1
            if part.exists() and part.stat().st_size == length:
                return length
            for attempt in range(5):
                try:
                    with requests.get(url, headers={**headers, 'Range': f'bytes={start}-{end}'},
                                      stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise ValueError('Server did not return the requested byte range')
                        temp = part.with_suffix('.tmp')
                        with temp.open('wb') as stream:
                            for chunk in response.iter_content(1024**2):
                                stream.write(chunk)
                    if temp.stat().st_size != length:
                        raise ValueError('Incomplete range')
                    temp.replace(part)
                    return length
                except (requests.RequestException, ValueError):
                    if attempt == 4:
                        raise
                    time.sleep(min(2**attempt, 8))

        done = 0
        starts = list(range(0, size, block_size))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(one, start) for start in starts]
            for future in as_completed(jobs):
                done += future.result()
                print(f'{name}: {done/1e6:.1f}/{size/1e6:.1f} MB; {time.monotonic()-started:.1f}s', flush=True)
        temporary = path.with_suffix(path.suffix + '.partial')
        with temporary.open('wb') as out:
            for start in starts:
                with (parts / f'{start:012d}').open('rb') as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
                        out.write(chunk)
        temporary.replace(path)
    else:
        temporary = path.with_suffix(path.suffix + '.partial')
        with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(1024**2):
                    stream.write(chunk)
        temporary.replace(path)
    record = {'source': name, 'url': url, 'filename': filename, 'bytes': path.stat().st_size,
              'sha256': digest(path), 'etag': etag,
              'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
              'seconds': round(time.monotonic()-started, 2)}
    if name == 'bindingdb':
        checksum_url = url.removesuffix('.zip') + '.md5'
        response = requests.get(checksum_url, headers=headers, timeout=90)
        response.raise_for_status()
        checksum = re.search(r'\b[0-9a-fA-F]{32}\b', response.text)
        if not checksum or digest(path, 'md5').lower() != checksum[0].lower():
            raise ValueError('BindingDB published MD5 checksum mismatch')
        record['upstream_md5'] = checksum[0].lower()
        record['checksum_url'] = checksum_url
    record_path.write_text(json.dumps(record, indent=2))
    print(f'{name}: DONE {record["bytes"]/1e6:.1f} MB in {record["seconds"]:.1f}s', flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', nargs='*', choices=list(FILES))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    for name in args.sources or FILES:
        download(name, args.workers)
