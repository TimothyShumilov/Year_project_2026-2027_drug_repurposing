"""Check snapshot integrity, scientific invariants and deterministic replay offline."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import re
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import unquote

from public_api import ROOT


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_text())


def read_csv(name):
    with (ROOT / 'data/processed' / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    checks = []

    def check(label, condition):
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    cfg = read_json('config/pilot.json')
    summary = read_json('data/processed/summary.json')
    raw = {}
    identities = {
        'associations': lambda r: r['target']['id'],
        'evidence': lambda r: r['id'],
        'ot_clinical_drugs': lambda r: r['drug']['id'],
        'chembl_molecule': lambda r: r['molecule_chembl_id'],
        'chembl_clinical_molecules': lambda r: r['molecule_chembl_id'],
        'chembl_mechanism': lambda r: r['mec_id'],
        'chembl_target': lambda r: r['target_chembl_id'],
        'chembl_drug_indication': lambda r: r['drugind_id'],
        'clinicaltrials': lambda r: r['protocolSection']['identificationModule']['nctId'],
    }
    for name, identity in identities.items():
        data = read_json(f'data/raw/{name}.json')
        raw[name] = data['rows']
        check(f'{name}: all reported records downloaded', len(data['rows']) == data['count'])
        check(f'{name}: unique record identities', len({identity(r) for r in data['rows']}) == data['count'])

    check('download completed without errors', not read_json('data/raw/fetch_status.json')['failures'])
    check('release metadata matches summary',
          summary['opentargets_release'] == read_json('data/raw/release_metadata.json')['meta']['dataVersion']
          and summary['chembl_status'] == read_json('data/raw/chembl_status.json'))
    check('configured date and disease match summary',
          cfg['as_of_date'] == summary['as_of_date'] and cfg['disease_id'] == summary['disease_id'])
    scope = {cfg['disease_id'], *read_json('data/raw/disease.json')['descendants']}
    check('evidence belongs only to Crohn and its descendants',
          all(r['disease']['id'] in scope for r in raw['evidence']))
    check('raw genetic evidence contains only allowed nonclinical sources',
          all(r['datasourceId'] in cfg['genetic_sources'] for r in raw['evidence']))
    check('global molecule query contains only phase-4 records',
          all(float(r['max_phase']) == 4 for r in raw['chembl_molecule']))
    check('ChEMBL target query contains only human targets',
          all(r['tax_id'] == 9606 for r in raw['chembl_target']))
    check('indication query contains only Crohn MeSH records',
          all(r['mesh_id'] == 'D003424' for r in raw['chembl_drug_indication']))
    check('registry query avoids person/contact fields',
          not any(re.search('Contact|Investigator|Location|Facility|Participant', field, re.I)
                  for field in read_json('data/raw/clinicaltrials.json')['requested_fields']))

    approved = read_csv('approved_drugs.csv')
    candidates = read_csv('candidate_ranking.csv')
    paths = read_csv('drug_target_paths.csv')
    targets = read_csv('target_features.csv')
    evidence = read_csv('genetic_evidence.csv')
    trials = read_csv('clinical_trials.csv')
    matches = read_csv('trial_intervention_matches.csv')
    clinical = read_csv('clinical_mechanisms.csv')
    candidate_ids = {r['drug_id'] for r in candidates}
    control_ids = {r['drug_id'] for r in approved if r['known_crohn_indication'] == 'True'}
    eligible_controls = {r['drug_id'] for r in approved
                         if r['drug_id'] in control_ids and json.loads(r['linked_genetic_genes'])}
    check('candidate IDs are unique', len(candidate_ids) == len(candidates))
    check('candidate pool excludes known Crohn indications', candidate_ids.isdisjoint(control_ids))
    check('candidate pool has positive genetic evidence',
          all(json.loads(r['linked_genetic_genes']) and float(r['genetic_score']) > 0 for r in candidates))
    check('approved and candidate counts agree with summary',
          len(approved) == summary['nonwithdrawn_approved_parent_molecules']
          and len(candidates) == summary['genetic_repurposing_candidates'])
    check('genetic eligible control count agrees with summary',
          len(eligible_controls) == summary['genetically_eligible_control_drugs'])
    check('genetic evidence and feature counts agree with summary',
          len(evidence) == summary['nonclinical_genetic_evidence']
          and len(targets) == summary['ot_associated_targets']
          and sum(float(r['genetic_score']) > 0 for r in targets) == summary['genetic_associated_targets'])
    check('source-level features exclude clinical/literature scores',
          all(set(json.loads(r['sources'])) <= set(cfg['genetic_sources'] + cfg['functional_sources']) for r in targets + paths))
    for score in ('genetic', 'combined', 'direction'):
        expected = sorted(candidates, key=lambda r: (-float(r[f'{score}_score']), r['drug_id']))
        check(f'{score} ranks follow declared score and tie rule',
              all(int(r[f'{score}_rank']) == rank for rank, r in enumerate(expected, 1)))
    check('direction coverage count agrees with target features',
          sum(r['genetic_therapeutic_direction'] in {'activate', 'inhibit'} for r in targets)
          == summary['direction_assessable_genes'])
    check('direction coverage count agrees with candidate rows',
          sum(bool(json.loads(r['direction_assessable_genes'])) for r in candidates)
          == summary['direction_assessable_repurposing_candidates'])
    if not summary['direction_assessable_repurposing_candidates']:
        check('absent direction coverage cannot change ranks',
              all(r['combined_rank'] == r['direction_rank'] for r in candidates))
    check('complete direction count agrees with genetic evidence',
          sum(r['therapeutic_direction'] in {'activate', 'inhibit'} for r in evidence)
          == summary['complete_direction_evidence'])
    check('registry count and inclusion funnel agree with summary',
          len(trials) == summary['registry_search_records']
          and dict(Counter(r['inclusion_status'] for r in trials)) == summary['trial_inclusion_counts'])
    included = {r['nct_id'] for r in trials if r['inclusion_status'] == 'included'}
    check('registry matching uses only included studies', all(r['nct_id'] in included for r in matches))
    check('ambiguous/placebo/unmatched interventions excluded',
          all(r['included_in_counts'] == 'False' for r in matches
              if r['matching_method'] in {'ambiguous', 'placebo', 'unmatched'}))
    check('matching funnel agrees with summary',
          dict(Counter(r['matching_method'] for r in matches)) == summary['intervention_matching_counts'])
    check('eligible and matched trial counts agree with summary',
          len(included) == summary['eligible_drug_trials']
          and sum(r['nct_id'] in included and bool(json.loads(r['matched_drug_ids'])) for r in trials)
          == summary['eligible_trials_with_mapped_intervention'])
    check('clinical signature count agrees with summary', len(clinical) == summary['clinical_human_moa_signatures'])
    check('clinical candidate signature count agrees with summary',
          sum(bool(json.loads(r['candidate_drug_ids'])) for r in clinical) == summary['candidate_clinical_human_moa_signatures'])
    check('temporal testing is not claimed', not summary['temporal_evaluation_performed'])

    # Fail on byte-level differences, including unordered set iteration and stale
    # reports. Replay happens without reading network clients or invoking fetch.
    outputs = sorted((ROOT / 'data/processed').glob('*.csv')) + [ROOT / 'data/processed/summary.json', ROOT / 'reports/pilot_report.html']
    before = {str(p.relative_to(ROOT)): digest(p) for p in outputs}
    from analyze_pilot import build_analysis
    from render_report import main as render
    build_analysis()
    with contextlib.redirect_stdout(io.StringIO()):
        render()
    check('byte-identical offline recomputation of all tables, summary and report',
          before == {str(p.relative_to(ROOT)): digest(p) for p in outputs})

    report = (ROOT / 'reports/pilot_report.html').read_text()
    local_links = [unquote(url) for url in re.findall(r'href="([^"]+)"', report) if not url.startswith(('https:', 'http:', '#'))]
    check('all local report links exist or are the manifest being created',
          all((ROOT / 'reports' / url).resolve().exists() or url == '../data/processed/manifest.json' for url in local_links))

    # Verify response checksums and cache identity before publishing provenance.
    cache_records = []
    for path in sorted((ROOT / 'data/raw/cache').glob('*.json')):
        record = json.loads(path.read_text())
        payload = record['request']
        body = b'' if payload is None else json.dumps(payload, sort_keys=True).encode()
        key = hashlib.sha256(record['url'].encode() + body).hexdigest()
        response_sha = hashlib.sha256(json.dumps(record['response'], sort_keys=True).encode()).hexdigest()
        if path.stem != key or record['response_sha256'] != response_sha:
            raise AssertionError(f'Cache checksum mismatch: {path.name}')
        cache_records.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                              'sha256': digest(path), 'retrieved_at_utc': record['retrieved_at_utc'],
                              'url': record['url'], 'response_sha256': response_sha})
    check('all cached public API responses retain valid request and response hashes', bool(cache_records))
    artifacts = sorted({ROOT / 'README.md', ROOT / '.gitignore', ROOT / 'config/pilot.json',
                        *ROOT.glob('scripts/*.py'), *ROOT.glob('tests/*.py'),
                        *ROOT.glob('data/raw/*.json'), *outputs})
    manifest = {
        'validated_at_utc': datetime.now(timezone.utc).isoformat(),
        'as_of_date': cfg['as_of_date'], 'disease_id': cfg['disease_id'],
        'checks_passed': checks, 'offline_recomputation_byte_identical': True,
        'artifacts': [{'path':str(p.relative_to(ROOT)), 'bytes':p.stat().st_size, 'sha256':digest(p)} for p in artifacts],
        'cached_api_responses': cache_records,
    }
    (ROOT / 'data/processed/manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f'PASS: {len(checks)} checks; {len(outputs)} byte-identical recomputed outputs; {len(cache_records)} intact cached responses.')
    print(ROOT / 'data/processed/manifest.json')


if __name__ == '__main__':
    main()
