"""Download a bounded, credential-free feasibility snapshot for Crohn disease."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode, urljoin

from public_api import ROOT, fetch_json, graphql, save_json

CONFIG = json.loads((ROOT / "config/pilot.json").read_text())
DISEASE = CONFIG["disease_id"]
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/"


def chembl_pages(resource, params, offline, filename=None):
    url = CHEMBL + resource + ".json?" + urlencode({"limit": 1000, **params})
    rows, count = [], None
    key = {"molecule": "molecules", "mechanism": "mechanisms", "target": "targets", "drug_indication": "drug_indications"}[resource]
    while url:
        page = fetch_json(url, offline=offline)
        count = page["page_meta"]["total_count"] if count is None else count
        rows.extend(page[key])
        nxt = page["page_meta"]["next"]
        url = urljoin(CHEMBL, nxt) if nxt else None
        print(f"ChEMBL {resource}: {len(rows)}/{count}", flush=True)
    assert len(rows) == count, (resource, len(rows), count)
    save_json(filename or f"chembl_{resource}.json", {"count": count, "rows": rows})
    return rows


def fetch_trials(offline):
    # Fetch only scientific/administrative fields, deliberately excluding contacts,
    # investigators, facilities, free-text descriptions and participant data.
    fields = ["NCTId", "BriefTitle", "Condition", "InterventionName", "InterventionOtherName",
              "InterventionType", "Phase", "OverallStatus", "StudyType", "StudyFirstPostDate",
              "HasResults", "WhyStopped", "DesignPrimaryPurpose"]
    params = {"query.cond": "Crohn Disease", "format": "json", "countTotal": "true",
              "pageSize": 1000, "fields": ",".join(fields)}
    rows, count = [], None
    while True:
        page = fetch_json("https://clinicaltrials.gov/api/v2/studies?" + urlencode(params), offline=offline)
        count = page.get("totalCount", count)
        rows.extend(page.get("studies", []))
        print(f"ClinicalTrials.gov: {len(rows)}/{count}", flush=True)
        if not page.get("nextPageToken"):
            break
        params["pageToken"] = page["nextPageToken"]
    assert len(rows) == count
    assert len({s["protocolSection"]["identificationModule"]["nctId"] for s in rows}) == len(rows)
    save_json("clinicaltrials.json", {"count": count, "rows": rows, "requested_fields": fields})


def fetch_associations(offline):
    meta = graphql('{meta{apiVersion{x y z} dataVersion{year month iteration}}}', offline=offline)
    save_json("release_metadata.json", meta)
    query = '''query($id:String!,$page:Int!,$size:Int!,$indirect:Boolean!){ disease(efoId:$id){
      id name descendants ancestors synonyms { terms }
      associatedTargets(enableIndirect:$indirect,page:{index:$page,size:$size}){
        count rows { score target{id approvedSymbol proteinIds{id source}}
          datatypeScores{id score} datasourceScores{id score} }
      }
    }}'''
    # Score ties in this API are not stably ordered across pages. Merge traversals
    # with different page boundaries and require exact reported unique coverage.
    by_id, pagination_audit = {}, []
    for size in (2500, 1500, 2000, 1000):
        page_index, traversal_rows = 0, []
        while True:
            disease = graphql(query, {"id": DISEASE, "page": page_index, "size": size, "indirect": True}, offline=offline)["disease"]
            section = disease["associatedTargets"]
            traversal_rows.extend(section["rows"])
            for row in section["rows"]:
                by_id.setdefault(row["target"]["id"], row)
            print(f"Open Targets associations: {len(by_id)} unique/{section['count']} (page size {size})", flush=True)
            if len(traversal_rows) >= section["count"]:
                break
            page_index += 1
        pagination_audit.append({"page_size": size, "returned_rows": len(traversal_rows), "unique_in_traversal": len({r['target']['id'] for r in traversal_rows})})
        if len(by_id) == section["count"]:
            break
    rows = list(by_id.values())
    assert len(rows) == section["count"], (len(rows), section["count"])
    disease.pop("associatedTargets")
    save_json("disease.json", disease)
    save_json("associations.json", {"count": section["count"], "rows": rows, "enableIndirect": True, "pagination_audit": pagination_audit})
    sources = sorted({s["id"] for row in rows for s in row["datasourceScores"]})
    # Explicitly exclude clinical precedence, somatic cancer sources, and mined
    # literature. Expression is descriptive and never determines effect sign.
    allowed = [s for s in sources if s in set(CONFIG["genetic_sources"] + CONFIG["functional_sources"])]
    direction_sources = [s for s in allowed if s in CONFIG["genetic_sources"]]
    save_json("source_policy.json", {"present_sources": sources, "included_sources": allowed,
                                    "direction_evidence_sources": direction_sources,
                                    "excluded_sources": sorted(set(sources) - set(allowed))})
    # Expression provides a source-level feature but cannot determine therapeutic
    # direction; its thousands of full records need not be downloaded for this audit.
    ids = [row["target"]["id"] for row in rows if any(s["id"] in direction_sources for s in row["datasourceScores"])]
    query = '''query($id:String!,$ids:[String!]!,$sources:[String!],$cursor:String){disease(efoId:$id){
      evidences(ensemblIds:$ids,enableIndirect:true,datasourceIds:$sources,size:1000,cursor:$cursor){
        count cursor rows{id datasourceId datatypeId score directionOnTarget directionOnTrait
          target{id approvedSymbol} disease{id name} studyId variantRsId
          variantFunctionalConsequence{id label} variantFunctionalConsequenceFromQtlId{id label}
          beta oddsRatio clinicalSignificances literature evidenceDate publicationYear publicationDate releaseDate}
      }
    }}'''
    evidence, cursor, seen = [], None, set()
    while True:
        section = graphql(query, {"id": DISEASE, "ids": ids, "sources": direction_sources, "cursor": cursor}, offline=offline)["disease"]["evidences"]
        evidence.extend(section["rows"])
        print(f"Open Targets nonclinical evidence: {len(evidence)}/{section['count']}", flush=True)
        cursor = section["cursor"]
        if not cursor:
            break
        if cursor in seen:
            raise ValueError("Repeated evidence cursor")
        seen.add(cursor)
    assert len(evidence) == section["count"]
    assert len({r["id"] for r in evidence}) == len(evidence)
    save_json("evidence.json", {"count": section["count"], "rows": evidence, "enableIndirect": True})
    clinical = graphql('''query($id:String!){disease(efoId:$id){drugAndClinicalCandidates{
      count rows {maxClinicalStage drug{id name maximumClinicalStage synonyms{label} tradeNames{label}
        mechanismsOfAction{rows{actionType mechanismOfAction targetName targets{id approvedSymbol}}}}
        clinicalReports{id origin provider source type clinicalStage url trialStartDate trialOverallStatus trialWhyStopped}
      }
    }}}''', {"id": DISEASE}, offline=offline)["disease"]["drugAndClinicalCandidates"]
    assert len(clinical["rows"]) == clinical["count"]
    save_json("ot_clinical_drugs.json", clinical)
    print(f"Open Targets clinical drugs: {clinical['count']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--only", nargs="+", help="Fetch selected sources only")
    args = parser.parse_args()
    jobs = {
        "opentargets": lambda: fetch_associations(args.offline),
        "trials": lambda: fetch_trials(args.offline),
        "chembl_molecule": lambda: chembl_pages("molecule", {"max_phase": 4, "only": "molecule_chembl_id,pref_name,max_phase,first_approval,molecule_type,molecule_synonyms,molecule_hierarchy,withdrawn_flag"}, args.offline),
        "chembl_mechanism": lambda: chembl_pages("mechanism", {}, args.offline),
        "chembl_indications": lambda: chembl_pages("drug_indication", {"mesh_id": "D003424"}, args.offline),
        "chembl_targets": lambda: chembl_pages("target", {"tax_id": 9606, "only": "target_chembl_id,pref_name,target_type,tax_id,organism,target_components"}, args.offline),
        "chembl_status": lambda: save_json("chembl_status.json", fetch_json(CHEMBL+"status.json", offline=args.offline)),
    }
    if args.only:
        jobs = {name: jobs[name] for name in args.only}
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(job): name for name, job in jobs.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                print(f"Completed: {name}", flush=True)
            except Exception as error:
                failures.append({"source": name, "error": type(error).__name__ + ": " + str(error)[:3000]})
                print(f"FAILED {name}: {type(error).__name__}: {error}", flush=True)
    if "opentargets" in jobs and not any(f['source'] == 'opentargets' for f in failures):
        clinical_ids = sorted({r['drug']['id'] for r in json.loads((ROOT/'data/raw/ot_clinical_drugs.json').read_text())['rows']})
        try:
            chembl_pages("molecule", {"molecule_chembl_id__in": ",".join(clinical_ids),
                "only": "molecule_chembl_id,pref_name,max_phase,first_approval,molecule_type,molecule_synonyms,molecule_hierarchy,withdrawn_flag"},
                args.offline, filename="chembl_clinical_molecules.json")
        except Exception as error:
            failures.append({"source":"chembl_clinical_molecules","error":type(error).__name__+": "+str(error)[:3000]})
    save_json("fetch_status.json", {"failures": failures})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
