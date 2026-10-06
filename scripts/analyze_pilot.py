"""Dependency-free analysis of the saved pilot snapshot. No network access."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict

from public_api import ROOT

LOSS_ACTIONS = {"INHIBITOR", "ANTAGONIST", "BLOCKER", "INVERSE AGONIST", "NEGATIVE ALLOSTERIC MODULATOR"}
GAIN_ACTIONS = {"ACTIVATOR", "AGONIST", "PARTIAL AGONIST", "POSITIVE ALLOSTERIC MODULATOR"}


def action_direction(action):
    if action in LOSS_ACTIONS:
        return "inhibit"
    if action in GAIN_ACTIONS:
        return "activate"
    return "unknown"


def therapeutic_direction(target_direction, trait_direction):
    """Invert a risk-inducing perturbation; retain a protective perturbation."""
    if (target_direction, trait_direction) in {("LoF", "protective"), ("GoF", "risk")}:
        return "inhibit"
    if (target_direction, trait_direction) in {("LoF", "risk"), ("GoF", "protective")}:
        return "activate"
    return "unknown"


def collapse_directions(values):
    known = set(values) - {"unknown"}
    if len(known) > 1:
        return "conflicting"
    return next(iter(known)) if known else "unknown"


def normalized(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def match_intervention(name, other_names, aliases, preferred_aliases=None):
    """Conservative exact matching; never infer a combination's components."""
    if "placebo" in str(name).lower():
        return set(), "placebo"
    preferred_aliases = preferred_aliases or {}
    primary = preferred_aliases.get(normalized(name), set()) or aliases.get(normalized(name), set())
    if primary:
        return set(primary), "exact" if len(primary) == 1 else "ambiguous"
    variants = other_names or []
    exact = set()
    for text in variants:
        exact.update(preferred_aliases.get(normalized(text), set()) or aliases.get(normalized(text), set()))
    if exact:
        return exact, "exact" if len(exact) == 1 else "ambiguous"
    # Only remove trailing dosage and familiar route suffixes, not drug names.
    cleaned = re.sub(r"\s+\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|%)\b.*$", "", str(name), flags=re.I)
    cleaned = re.sub(r"\s+(?:oral|subcutaneous|intravenous|injection|tablets?|capsules?)$", "", cleaned, flags=re.I)
    found = preferred_aliases.get(normalized(cleaned), set()) or aliases.get(normalized(cleaned), set())
    if found:
        return set(found), "dose_route_cleaned" if len(found) == 1 else "ambiguous"
    return set(), "unmatched"


def is_crohn(conditions):
    synonyms = {"regionalenteritis", "colitisgranulomatous", "enteritisregional", "granulomatouscolitis"}
    return any("crohn" in str(c).lower() or normalized(c) in synonyms for c in conditions)


def load(name):
    return json.loads((ROOT / "data" / "raw" / name).read_text())


def write_csv(name, rows, fields):
    path = ROOT / "data" / "processed" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict, set, tuple)) else value for key, value in row.items() if key in fields})


def build_analysis():
    cfg = json.loads((ROOT / "config" / "pilot.json").read_text())
    associations = load("associations.json")["rows"]
    evidence = load("evidence.json")["rows"]
    molecules = load("chembl_molecule.json")["rows"]
    mechanisms = load("chembl_mechanism.json")["rows"]
    human_targets = load("chembl_target.json")["rows"]
    indications = load("chembl_drug_indication.json")["rows"]
    clinical = load("ot_clinical_drugs.json")["rows"]
    trials = load("clinicaltrials.json")["rows"]
    genetic_sources = set(cfg["genetic_sources"])
    included_sources = genetic_sources | set(cfg["functional_sources"])
    assert not included_sources & set(cfg["exclude_features"])
    disease_scope = {cfg["disease_id"], *load("disease.json")["descendants"]}
    assert all(row["disease"]["id"] in disease_scope for row in evidence)
    assert all(row["datasourceId"] in genetic_sources for row in evidence)

    symbols, gene_features, protein_to_genes = {}, {}, defaultdict(set)
    for row in associations:
        gene = row["target"]["id"]
        symbols[gene] = row["target"]["approvedSymbol"]
        source_scores = {s["id"]: s["score"] for s in row["datasourceScores"]}
        gene_features[gene] = {
            "genetic_score": max([source_scores.get(s, 0) for s in genetic_sources], default=0),
            "expression_score": source_scores.get("expression_atlas", 0),
            "sources": {s: score for s, score in source_scores.items() if s in included_sources},
        }
        for protein in row["target"]["proteinIds"]:
            protein_to_genes[protein["id"].split("-")[0]].add(gene)
    direction_evidence = defaultdict(list)
    for row in evidence:
        direction_evidence[row["target"]["id"]].append(therapeutic_direction(row["directionOnTarget"], row["directionOnTrait"]))
    gene_directions = {gene: collapse_directions(direction_evidence[gene]) for gene in symbols}

    parent_of, families = {}, defaultdict(list)
    # Clinical molecules provide IDs and aliases for non-approved compounds too.
    extra_path = ROOT / "data/raw/chembl_clinical_molecules.json"
    extra_molecules = json.loads(extra_path.read_text())["rows"] if extra_path.exists() else []
    for molecule in [*molecules, *extra_molecules]:
        parent_of[molecule["molecule_chembl_id"]] = (molecule.get("molecule_hierarchy") or {}).get("parent_chembl_id") or molecule["molecule_chembl_id"]
    for mechanism in mechanisms:
        parent_of.setdefault(mechanism["molecule_chembl_id"], mechanism["parent_molecule_chembl_id"] or mechanism["molecule_chembl_id"])
    for molecule in molecules:
        families[parent_of[molecule["molecule_chembl_id"]]].append(molecule)
    approved_parents = set(families)
    active_approved = {p for p, rows in families.items() if any(not r["withdrawn_flag"] for r in rows)}
    approved_cd_chembl = {parent_of.get(r["molecule_chembl_id"], r.get("parent_molecule_chembl_id") or r["molecule_chembl_id"])
                   for r in indications if float(r.get("max_phase_for_ind") or 0) == 4}
    # Label/regulator records are an independent annotation channel. They are
    # used only to construct controls/exclusions, never as ranking features.
    approved_cd_label, approval_records = set(), []
    for row in clinical:
        for report in row["clinicalReports"]:
            if report["clinicalStage"] == "APPROVAL" and report["origin"] in {"DRUG_LABEL", "REGULATORY_AGENCY"} and report["type"] == "INDICATION":
                parent = parent_of.get(row['drug']['id'], row['drug']['id'])
                approved_cd_label.add(parent)
                approval_records.append({"drug_id":parent,"drug_name":row['drug']['name'],"report_id":report['id'],
                    "origin":report['origin'],"source":report['source'],"url":report['url']})
    approved_cd = approved_cd_chembl | approved_cd_label
    names, aliases, preferred_aliases = {}, defaultdict(set), defaultdict(set)

    def add_alias(text, parent):
        if not text:
            return
        key = normalized(text)
        if len(key) >= 3:
            aliases[key].add(parent)

    for molecule in [*molecules, *extra_molecules]:
        parent = parent_of[molecule["molecule_chembl_id"]]
        if parent not in names or molecule["molecule_chembl_id"] == parent:
            names[parent] = molecule.get("pref_name") or parent
        add_alias(molecule.get("pref_name"), parent)
        if molecule.get("pref_name"):
            preferred_aliases[normalized(molecule["pref_name"])].add(parent)
        for synonym in molecule.get("molecule_synonyms") or []:
            add_alias(synonym["molecule_synonym"], parent)
    for row in clinical:
        drug = row["drug"]
        parent = parent_of.get(drug["id"], drug["id"])
        names.setdefault(parent, drug["name"])
        add_alias(drug["name"], parent)
        preferred_aliases[normalized(drug["name"])].add(parent)
        for alias in drug["synonyms"] + drug["tradeNames"]:
            add_alias(alias["label"], parent)

    target_by_id = {row["target_chembl_id"]: row for row in human_targets
                    if "PROTEIN" in row["target_type"] or row["target_type"] == "SELECTIVITY GROUP"}
    target_genes = {}
    for target_id, row in target_by_id.items():
        genes = set()
        for component in row.get("target_components") or []:
            accession = component.get("accession")
            if accession:
                genes.update(protein_to_genes.get(accession.split("-")[0], set()))
        target_genes[target_id] = genes

    mechanisms_by_parent = defaultdict(list)
    human_mechanisms = defaultdict(list)
    projected_edges = defaultdict(set)
    mechanism_rows = []
    for mechanism in mechanisms:
        parent = parent_of.get(mechanism["molecule_chembl_id"], mechanism["parent_molecule_chembl_id"] or mechanism["molecule_chembl_id"])
        mechanisms_by_parent[parent].append(mechanism)
        target_id = mechanism.get("target_chembl_id")
        if target_id in target_by_id:
            human_mechanisms[parent].append(mechanism)
            for gene in target_genes[target_id]:
                projected_edges[parent].add((gene, mechanism.get("action_type") or "UNKNOWN", target_id))
        mechanism_rows.append({"drug_id": parent, "drug_name": names.get(parent, parent),
            "target_chembl_id": target_id, "target_type": target_by_id.get(target_id, {}).get("target_type", "nonhuman_or_unresolved"),
            "action_type": mechanism.get("action_type"), "mechanism_of_action": mechanism["mechanism_of_action"],
            "mapped_crohn_gene_ids": sorted(target_genes.get(target_id, set())),
            "mapped_crohn_gene_symbols": sorted(symbols[g] for g in target_genes.get(target_id, set())),
            "direct_interaction": mechanism.get("direct_interaction"), "mec_id": mechanism["mec_id"]})

    def signature(parent):
        return tuple(sorted({(r["target_chembl_id"], r.get("action_type") or "UNKNOWN") for r in human_mechanisms[parent]}))

    clinical_drug_trials, trial_rows, intervention_rows = defaultdict(set), [], []
    exclusions, matching, unmatched = Counter(), Counter(), Counter()
    registry_crohn_count, eligible_count = 0, 0
    for study in trials:
        section = study["protocolSection"]
        ident = section["identificationModule"]
        status = section["statusModule"]
        design = section.get("designModule") or {}
        interventions = (section.get("armsInterventionsModule") or {}).get("interventions") or []
        conditions = (section.get("conditionsModule") or {}).get("conditions") or []
        is_cd = is_crohn(conditions)
        registry_crohn_count += is_cd
        purpose = (design.get("designInfo") or {}).get("primaryPurpose")
        date = (status.get("studyFirstPostDateStruct") or {}).get("date") or ""
        reason = "included"
        if date and date > cfg["as_of_date"]:
            reason = "first_posted_after_cutoff"
        elif not is_cd:
            reason = "no_explicit_crohn_condition"
        elif design.get("studyType") != "INTERVENTIONAL":
            reason = "not_interventional"
        elif purpose not in {"TREATMENT", None}:
            reason = "not_treatment_purpose"
        elif not any(i["type"] in {"DRUG", "BIOLOGICAL"} for i in interventions):
            reason = "no_drug_or_biological_intervention"
        exclusions[reason] += 1
        eligible_count += reason == "included"
        drug_ids = set()
        if reason == "included":
            for intervention in interventions:
                if intervention["type"] not in {"DRUG", "BIOLOGICAL"}:
                    continue
                found, method = match_intervention(intervention["name"], intervention.get("otherNames"), aliases, preferred_aliases)
                matching[method] += 1
                usable = found if method in {"exact", "dose_route_cleaned"} else set()
                drug_ids.update(usable)
                if method in {"unmatched", "ambiguous"}:
                    unmatched[intervention["name"]] += 1
                intervention_rows.append({"nct_id": ident["nctId"], "intervention_name": intervention["name"],
                    "intervention_type": intervention["type"], "other_names": intervention.get("otherNames") or [],
                    "matching_method": method, "matched_drug_ids": sorted(found), "included_in_counts": bool(usable)})
            for drug in drug_ids:
                clinical_drug_trials[drug].add(ident["nctId"])
        trial_rows.append({"nct_id": ident["nctId"], "title": ident["briefTitle"], "conditions": conditions,
            "first_posted": date, "study_type": design.get("studyType"), "primary_purpose": purpose,
            "status": status["overallStatus"], "phases": design.get("phases") or [],
            "has_results": study.get("hasResults", False), "why_stopped": status.get("whyStopped"),
            "inclusion_status": reason, "matched_drug_ids": sorted(drug_ids)})

    ranking_weights = cfg["ranking"]
    drug_rows, edge_rows = [], []
    for parent in sorted(active_approved):
        drug_edges = projected_edges[parent]
        paths = []
        for gene, action, target_id in sorted(drug_edges):
            f = gene_features[gene]
            drug_direction, gene_direction = action_direction(action), gene_directions[gene]
            assessable = drug_direction != "unknown" and gene_direction in {"inhibit", "activate"}
            consistency = ("consistent" if drug_direction == gene_direction else "inconsistent") if assessable else ("conflicting" if gene_direction == "conflicting" else "unknown")
            factor = ranking_weights["consistent_factor"] if consistency == "consistent" else ranking_weights["inconsistent_factor"] if consistency == "inconsistent" else ranking_weights["unknown_or_conflicting_factor"]
            base = ranking_weights["genetic_weight"] * f["genetic_score"] + ranking_weights["expression_weight"] * f["expression_score"]
            path = {"drug_id": parent, "drug_name": names[parent], "target_chembl_id": target_id,
                "target_type": target_by_id[target_id]["target_type"], "gene_id": gene, "gene_symbol": symbols[gene],
                "action_type": action, "drug_direction": drug_direction, "genetic_therapeutic_direction": gene_direction,
                "consistency": consistency, "genetic_score": f["genetic_score"], "expression_score": f["expression_score"],
                "combined_score": base, "direction_score": base * factor, "sources": f["sources"]}
            paths.append(path)
            edge_rows.append(path)
        genetic_score = max([p["genetic_score"] for p in paths], default=0)
        combined_score = max([p["combined_score"] for p in paths], default=0)
        direction_score = max([p["direction_score"] for p in paths], default=0)
        affected = [p for p in paths if p["consistency"] in {"consistent", "inconsistent"}]
        genes = {p["gene_id"] for p in paths}
        approved_years = [m["first_approval"] for m in families[parent] if m.get("first_approval")]
        drug_rows.append({"drug_id": parent, "drug_name": names[parent], "molecule_types": sorted({m["molecule_type"] for m in families[parent]}),
            "first_approval": min(approved_years) if approved_years else None, "approved_crohn_chembl": parent in approved_cd_chembl,
            "approved_crohn_label_or_regulator":parent in approved_cd_label,"known_crohn_indication":parent in approved_cd,
            "has_curated_moa": bool(mechanisms_by_parent[parent]), "has_human_moa": bool(human_mechanisms[parent]),
            "linked_nc_genes": sorted(symbols[g] for g in genes if gene_features[g]["sources"]),
            "linked_genetic_genes": sorted(symbols[g] for g in genes if gene_features[g]["genetic_score"] > 0),
            "direction_assessable_genes": sorted({p["gene_symbol"] for p in affected}),
            "direction_consistency": sorted({p["consistency"] for p in affected}),
            "human_moa_signature": signature(parent), "matched_crohn_trial_count": len(clinical_drug_trials[parent]),
            "genetic_score": genetic_score, "combined_score": combined_score, "direction_score": direction_score})

    candidates = [r for r in drug_rows if r["linked_genetic_genes"] and not r["known_crohn_indication"]]
    pool_with_controls = [r for r in drug_rows if r["linked_genetic_genes"]]
    for field in ["genetic_score", "combined_score", "direction_score"]:
        for rank, row in enumerate(sorted(candidates, key=lambda r: (-r[field], r["drug_id"])), 1):
            row[field.replace("score", "rank")] = rank
    candidate_ids = {r["drug_id"] for r in candidates}
    clinically_matched = set(clinical_drug_trials) - {d for d, ids in clinical_drug_trials.items() if not ids}
    clinical_human = {p for p in clinically_matched if human_mechanisms[p]}
    clinical_signatures = {signature(p) for p in clinical_human}
    directed_clinical = {p for p in clinical_human if any(gene_directions.get(gene) in {"activate", "inhibit"} and action_direction(action) != "unknown" for gene, action, _ in projected_edges[p])}
    projected_directed_genes = {gene for parent in active_approved for gene, action, _ in projected_edges[parent]
                               if gene_directions[gene] in {"activate", "inhibit"} and action_direction(action) != "unknown"}
    approved_gene_ids = {gene for parent in active_approved for gene, _, _ in projected_edges[parent] if gene_features[gene]["genetic_score"] > 0}
    directed_genes = {gene for gene, direction in gene_directions.items() if direction in {"inhibit", "activate"}}
    complete_direction = [row for row in evidence if therapeutic_direction(row["directionOnTarget"], row["directionOnTrait"]) != "unknown"]

    quality = []
    extra_by_id = {m["molecule_chembl_id"]: m for m in extra_molecules}
    for row in clinical:
        drug = row["drug"]
        parent = parent_of.get(drug["id"], drug["id"])
        if drug["maximumClinicalStage"] == "APPROVAL" and parent not in approved_parents:
            quality.append({"drug_id": drug["id"], "drug_name": drug["name"], "ot_global_stage": drug["maximumClinicalStage"],
                "chembl_max_phase": extra_by_id.get(drug["id"], {}).get("max_phase"),
                "chembl_first_approval": extra_by_id.get(drug["id"], {}).get("first_approval"),
                "note": "OT approval proxy disagrees with ChEMBL approved molecule pool; not proof of regulatory status."})

    summary = {
        "as_of_date": cfg["as_of_date"], "opentargets_release": load("release_metadata.json")["meta"]["dataVersion"],
        "chembl_status": load("chembl_status.json"), "disease_id": cfg["disease_id"],
        "ot_associated_targets": len(associations),
        "nc_associated_targets": sum(bool(f["sources"]) for f in gene_features.values()),
        "genetic_associated_targets": sum(f["genetic_score"] > 0 for f in gene_features.values()),
        "nonclinical_genetic_evidence": len(evidence), "complete_direction_evidence": len(complete_direction),
        "direction_assessable_genes": len(directed_genes), "direction_gene_symbols": sorted(symbols[g] for g in directed_genes),
        "direction_conflicting_genes": sum(d == "conflicting" for d in gene_directions.values()),
        "approved_molecule_records": len(molecules), "approved_parent_molecules": len(approved_parents),
        "nonwithdrawn_approved_parent_molecules": len(active_approved),
        "approved_with_curated_moa": sum(bool(mechanisms_by_parent[p]) for p in active_approved),
        "approved_with_human_moa": sum(bool(human_mechanisms[p]) for p in active_approved),
        "approved_with_nc_target": sum(bool(r["linked_nc_genes"]) for r in drug_rows),
        "approved_with_genetic_target": len(pool_with_controls), "approved_crohn_indication_parent_molecules": len(approved_cd),
        "approved_crohn_chembl_parent_molecules":len(approved_cd_chembl),
        "approved_crohn_label_or_regulator_parent_molecules":len(approved_cd_label),
        "additional_crohn_approved_by_label_or_regulator":sorted(approved_cd_label-approved_cd_chembl),
        "approved_genetically_supported_target_genes": len(approved_gene_ids),
        "approved_direction_assessable_target_genes": len(projected_directed_genes),
        "approved_targets_with_signed_genetic_evidence_ignoring_drug_action":len(directed_genes & {g for p in active_approved for g,_,_ in projected_edges[p]}),
        "approved_direction_assessable_drugs": sum(bool(r["direction_assessable_genes"]) for r in drug_rows),
        "genetic_repurposing_candidates": len(candidates),
        "direction_assessable_repurposing_candidates": sum(bool(r["direction_assessable_genes"]) for r in candidates),
        "direction_ranking_changed_candidates": sum(r["combined_rank"] != r["direction_rank"] for r in candidates),
        "registry_search_records": len(trials), "explicit_crohn_condition_records": registry_crohn_count,
        "eligible_drug_trials": eligible_count, "trial_inclusion_counts": dict(exclusions),
        "eligible_trials_with_mapped_intervention": sum(r["inclusion_status"] == "included" and bool(r["matched_drug_ids"]) for r in trial_rows),
        "intervention_matching_counts": dict(matching),
        "mapped_clinical_drugs": len(clinically_matched), "mapped_clinical_drugs_with_human_moa": len(clinical_human),
        "clinical_human_moa_signatures": len(clinical_signatures),
        "approved_clinical_human_moa_signatures": len({signature(p) for p in clinical_human & active_approved}),
        "candidate_clinical_human_moa_signatures": len({signature(p) for p in clinical_human & candidate_ids}),
        "candidate_clinically_listed_drugs":len(clinically_matched & candidate_ids),
        "genetically_eligible_control_drugs":len(approved_cd & {r['drug_id'] for r in pool_with_controls}),
        "genetically_eligible_control_moa_signatures":len({signature(p) for p in approved_cd & {r['drug_id'] for r in pool_with_controls}}),
        "single_protein_genetic_control_drugs":len({p['drug_id'] for p in edge_rows if p['drug_id'] in approved_cd and p['target_type']=='SINGLE PROTEIN' and p['genetic_score']>0}),
        "candidate_pool_sensitivity": {
            str(threshold):sum(r['genetic_score']>=threshold for r in candidates) for threshold in [0.1,0.3,0.5,0.7]
        },
        "single_protein_genetic_candidates":sum(any(p['target_type']=='SINGLE PROTEIN' and p['genetic_score']>0 for p in edge_rows if p['drug_id']==r['drug_id']) for r in candidates),
        "direction_covered_clinical_drugs": len(directed_clinical),
        "direction_covered_clinical_signatures": len({signature(p) for p in directed_clinical}),
        "ot_approval_disagreements": len(quality),
        "evidence_with_evidence_date": sum(bool(r.get("evidenceDate")) for r in evidence),
        "evidence_with_publication_date": sum(bool(r.get("publicationDate") or r.get("publicationYear")) for r in evidence),
        "active_approved_with_first_approval_year": sum(r["first_approval"] is not None for r in drug_rows),
        "historical_drug_target_snapshot_available": False,
        "temporal_evaluation_performed": False,
        "source_evidence_counts": dict(Counter(r["datasourceId"] for r in evidence)),
        "source_complete_direction_counts": dict(Counter(r["datasourceId"] for r in complete_direction)),
        "direction_pairs": {f"{a or 'missing'}/{b or 'missing'}": count for (a, b), count in Counter((r['directionOnTarget'], r['directionOnTrait']) for r in evidence).items()},
        "illustrative_recovery": {},
    }
    # Retrospective recovery only; no fitting, no future prediction or efficacy label.
    controls = approved_cd & {r["drug_id"] for r in pool_with_controls}
    for field in ["genetic_score", "combined_score", "direction_score"]:
        ordered = sorted(pool_with_controls, key=lambda r: (-r[field], r["drug_id"]))
        summary["illustrative_recovery"][field] = {"control_count_in_eligible_pool": len(controls),
            "top20_controls": sum(r["drug_id"] in controls for r in ordered[:20]),
            "recall_at_20_within_eligible_pool": sum(r["drug_id"] in controls for r in ordered[:20])/len(controls) if controls else None}
    gates = cfg["feasibility_gates"]
    summary["feasibility"] = {
        "genetic_ranking_pool_sufficient": len(candidates) >= gates["minimum_genetic_repurposing_candidates"],
        "direction_candidate_pool_sufficient": summary["direction_assessable_repurposing_candidates"] >= gates["minimum_direction_assessable_candidates"],
        "clinical_mechanism_proxy_sufficient": len(clinical_signatures) >= gates["minimum_clinical_human_moa_signatures"],
        "direction_clinical_mechanism_proxy_sufficient": summary["direction_covered_clinical_signatures"] >= gates["minimum_direction_covered_clinical_signatures"],
    }
    summary["feasibility"]["original_direction_centered_topic_ready"] = all(summary["feasibility"].values())
    summary["feasibility"]["genetic_ranking_topic_ready_for_next_stage"] = summary["feasibility"]["genetic_ranking_pool_sufficient"] and summary["feasibility"]["clinical_mechanism_proxy_sufficient"]

    clinical_mechanism_rows = []
    for sig in sorted(clinical_signatures):
        parents = sorted(p for p in clinical_human if signature(p) == sig)
        clinical_mechanism_rows.append({"mechanism_signature": sig,
            "drug_ids": parents, "drug_names": [names.get(p, p) for p in parents],
            "nct_ids": sorted({n for p in parents for n in clinical_drug_trials[p]}),
            "approved_drug_ids": sorted(set(parents) & active_approved),
            "candidate_drug_ids": sorted(set(parents) & candidate_ids),
            "direction_covered_drug_ids": sorted(set(parents) & directed_clinical)})
    source_rows = []
    for source, n in sorted(summary["source_evidence_counts"].items()):
        src_rows = [r for r in evidence if r["datasourceId"] == source]
        source_rows.append({"source": source, "evidence_count": n, "unique_genes": len({r["target"]["id"] for r in src_rows}),
            "complete_direction_evidence": summary["source_complete_direction_counts"].get(source, 0),
            "complete_direction_genes": len({r["target"]["id"] for r in src_rows if therapeutic_direction(r["directionOnTarget"], r["directionOnTrait"]) != "unknown"})})
    evidence_rows = [{"evidence_id":r["id"], "source":r["datasourceId"], "gene_id":r["target"]["id"],
                     "gene_symbol":r["target"]["approvedSymbol"], "disease_id":r["disease"]["id"], "score":r["score"],
                     "direction_on_target":r["directionOnTarget"], "direction_on_trait":r["directionOnTrait"],
                     "therapeutic_direction":therapeutic_direction(r["directionOnTarget"],r["directionOnTrait"]),
                     "study_id":r["studyId"], "evidence_date":r["evidenceDate"]} for r in evidence]
    target_rows = [{"gene_id":g, "gene_symbol":symbols[g], **f, "genetic_therapeutic_direction":gene_directions[g],
                    "direction_evidence_count":sum(d != "unknown" for d in direction_evidence[g]),
                    "linked_active_approved_drugs":sorted(p for p in active_approved if any(e[0] == g for e in projected_edges[p]))} for g,f in gene_features.items()]
    for name, rows in [("approved_drugs.csv",drug_rows), ("candidate_ranking.csv",sorted(candidates,key=lambda r:r['combined_rank'])),
                       ("drug_target_paths.csv",edge_rows), ("clinical_trials.csv",trial_rows), ("trial_intervention_matches.csv",intervention_rows),
                       ("clinical_mechanisms.csv",clinical_mechanism_rows), ("source_coverage.csv",source_rows),
                       ("approval_disagreements.csv",quality), ("genetic_evidence.csv",evidence_rows), ("target_features.csv",target_rows),
                       ("drug_mechanisms.csv",mechanism_rows),("crohn_approval_records.csv",approval_records)]:
        write_csv(name, rows, list(rows[0]) if rows else ["no_rows"])
    unmatched_rows = [{"intervention_name":name,"occurrences":n} for name,n in unmatched.most_common()]
    write_csv("unmatched_interventions.csv", unmatched_rows, ["intervention_name","occurrences"])
    out = ROOT / "data/processed/summary.json"
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary, candidates, source_rows, quality, unmatched_rows


if __name__ == "__main__":
    summary, _, _, _, _ = build_analysis()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
