"""Count actual pKi label contributions after curation and target selection."""
from __future__ import annotations

import json
import pandas as pd

from affinity_eda import save_json
from multisource_data import DATA, RESULTS
from multisource_models import original_keys, eligible

SOURCES = ['ChEMBL', 'BindingDB', 'PubChem', 'DrugCentral']
PREPARATION = {
    'ChEMBL': 'Точные положительные Ki в нМ; тест связывания с отдельным человеческим белком, confidence 9, без аннотированной мутации. Пересчёт Ki → pKi.',
    'BindingDB': 'Только Ki из статей и патентов; импорты ChEMBL исключены. Отдельный человеческий белок; последовательность экспериментального белка должна совпадать с участком канонической последовательности UniProt. Пересчёт Ki в нМ → pKi.',
    'PubChem': 'Только количественные Ki из курируемого импорта BindingDB; ID теста проверен напрямую в PubChem. Те же проверки белка, что для BindingDB. Пересчёт Ki в нМ → pKi.',
    'DrugCentral': 'Соединение таблицы активностей со структурами SDF по STRUCT_ID; только человеческие мишени, ACT_TYPE=Ki и RELATION="=". ACT_VALUE уже содержит pKi: повторный логарифм не нужен.',
}
URLS = {
    'ChEMBL': 'https://www.ebi.ac.uk/chembl/',
    'BindingDB': 'https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp',
    'PubChem': 'https://pubchem.ncbi.nlm.nih.gov/docs/bioassays',
    'DrugCentral': 'https://www.drugcentral.org/download',
}


def build_report():
    pairs = pd.read_parquet(DATA/'pairs.parquet').query("endpoint == 'pKi'").copy()
    counts_path = RESULTS/'pKi_target_counts.csv'
    if not counts_path.exists():
        eligible(pairs, 'pKi')
    counts = pd.read_csv(counts_path)
    eligible_targets = set(counts.loc[counts.eligible, 'target_id'])
    used = pairs.loc[pairs.target_id.isin(eligible_targets)].copy()
    reference = original_keys()[['compound_id', 'target_id', 'scaffold_split']].rename(
        columns={'scaffold_split': 'original_part'})
    used = used.merge(reference, on=['compound_id', 'target_id'], how='left', validate='one_to_one')
    observations = pd.read_parquet(DATA/'observations.parquet').query("endpoint == 'pKi'")
    rows = []
    for source in SOURCES:
        block = used.loc[used.source.eq(source)]
        row = {'source': source, 'total_pairs': len(block), 'used_for_training': bool(block.scaffold_split.eq('train').any()),
               'validation_selection_pairs': int(block.original_part.eq('validation').sum()),
               'prepared_observations': int(observations.source.eq(source).sum()),
               'curated_pairs_all_targets': int(pairs.source.eq(source).sum()),
               'preparation': PREPARATION[source], 'url': URLS[source]}
        for split in ['train', 'validation', 'test']:
            row[split+'_pairs'] = int(block.scaffold_split.eq(split).sum())
            row[split+'_percent'] = 100*row[split+'_pairs']/len(block)
        row['diagnostic_validation_pairs'] = row['validation_pairs']-row['validation_selection_pairs']
        row['used_for_validation_selection'] = row['validation_selection_pairs'] > 0
        rows.append(row)
    pubchem_aids = sorted(set(observations.loc[observations.source.eq('PubChem'), 'pubchem_aid'].dropna()))
    report = {'endpoint': 'pKi', 'curated_pairs': len(pairs), 'curated_molecules': pairs.compound_id.nunique(),
              'benchmark_pairs': len(used), 'benchmark_targets': used.target_id.nunique(),
              'benchmark_molecules': used.compound_id.nunique(),
              'split_pairs': used.scaffold_split.value_counts().to_dict(),
              'validation_selection_pairs': int(used.original_part.eq('validation').sum()),
              'pubchem_pki_assay_ids': pubchem_aids,
              'counting_rule': 'One selected label per compound/target; source is the chosen label provider, not every overlapping provenance source.',
              'percentage_denominator': 'Retained benchmark pairs from that source, across train/validation/test.',
              'split_policy': 'Common scaffold groups; original ChEMBL assignments fixed; new groups SHA256(seed|scaffold), nominal 80/10/10.',
              'selection_policy': 'Hyperparameters use original ChEMBL validation only. Remaining validation rows are diagnostics; all test rows are evaluation only.',
              'sources': rows}
    save_json(RESULTS/'pki_source_usage.json', report)
    pd.DataFrame(rows).to_csv(RESULTS/'pki_source_usage.csv', index=False)
    return report


def usage_markdown(report, include_selection=True):
    if include_selection:
        text = ('| Источник выбранной метки | Обучение: пар (%) | Validation: пар (%) | Test: пар (%) | Из validation для выбора параметров |\n'
                '|---|---:|---:|---:|---:|\n')
    else:
        text = ('| Источник данных | Train: пар (%) | Validation: пар (%) | Test: пар (%) |\n'
                '|---|---:|---:|---:|\n')
    for row in report['sources']:
        cells = [f"{row[s+'_pairs']:,} ({row[s+'_percent']:.1f}%)".replace(',', ' ') for s in ['train', 'validation', 'test']]
        selection = f" {row['validation_selection_pairs']:,} |" if include_selection else ''
        text += f"| [{row['source']}]({row['url']}) | {' | '.join(cells)} |{selection}\n".replace(',', ' ')
    return text.strip()


def preparation_markdown(report):
    return '| Источник | Что делаем перед использованием |\n|---|---|\n'+'\n'.join(
        f"| {r['source']} | {r['preparation']} |" for r in report['sources'])


if __name__ == '__main__':
    print(json.dumps(build_report(), ensure_ascii=False, indent=2))
