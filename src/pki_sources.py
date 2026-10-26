"""Count actual pKi label contributions after curation and target selection."""
from __future__ import annotations

import json
import pandas as pd

from affinity_eda import save_json
from multisource_data import DATA, RESULTS, INCLUDED_SOURCES
from multisource_models import original_keys, eligible

SOURCES = list(INCLUDED_SOURCES)
PREPARATION = {
    'ChEMBL': 'Точные положительные Ki в нМ; тест связывания с отдельным человеческим белком, confidence 9, без аннотированной мутации. Пересчёт Ki → pKi.',
    'BindingDB': 'Только Ki из статей и патентов; импорты ChEMBL исключены. Отдельный человеческий белок; последовательность экспериментального белка должна совпадать с участком канонической последовательности UniProt. Пересчёт Ki в нМ → pKi.',
    'DrugCentral': 'Соединение таблицы активностей со структурами SDF по STRUCT_ID; только человеческие мишени, ACT_TYPE=Ki и RELATION="=". ACT_VALUE уже содержит pKi: повторный логарифм не нужен.',
}
URLS = {
    'ChEMBL': 'https://www.ebi.ac.uk/chembl/',
    'BindingDB': 'https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp',
    'DrugCentral': 'https://www.drugcentral.org/download',
}

SELECTION_CRITERIA = [
    'Соответствие задаче: количественные Ki для пары «малая молекула — отдельный человеческий белок». Kd, IC50 и составные оценки не заменяют Ki.',
    'Готовность к обучению: есть корректная структура SMILES, однозначный ID белка, точное положительное измерение и известные единицы. Ki переводим в pKi; уже готовое pKi повторно не логарифмируем.',
    'Качество и происхождение: можно проследить источник измерения, проверить повторы и противоречия. Копии между базами не считаем дополнительными независимыми примерами.',
    'Полезное покрытие после очистки и устранения пересечений. Смотрим на оставшиеся уникальные пары и заполненность частей выборки. Для мишени бейзлайна заранее заданы минимум 300 пар, из них ≥100 train, ≥10 validation и ≥10 test; это пороги мишени, а не размера всей базы.',
    'Воспроизводимый открытый доступ: данные можно получить, зафиксировать версию и опубликовать подготовленную выборку на Hugging Face с соблюдением условий источника.',
]
EXCLUDED_DATASETS = [
    {'source':'PubChem BioAssay','reason':'После очистки и разрешения пересечений в нашем бейзлайне остались только 46 пар: 42 train, 0 validation, 4 test. Малый самостоятельный вклад и отсутствие validation-части.',
     'url':'https://pubchem.ncbi.nlm.nih.gov/docs/bioassays'},
    {'source':'DAVIS','reason':'Метки Kd/pKd вместо Ki/pKi; отдельная задача, не используется для обучения или валидации текущего бейзлайна.',
     'url':'https://tdcommons.ai/multi_pred_tasks/dti/#davis'},
    {'source':'KIBA','reason':'Составной KIBA score, объединяющий разные виды активности; его нельзя считать pKi.',
     'url':'https://tdcommons.ai/multi_pred_tasks/dti/#kiba'},
    {'source':'PDBbind','reason':'Не получена выгрузка с разрешением на использование и публикацию подготовленных данных на Hugging Face.',
     'url':'https://www.pdbbind-plus.org.cn/termofuse'},
    {'source':'DrugBank','reason':'Академический доступ предоставляется по заявке; разрешённая для публикации выгрузка в проекте не получена.',
     'url':'https://go.drugbank.com/academic_research/'},
    {'source':'Open Targets','reason':'Связи мишеней с заболеваниями и биомедицинские доказательства не являются количественными метками Ki для этой регрессии.',
     'url':'https://platform-docs.opentargets.org/'},
    {'source':'ClinicalTrials.gov','reason':'Сведения о клинических исследованиях и вмешательствах не дают требуемые измерения Ki для пар «молекула — белок».',
     'url':'https://clinicaltrials.gov/data-api/about-api'},
]


def criteria_markdown():
    return '\n'.join(f'{i}. {text}' for i,text in enumerate(SELECTION_CRITERIA,1))


def exclusions_markdown():
    return '| Исключённый датасет | Причина |\n|---|---|\n'+'\n'.join(
        f"| [{r['source']}]({r['url']}) | {r['reason']} |" for r in EXCLUDED_DATASETS)


def build_report():
    pairs = pd.read_parquet(DATA/'pairs.parquet').query("endpoint == 'pKi'").copy()
    counts_path = RESULTS/'pKi_target_counts.csv'
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
    report = {'endpoint': 'pKi', 'curated_pairs': len(pairs), 'curated_molecules': pairs.compound_id.nunique(),
              'benchmark_pairs': len(used), 'benchmark_targets': used.target_id.nunique(),
              'benchmark_molecules': used.compound_id.nunique(),
              'split_pairs': used.scaffold_split.value_counts().to_dict(),
              'validation_selection_pairs': int(used.original_part.eq('validation').sum()),
              'selection_criteria': SELECTION_CRITERIA,
              'excluded_datasets': EXCLUDED_DATASETS,
              'counting_rule': 'One selected label per compound/target; source is the chosen label provider, not every overlapping provenance source.',
              'percentage_denominator': 'Retained benchmark pairs from that source, across train/validation/test.',
              'split_policy': 'Common scaffold groups; original ChEMBL assignments fixed; new groups SHA256(seed|scaffold), nominal 80/10/10.',
              'selection_policy': 'Hyperparameters use original ChEMBL validation only. Remaining validation rows are diagnostics; all test rows are evaluation only.',
              'sources': rows}
    save_json(RESULTS/'pki_source_usage.json', report)
    save_json(RESULTS/'excluded_datasets.json',EXCLUDED_DATASETS)
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
