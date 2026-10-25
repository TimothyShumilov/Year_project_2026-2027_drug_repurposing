"""Build a concise, readable checkpoint notebook; calculations live in src/."""
import json
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def build():
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text):cells.append(nbf.v4.new_code_cell(text.strip()))
    md('''# EDA и бейзлайн прогнозирования аффинности

По структуре малой молекулы предсказываем силу её взаимодействия с белковой мишенью. **Мишень** здесь — отдельный человеческий белок; **pKi** — логарифмическая мера взаимодействия. pKi = 7 соответствует Ki = 100 нМ, pKi = 6 — Ki = 1 мкМ. Большее pKi означает более сильное взаимодействие.

В ноутбуке: качество и состав данных, проверка разбиений, Random Forest на Morgan fingerprints, интерпретация и проверка аугментаций. Модель оценивает молекулярное взаимодействие; терапевтический эффект требует отдельной проверки.''')
    code('''from pathlib import Path
import sys, json
ROOT = Path.cwd()
if not (ROOT / 'src').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
from affinity_eda import (prepare_data, get_fingerprints, fingerprint_audit, fingerprint_sensitivity, approved_coverage, train_benchmark,
                          augmentation_invariance, explain_baseline, CFG, RESULTS, DATA)
from eda_figures import generate_figures
pd.set_option('display.max_colwidth', 55)
pd.set_option('display.precision', 3)
pairs, molecules, eda = prepare_data()
display(Markdown(f"**Источник:** {eda['source']['release']}. "
                 f"**Срез:** 25 октября 2026. **Случайное зерно:** {CFG['seed']}."))
display(pd.DataFrame({'Показатель': ['Исходных измерений', 'Пар после очистки', 'Молекул',
                                    'Мишеней после очистки', 'Мишеней для бейзлайна'],
                      'Число': [eda['raw_rows'], eda['curated_pairs'], eda['curated_molecules'],
                                eda['curated_targets'], eda['benchmark_targets']]}))
if (RESULTS / 'baseline_summary.json').exists():
    headline = json.loads((RESULTS / 'baseline_summary.json').read_text())
    display(Markdown(f"**Основной результат:** на новых химических каркасах MAE Random Forest = "
                     f"**{headline['test_macro_MAE']:.3f} pKi**; у среднего по мишени — "
                     f"**{headline['dummy_test_macro_MAE']:.3f}**. Проверка охватывает "
                     f"**Мишеней в проверке:** {headline['targets']}; **пар test:** {headline['test_pairs']:,}."))''')
    md('''## Данные и качество

Источник — ChEMBL. Используем только Ki из тестов связывания, поэтому разные типы активности не смешиваются. Сохраняем прямое сопоставление с отдельным человеческим белком, уверенность аннотации 9, корректную структуру и точное положительное измерение в нМ. Числовые значения pKi пересчитываются из Ki.

Пропуски в условиях экспериментов означают отсутствие аннотации. Пропуск в матрице «молекула — мишень» означает **неизвестную активность**, его нельзя заменять нулём.''')
    code('''missing = pd.read_csv(RESULTS / 'raw_missingness.csv', index_col=0)
fields = {'canonical_smiles': 'Структура молекулы', 'standard_value': 'Числовое значение Ki',
          'pchembl_value': 'Готовое pChEMBL', 'document_year': 'Год публикации'}
display(missing.loc[list(fields), ['missing_fraction']].rename(index=fields, columns={'missing_fraction': 'Доля пропусков'}))
relation_missing = pd.read_csv(RESULTS / 'pchembl_missing_by_relation.csv', index_col=0)
display(Markdown(f"Доля незаполненных клеток матрицы взаимодействий: "
                 f"**{eda['matrix_missing_fraction']:.2%}**. "
                 f"Готовое pChEMBL отсутствует у **{relation_missing.loc['=', 'True']:.1%}** "
                 "точных измерений и у всех пороговых записей (<, > и т. п.). "
                 "Пороговые записи исключены: они не задают точное значение для обычной регрессии. "
                 "Отсутствие аннотации мутации не доказывает отсутствие мутации."))''')
    code('''display(pd.read_csv(RESULTS / 'cleaning_funnel.csv').rename(
    columns={'step': 'Шаг', 'remaining': 'Осталось измерений', 'removed': 'Исключено'}))
display(Markdown(f"После проверки качества **{eda['pki_check']['with_provided_pchembl']:,}** "
                 "пересчитанных pKi согласуются с pChEMBL (допуск 0,05). "
                 f"Из структур исключены: >100 тяжёлых атомов — {eda['structure_status']['over_100_heavy_atoms']:,}; "
                 f"несколько крупных органических компонентов — {eda['structure_status']['multiple_organic_components']:,}; "
                 "остальные причины — отсутствие органического фрагмента, неподдерживаемый элемент или ошибка SMILES."))''')
    md('''Соли обрабатываются выбором основного органического фрагмента; неоднозначные записи с двумя крупными органическими компонентами откладываются. Заряды, таутомерная форма и стереохимия сохраняются. Ограничение до 100 тяжёлых атомов задаёт область рассматриваемых органических соединений.

Повторы одной структуры и мишени объединяются медианой. Если разброс превышает 2 единицы pKi (более чем стократное различие Ki), пару исключаем из бейзлайна как противоречивую. Такие пары сохранены для разбора. Статистически необычное значение само по себе не удаляем: редкое сильное связывание может быть реальным.''')
    code('''display(Markdown(f"Пар с повторными измерениями: **{eda['repeats']['repeated_pairs']:,}**; "
                 f"исключено противоречивых пар: **{eda['repeats']['conflicting_pairs_over_2_pki']:,}**."))
display(Markdown(f"По правилу 1,5 межквартильного размаха отмечено "
                 f"**{eda['iqr_outlier_pairs']:,}** необычных значений pKi; это диагностический флаг. "
                 f"Диапазон после очистки: {eda['pki_min']:.2f}–{eda['pki_max']:.2f}."))
# Полные описательные статистики вычислены в prepare_data через describe(include='all').
statistics = pd.read_csv(RESULTS / 'molecule_describe.csv', index_col=0).loc[
    ['molecular_weight','logp','tpsa','heavy_atoms'], ['min','25%','50%','75%','max']]
statistics = pd.concat([pd.read_csv(RESULTS / 'pair_describe.csv', index_col=0).loc[['pki'], statistics.columns], statistics])
display(statistics.rename(index={'pki':'pKi','molecular_weight':'Масса, Да','logp':'logP',
                                  'tpsa':'Полярная площадь, Å²','heavy_atoms':'Тяжёлые атомы'},
                          columns={'min':'Минимум','25%':'Q1','50%':'Медиана','75%':'Q3','max':'Максимум'}).round(2))''')
    md('''## Распределения и покрытие

Это регрессия, поэтому баланса классов нет. Важны распределение pKi и дисбаланс числа измерений между мишенями. Тяжёлые атомы — все атомы, кроме водорода; масса, logP и полярная площадь описывают размер и физико-химические свойства молекулы. Дескрипторы рассчитаны для анализа данных; лес использует только Morgan fingerprints.''')
    code('''X = get_fingerprints(molecules)
fp_audit = fingerprint_audit(molecules, X)
display(pd.Series({'Битов в векторе': fp_audit['bits'],
                   'Медианное число активных битов': fp_audit['active_bits_median'],
                   'Постоянных признаков': fp_audit['constant_bits'],
                   'Разных структур сверх числа уникальных fingerprints': fp_audit['different_structures_with_same_fingerprint_excess']}, name='Значение').to_frame())
coverage = approved_coverage(pairs)
display(pd.Series({'Допущенных структур одобренных малых молекул': coverage['eligible_unique_small_molecule_structures'],
                   'Из них с измерением Ki в очищенном наборе': coverage['with_curated_Ki_measurement']}, name='Число').to_frame())
display(Markdown(f"Для экспериментов: **мишеней — {eda['benchmark_targets']}**, "
                 f"**размеченных пар — {eda['benchmark_pairs']:,}**. "
                 f"Требуется не менее {CFG['minimum_pairs_per_target']} пар на мишень "
                 "и достаточное покрытие во всех частях обоих разбиений. "
                 f"Число мишеней, исключённых из-за покрытия: {eda['targets_dropped_for_split_coverage']}."))''')
    md('''Разные структуры могут иметь одинаковый Morgan fingerprint: компактный вектор не сохраняет всю информацию о молекуле. Это ограничение представления. Ниже дополнительно проверяем ошибку без совпадающих отпечатков.

## Как проверяем модель

Во всех задачах одна молекула относится к одной части выборки. Случайное разбиение группирует соединения с одинаковой структурой без учёта стереохимии. Основное разбиение группирует по Bemis–Murcko-каркасу — центральной кольцевой структуре и связующим фрагментам. Все бескольцевые молекулы образуют отдельную общую группу.

Каркасы между train, validation и test не пересекаются. Это более строгая проверка переноса на новые структуры, хотя разные каркасы тоже могут быть химически похожи. Доли около 80/10/10 относятся к группам и могут отличаться по числу молекул.''')
    code('''split_rows = []
for split in ['random_split', 'scaffold_split']:
    for part in ['train','validation','test']:
        selected = molecules[split].eq(part)
        split_rows.append({'Разбиение': split, 'Часть': part,
                           'Молекул': int(selected.sum()),
                           'Пар бейзлайна': int((pairs.benchmark_target & pairs[split].eq(part)).sum())})
display(pd.DataFrame(split_rows).replace({'random_split':'Случайное','scaffold_split':'По каркасам',
                                         'train':'Обучение','validation':'Валидация','test':'Тест'}))
for split in ['random_split','scaffold_split']:
    groups = {part: set(molecules.loc[molecules[split].eq(part),'compound_id'])
              for part in ['train','validation','test']}
    assert not (groups['train'] & groups['validation'] or groups['train'] & groups['test'] or groups['validation'] & groups['test'])
for left,right in [('train','test'),('train','validation'),('validation','test')]:
    assert set(molecules.loc[molecules.scaffold_split.eq(left),'scaffold']).isdisjoint(
        molecules.loc[molecules.scaffold_split.eq(right),'scaffold'])
assert not pairs.duplicated(['compound_id','target_chembl_id']).any()
print('Повторяющихся пар и пересечений молекул нет; каркасы в основном разбиении не пересекаются.')''')
    md('''## Random Forest на Morgan fingerprints

Morgan fingerprint — 2048 признаков наличия локальных окружений атомов в радиусе двух связей, с учётом стереохимии. Для каждой мишени обучается отдельный лес из 100 деревьев. Проверяем минимальный размер листа 1, 3 и 10; остальные параметры одинаковы. Выбор — по среднему MAE на validation основного разбиения.

**MAE** — средняя абсолютная ошибка pKi. Ошибка на 1 единицу соответствует десятикратному расхождению Ki. **RMSE** сильнее учитывает крупные ошибки. **Спирмен** показывает качество порядка предсказаний; для постоянного прогноза он не определён. Все основные метрики усредняются сначала внутри каждой мишени, затем между мишенями: большие задачи не подавляют малые. Простой контроль всегда предсказывает средний pKi обучающей части своей мишени.''')
    code('''# При первом запуске функция действительно обучает модели; повторный запуск использует
# завершённые эксперименты только при совпадении хешей данных и конфигурации.
baseline = train_benchmark(pairs, molecules, X)
metrics = pd.read_csv(RESULTS / 'metrics_macro.csv')
selection = json.loads((RESULTS / 'selection.json').read_text())
display(pd.Series(selection['validation_macro_MAE'], name='Validation MAE').to_frame())
display(Markdown(f"Выбрана конфигурация **{baseline['selected']}**. "
                 "Тестовые метрики не используются при выборе параметров."))
experiments = [baseline['selected'], 'RF_random', 'RF_dropout5']
table = metrics.loc[metrics.part.eq('test') & metrics.experiment.isin(experiments)].copy()
table = table.loc[table.model.eq('RF') | table.experiment.eq(baseline['selected'])]
table['Проверка'] = table['split'].map({'scaffold_split': 'Новые каркасы', 'random_split': 'Случайные молекулы'})
table['Модель'] = np.where(table.model.eq('Mean'), 'Среднее по мишени',
                           np.where(table.experiment.eq('RF_dropout5'), 'RF с шумом 5%', 'Random Forest'))
table = table[['Модель','Проверка','MAE','RMSE','Spearman']].round(3).reset_index(drop=True)
table['Spearman'] = table.Spearman.astype(object).where(table.Spearman.notna(), '—')
display(table.rename(columns={'Spearman':'Спирмен'}))''')
    code('''detail = pd.read_csv(RESULTS / 'metrics_by_target.csv')
main = detail.loc[detail.experiment.eq(baseline['selected']) & detail.part.eq('test') & detail.model.eq('RF')]
ordered = main.sort_values('MAE')
examples = pd.concat([ordered.head(3), ordered.tail(3)]).drop_duplicates('target_id')
display(examples[['target_name','n','MAE','RMSE','Spearman']].rename(
    columns={'target_name':'Мишень','n':'Пар test','Spearman':'Спирмен'}).round(3).reset_index(drop=True))
random_mae = metrics.loc[metrics.experiment.eq('RF_random') & metrics.part.eq('test') & metrics.model.eq('RF'),'MAE'].iloc[0]
delta = baseline['test_macro_MAE'] - random_mae
display(Markdown(f"На новых каркасах MAE = **{baseline['test_macro_MAE']:.3f}**, "
                 f"у среднего по мишени — **{baseline['dummy_test_macro_MAE']:.3f}**. "
                 f"Отличие от случайного разбиения: **{delta:+.3f}**. "
                 f"Для трёх инициализаций MAE: **{baseline['three_seed_macro_MAE_mean']:.3f} ± "
                 f"{baseline['three_seed_macro_MAE_sd']:.3f}** (стандартное отклонение). "
                 "Это разброс обучения на одном разбиении, а не доверительный интервал обобщения."))''')
    code('''sensitivity = fingerprint_sensitivity(pairs, molecules, X, baseline)
display(Markdown(f"В основном тесте **{sensitivity['test_pairs_with_fingerprint_seen_for_same_target']}** пар "
                 "имеют такой же fingerprint, как обучающий пример для той же мишени. "
                 f"После исключения этих случаев только из диагностического расчёта MAE = "
                 f"**{sensitivity['macro_MAE_without_shared_fingerprints']:.3f}** "
                 f"Число мишеней осталось прежним: {sensitivity['targets_remaining']}. "
                 "Основное разбиение и выбор параметров не меняются."))''')
    md('''## Интерпретация признаков

Для наиболее представленной мишени проверяем важные биты: перемешиваем каждый признак на validation и смотрим, насколько ухудшится MAE. Положительное изменение означает, что признак помогал модели на этой выборке. Один бит может кодировать несколько подструктур из-за хеширования; его важность не доказывает механизм связывания.''')
    code('''interpretation = explain_baseline(pairs, molecules, X, baseline)
display(Markdown(f"Мишень для примера: **{interpretation['target_name']}**."))
display(pd.read_csv(RESULTS / 'feature_importances.csv').sort_values('permutation_MAE_increase', ascending=False)
        .head(5)[['bit','permutation_MAE_increase']].rename(
            columns={'bit':'Бит','permutation_MAE_increase':'Увеличение MAE'}).round(4))
display(pd.read_csv(RESULTS / 'fingerprint_bit_examples.csv').drop_duplicates('bit')
        .head(5)[['bit','radius','fragment']].rename(columns={'bit':'Бит','radius':'Радиус','fragment':'Пример фрагмента (SMILES)'}))''')
    md('''## Проверка аугментаций

1. Случайные SMILES меняют текстовую запись той же молекулы.
2. Перестановка индексов атомов меняет порядок хранения того же графа.
3. Дополнительно обучаем лес с удалением 5% активных битов train. Это шум в признаках; он не гарантирует соответствия реальной молекуле и рассматривается только как диагностический эксперимент.''')
    code('''augmentation = augmentation_invariance(molecules)
display(pd.DataFrame({'Способ': ['Случайные SMILES','Перестановка атомов'],
                      'Проверок': [augmentation['trials_per_method']]*2,
                      'Совпавших fingerprints': [augmentation['random_smiles_identical_fingerprints'],
                                               augmentation['atom_permutation_identical_fingerprints']]}))
comparison = metrics.loc[metrics.experiment.isin([baseline['selected'],'RF_dropout5']) & metrics.model.eq('RF'),
                         ['experiment','part','MAE','RMSE']]
display(comparison.rename(columns={'experiment':'Эксперимент','part':'Часть'}).round(3))
noise_mae = comparison.loc[comparison.experiment.eq('RF_dropout5') & comparison.part.eq('test'), 'MAE'].iloc[0]
display(Markdown(f"Шум в битах изменил тестовую MAE на **{noise_mae - baseline['test_macro_MAE']:+.3f} pKi** "
                 "(плюс означает ухудшение)."))
display(Markdown('Одинаковые fingerprints не добавляют лесу новых структурных признаков. '
                 'Для будущей молекулярной GNN эти преобразования полезны как проверка '
                 'инвариантности; случайные SMILES могут быть полезны модели, читающей SMILES. '
                 'Основной бейзлайн фиксируется без искусственного удаления битов.'))''')
    md('''## Графики''')
    code('''generate_figures(pairs, molecules, eda, baseline)
for filename in ['01_affinity_and_targets.png','02_molecule_properties.png','03_correlations.png',
                 '04_pairplot.png','05_splits.png','06_prediction_errors.png','07_feature_importance.png']:
    display(Image(filename=str(RESULTS / 'figures' / filename)))''')
    md('''Корреляции и попарные распределения с pKi показаны внутри одной мишени, чтобы не смешивать разные белки. В этом примере отдельные дескрипторы слабо связаны с pKi (|ρ| ≤ 0,15). Общая корреляция размера молекулы с активностью могла бы отражать различия изученных наборов, а не универсальное правило.

## Выводы и следующий этап''')
    code(r'''difference = baseline['dummy_test_macro_MAE'] - baseline['test_macro_MAE']
conclusion = ('Лес улучшил MAE относительно среднего по мишени' if difference > 0
              else 'Лес не улучшил MAE относительно среднего по мишени')
display(Markdown(f"- После очистки осталось **{eda['curated_pairs']:,} пар**; "
                 f"**число зафиксированных для сравнения мишеней — {eda['benchmark_targets']}**.\n"
                 f"- {conclusion}: разница **{difference:.3f} pKi**. "
                 "Это результат на новых каркасах, а не оценка эффективности лечения.\n"
                 "- Покрытие мишеней неравномерно. Будущую многозадачную GNN сравниваем "
                 "на тех же 131 мишени и разбиениях, отдельно проверяя пользу для небольших задач. "
                 "Обучение на дополнительных мишенях — отдельный эксперимент.\n"
                 "- Пропуски активности не являются отрицательными метками. "
                 "Шум измерений и противоречивые повторы требуют отдельного контроля.\n"
                 "- Этот бейзлайн поддерживает известный набор мишеней. "
                 "Для переноса на новые белки потребуется модель, использующая их признаки.\n"
                 "- Практическая ценность и ограничения описаны в [одностраничном PDF]"
                 "(output/pdf/practical_value.pdf)."))''')
    md('''## Воспроизведение и источники

Python 3.12, версии библиотек — `requirements.txt`. Код очистки и экспериментов — `src/affinity_eda.py`; параметры — `config/eda.json`; зафиксированные данные и разбиения — `data/processed/`; метрики, предсказания и проверки — `results/`. В `results/manifest.json` записаны контрольные суммы. Сырые ответы API сохраняются локально и повторно используются; для новой выгрузки нужен интернет.

- [ChEMBL API](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services)
- [Типы измерений ChEMBL](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions)
- [Morgan fingerprints в RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints)
- [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

Требования чекпойнта: EDA для DL и Research, базовая модель, интерпретация, проверка аугментаций и отдельный PDF о практической ценности.''')
    notebook=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3 (affinity-eda)','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12.14'}})
    output=ROOT/'EDA.ipynb';nbf.write(notebook,output);print(output)


if __name__=='__main__':build()
