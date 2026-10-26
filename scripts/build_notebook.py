"""Build a concise, readable checkpoint notebook; calculations live in src/."""
import json
import sys
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pki_sources import build_report, usage_markdown, preparation_markdown


def build():
    source_usage = build_report()
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text):cells.append(nbf.v4.new_code_cell(text.strip()))
    md(r'''# EDA и бейзлайн прогнозирования аффинности

По структуре малой молекулы предсказываем **pKi для выбранной белковой мишени** — числовую меру силы их взаимодействия.

**Малая молекула** в этом проекте — органическое соединение с относительно небольшой молекулярной массой. Её **структура** описывает, какие атомы входят в состав молекулы и какими химическими связями они соединены.

**SMILES** — текстовый формат записи химической структуры: буквы обозначают атомы, специальные символы — связи, скобки — ответвления, а цифры указывают замыкание колец.

Например, [кофеин](https://pubchem.ncbi.nlm.nih.gov/compound/caffeine) состоит из атомов углерода, водорода, азота и кислорода; его структуру можно записать в формате SMILES: `Cn1c(=O)c2c(ncn2C)n(C)c1=O`.

**Белковая мишень** — белок, связывание с которым может изменять его активность. Белок представляет собой молекулу, построенную из аминокислот. В проекте рассматриваем отдельные человеческие белки. Например, [аденозиновый рецептор A2A](https://www.ncbi.nlm.nih.gov/gene/135) — белок на мембране клетки, который воспринимает сигнал аденозина и передаёт его внутрь клетки.

**Ki** — константа ингибирования: чем она меньше, тем сильнее взаимодействие. **Целевая величина основной модели — именно pKi**, логарифмическое преобразование Ki:

$$
pK_i = -\log_{10}(K_i\text{ в моль/л}) = 9 - \log_{10}(K_i\text{ в нМ}).
$$

Например, Ki = 100 нМ даёт pKi = 9 − log10(100) = 7; Ki = 1 мкМ = 1000 нМ даёт pKi = 6. Увеличение pKi на 1 означает уменьшение Ki в 10 раз. Модель предсказывает pKi; большее значение означает более сильное взаимодействие.''')
    md('''## Химический каркас и разбиение данных

**Химический каркас** — основа строения молекулы. В проекте используем каркас Bemis–Murcko в реализации RDKit: упрощённо, это кольца и участки, соединяющие их, после удаления боковых цепей. Кольцо — замкнутая цепочка атомов; боковые группы присоединены к этой основе. Типы атомов и связей в каркасе сохраняются, стереохимия при группировке не учитывается.

Например, толуол и фенол различаются боковой группой, но имеют один каркас:

| Молекула | Структура упрощённо | Каркас |
|---|---|---|
| Толуол | Бензольное кольцо + CH₃ | Бензольное кольцо |
| Фенол | Бензольное кольцо + OH | Бензольное кольцо |

Одинаковый каркас не означает одинаковый pKi: боковые группы тоже влияют на связывание с белком. **Каркас используем только для разбиения. Morgan fingerprints считаем по полной обработанной молекуле, включая боковые группы.**

**Зачем группировать по каркасу.** При случайном разбиении близкие аналоги могут оказаться одновременно в train и test. Их fingerprints часто похожи, поэтому ошибка на таком тесте может быть слишком оптимистичной для поиска соединений с новой основой. Разбиение по каркасам проверяет, насколько хорошо модель предсказывает pKi для структур с каркасами, которых не было в обучении.

**Как это работает у нас.** Всем молекулам с одним каркасом назначаем одну часть: train, validation или test. Назначение общее для всех источников и белковых мишеней. Если бензольный каркас попал в train, толуол и фенол попадут туда вместе. Целевое соотношение групп — около 80/10/10; группы имеют разный размер, поэтому доли молекул и измеренных пар могут отличаться. Исходное разбиение ChEMBL сохранено, новые каркасы распределяются воспроизводимым хешем без использования pKi.

Молекулы без колец дают пустой каркас; наш код объединяет их в одну группу `ACYCLIC`. Это техническое упрощение: такие молекулы могут сильно различаться. Разные непустые каркасы тоже могут быть похожи. Тест проверяет перенос на новые каркасы для известных белковых мишеней.

Точное выделение каркаса выполняет [RDKit GetScaffoldForMol](https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html).''')
    md('''## Выбор данных

'''+usage_markdown(source_usage, include_selection=False)+'''

Гиперпараметры подбираю по validation-парам исходного ChEMBL.

Разбиение общее для всех источников: один химический каркас попадает только в одну часть.

'''+preparation_markdown(source_usage)+'''

Для всех источников проверяем SMILES, удаляем фрагменты солей по единому правилу и вычисляем Morgan fingerprints. Повторы одной пары внутри источника объединяем медианой pKi; разброс более 2 pKi исключаем. При пересечении баз выбираем одну метку: ChEMBL → BindingDB → PubChem → DrugCentral. Сохраняем происхождение и расхождения; исходные ответы ChEMBL не меняем. В таблице указан источник выбранной метки, поэтому каждая пара посчитана один раз.''')
    md('''## Что есть в данных и что использует модель

**Исходная строка — результат одного эксперимента для молекулы и биологической мишени.** Для основной задачи отбираем отдельные человеческие белки. После очистки одна строка — уникальная пара «структура молекулы — мишень» для определённого типа активности. Структуры и их свойства хранятся отдельно в `molecules.parquet`. Исходный срез ChEMBL находится в `data/processed/`, объединённые данные — в `data/multisource/`.

Основные исходные поля ChEMBL и присоединённых аннотаций:

| Поле | Что означает | Как используем и почему |
|---|---|---|
| `canonical_smiles` → `smiles` | Текстовая запись атомов, связей и стереохимии молекулы; `smiles` — результат нашей обработки | Источник входных признаков: структура определяет, какие взаимодействия молекула может образовывать с белком |
| `standard_type`, `standard_value`, `standard_units`, `standard_relation` | Вид, числовое значение, единицы и знак измерения; например, Ki = 100 нМ | Отбираем точные Ki в нМ и рассчитываем **целевую величину** `pki = 9 − log10(Ki в нМ)` |
| `pchembl_value` | Готовая логарифмическая мера активности ChEMBL; для выбранных Ki соответствует pKi | Проверяем пересчёт. Результаты измерения используются как ответ модели, а подача их на вход создала бы утечку ответа |
| `target_chembl_id`, название мишени | Какой белок исследовали | Выбирают отдельный лес или выход будущей многозадачной сети. ID служит меткой задачи и не описывает строение белка |
| `assay_type`, `target_type`, организм, `confidence_score`, аннотация мутации | Тип теста, тип и вид мишени, уверенность её сопоставления и сведения о варианте белка | Фильтруем сопоставимые эксперименты: связывание, отдельный человеческий белок, уверенность 9, без аннотированной мутации |
| `data_validity_comment`, `standard_flag`, `potential_duplicate` | Комментарии и флаги качества измерения и возможного дубликата | Используем при очистке; они описывают качество записи |
| ID молекулы, измерения, теста и документа; `document_year` | Идентификаторы происхождения записи и год публикации | Связываем таблицы, проверяем повторы и историю данных. На вход модели не подаём: по ним нельзя описать новую структуру |

В объединённом наборе `target_id` задаёт UniProt-белок, `endpoint` — шкалу ответа, а `label` — числовую метку. `source` указывает выбранный источник; `sources`, `source_count`, `cross_source_range` — все источники пары и их расхождение. В таблице наблюдений `provider` обозначает поставщика выгрузки, `upstream_source` — исходную базу, `source_record_id` и `source_study_id` — происхождение записи. Эти поля нужны для отбора задачи и аудита; входом леса остаются только признаки структуры молекулы.

Из SMILES с помощью RDKit дополнительно рассчитываем свойства молекул. Они нужны для EDA: показывают, какие соединения представлены и где встречаются необычные значения.

| Поле в `molecules.parquet` | Что означает | Что помогает проверить |
|---|---|---|
| `molecular_weight`, `heavy_atoms` | Молекулярная масса в Да и число атомов, кроме водорода | Размер молекул; чрезмерно крупные соединения |
| `logp` | Расчётный logP — мера предпочтения жировой среды водной | Липофильность и её диапазон |
| `tpsa` | Расчётная топологическая полярная площадь, Å² | Выраженность полярных участков молекулы |
| `hbd`, `hba` | Число доноров и акцепторов водородной связи | Возможности образования водородных связей |
| `rotatable_bonds`, `rings` | Число вращаемых связей и колец | Гибкость и кольцевое строение молекул |

**Вход Random Forest — только 2048 Morgan-битов**, рассчитанных из `smiles`: 0/1 кодируют наличие локальных окружений атомов в радиусе двух связей, с учётом стереохимии. Это фиксированный числовой вектор, удобный для леса и сохраняющий сведения о подструктурах. Дескрипторы выше используем для анализа; такой выбор оставляет одно простое представление в бейзлайне. Пользу добавления дескрипторов можно проверить отдельным экспериментом. Сам SMILES лес не читает.

**Для будущей GNN** планируем использовать ту же структуру как граф: узлы — атомы (элемент, заряд, ароматичность), рёбра — связи (тип, стереохимия). Сеть будет сама обучать представление молекулы и выдавать pKi для фиксированного набора мишеней. Последовательности и 3D-структуры белков в текущий вход не входят.

Остальные поля — служебные: `compound_id`, `molecule_index` связывают таблицы и строки fingerprints; `connectivity`, `scaffold` и поля `*_split` задают разбиения; `benchmark_target` отмечает включённые в сравнение мишени. `n_measurements`, `n_assays`, `n_documents`, `pki_min`, `pki_max`, `repeat_range`, `first_year`, `last_year` описывают повторы и публикации, а `fragment_removed` — удаление фрагмента соли. Эти поля помогают аудиту и оценке и не подаются в модель.

Определения: [поля измерений ChEMBL](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions), [Morgan fingerprints в RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints).''')
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
display(Markdown('**Исходный срез ChEMBL:**'))
display(pd.DataFrame({'Показатель': ['Исходных измерений', 'Пар после очистки', 'Молекул',
                                    'Мишеней после очистки', 'Мишеней для бейзлайна'],
                      'Число': [eda['raw_rows'], eda['curated_pairs'], eda['curated_molecules'],
                                eda['curated_targets'], eda['benchmark_targets']]}))
''')
    md('''## Объединённые данные и качество

В таблице ниже посчитаны подготовленные строки каждого источника до разрешения пересечений между базами. Строки ChEMBL уже объединяют повторы исходного среза; их нельзя считать независимыми от импортов в другие базы.''')
    code('''MULTI_RESULTS = ROOT / 'results/multisource'
MULTI_DATA = ROOT / 'data/multisource'
multi_eda = json.loads((MULTI_RESULTS / 'data_summary.json').read_text())
multi_baseline = json.loads((MULTI_RESULTS / 'baseline_summary.json').read_text())
source_usage = json.loads((MULTI_RESULTS / 'pki_source_usage.json').read_text())
display(pd.DataFrame(source_usage['sources'])[['source','prepared_observations','curated_pairs_all_targets']]
    .rename(columns={'source':'Источник','prepared_observations':'Строк pKi до согласования',
                     'curated_pairs_all_targets':'Пар pKi после согласования'}))
multi_pairs = pd.read_parquet(MULTI_DATA / 'pairs.parquet').query("endpoint == 'pKi'")
display(multi_pairs.groupby('endpoint').agg(Пар=('label','size'), Мишеней=('target_id','nunique'),
    Минимум=('label','min'), Медиана=('label','median'), Максимум=('label','max')).round(3))
scale_audit = json.loads((MULTI_RESULTS / 'drugcentral_scale_audit.json').read_text())
display(Markdown('Дополнительные Ki из BindingDB/PubChem оставлены только при совпадении '
    'экспериментальной последовательности с участком канонического человеческого белка. '
    f"Напрямую проверены ID **{len(source_usage['pubchem_pki_assay_ids'])}** тестов PubChem с Ki."))
display(Markdown(f"DrugCentral уже хранит отрицательный логарифм концентрации. "
    f"На **{scale_audit['overlapping_exact_structure_target_pairs']}** совпадающих парах медианное "
    f"расхождение с ChEMBL — **{scale_audit['median_absolute_difference_if_ACT_VALUE_is_pKi']:.3f} pKi**. "
    "Повторное логарифмирование было бы ошибкой."))
display(Markdown(f"Всего после согласования: **{multi_eda['pairs_by_endpoint']['pKi']:,} пар pKi**, "
    f"**{source_usage['curated_molecules']:,} структур**. "
    f"В бейзлайне используются **{source_usage['benchmark_pairs']:,} пар** для "
    f"**{source_usage['benchmark_targets']} мишеней**, прошедших пороги наполненности выборок. "
    "Расхождения между источниками сохранены во флаге cross_source_range; оригинальные метки ChEMBL не меняются."))''')
    md('''### Качество исходного среза ChEMBL

Сохраняем только Ki из тестов связывания: прямое сопоставление с отдельным человеческим белком, уверенность аннотации 9, корректную структуру и точное положительное измерение в нМ.

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
    md('''## Распределения и покрытие исходного среза

Это регрессия, поэтому баланса классов нет. Важны распределение pKi и дисбаланс числа измерений между мишенями.''')
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

Одна молекула относится только к одной части выборки. Для сравнения используем случайное разбиение, которое группирует одинаковые структуры без учёта стереохимии, и основное разбиение по химическим каркасам, описанное выше.

Ниже проверяем, что каркасы основного разбиения не пересекаются между train, validation и test, и показываем фактическое число молекул и размеченных пар в каждой части.''')
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
    md('''## Исходный бейзлайн ChEMBL: Random Forest на Morgan fingerprints

Для каждой мишени обучается отдельный лес из 100 деревьев. Проверяем минимальный размер листа 1, 3 и 10; остальные параметры одинаковы. Выбор — по среднему MAE на validation основного разбиения.

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

## Результаты после объединения источников

Все источники используют общие группы молекул. Сравнение с прежним лесом проводится на тех же 16 665 парах теста ChEMBL; метки теста и разбиение не изменены. Число деревьев и варианты размера листа одинаковы; параметры выбираются по прежней validation-выборке ChEMBL.''')
    code('''from multisource_figures import generate as generate_source_figure
display(Image(filename=str(generate_source_figure())))
display(pd.read_csv(MULTI_RESULTS / 'original_test_comparison.csv').rename(
    columns={'training_sources':'Источники обучения','targets':'Мишеней','test_pairs':'Пар test'}).round(3))
rows = []
for result in multi_baseline['datasets']:
    if result['dataset'] != 'pKi':
        continue
    rf = next(r for r in result['test_results'] if r['part']=='test' and r['model']=='RF')
    mean = next(r for r in result['test_results'] if r['part']=='test' and r['model']=='Mean')
    rows.append({'Набор':result['dataset'],'Шкала':result['endpoint'],'Мишеней':result['eligible_targets'],
                 'Пар train/val/test':result['eligible_pairs'],'Пар test':result['test_pairs'],
                 'MAE RF':rf['MAE'],'MAE среднего':mean['MAE']})
display(pd.DataFrame(rows).round(3))
display(pd.read_csv(MULTI_RESULTS / 'source_test_slices.csv').query("endpoint == 'pKi'")[['source','targets','pairs','macro_MAE']]
        .rename(columns={'endpoint':'Шкала','source':'Источник','targets':'Мишеней','pairs':'Пар','macro_MAE':'MAE'}).round(3))
display(Markdown('Срезы по источнику могут содержать те же пары, что основной тест. '
    'Малые срезы, особенно четыре пары PubChem pKi, не позволяют делать устойчивые выводы о качестве на всей базе.'))''')
    md('''## Датасеты для Hugging Face''')
    code('''package = json.loads((MULTI_RESULTS / 'hf_package.json').read_text())
display(pd.DataFrame(package['configurations']).pivot(index='config',columns='split',values='rows').fillna(0).astype(int))
publication_file = MULTI_RESULTS / 'hf_publication.json'
if publication_file.exists():
    publication = json.loads(publication_file.read_text())
    display(Markdown(f"Опубликовано: [{publication['repository']}]({publication['url']}). "
                     f"Версия: `{publication['revision']}`."))
else:
    display(Markdown(f"Пакет для `{package['repository']}` подготовлен локально; "
                     "публикация ещё не подтверждена."))
display(Markdown('В пакет входят готовые train/validation/test для pKi, структуры, '
    'происхождение меток, лицензии, контрольные суммы и результаты экспериментов. '
    'Флаг benchmark_target отмечает 156 мишеней текущего бейзлайна.'))''')
    md('''## Выводы и следующий этап''')
    code(r'''main_result = next(r for r in multi_baseline['datasets'] if r['dataset']=='pKi')
main_test = next(r for r in main_result['test_results'] if r['part']=='test' and r['model']=='RF')
comparison = multi_baseline['original_test_comparison']
display(Markdown(f"- Основной очищенный набор — **{multi_eda['pairs_by_endpoint']['pKi']:,} пар pKi**; "
    f"для бенчмарка подходят **{main_result['eligible_targets']} мишеней** и **{main_result['eligible_pairs']:,} пар**.\n"
    f"- MAE леса на новых каркасах — **{main_test['MAE']:.3f} pKi**. На фиксированном тесте ChEMBL "
    f"после расширения — **{comparison[1]['MAE']:.3f}** против **{comparison[0]['MAE']:.3f}** прежде: "
    "само добавление источников улучшения не дало; причина этим сравнением не установлена.\n"
    "- Следующий эксперимент — многозадачная GNN на тех же метках и разбиениях, "
    "с отдельным анализом небольших задач. Модель пока поддерживает фиксированный набор белков.\n"
    "- Пропуски не являются отрицательными метками; повторы между базами "
    "не считаются дополнительными независимыми парами.\n"
    "- Практическая ценность и ограничения — в [одностраничном PDF](output/pdf/practical_value.pdf)."))''')
    md('''## Воспроизведение и источники

Python 3.12, версии библиотек — `requirements.txt`. Исходный срез ChEMBL: `src/affinity_eda.py`, `config/eda.json`, `data/processed/`. Объединение: `src/multisource_data.py`; эксперименты: `src/multisource_models.py`; параметры: `config/multisource.json`; результаты: `results/multisource/`. Данные [опубликованы на Hugging Face](https://huggingface.co/datasets/tim-shu/drug-affinity-multisource); локальные копии — в `data/multisource/` и `output/huggingface/`. Команды воспроизведения из готового пакета — в `readme.md`. Сырые выгрузки кэшируются. Воспроизведение из источников: `scripts/download_sources.py`, затем `scripts/run_multisource.py`. Проверки: `scripts/verify_results.py` и `scripts/verify_multisource.py`.

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
