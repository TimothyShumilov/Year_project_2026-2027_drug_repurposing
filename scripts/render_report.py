"""Render a standalone Russian HTML report from computed local artifacts."""
from __future__ import annotations

import csv
import html
import json

from public_api import ROOT


def read_csv(name):
    with (ROOT/'data/processed'/name).open() as stream:
        return list(csv.DictReader(stream))


def esc(value):
    return html.escape(str(value))


def table(headers, rows, *, identifier=None):
    attrs = f' id="{identifier}"' if identifier else ''
    head = ''.join(f'<th>{esc(h)}</th>' for h in headers)
    body = ''.join('<tr>'+''.join(f'<td>{esc(c)}</td>' for c in row)+'</tr>' for row in rows)
    return f'<div class="scroll"><table{attrs}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def main():
    s=json.loads((ROOT/'data/processed/summary.json').read_text())
    ot_release=s['opentargets_release']['year']+'.'+s['opentargets_release']['month']
    chembl_release=s['chembl_status']['chembl_db_version'].replace('_',' ')
    drug_rows=read_csv('candidate_ranking.csv')
    coverage=read_csv('source_coverage.csv')
    quality=read_csv('approval_disagreements.csv')
    unmatched=read_csv('unmatched_interventions.csv')
    pct=lambda a,b:f'{100*a/b:.1f}%' if b else '—'
    decision='Исходную тему с направлением эффекта в центре исследования нужно скорректировать.'
    metrics=[
        ('Препараты в пуле ChEMBL, после объединения форм и исключения отозванных',s['nonwithdrawn_approved_parent_molecules']),
        ('Из них с курируемым механизмом действия, включая неполные/нечеловеческие мишени',s['approved_with_curated_moa']),
        ('С человеческой белковой мишенью / комплексом / семейством',s['approved_with_human_moa']),
        ('С мишенью, имеющей неклинические доказательства для болезни Крона',s['approved_with_nc_target']),
        ('С мишенью, имеющей генетические доказательства для болезни Крона',s['approved_with_genetic_target']),
        ('После исключения найденных известных показаний для болезни Крона',s['genetic_repurposing_candidates']),
        ('Из этого пула со связью через SINGLE PROTEIN',s['single_protein_genetic_candidates']),
        ('С генетическим source score ≥ 0.5',s['candidate_pool_sensitivity']['0.5']),
    ]
    source_table=table(['Источник','Свидетельств','Генов','Полное направление: свидетельств','Полное направление: генов'],
        [(r['source'],r['evidence_count'],r['unique_genes'],r['complete_direction_evidence'],r['complete_direction_genes']) for r in coverage])
    clinical_table=table(['Показатель','Число'],[
        ('Широкий поиск ClinicalTrials.gov',s['registry_search_records']),
        ('Явное указание болезни Крона в conditions',s['explicit_crohn_condition_records']),
        ('Интервенционные исследования DRUG/BIOLOGICAL, лечебная цель или цель не указана',s['eligible_drug_trials']),
        ('Из них с хотя бы одним консервативно сопоставленным препаратом',s['eligible_trials_with_mapped_intervention']),
        ('Различных сопоставленных препаратов / родительских молекул',s['mapped_clinical_drugs']),
        ('Различных аннотированных наборов человеческих мишеней и action_type',s['clinical_human_moa_signatures']),
        ('Кандидатов из генетического пула, перечисленных как вмешательство',s['candidate_clinically_listed_drugs']),
        ('Наборов мишеней / действий среди этих кандидатов',s['candidate_clinical_human_moa_signatures']),
        ('Известных препаратов болезни Крона внутри генетически поддержанного пула',s['genetically_eligible_control_drugs']),
        ('Наборов мишеней / действий среди этого контрольного набора',s['genetically_eligible_control_moa_signatures']),
        ('Контрольных препаратов со связью через SINGLE PROTEIN',s['single_protein_genetic_control_drugs']),
    ])
    candidate_table=table(['Ранг*','Препарат / parent ID','Генетически поддержанные гены**','Тип препарата','Score*','Исследований с этим вмешательством'],
        [(r['combined_rank'],r['drug_name']+' / '+r['drug_id'],', '.join(json.loads(r['linked_genetic_genes'])),
          ', '.join(json.loads(r['molecule_types'])),f"{float(r['combined_score']):.4f}",r['matched_crohn_trial_count']) for r in drug_rows],identifier='candidates')
    disagreement_table=table(['Препарат','Open Targets: global stage','ChEMBL: max_phase'],
        [(r['drug_name'],r['ot_global_stage'],r['chembl_max_phase'] or 'нет записи') for r in quality])
    unmatched_table=table(['Не сопоставленное / неоднозначное название','Вхождений'],[(r['intervention_name'],r['occurrences']) for r in unmatched[:25]])
    exclude_table=table(['Причина исключения / включения','Исследований'],sorted(s['trial_inclusion_counts'].items()))
    recall=table(['Модель','Контролей в top-20','Контролей в доступном пуле','Recall@20 внутри пула'],
        [(name,v['top20_controls'],v['control_count_in_eligible_pool'],pct(v['top20_controls'],v['control_count_in_eligible_pool']))
         for name,v in s['illustrative_recovery'].items()])
    facts={
        'Полные пары directionOnTarget / directionOnTrait': f"{s['complete_direction_evidence']} / {s['nonclinical_genetic_evidence']} ({pct(s['complete_direction_evidence'],s['nonclinical_genetic_evidence'])})",
        'Гены с полным направлением':f"{s['direction_assessable_genes']} / {s['genetic_associated_targets']} ({pct(s['direction_assessable_genes'],s['genetic_associated_targets'])})",
        'Гены с полным направлением и мишенью одобренного препарата':s['approved_targets_with_signed_genetic_evidence_ignoring_drug_action'],
        'Кандидаты с полным направлением и распознаваемым action_type':s['direction_assessable_repurposing_candidates'],
        'Клинические наборы мишеней с автоматически сопоставимым направлением':s['direction_covered_clinical_signatures'],
    }
    content=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Пилот: перепрофилирование препаратов при болезни Крона</title>
<style>
:root{{--ink:#182b39;--muted:#566575;--line:#dce3e8;--accent:#125b73;--bg:#f3f6f8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1160px;margin:36px auto;padding:36px;background:white;border-radius:16px;box-shadow:0 8px 40px #182b3908}}
h1{{font-size:34px;line-height:1.2;margin:12px 0}}h2{{margin:40px 0 12px;font-size:24px}}h3{{margin:24px 0 8px}}p{{max-width:1000px}}
.meta{{color:var(--muted);font-size:14px}}.decision{{border-left:5px solid #bc7630;background:#fff4e6;padding:18px 22px;margin:24px 0}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}.card{{padding:18px;border:1px solid var(--line);border-radius:10px}}
.number{{display:block;font-size:32px;font-weight:700;color:var(--accent)}}.label{{font-size:14px;color:var(--muted)}}
.scroll{{overflow:auto;border:1px solid var(--line);border-radius:8px;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:left;padding:11px 13px;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#eaf1f5}}tbody tr:last-child td{{border-bottom:0}}
tbody tr:nth-child(even){{background:#fafcfd}}a{{color:var(--accent)}}code{{background:#eef3f6;padding:2px 5px;border-radius:4px}}
.note{{font-size:14px;color:var(--muted)}}input{{width:100%;padding:12px;border:1px solid var(--line);border-radius:8px;font-size:16px}}
details{{border:1px solid var(--line);padding:12px 16px;border-radius:8px;margin:16px 0}}summary{{cursor:pointer;font-weight:600}}pre{{overflow:auto;background:#eef3f6;padding:16px;border-radius:8px}}
@media(max-width:760px){{main{{margin:0;padding:22px;border-radius:0}}.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:28px}}}}
@media print{{body{{background:white}}main{{margin:0;padding:0;box-shadow:none}}.scroll{{overflow:visible}}details>div{{display:block}}input{{display:none}}}}
</style><main>
<div class="meta">Срез на {esc(s['as_of_date'])} · Open Targets {esc(ot_release)} · {esc(chembl_release)} · ClinicalTrials.gov API v2</div>
<h1>Пилот выполнимости проекта<br>Болезнь Крона и перепрофилирование препаратов</h1>
<p>Проверены размер лекарственного пула, доступность готового направления генетического эффекта, связывание клинических исследований с препаратами и число различимых аннотаций механизмов.</p>
<div class="decision"><strong>{decision}</strong><p>Для объяснимого ранжирования по генетическим и функциональным доказательствам есть рабочий набор. Для сравнения моделей по направлению эффекта готовых аннотаций недостаточно: ни один кандидат не имеет одновременно полной генетической пары направления и распознаваемого типа действия препарата. Обучение и оценка сложной ML-модели также ограничены малым контрольным набором.</p></div>
<div class="cards">
<div class="card"><span class="number">{s['genetic_associated_targets']}</span><span class="label">мишеней с генетическими доказательствами</span></div>
<div class="card"><span class="number">{s['genetic_repurposing_candidates']}</span><span class="label">кандидатов после исключения известных показаний</span></div>
<div class="card"><span class="number">{s['eligible_drug_trials']}</span><span class="label">исследований, прошедших фильтры</span></div>
<div class="card"><span class="number">{s['direction_assessable_repurposing_candidates']}</span><span class="label">кандидатов с полностью сопоставимым направлением</span></div>
</div>
<h2>1. Размер доступного пула</h2>
<p>Open Targets содержит {s['ot_associated_targets']} мишеней для болезни Крона и её подтипов. После исключения клинических, литературных и онкологических источников остаются {s['nc_associated_targets']} мишеней; {s['genetic_associated_targets']} имеют генетические доказательства. Данные родительского термина IBD и язвенного колита в этот срез не переносились.</p>
{table(['Шаг отбора','Число'],metrics)}
<p class="note">«Одобренный» здесь означает <code>max_phase = 4</code> в ChEMBL; формы объединены через <code>parent_chembl_id</code>. Это определение базы, а не самостоятельная регуляторная экспертиза. Отозванные семейства исключены. Из {s['approved_with_genetic_target']} генетически поддержанных препаратов {s['genetically_eligible_control_drugs']} отнесены к известным показаниям и исключены из кандидатов.</p>
<h2>2. Покрытие направления эффекта</h2>
{table(['Показатель','Результат'],facts.items())}
{source_table}
<p>Полное направление найдено для генов <strong>{', '.join(s['direction_gene_symbols'])}</strong>. Все {s['complete_direction_evidence']} полных записей в этом срезе соответствуют <code>LoF / risk</code>, то есть генетической гипотезе об активации. У {s['source_evidence_counts']['gwas_credible_sets']} GWAS-свидетельств готовые поля направления пусты. Это ограничение выбранной выгрузки: оно не доказывает отсутствия направления в первичных GWAS/molQTL-данных.</p>
<p>Единственное пересечение готового генетического направления с одобренным лекарственным пулом — NOD2 / mifamurtide. ChEMBL обозначает действие как <code>OTHER</code>, а не AGONIST/ACTIVATOR. Этот случай оставлен неизвестным. Ручная фармакологическая проверка одного случая не создаст достаточного набора для сравнительного эксперимента.</p>
<p>Направление нельзя выводить из повышенной экспрессии. Отсутствующее направление сохраняется как неизвестное; несовместимые направления разных записей считаются конфликтом. Клиническое направление «препарат защищает от заболевания» из Open Targets в признаки не включалось.</p>
<h2>3. Клиническая проверка и независимость наблюдений</h2>
{clinical_table}
<p>Консервативное сопоставление покрывает {pct(s['eligible_trials_with_mapped_intervention'],s['eligible_drug_trials'])} подходящих исследований. Приоритет имеет основное название вмешательства; затем проверяются синонимы, дозировка и путь введения. Неоднозначные совпадения и placebo исключены. Состав комбинации не угадывался.</p>
<p><strong>{s['clinical_human_moa_signatures']} — число различимых аннотированных наборов мишеней и action_type, а не доказанное число независимых терапевтических механизмов.</strong> Семейства, отдельные белки и близкие комплексы могут обозначать пересекающиеся механизмы. В самом пуле кандидатов найдено {s['candidate_clinically_listed_drugs']} клинически перечисленных препаратов и {s['candidate_clinical_human_moa_signatures']} таких наборов. Для эффективности нужны результаты и разбор роли вмешательства: препарат мог быть компаратором, фоновой терапией или вспомогательным средством.</p>
<p>Контрольный набор внутри генетически поддержанного пула содержит {s['genetically_eligible_control_drugs']} препаратов и {s['genetically_eligible_control_moa_signatures']} аннотированных наборов. При ограничении SINGLE PROTEIN остаётся {s['single_protein_genetic_control_drugs']} контрольный препарат. Такая выборка требует осторожного дизайна сравнения и не обосновывает большую GNN.</p>
<details><summary>Воронка исключения исследований</summary>{exclude_table}</details>
<details><summary>Частые названия, оставленные для ручного сопоставления</summary>{unmatched_table}<p class="note">Полная таблица: <a href="../data/processed/unmatched_interventions.csv">unmatched_interventions.csv</a>. Отсутствие совпадения не означает отсутствие исследуемого препарата в биомедицинских источниках.</p></details>
<h2>4. Качество лекарственных и клинических аннотаций</h2>
<p>У {s['ot_approval_disagreements']} препаратов из клинического набора Open Targets глобальный статус APPROVAL расходится с пулом одобренных молекул ChEMBL. Например, бриакинумаб: APPROVAL в Open Targets и фаза 3 в ChEMBL. Поэтому статус Open Targets не использован для формирования глобального одобренного пула.</p>
<details><summary>Все расхождения глобального статуса</summary>{disagreement_table}</details>
<p>ChEMBL не содержит всех текущих одобренных показаний болезни Крона. Записи регуляторов и drug labels из клинических аннотаций Open Targets дополнительно исключили из кандидатов гуселькумаб, рисанкизумаб и упадацитиниб. Также добавлена азатиоприновая аннотация показания. Сохранены первичные ссылки и происхождение каждой записи: <a href="../data/processed/crohn_approval_records.csv">crohn_approval_records.csv</a>. Эти записи используются только для меток и исключений.</p>
<p>Мишени PROTEIN COMPLEX и PROTEIN FAMILY проецируются на компоненты для поиска путей. Такая проекция не доказывает прямого связывания препарата с каждым генным продуктом. Например, запись о комплексе IL‑23 может перенести поддержку IL12B на препарат, действующий на другой компонент комплекса. Это отдельная проверяемая проблема для статьи. Только {s['single_protein_genetic_candidates']} из {s['genetic_repurposing_candidates']} кандидатов имеют хотя бы одну генетически поддержанную связь через SINGLE PROTEIN.</p>
<h2>5. Прототип ранжирования</h2>
<p>Сравнены три детерминированные эвристики без обучения: максимальный генетический source score; максимум <code>0.8 × genetics + 0.2 × expression</code> по связанным мишеням; тот же score с множителем 1.5 для согласованного направления и 0.5 для противоречащего. Веса заданы в конфигурации. Unknown/conflicting даёт множитель 1.0. Из-за отсутствия полностью сопоставимых направлений третья модель совпала со второй для всех кандидатов.</p>
<p class="note">* Этот рейтинг проверяет работу пайплайна и не представляет список доказанных новых терапий. Равные значения упорядочены по ChEMBL ID. ** Гены могут быть компонентами аннотированного комплекса/семейства. Число исследований отражает присутствие препарата в списке вмешательств, без вывода об эффективности.</p>
<input id="filter" type="search" placeholder="Поиск препарата, гена или ChEMBL ID" aria-label="Фильтр таблицы кандидатов">
{candidate_table}
<details><summary>Ретроспективное восстановление контрольного набора</summary>{recall}<p class="note">Знаменатель — только {s['genetically_eligible_control_drugs']} известных препаратов внутри заранее отобранного генетического пула, а не все известные средства для болезни Крона. Генетический отбор уже повышает обогащение. Веса не обучались; сравнение не является временным тестом, оценкой клинической эффективности или статистическим доказательством превосходства.</p></details>
<h2>6. Решение о теме и следующем эксперименте</h2>
<p><strong>Рекомендуемая корректировка:</strong> «Объяснимое ранжирование препаратов для перепрофилирования при болезни Крона: вклад генетических и функциональных доказательств и влияние детализации аннотаций мишеней».</p>
<p>Основной следующий эксперимент: сравнить генетический рейтинг и добавление экспрессии, отдельно для SINGLE PROTEIN и для комплексов/семейств; оценить устойчивость к порогу генетической поддержки и разметить клинически исследованные механизмы вручную. Это позволит исследовать реальную проблему качества данных. Направление эффекта оставить дополнительным разбором либо отдельным более сложным этапом реконструкции из колокализаций/GWAS/molQTL.</p>
<p>Пороговые инженерные критерии в <code>config/pilot.json</code> дают положительное решение о переходе к следующему этапу генетического ранжирования и отрицательное — о центральном эксперименте с готовым направлением. Это не расчёт статистической мощности и не подтверждение готовности статьи к публикации.</p>
<h2>7. Воспроизводимость и границы результата</h2>
<p>Срез открытых API сохранён вместе с исходными ответами, параметрами запросов, датами получения и SHA‑256. Пагинация проверена на полноту и уникальность: для Open Targets разные границы страниц позволили получить все {s['ot_associated_targets']} уникальных мишеней. Анализ выполняется локально без доступа к сети и без дополнительных библиотек.</p>
<pre>python3 scripts/fetch_pilot.py --offline
python3 scripts/analyze_pilot.py
python3 scripts/render_report.py
python3 -m unittest discover -s tests -v
python3 scripts/verify_pilot.py</pre>
<p>Временная проверка в пилоте <strong>не выполнена</strong>. У {s['evidence_with_evidence_date']} из {s['nonclinical_genetic_evidence']} свидетельств есть evidenceDate, у {s['evidence_with_publication_date']} — публикационная дата/год, у {s['active_approved_with_first_approval_year']} из {s['nonwithdrawn_approved_parent_molecules']} препаратов — год первого одобрения. Эти поля не заменяют исторические версии drug–target-связей, молекулярных аннотаций и моделей L2G. Текущие признаки нельзя объявлять доступными в прошлом только по дате публикации.</p>
<p>Ни обучение ML, ни проверка будущих показаний, ни фармакологическая/экспериментальная валидация новых кандидатов здесь не заявляются. Неизвестные связи не превращены в отрицательные примеры; статус COMPLETED/TERMINATED не превращён в метку успеха/неудачи.</p>
<p><a href="../data/processed/summary.json">Все показатели JSON</a> · <a href="../data/processed/candidate_ranking.csv">Рейтинг CSV</a> · <a href="../data/processed/clinical_mechanisms.csv">Клинические наборы мишеней CSV</a> · <a href="../data/processed/manifest.json">Манифест SHA‑256</a> · <a href="../README.md">Инструкция проекта</a></p>
<h3>Источники</h3><p class="note">
<a href="https://platform.opentargets.org/disease/MONDO_0005011/associations">Open Targets: Crohn disease</a> ·
<a href="https://platform-docs.opentargets.org/evidence">Схема доказательств и направления эффекта</a> ·
<a href="https://platform-docs.opentargets.org/data-access/graphql-api">Open Targets GraphQL API</a> ·
<a href="https://www.ebi.ac.uk/chembl/api/data/docs">EMBL‑EBI ChEMBL API</a> ·
<a href="https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/drug-and-compound-questions">Аннотации ChEMBL</a> ·
<a href="https://clinicaltrials.gov/data-api/api">NIH ClinicalTrials.gov API v2</a>.</p>
</main><script>document.getElementById('filter').addEventListener('input',function(){{const q=this.value.toLowerCase();document.querySelectorAll('#candidates tbody tr').forEach(tr=>{{tr.hidden=!tr.textContent.toLowerCase().includes(q)}})}});</script></html>'''
    out=ROOT/'reports/pilot_report.html'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(content)
    print(out)


if __name__=='__main__':
    main()
