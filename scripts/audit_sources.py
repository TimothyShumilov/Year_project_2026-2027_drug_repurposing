"""Record source applicability, actual use, access limits and label-scale checks."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from affinity_eda import save_json
from multisource_data import DATA,RESULTS


def main():
    s=json.loads((RESULTS/'data_summary.json').read_text())
    entries=[
        {'source':'ChEMBL','status':'used','role':'Основная регрессия pKi и исходный фиксированный тест',
         'license':'CC BY-SA 3.0; integrated adaptation BY-SA 4.0','evidence':'https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions'},
        {'source':'BindingDB','status':'used','role':'Дополнительные Ki из статей и патентов; импорт ChEMBL повторно не добавлен',
         'license':'Native data: CC BY; upstream notices retained','evidence':'https://www.bindingdb.org/rwd/bind/info.jsp'},
        {'source':'PubChem BioAssay','status':'used','role':'Ki и IC50/Kd через курируемый импорт BindingDB; AID проверены в PubChem',
         'license':'Source/depositor terms retained; factual observations only','evidence':'https://pubchem.ncbi.nlm.nih.gov/docs/bioassays'},
        {'source':'DrugCentral','status':'used','role':'Количественные активности одобренных препаратов, отдельные оценки срезов по источнику',
         'license':'CC BY-SA 4.0','evidence':'https://www.drugcentral.org/privacy'},
        {'source':'UniProt','status':'used','role':'Проверка вида и последовательностей белков; признаки для будущей DL-модели',
         'license':'CC BY 4.0','evidence':'https://www.uniprot.org/help/license'},
        {'source':'DAVIS','status':'used','role':'Отдельный бенчмарк pKd с флагом предела измерения',
         'license':'TDC DOI: CC0 1.0; TDC page also links CC BY 4.0 with Not Specified notice',
         'evidence':'https://api.datacite.org/dois/10.7910/DVN/21LKWG'},
        {'source':'KIBA','status':'used','role':'Отдельный бенчмарк в собственной шкале KIBA score',
         'license':'TDC DOI: CC0 1.0; TDC page also links CC BY 4.0 with Not Specified notice',
         'evidence':'https://api.datacite.org/dois/10.7910/DVN/21LKWG'},
        {'source':'PDBbind','status':'applicable_but_access_and_redistribution_blocked',
         'role':'Может дать дополнительные аффинности; разрешённая для публикации копия не получена',
         'license':'Redistribution of raw/derived data requires written permission',
         'evidence':'https://www.pdbbind-plus.org.cn/termofuse'},
        {'source':'DrugBank','status':'applicable_but_authorised_download_missing',
         'role':'Потенциальные количественные активности и известные взаимодействия; доступной авторизованной выгрузки нет',
         'license':'Dataset-specific academic access; terms must be verified on the supplied release',
         'evidence':'https://go.drugbank.com/academic_research/'},
        {'source':'Open Targets','status':'outside_current_affinity_endpoint',
         'role':'Связи мишеней с заболеваниями для последующего анализа гипотез, а не метки Ki текущей регрессии',
         'evidence':'https://platform-docs.opentargets.org/'},
        {'source':'ClinicalTrials.gov','status':'outside_current_affinity_endpoint',
         'role':'Клиническая проверка гипотез; записи испытаний не являются измерениями Ki',
         'evidence':'https://clinicaltrials.gov/data-api/about-api'}
    ]
    for entry in entries:
        match='PubChem' if entry['source']=='PubChem BioAssay' else entry['source']
        entry['prepared_rows_by_endpoint']={r['endpoint']:r['rows'] for r in s['rows_by_source_endpoint'] if r['source']==match}
    save_json(RESULTS/'source_registry.json',entries)
    estimates=pd.read_parquet(DATA/'source_estimates.parquet')
    keys=['compound_id','target_id','endpoint']
    dc=estimates.loc[estimates.source.eq('DrugCentral'),keys+['label']]
    chembl=estimates.loc[estimates.source.eq('ChEMBL'),keys+['label']]
    overlap=dc.merge(chembl,on=keys,suffixes=('_dc','_chembl'),validate='one_to_one')
    correct=np.abs(overlap.label_dc-overlap.label_chembl)
    incorrect=np.abs((9-np.log10(overlap.label_dc))-overlap.label_chembl)
    audit={'overlapping_exact_structure_target_pairs':len(overlap),
           'median_absolute_difference_if_ACT_VALUE_is_pKi':float(correct.median()),
           'median_absolute_difference_if_misread_as_nM':float(incorrect.median()),
           'fraction_within_0_1_pKi':float(correct.le(.1).mean()),
           'interpretation':'DrugCentral is already on the p-scale; disagreement between experiments still remains.'}
    if not audit['median_absolute_difference_if_ACT_VALUE_is_pKi'] < audit['median_absolute_difference_if_misread_as_nM']:
        raise ValueError('DrugCentral scale sanity check failed')
    save_json(RESULTS/'drugcentral_scale_audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
