"""Check source BioAssay identifiers without duplicating depositor protocols."""
import gzip
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/multisource/pubchem'
DATA=ROOT/'data/multisource'
RESULTS=ROOT/'results/multisource'


def main():
    RAW.mkdir(parents=True,exist_ok=True)
    observations=pd.read_parquet(DATA/'observations.parquet').query("endpoint == 'pKi'")
    values=observations.loc[observations.source.eq('PubChem'),'pubchem_aid'].dropna().unique()
    aids=sorted({int(v) for text in values for v in re.findall(r'\d+',text)})
    groups=[aids[i:i+20] for i in range(0,len(aids),20)]
    def one(group):
        url='https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/'+','.join(map(str,group))+'/description/JSON'
        path=RAW/(hashlib.sha256(url.encode()).hexdigest()+'.json.gz')
        if path.exists():
            with gzip.open(path,'rt') as stream:return json.load(stream)
        for attempt in range(4):
            try:
                r=requests.get(url,timeout=90);r.raise_for_status();d=r.json()
                with gzip.open(path,'wt') as stream:json.dump(d,stream)
                return d
            except requests.RequestException:
                if attempt==3:return {'error':'source metadata unavailable','requested_aids':group}
                time.sleep(2**attempt)
    records=[];errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(one,groups),1):
            if 'error' in result:errors.append(result)
            for container in result.get('PC_AssayContainer',[]):
                d=container.get('assay',{}).get('descr',{})
                records.append({'aid':d.get('aid',{}).get('id'),'version':d.get('aid',{}).get('version'),
                    'assay_name':d.get('name'),'activity_outcome_method':d.get('activity_outcome_method'),
                    'numeric_result_names':[v.get('name','') for v in d.get('results',[]) if v.get('type') in [1,2]],
                    'provider':'PubChem PUG REST'})
            print('PubChem assay metadata',i,'/',len(groups),flush=True)
    pd.DataFrame(records).drop_duplicates('aid').to_parquet(DATA/'pubchem_assays.parquet',index=False)
    output={'requested_aids':len(aids),'retrieved_aids':len({r['aid'] for r in records}),'failed_batches':errors,
            'data_provider':'Quantitative observations imported through BindingDB; assay identities checked directly with PubChem.'}
    (RESULTS/'pubchem_metadata_audit.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output),flush=True)


if __name__=='__main__':main()
