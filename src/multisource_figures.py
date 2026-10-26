"""Source distributions for the expanded affinity corpus."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multisource_data import DATA,RESULTS


def generate():
    estimates=pd.read_parquet(DATA/'source_estimates.parquet')
    pki=estimates.loc[estimates.endpoint.eq('pKi')]
    selected=pd.read_parquet(DATA/'pairs.parquet')
    chosen=selected.loc[selected.endpoint.eq('pKi')].source.value_counts()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    bins=np.linspace(pki.label.min(),pki.label.max(),45)
    for (source,block),color in zip(pki.groupby('source'),['#b17144','#4c6b83','#8d969d','#6f8c66']):
        axes[0].hist(block.label,bins=bins,density=True,histtype='step',label=source,color=color)
    axes[0].set(xlabel='pKi',ylabel='Плотность',title='Метки до разрешения пересечений')
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].barh(chosen.index[::-1],chosen.values[::-1],color='#4c6b83')
    axes[1].set(xlabel='Число пар',title='Источник выбранной метки pKi',xscale='log')
    fig.tight_layout()
    path=RESULTS/'source_distributions.png'
    fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)
    return path
