"""Plain scientific figures for the executed EDA notebook."""
from __future__ import annotations

import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, StrMethodFormatter
import numpy as np
import pandas as pd

from affinity_eda import RESULTS,DATA


def generate_figures(pairs,molecules,summary,baseline):
    destination=RESULTS/'figures';destination.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.dpi':110,'savefig.dpi':150})
    def save(fig,name):
        fig.tight_layout();fig.savefig(destination/name,bbox_inches='tight');plt.close(fig)
    counts=pairs.groupby('target_name').size().sort_values(ascending=False)
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    ax[0].hist(pairs.pki,bins=50,color='#4c6b83');ax[0].set(xlabel='pKi — больше означает сильнее связывание',ylabel='Число пар')
    ax[1].barh(counts.head(8).index[::-1],counts.head(8).values[::-1],color='#4c6b83')
    ax[1].set(xlabel='Число уникальных молекул',title='Наиболее изученные мишени')
    save(fig,'01_affinity_and_targets.png')
    fig,axes=plt.subplots(2,2,figsize=(9,5.5))
    for ax,column,label in zip(axes.ravel(),['heavy_atoms','molecular_weight','logp','tpsa'],
                              ['Число тяжёлых атомов','Молекулярная масса, Да','Расчётный logP','Полярная площадь, Å²']):
        ax.hist(molecules[column],bins=40,color='#4c6b83');ax.set(xlabel=label,ylabel='Число молекул')
    save(fig,'02_molecule_properties.png')
    target=pairs.loc[pairs.benchmark_target].groupby('target_chembl_id').size().idxmax()
    block=pairs.loc[pairs.target_chembl_id.eq(target)].merge(molecules,on='compound_id',suffixes=('','_mol'))
    columns=['pki','molecular_weight','logp','tpsa','heavy_atoms']
    correlations=block[columns].corr(method='spearman')
    correlations.to_csv(RESULTS/'within_target_correlations.csv')
    fig,ax=plt.subplots(figsize=(6,4.8))
    image=ax.imshow(correlations,vmin=-1,vmax=1,cmap='RdBu_r')
    labels=['pKi','Масса','logP','Полярная площадь','Тяжёлые атомы']
    ax.set_xticks(range(5),labels,rotation=35,ha='right');ax.set_yticks(range(5),labels)
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{correlations.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9)
    ax.set_title('Связи признаков внутри одной мишени\n'+block.target_name.iloc[0],fontsize=10)
    fig.colorbar(image,ax=ax,shrink=.75,label='Корреляция Спирмена')
    save(fig,'03_correlations.png')
    # A compact pairplot of descriptors and pKi for one target; avoids mixing target effects.
    sample=block.sample(min(1500,len(block)),random_state=42)
    selected=['molecular_weight','logp','pki'];labels=['Масса','logP','pKi']
    fig,axes=plt.subplots(3,3,figsize=(7,6))
    for i,col_y in enumerate(selected):
        for j,col_x in enumerate(selected):
            ax=axes[i,j]
            if i==j:ax.hist(sample[col_x],bins=25,color='#4c6b83')
            else:ax.scatter(sample[col_x],sample[col_y],s=3,alpha=.2,color='#4c6b83',rasterized=True)
            if i==2:ax.set_xlabel(labels[j])
            if j==0:ax.set_ylabel(labels[i])
    save(fig,'04_pairplot.png')
    fig,axes=plt.subplots(1,2,figsize=(9,3.2))
    for part,color in zip(['train','validation','test'],['#4c6b83','#999999','#bb7a4b']):
        data=pairs.loc[pairs.benchmark_target & pairs.scaffold_split.eq(part),'pki']
        axes[0].hist(data,bins=np.linspace(pairs.pki.min(),pairs.pki.max(),45),density=True,histtype='step',label=part,color=color)
    axes[0].set(xlabel='pKi',ylabel='Плотность');axes[0].legend(frameon=False)
    overlap=[summary['random_split_scaffold_overlap_train_test'],summary['scaffold_split_scaffold_overlap_train_test']]
    axes[1].bar(['Случайное','По каркасам'],overlap,color=['#999999','#4c6b83'])
    axes[1].set(ylabel='Общих каркасов train и test')
    save(fig,'05_splits.png')
    predictions=pd.read_parquet(RESULTS/'test_predictions.parquet')
    test=predictions.loc[predictions.experiment.eq(baseline['selected'])]
    sample=test.sample(min(12000,len(test)),random_state=42)
    detail=pd.read_csv(RESULTS/'metrics_by_target.csv')
    selected=detail.loc[detail.experiment.eq(baseline['selected'])&detail.part.eq('test')&detail.model.eq('RF')].copy()
    train_counts=pairs.loc[pairs.scaffold_split.eq('train')].groupby('target_chembl_id').size()
    selected['n_train']=selected.target_id.map(train_counts)
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    axes[0].scatter(sample.pki,sample.prediction,s=3,alpha=.15,color='#4c6b83',rasterized=True)
    lo=min(sample.pki.min(),sample.prediction.min());hi=max(sample.pki.max(),sample.prediction.max())
    axes[0].plot([lo,hi],[lo,hi],color='black',lw=1);axes[0].set(xlabel='Измеренный pKi',ylabel='Предсказанный pKi')
    axes[1].scatter(selected.n_train,selected.MAE,s=18,color='#4c6b83',alpha=.7)
    axes[1].set(xscale='log',xlabel='Измерений в train для мишени',ylabel='MAE на новых каркасах')
    axes[1].xaxis.set_major_locator(FixedLocator([200,500,1000,2000,5000]))
    axes[1].xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    save(fig,'06_prediction_errors.png')
    importance=pd.read_csv(RESULTS/'feature_importances.csv').sort_values('permutation_MAE_increase',ascending=False).head(10)
    fig,ax=plt.subplots(figsize=(7,3.2))
    ax.barh(importance.bit.astype(str).iloc[::-1],importance.permutation_MAE_increase.iloc[::-1],color='#4c6b83')
    ax.axvline(0,color='black',lw=.8);ax.set(xlabel='Изменение MAE после перемешивания признака',ylabel='Номер бита Morgan')
    save(fig,'07_feature_importance.png')
    return sorted(p.name for p in destination.glob('*.png'))
