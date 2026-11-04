"""Local scientific figures for separated HP2 similarity axes; read-only inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

VIEW_NAMES={'structure':'作動の仕方','numeric':'トルク・ペダル等','combined':'構造＋数値','context':'前後の文脈'}
SIGNAL_NAMES={'rpm':'回転数','torque':'トルク','pedal2':'ペダル2','bulk':'バルク開量'}
VIEW_COLORS={'structure':'#397ea9','numeric':'#b56e46','combined':'#6b8d52','context':'#9a73ac'}


def safe(value):return re.sub(r'[^A-Za-z0-9_-]+','_',str(value))


def shortcase(value):
    parts=str(value).split('__')
    return parts[0]+' #'+parts[-1].replace('Chair_log_seq_','') if len(parts)>=3 else str(value)


def read(out,name,required=False):
    p=out/name
    if not p.exists():
        if required:raise FileNotFoundError(p)
        return pd.DataFrame()
    try:return pd.read_csv(p,low_memory=False)
    except pd.errors.EmptyDataError:
        if required:raise
        return pd.DataFrame()


def draw_profiles(plt,out,catalog,models,profiles,save):
    cat=catalog.set_index('feature');casecat=read(out,'case_feature_catalog.csv').set_index('feature')
    for mid,g in profiles.groupby('model_id',sort=False):
        model=models[models.model_id==mid].iloc[0]
        view=str(model['view']);level=str(model['level'])
        fullcat=casecat.reset_index() if level=='case' else catalog
        selected_views=['structure','numeric'] if view=='combined' else [view]
        wantedcat=fullcat[fullcat.view.isin(selected_views)]
        # Bulk remains an auxiliary signal; use exactly the principal numeric families.
        if view in ['numeric','combined']:
            wantedcat=wantedcat[(wantedcat.view!='numeric')|wantedcat.block.str.contains('torque|pedal2')]
        wanted=wantedcat.feature.tolist()
        selected=g[g.feature.isin(wanted)].copy()
        if selected.empty:continue
        metric='case_balanced_median' if level=='block' and 'case_balanced_median' in selected else 'raw_median'
        median=selected.pivot(index='feature',columns='cluster',values=metric)
        median=median.reindex([f for f in wanted if f in median.index]).dropna(how='all')
        if median.empty:continue
        median=median.iloc[:22]
        vals=median.to_numpy(float);center=np.nanmedian(vals,axis=1,keepdims=True)
        spread=np.nanmax(vals,axis=1,keepdims=True)-np.nanmin(vals,axis=1,keepdims=True)
        z=np.divide(vals-center,spread,out=np.zeros_like(vals),where=spread>1e-10);z[~np.isfinite(vals)]=np.nan
        fig=plt.figure(figsize=(14,max(7,len(median)*.68)))
        ax=fig.add_axes([.43,.055,.43,.85]);cax=fig.add_axes([.89,.13,.02,.65])
        im=ax.imshow(np.ma.masked_invalid(z),cmap='RdBu_r',vmin=-.6,vmax=.6,aspect='auto')
        labels=[]
        for f in median.index:
            label=str(cat.loc[f,'label']) if f in cat.index else str(casecat.loc[f,'label']) if f in casecat.index else f
            for a,b in [('torque','トルク'),('pedal2','ペダル2'),('bulk','バルク開量')]:label=label.replace(a,b)
            labels.append('\n'.join(label[i:i+28] for i in range(0,len(label),28)))
        ax.set_yticks(range(len(labels)),labels,fontsize=9)
        ax.set_xticks(range(median.shape[1]),[f'C{int(c)}' for c in median.columns],fontsize=11)
        for i,f in enumerate(median.index):
            for j,c in enumerate(median.columns):
                record=selected[(selected.feature==f)&(selected.cluster==c)];v=median.loc[f,c]
                n=f'\nn={int(record.n_observed.iloc[0])}' if len(record) else ''
                if level=='block' and 'n_supported_cases' in record and len(record):n+=f' / {int(record.n_supported_cases.iloc[0])}症例'
                ax.text(j,i,('—' if pd.isna(v) else f'{v:.3g}')+n,ha='center',va='center',fontsize=9)
        invalid=str(model.get('valid',True)).lower() in ['false','0','0.0']
        fig.suptitle(mid+('【基準不合格・非推奨】' if invalid else '')+'\n原単位の'+
            ('症例均等重み中央値' if metric=='case_balanced_median' else '中央値')+'、色は特徴ごとの相対差',fontsize=15,y=.98)
        fig.colorbar(im,cax=cax,label='特徴ごとの相対差（表示用）')
        save(fig,'profile_'+safe(mid)+'.png','候補群の原単位プロファイル：'+mid,
            '色は表示用で、数値は原単位。数値は3対応bin以上の支持から集計。nは支持単位数、併記する症例数も確認する。未観測を0表示しない。',view=view,level=level,kind='model')


def draw_final_supplements(plt,out,models,assign,nstable,save):
    cases=read(out,'case_features.csv').set_index('case_id')
    candidates=models[(models.level=='case')&(models['view']=='structure')&(models.method=='kmedoids')]
    if len(candidates):
        mid=candidates.iloc[0].model_id
        group=assign[(assign.model_id==mid)&assign.cluster.notna()].set_index('case_id')
        values=cases.join(group[['cluster']],how='inner')
        fig,axes=plt.subplots(1,2,figsize=(14,6));colors=['#457fa8','#c57b49','#7e9f59','#9d75ac']
        for cl,g in values.groupby('cluster'):
            color=colors[(int(cl)-1)%len(colors)]
            axes[0].scatter(g.run_q50_s__median,g.n_runs__median,s=60,color=color,alpha=.72,edgecolors='white',linewidths=.5,label=f'C{int(cl)}: {len(g)}症例')
            ok=g.torque_q50__median.notna()
            axes[1].scatter(g.loc[ok,'n_runs__median'],g.loc[ok,'torque_q50__median'],s=60,color=color,alpha=.72,edgecolors='white',linewidths=.5,label=f'C{int(cl)}: トルク支持{int(ok.sum())}/{len(g)}症例')
        axes[0].set_xlabel('典型的な1回の作動長（秒）\n各区間のrun長中央値 → 症例内中央値')
        axes[0].set_ylabel('典型的な区間内作動回数\n症例内ブロック中央値')
        axes[1].set_xlabel('典型的な区間内作動回数\n症例内ブロック中央値')
        axes[1].set_ylabel('トルクq50の症例内中央値\n3対応bin以上の支持区間のみ')
        for ax in axes:ax.legend(fontsize=9);ax.grid(alpha=.18)
        fig.suptitle('症例構造2群を原単位で確認する：各点は1症例',fontsize=16)
        fig.tight_layout(rect=[0,0,1,.92]);save(fig,'12_case_structure_raw_axes.png','症例構造候補を原単位の2軸で見る',
            '色は30秒ブロック定義内のcase構造k-medoids候補。構造が異なってもトルクの差が一方向とは限らない。未観測トルクは点を省略し、支持Nを凡例へ示す。',view='structure',level='case')
    equal=read(out,'equal_opportunity_neighbor_sensitivity.csv')
    if len(equal) and len(nstable):
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        for view,g in nstable[nstable.level=='block'].groupby('view'):
            vals=g.groupby('case_id').conditional_neighbor_retention.median()
            axes[0].hist(vals,bins=np.linspace(0,1,16),histtype='step',lw=2,color=VIEW_COLORS.get(view),label=VIEW_NAMES.get(view,view))
        for view,g in equal.groupby('view'):
            vals=g.groupby('case_id').mean_top3_jaccard.mean()
            axes[1].hist(vals,bins=np.linspace(0,1,16),histtype='step',lw=2,color=VIEW_COLORS.get(view),label=VIEW_NAMES.get(view,view))
        axes[0].set_title('候補機会を保ち、尺度を再推定\n元近隣が残存する反復への条件付き保持率')
        axes[1].set_title('候補を1区間/症例へ制限\n元top3とのJaccard：機会数と症例内異質性への感度')
        for ax in axes:ax.set_xlim(0,1);ax.set_xlabel('一致・保持指標（別定義）');ax.set_ylabel('症例数');ax.legend(fontsize=9)
        fig.suptitle('高い尺度再推定安定性と、候補構成への頑健性は別の評価',fontsize=15)
        fig.tight_layout(rect=[0,0,1,.9]);save(fig,'13_equal_opportunity_sensitivity.png','近隣の安定性を2種類に分ける',
            '左は各症例の区間ごと保持率の中央値。右は1区間/症例に候補を制限した感度の症例平均。指標・候補集合が異なり、同一性能の直接比較ではない。')
    dependence=read(out,'numeric_signal_dependence.csv')
    if len(dependence):
        table=dependence.pivot(index='model_id',columns='signal_only',values='ari_to_two_signal_partition')
        fig,ax=plt.subplots(figsize=(10,5));im=ax.imshow(table,aspect='auto',cmap='RdBu_r',vmin=-.1,vmax=1)
        ax.set_yticks(range(len(table)),table.index)
        ax.set_xticks(range(len(table.columns)),[SIGNAL_NAMES.get(c,c) for c in table.columns])
        for i in range(len(table)):
            for j in range(len(table.columns)):ax.text(j,i,f'{table.iloc[i,j]:.3f}',ha='center',va='center')
        fig.colorbar(im,ax=ax,label='2信号の元分割とのARI');ax.set_title('数値群は、どちらの信号の分け方に近いか')
        fig.tight_layout();save(fig,'14_numeric_signal_dependence.png','トルクとペダル2への分割依存',
            '値が高いほど、元の数値分割がその単一信号の分割に近い。高い一致を外部検証や2信号共通の機序と解釈しない。',view='numeric')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--dpi',type=int,default=145);ap.add_argument('--html-only',action='store_true')
    ap.add_argument('--profiles-only',action='store_true')
    ap.add_argument('--supplements-only',action='store_true')
    args=ap.parse_args();out=args.output.resolve()
    (out/'figures').mkdir(exist_ok=True);(out/'process').mkdir(exist_ok=True)
    def log(event,**details):
        record=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),event=event,**details)
        with (out/'process/visualization_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps(record,ensure_ascii=False),flush=True)
    units=read(out,'units.csv',True);features=read(out,'features.csv',True).set_index('unit_id')
    catalog=read(out,'feature_catalog.csv',True);support=read(out,'signal_support.csv')
    models=read(out,'models.csv');assign=read(out,'assignments.csv');neighbors=read(out,'neighbors.csv')
    nstable=read(out,'neighbor_stability.csv');nagree=read(out,'neighbor_agreement.csv')
    stability=read(out,'stability.csv');coords=read(out,'view_coordinates.csv')
    profiles=read(out,'cluster_profiles_supported.csv')
    if profiles.empty:profiles=read(out,'cluster_profiles.csv')
    examples=read(out,'pair_examples.csv')
    scope=read(out,'scope_audit.csv');stageunits=read(out,'stage_units.csv')
    stagefeatures=read(out,'stage_features.csv');casefeatures=read(out,'case_features.csv')
    log('start',primary_units=len(units),primary_cases=units.case_id.nunique(),html_only=args.html_only)
    assert len(units)==203 and units.case_id.nunique()==67
    if args.html_only:
        index=read(out,'figure_index.csv',True).fillna('').to_dict('records')
        build_html(out,units,models,assign,neighbors,nstable,coords,index,log)
        return
    if args.profiles_only:
        index=read(out,'figure_index.csv',True).fillna('').to_dict('records')
        index=[r for r in index if not str(r['path']).startswith('figures/profile_')]
        with tempfile.TemporaryDirectory(prefix='hp2-profile-refresh-') as tmp:
            os.environ['MPLCONFIGDIR']=tmp
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False,
                'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
            def save_profile(fig,name,title,caption,view='',level='',kind='model'):
                fig.savefig(out/'figures'/name,dpi=args.dpi,bbox_inches='tight');plt.close(fig)
                index.append(dict(path='figures/'+name,title=title,caption=caption,view=view,level=level,kind=kind))
            draw_profiles(plt,out,catalog,models,profiles,save_profile)
        pd.DataFrame(index).to_csv(out/'figure_index.csv',index=False)
        build_html(out,units,models,assign,neighbors,nstable,coords,index,log)
        log('profile_refresh_complete',n_figures=len(index))
        return
    if args.supplements_only:
        index=read(out,'figure_index.csv',True).fillna('').to_dict('records')
        index=[r for r in index if not any(str(r['path']).startswith('figures/'+prefix) for prefix in ['12_','13_','14_'])]
        with tempfile.TemporaryDirectory(prefix='hp2-supplement-refresh-') as tmp:
            os.environ['MPLCONFIGDIR']=tmp
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False,
                'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
            def save_supplement(fig,name,title,caption,view='',level='',kind='overview'):
                fig.savefig(out/'figures'/name,dpi=args.dpi,bbox_inches='tight');plt.close(fig)
                index.append(dict(path='figures/'+name,title=title,caption=caption,view=view,level=level,kind=kind))
            draw_final_supplements(plt,out,models,assign,nstable,save_supplement)
        pd.DataFrame(index).to_csv(out/'figure_index.csv',index=False)
        build_html(out,units,models,assign,neighbors,nstable,coords,index,log)
        log('supplement_refresh_complete',n_figures=len(index))
        return
    figures=[]
    with tempfile.TemporaryDirectory(prefix='hp2-similarity-viz-') as tmp:
        os.environ['MPLCONFIGDIR']=tmp
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False,
            'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
        def save(fig,name,title,caption,view='',level='',kind='overview'):
            fig.savefig(out/'figures'/name,dpi=args.dpi,bbox_inches='tight');plt.close(fig)
            figures.append(dict(path='figures/'+name,title=title,caption=caption,view=view,level=level,kind=kind))
        # Scope and explicit numeric support counts.
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        axes[0].bar(['監査対象','HP2評価可能','高回転主体を含む'],[300,249,67],color=['#bdc7d1','#719ebc','#397ea9'])
        for x,y in enumerate([300,249,67]):axes[0].text(x,y+4,str(y),ha='center')
        axes[0].set_ylabel('症例数');axes[0].set_ylim(0,335);axes[0].tick_params(axis='x',labelrotation=12)
        axes[1].hist(units.groupby('case_id').size(),bins=np.arange(.5,14.5),color='#397ea9',edgecolor='white')
        axes[1].set_xlabel('1症例内の高回転主体ブロック数');axes[1].set_ylabel('症例数')
        if len(support):
            ss=support[support.get('unit_level',pd.Series('block',index=support.index))=='block']
            count=[]
            for signal in ['torque','pedal2','bulk']:
                s=ss[ss.signal==signal]
                count.append([int((s.n_bins>=n).sum()) for n in [1,3,5,10]])
            im=axes[2].imshow(count,cmap='Blues',vmin=0,vmax=203)
            axes[2].set_xticks(range(4),['1以上','3以上','5以上','10以上']);axes[2].set_yticks(range(3),['トルク','ペダル2','バルク開量'])
            for i in range(3):
                for j in range(4):axes[2].text(j,i,str(count[i][j]),ha='center',va='center',color='white' if count[i][j]>120 else '#24313c')
            axes[2].set_xlabel('同run・同秒binの対応観測数');fig.colorbar(im,ax=axes[2],label='支持ブロック数')
        fig.suptitle('比較対象と観測支持：203ブロック／67症例を固定',fontsize=16)
        fig.tight_layout(rect=[0,0,1,.92])
        save(fig,'01_scope_and_support.png','対象範囲と数値支持',
             '未記録をOFFや0に置き換えない。数値の支持が足りないブロックは数値距離の比較対象から区別する。')
        # Raw structural distributions use one point per block, nested in cases.
        struct=[f for f in ['active_s','active_duty','n_runs','run_q50_s','run_q90_s','off_gap_q50_s'] if f in features]
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        cat=catalog.set_index('feature')
        for ax,f in zip(axes.ravel(),struct):
            vals=features[f].dropna()
            ax.hist(vals,bins=25,color='#568faf',edgecolor='white');ax.set_xlabel(str(cat.loc[f,'label']) if f in cat.index else f)
            ax.set_ylabel('ブロック数');ax.grid(axis='y',alpha=.15)
            ax.text(.97,.95,f'n={len(vals)}\n中央値={vals.median():.3g}',transform=ax.transAxes,ha='right',va='top',fontsize=9)
        for ax in axes.ravel()[len(struct):]:ax.axis('off')
        fig.suptitle('作動構造の原単位分布：回転数が同じでも、長さ・反復・停止が異なる',fontsize=15)
        fig.tight_layout(rect=[0,0,1,.94])
        save(fig,'02_structure_raw_distributions.png','作動構造の分布',
             '各点・各件はブロック。同一症例内の複数ブロックを独立症例とは扱わず、安定性は症例単位で検証する。',view='structure')
        numeric=[f for f in catalog.loc[catalog.view=='numeric','feature'] if f in features]
        # Prefer central/upper/range features, maximum nine clear panels.
        numeric=numeric[:9]
        fig,axes=plt.subplots(3,3,figsize=(15,10))
        for ax,f in zip(axes.ravel(),numeric):
            vals=features[f].dropna();ax.hist(vals,bins=23,color='#be8662',edgecolor='white')
            ax.set_xlabel(str(cat.loc[f,'label']) if f in cat.index else f);ax.set_ylabel('記録ありブロック数')
            ax.text(.97,.95,f'n={len(vals)}/203\n中央値={vals.median():.3g}',transform=ax.transAxes,ha='right',va='top',fontsize=9)
        for ax in axes.ravel()[len(numeric):]:ax.axis('off')
        fig.suptitle('トルク・ペダル2・バルク開量：原単位の観測分布',fontsize=16);fig.tight_layout(rect=[0,0,1,.95])
        save(fig,'03_numeric_raw_distributions.png','数値水準の分布',
             '観測がある場合の分布。支持条件の判定前の原表を示すため、モデル採用件数とは異なる。バルク開量は補助比較。小さいRPM揺らぎを主距離へ過大に反映しない。',view='numeric')
        # Intercase nearest-neighbor agreement is a different question from partition agreement.
        if len(nagree):
            fig,axes=plt.subplots(1,2,figsize=(14,5))
            jcol=next((x for x in ['mean_top3_jaccard','mean_neighbor_jaccard','top3_jaccard'] if x in nagree),None)
            labels=[f'{r.level}: {r.view_a}↔{r.view_b}' for r in nagree.itertuples()]
            if jcol:axes[0].barh(range(len(nagree)),nagree[jcol],color='#668ca7')
            axes[0].set_yticks(range(len(labels)),labels);axes[0].set_xlim(0,1);axes[0].set_xlabel('共通候補集合での近隣top3の一致度')
            if 'distance_spearman' in nagree:axes[1].barh(range(len(nagree)),nagree.distance_spearman,color='#b68b65')
            axes[1].set_yticks(range(len(labels)),labels);axes[1].set_xlim(-1,1);axes[1].set_xlabel('距離順位のSpearman相関')
            fig.suptitle('作動が似ている相手と、数値が似ている相手は同じか',fontsize=15);fig.tight_layout(rect=[0,0,1,.92])
            save(fig,'04_neighbor_axis_agreement.png','類似性の軸による近隣の違い',
                 '異なる観測範囲を直接比較せず、両方の指標を計算できる共通集合で一致を調べる。近隣は必ず別症例。')
        # Pairwise distances are shown only on the shared eligible cohort.
        ds=read(out,'distances/block_structure.csv');dn=read(out,'distances/block_numeric.csv')
        if len(ds) and len(dn):
            ds=ds.set_index(ds.columns[0]);dn=dn.set_index(dn.columns[0]);shared=ds.index.intersection(dn.index)
            shared=shared.intersection(ds.columns).intersection(dn.columns)
            aa=ds.loc[shared,shared].to_numpy(float);bb=dn.loc[shared,shared].to_numpy(float)
            casesmap=units.set_index('unit_id').case_id
            tri=np.triu_indices(len(shared),1);mask=np.array([casesmap.loc[shared[i]]!=casesmap.loc[shared[j]] for i,j in zip(*tri)])
            fig,ax=plt.subplots(figsize=(7,6));ax.hexbin(aa[tri][mask],bb[tri][mask],gridsize=33,mincnt=1,cmap='Blues')
            ax.set_xlabel('作動構造の距離');ax.set_ylabel('数値水準の距離');ax.set_title(f'共通{len(shared)}ブロックの異症例ペア：距離が異なる軸')
            ax.grid(alpha=.15);fig.tight_layout()
            save(fig,'05_structure_numeric_distance.png','作動構造距離と数値距離',
                 '左下ほど両方が近い。左上は作動が近いが数値が異なる。右下は数値が近いが作動が異なる。ペアは独立標本ではない。')
        if len(nstable):
            fig,axes=plt.subplots(1,2,figsize=(13,5))
            levels=nstable.level.unique() if 'level' in nstable else ['block']
            for ax,level in zip(axes,levels):
                g=nstable[nstable.level==level] if 'level' in nstable else nstable
                for view,t in g.groupby('view'):
                    col=next((c for c in ['conditional_neighbor_retention','neighbor_retention'] if c in t),None)
                    if col:ax.hist(t[col].dropna(),bins=np.linspace(0,1,16),histtype='step',lw=2,label=VIEW_NAMES.get(view,view))
                ax.set_title('区間単位' if level=='block' else '症例単位');ax.set_xlim(0,1)
                ax.set_xlabel('候補症例が残る反復に条件付けた近隣保持率');ax.set_ylabel('比較対象数');ax.legend(fontsize=9)
            fig.suptitle('症例を入れ替えても、近い別症例が近くに残るか',fontsize=15);fig.tight_layout(rect=[0,0,1,.92])
            save(fig,'06_neighbor_stability.png','症例再抽出での近隣安定性',
                 '欠落した近隣症例の不在を失敗と数えず、存在する反復に条件付けて評価する。固定特徴・距離設計内の内部検証。')
        if len(models):
            fig,axes=plt.subplots(1,3,figsize=(19,max(6,len(models)*.38)))
            positions=np.arange(len(models));labels=models.model_id.tolist()
            for i,row in models.iterrows():
                color=VIEW_COLORS.get(row.get('view'),'#777')
                med=float(row.get('stability_median',np.nan));q1=float(row.get('stability_q25',med));q3=float(row.get('stability_q75',med))
                if np.isfinite(med):axes[0].errorbar(med,i,xerr=[[max(0,med-q1)],[max(0,q3-med)]],fmt='o',color=color,capsize=3)
                axes[1].barh(i,float(row.get('silhouette',np.nan)),color=color)
                if 'quality_ari' in row:axes[2].barh(i,float(row.quality_ari),color=color)
            for ax in axes:ax.set_yticks(positions,labels);ax.grid(axis='x',alpha=.15)
            axes[0].set_xlabel('症例再抽出ARI：中央値と四分位範囲');axes[0].set_xlim(-.1,1.05)
            axes[1].set_xlabel('Silhouette（同じ表現内の分離）');axes[2].set_xlabel('観測状況だけの分割とのARI')
            fig.suptitle('潜在的な群の候補：内部安定性・分離・観測条件依存',fontsize=16);fig.tight_layout(rect=[0,0,1,.95])
            save(fig,'07_candidate_model_metrics.png','分割候補の比較',
                 '異なる特徴空間のsilhouetteを直接ランキングしない。高ARIだけで自然な離散群や臨床的意味を証明しない。')
        if len(coords):
            for (level,view),g in coords.groupby(['level','view'],sort=False):
                display=g.copy();xcol='x' if 'x' in display else 'mds1';ycol='y' if 'y' in display else 'mds2'
                if len(assign) and len(models):
                    candidates=models[(models.level==level)&(models.view==view)]
                    if len(candidates):
                        preferred=candidates[candidates.method=='kmedoids']
                        mid=(preferred if len(preferred) else candidates).iloc[0].model_id
                        labs=assign[assign.model_id==mid][['unit_id','cluster']]
                        display=display.merge(labs,on='unit_id',how='left',validate='one_to_one')
                    else:mid='';display['cluster']=np.nan
                else:mid='';display['cluster']=np.nan
                fig,ax=plt.subplots(figsize=(8,6))
                for cl,part in display.groupby('cluster',dropna=False):
                    ax.scatter(part[xcol],part[ycol],s=40,alpha=.75,label='未分類' if pd.isna(cl) else f'C{int(cl)}')
                ax.set_xlabel('MDS 1');ax.set_ylabel('MDS 2');ax.legend(fontsize=9)
                ax.set_title(f'{"区間" if level=="block" else "症例"}：{VIEW_NAMES.get(view,view)} / {len(display)}件\n{mid}')
                fig.tight_layout();save(fig,f'map_{safe(level)}_{safe(view)}.png',f'{level}：{VIEW_NAMES.get(view,view)}の距離地図',
                    '距離を2次元へ近似した図。見た目の島だけを群の正しさの根拠にしない。',view=view,level=level,kind='model')
        if len(profiles):draw_profiles(plt,out,catalog,models,profiles,save)
        # One table may contain several grain checks; expose them without inventing a common denominator.
        stagecomparison=read(out,'stage_neighbor_comparison.csv')
        if len(stagecomparison):
            numericcols=[c for c in stagecomparison if c not in ['view','level'] and pd.api.types.is_numeric_dtype(stagecomparison[c])]
            if numericcols:
                fig,ax=plt.subplots(figsize=(12,max(4,len(stagecomparison)*.4)))
                ax.axis('off');cell=stagecomparison.head(20).round(3).fillna('—')
                table=ax.table(cellText=cell.values,colLabels=cell.columns,loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.8)
                ax.set_title('作動ブロックと大きな同HP段階の比較');fig.tight_layout()
                save(fig,'08_stage_granularity.png','区切り方による類似相手の変化',
                     '大きな段階は元の高回転ブロックを含むものを選ぶ。段階自身が高回転主体かどうかで対象症例を再選択しない。')
        sensitivity=read(out,'support_and_weight_sensitivity.csv')
        if len(sensitivity):
            table=sensitivity.pivot(index='model_id',columns='scenario',values='ari')
            fig,ax=plt.subplots(figsize=(max(12,len(table.columns)*1.4),max(6,len(table)*.48)))
            image=ax.imshow(np.ma.masked_invalid(table.to_numpy(float)),aspect='auto',cmap='RdYlGn',vmin=0,vmax=1)
            ax.set_yticks(range(len(table)),table.index);ax.set_xticks(range(len(table.columns)),table.columns,rotation=45,ha='right')
            for i in range(len(table)):
                for j in range(len(table.columns)):
                    value=table.iloc[i,j]
                    ax.text(j,i,'—' if pd.isna(value) else f'{value:.2f}',ha='center',va='center',fontsize=8)
            fig.colorbar(image,ax=ax,label='元の割当とのARI（共通対象上）')
            ax.set_title('支持条件・特徴群・組合せ重みを変えると、群は維持されるか')
            fig.tight_layout();save(fig,'09_support_weight_sensitivity.png','支持条件と特徴設計への感度',
                'これは元の分割との一致で、外部性能ではない。条件によって対象数が変わるため、n_common_unitsとn_casesをCSVで併記している。')
        opportunity=read(out,'neighbor_opportunity_audit.csv')
        if len(opportunity):
            fig,ax=plt.subplots(figsize=(9,5));names=[VIEW_NAMES.get(v,v) for v in opportunity.view]
            values=opportunity.spearman_case_blockcount_vs_neighbor_selections
            ax.barh(names,values,color=[VIEW_COLORS.get(v,'#888') for v in opportunity.view]);ax.set_xlim(-1,1)
            for i,v in enumerate(values):ax.text(v+.025,i,f'{v:.3f}',va='center')
            ax.set_xlabel('症例内ブロック数と近隣選出回数のSpearman相関');ax.grid(axis='x',alpha=.15)
            ax.set_title('近隣選出の機会数：複数区間を持つ症例への偏りを監査')
            fig.tight_layout();save(fig,'10_neighbor_opportunity.png','近隣選出の機会数による偏り',
                '多くの候補区間を持つ症例は最も近い区間を提供する機会が多い。区間ペア図だけでケース間再現性とせず、各症例1単位の集約解析も使う。')
        confound=read(out,'confound_diagnostics.csv')
        if len(confound):
            cols=[x for x in ['year_nmi','original_case_duration_eta2','source_case_highblock_count_eta2'] if x in confound]
            fig,ax=plt.subplots(figsize=(10,max(6,len(confound)*.4)))
            table=confound.set_index('model_id')[cols];im=ax.imshow(table,aspect='auto',cmap='Oranges',vmin=0,vmax=1)
            ax.set_yticks(range(len(table)),table.index);ax.set_xticks(range(len(cols)),['記録年NMI','症例長eta²','高回転block数eta²'][:len(cols)])
            for i in range(len(table)):
                for j in range(len(cols)):ax.text(j,i,f'{table.iloc[i,j]:.3f}',ha='center',va='center',fontsize=9)
            fig.colorbar(im,ax=ax,label='内部の関連指標');ax.set_title('記録年・症例長・観測単位数との関連')
            fig.tight_layout();save(fig,'11_recording_confound.png','記録条件・症例構成との関連',
                '単一機器内の関連監査。小さくても、未測定の交絡や外部妥当性を排除するものではない。')
        draw_final_supplements(plt,out,models,assign,nstable,save)
        # The nearest pair is an illustration; include source-selected controls when present.
        if len(examples):
            source=out.parent/'chair_hp_sequence_300_v3_20261104'
            points=pd.read_csv(source/'hp2_points.csv',low_memory=False)
            runs=pd.read_csv(source/'hp_runs.csv',low_memory=False)
            mapping=pd.read_csv(source/'run_to_block.csv',low_memory=False)
            runs=runs.merge(mapping[['case_id','run_id','block_id']],on=['case_id','run_id'],validate='one_to_one')
            unitmap=units.set_index('unit_id')
            highpoints=points.merge(units[['case_id','block_id']],on=['case_id','block_id'],validate='many_to_one')
            limits={'rpm':(0,45000),'pedal2':(0,1000),'bulk':(0,max(750,float(highpoints.bulk.max())*1.03)),
                    'torque':(0,max(5,float(highpoints.torque.max())*1.05))}
            example=examples.drop_duplicates(['view','query_unit_id','neighbor_unit_id','kind'])
            for ordinal,pair in enumerate(example.itertuples()):
                ids=[pair.query_unit_id,pair.neighbor_unit_id]
                if not all(i in unitmap.index for i in ids):continue
                a,b=[unitmap.loc[i] for i in ids]
                assert a.case_id!=b.case_id,'Cross-case examples must never select the same case'
                sources=[]
                for u in [a,b]:
                    data=points[(points.case_id==u.case_id)&(points.block_id==u.block_id)].sort_values('time_s')
                    rr=runs[(runs.case_id==u.case_id)&(runs.block_id==u.block_id)]
                    sources.append((u,data,rr))
                color=['#377da7','#d27b49'];fig,axes=plt.subplots(5,2,figsize=(15,11),sharex='col',gridspec_kw={'height_ratios':[2,1.5,1.5,1.5,.7]})
                for j,(u,data,rr) in enumerate(sources):
                    for i,signal in enumerate(['rpm','torque','pedal2','bulk']):
                        ok=data[signal].notna();ax=axes[i,j]
                        ax.scatter(data.loc[ok,'time_s']-u.start_s,data.loc[ok,signal],s=13,color=color[j],alpha=.75)
                        ax.set_ylabel(SIGNAL_NAMES[signal]);ax.set_ylim(*limits[signal]);ax.grid(alpha=.15)
                        ax.text(.98,.93,f'観測bin n={int(ok.sum())}',transform=ax.transAxes,ha='right',va='top',fontsize=8)
                        if not ok.any():ax.text(.5,.5,'未観測（0ではない）',transform=ax.transAxes,ha='center',va='center')
                        if signal=='rpm':ax.axhspan(36000,44000,color='#d86a50',alpha=.08)
                    for run in rr.itertuples():axes[4,j].broken_barh([(run.start_s-u.start_s,run.active_s)],(0,.8),facecolors=color[j])
                    axes[4,j].set_ylim(-.1,1);axes[4,j].set_yticks([.4],['HP2作動']);axes[4,j].set_xlabel('ブロック開始からの実時間（秒）')
                    axes[0,j].set_title(shortcase(u.case_id)+f' B{int(u.block_id):03d}\n作動{u.active_s:.1f}s / {int(u.n_runs)}run')
                    axes[4,j].set_xlim(0,max(a.elapsed_s,b.elapsed_s)*1.02)
                view=getattr(pair,'view','');kind=getattr(pair,'kind','nearest');distance=float(getattr(pair,'distance',np.nan))
                kindlabel={'nearest':'近隣','supportmatchedrandom':'観測支持を合わせた無作為対照',
                    'support_matched_random':'観測支持を合わせた無作為対照',
                    'structure_close_numeric_far':'作動は近いが数値は遠い',
                    'numeric_close_structure_far':'数値は近いが作動は遠い'}.get(kind,kind)
                distance_label=f'{distance:.3f}' if np.isfinite(distance) else '未評価'
                title=f'{VIEW_NAMES.get(view,view)}：別症例の{kindlabel} / 距離 {distance_label}'
                fig.suptitle(title,fontsize=16);fig.tight_layout(rect=[0,0,1,.94])
                stem=f'pair_{ordinal:02d}_{safe(view)}_{safe(kind)}'
                save(fig,stem+'_actual.png',title+'（実時間）',
                    '原単位の観測点のみ。白い間隔を補間しない。実作動バーと経過時間を分ける。距離は確率ではなく、小さいほど近い。',view=view,level='block',kind='pair')
                fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
                for j,(u,data,rr) in enumerate(sources):
                    for i,signal in enumerate(['rpm','torque','pedal2','bulk']):
                        ok=data[signal].notna();axes[i].scatter(data.loc[ok,'active_progress']*100,data.loc[ok,signal],s=15,color=color[j],alpha=.6,label=shortcase(u.case_id))
                        axes[i].set_ylabel(SIGNAL_NAMES[signal]);axes[i].set_ylim(*limits[signal]);axes[i].grid(alpha=.15)
                axes[0].axhspan(36000,44000,color='#d86a50',alpha=.08);axes[0].legend(fontsize=9)
                axes[-1].set_xlim(0,100);axes[-1].set_xlabel('HP2累積作動時間の進捗（%）— 停止を除く。DTW整列ではない')
                fig.suptitle(title,fontsize=15);fig.tight_layout(rect=[0,0,1,.95])
                save(fig,stem+'_active.png',title+'（累積作動進捗）',
                    '横軸だけを各ブロックの累積HP2作動時間で0〜100%へ換算。数値は正規化せず、欠測を補間しない。',view=view,level='block',kind='pair')
                log('pair_complete',ordinal=ordinal,view=view,query_unit=ids[0],neighbor_unit=ids[1],kind=kind)
        pd.DataFrame(figures).to_csv(out/'figure_index.csv',index=False)
        build_html(out,units,models,assign,neighbors,nstable,coords,figures,log)
        log('complete',n_figures=len(figures))


def build_html(out,units,models,assign,neighbors,nstable,coords,figures,log):
    h=html.escape
    note=(out/'report.txt').read_text() if (out/'report.txt').exists() else '数値・原単位図を作成中。科学的な最終評価文は後で更新します。'
    rows=[];unitmap=units.set_index('unit_id')
    case_features=read(out,'case_features.csv')
    if len(case_features):case_features=case_features.set_index(case_features.columns[0])
    for row in coords.itertuples():
        uid=str(row.unit_id);case=str(row.case_id);u=unitmap.loc[uid] if uid in unitmap.index else None
        raw={}
        if u is not None:
            raw=dict(active_s=float(u.active_s),n_runs=int(u.n_runs),duty=float(u.active_s/u.elapsed_s))
        elif len(case_features) and uid in case_features.index:
            c=case_features.loc[uid]
            raw=dict(active_s=float(c.get('active_s__median',np.nan)),n_runs=float(c.get('n_runs__median',np.nan)),duty=float(c.get('active_duty__median',np.nan)))
            raw={key:None if not np.isfinite(value) else value for key,value in raw.items()}
        x=getattr(row,'x',getattr(row,'mds1',np.nan));y=getattr(row,'y',getattr(row,'mds2',np.nan))
        if np.isfinite(x) and np.isfinite(y):rows.append(dict(id=uid,case=case,level=str(row.level),view=str(row.view),x=float(x),y=float(y),raw=raw))
    neighborrows=[]
    for row in neighbors.to_dict('records'):
        neighborrows.append({key:(None if isinstance(v,float) and not np.isfinite(v) else v) for key,v in row.items()})
    payload=json.dumps(dict(points=rows,neighbors=neighborrows),ensure_ascii=False,allow_nan=False)
    columns=[c for c in ['model_id','level','view','method','k','n_units','n_cases','silhouette','stability_median','stability_q25','stability_q75','quality_ari','status'] if c in models]
    chunks=['<!doctype html><html lang="ja"><meta charset="utf-8"><title>高回転HP2の類似性</title>',
        '<style>body{font:16px/1.65 system-ui,sans-serif;color:#23313c;max-width:1320px;margin:36px auto;padding:0 24px}'
        'h1,h2,h3{line-height:1.3}img{max-width:100%;height:auto}figure{margin:26px 0}figcaption,.small{font-size:13px;color:#576775}'
        'table{border-collapse:collapse;font-size:12px;width:100%}td,th{border:1px solid #d9e1e8;padding:7px;text-align:left}'
        'th{background:#edf3f7}select,input{font:inherit;padding:6px;margin:4px}canvas{max-width:100%;height:auto;border:1px solid #dce4ea}'
        '.scroll{overflow:auto}.note{padding:16px;background:#fff4df;border-left:5px solid #c88c3d}.report{white-space:pre-wrap;font-family:inherit;font-size:15px}'
        '.card{background:#f5f8fa;padding:16px;border-radius:8px;margin:18px 0}a{color:#23658c}</style>',
        '<h1>高回転HP2の中で、何が似ているか</h1><p>固定203ブロック・67症例 ／ 作動構造・数値水準・文脈を分離した探索</p>',
        '<div class="note">近隣は必ず別症例から選びます。未観測は0ではありません。'
        '内部の近さ・安定性は、処置名や外部再現性の証明ではありません。異なる軸で「似た相手」が変わることを比較します。</div>',
        '<p><a href="#explore">対話地図・近隣検索</a> ／ <a href="#overview">概要図</a> ／ <a href="#pairs">実測ペア</a> ／ '
        '<a href="#models">候補群</a> ／ <a href="#report">評価全文</a> ／ <a href="#downloads">監査CSV</a></p>',
        '<h2 id="explore">指標を切り替えて、近い別症例を確認する</h2>',
        '<div class="card"><label>単位 <select id="level"><option value="case">症例集約</option><option value="block">ブロック</option></select></label>'
        '<label>軸 <select id="view"><option value="structure">作動の仕方</option><option value="numeric">数値水準</option><option value="combined">構造＋数値</option><option value="context">前後の文脈</option></select></label>'
        '<label>症例・区間検索 <input id="search" placeholder="例: 1801、日付、B002"></label><p id="count"></p>'
        '<canvas id="scatter" width="1240" height="620"></canvas><p id="hover" class="small">点へマウスを近づけると症例・区間情報を表示。クリックすると近隣一覧を表示します。</p>'
        '<p id="selected"></p><div class="scroll"><table id="nearest"></table></div></div>',
        '<p class="small">MDSは距離を2次元へ近似した参考図です。地図上の見た目だけで離散群があるとは判断しません。数値支持が足りない区間は数値地図に表示されません。</p>',
        '<h2 id="overview">概要・支持・類似軸の比較</h2>']
    def render(item):
        return f'<figure data-view="{h(str(item.get("view","")))}"><h3>{h(str(item["title"]))}</h3><a href="{h(item["path"])}"><img loading="lazy" src="{h(item["path"])}" alt="{h(str(item["title"]))}"></a><figcaption>{h(str(item.get("caption","")))}</figcaption></figure>'
    for f in figures:
        if f.get('kind')=='overview':chunks.append(render(f))
    chunks.extend(['<h2 id="pairs">異なる症例の実測ペア</h2>',
        '<p>近隣ペアの表示は、選んだ指標の解釈を確認するためのものです。近隣を選んだ後の図だけで、群の妥当性が独立に検証されたことにはなりません。</p>'])
    for f in figures:
        if f.get('kind')=='pair':chunks.append(render(f))
    chunks.extend(['<h2 id="models">分割候補と原単位プロファイル</h2>',
        '<p>群のIDは暫定です。サイズ・独立症例数・内部安定性・観測条件への依存を合わせて判断します。</p>',
        '<div class="scroll">'+models[columns].to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}',na_rep='—')+'</div>'])
    for f in figures:
        if f.get('kind')=='model':chunks.append(render(f))
    chunks.extend(['<h2 id="report">科学的評価と判断の過程</h2>',f'<div class="report">{h(note)}</div>',
        '<h2 id="downloads">比較表・監査・過程</h2><ul>'])
    for name in ['report.txt','evaluation_summary.json','units.csv','features.csv','feature_catalog.csv','signal_support.csv',
        'models.csv','assignments.csv','neighbors.csv','neighbor_stability.csv','neighbor_agreement.csv','stability.csv',
        'cluster_profiles_supported.csv','cluster_profiles.csv','pair_examples.csv','neighbor_feature_differences.csv','scope_audit.csv','stage_units.csv',
        'stage_neighbor_comparison.csv','support_and_weight_sensitivity.csv','confound_diagnostics.csv','neighbor_opportunity_audit.csv',
        'numeric_signal_dependence.csv','equal_opportunity_neighbor_sensitivity.csv','sensitivity.csv','stage_sensitivity.csv','analysis_plan.json']:
        if (out/name).exists():chunks.append(f'<li><a href="{name}">{name}</a></li>')
    for p in sorted((out/'process').glob('*')):
        if p.is_file() and p.suffix in ['.txt','.json','.jsonl']:
            name=str(p.relative_to(out));chunks.append(f'<li><a href="{h(name)}">{h(name)}</a></li>')
    chunks.append('</ul><script>const DATA='+payload+';'+JS+'</script></html>')
    (out/'report.html').write_text('\n'.join(chunks),encoding='utf-8')
    log('html_written',figures=len(figures),interactive_points=len(rows),bytes=(out/'report.html').stat().st_size,scientific_report_embedded=(out/'report.txt').exists())


JS=r"""
const el=id=>document.getElementById(id);let shown=[],dots=[],selected=null;
function fmt(x){return x===null||x===undefined?'—':Number(x).toLocaleString(undefined,{maximumFractionDigits:3});}
function draw(){const level=el('level').value,view=el('view').value,q=el('search').value.toLowerCase();shown=DATA.points.filter(d=>d.level===level&&d.view===view&&(d.id.toLowerCase().includes(q)||d.case.toLowerCase().includes(q)));const c=el('scatter'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);dots=[];el('count').textContent=`表示 ${shown.length} 件（${level==='block'?'ブロック':'症例'} / ${view}）。比較可能な支持がある対象のみ。`;if(!shown.length){ctx.fillText('該当する比較対象がありません',35,45);return;}
const xs=shown.map(d=>d.x),ys=shown.map(d=>d.y),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),pad=45;ctx.strokeStyle='#d9e2e8';ctx.strokeRect(pad,pad,c.width-2*pad,c.height-2*pad);for(const d of shown){const x=pad+(d.x-xmin)/(xmax-xmin||1)*(c.width-2*pad),y=c.height-pad-(d.y-ymin)/(ymax-ymin||1)*(c.height-2*pad);ctx.beginPath();ctx.arc(x,y,d.id===selected?7:4.5,0,2*Math.PI);ctx.fillStyle=d.id===selected?'#c77046':view==='numeric'?'#aa7754':view==='combined'?'#708c54':'#4683a8';ctx.globalAlpha=.8;ctx.fill();dots.push({x,y,d});}ctx.globalAlpha=1;}
function nearest(d){selected=d.id;const rows=DATA.neighbors.filter(n=>n.level===d.level&&n.view===d.view&&n.unit_id===d.id).sort((a,b)=>a.rank-b.rank);el('selected').textContent=`選択: ${d.id} ／ 近隣は別症例のみ。距離は小さいほど近い（確率ではありません）。`;const table=el('nearest');table.replaceChildren();let tr=document.createElement('tr');for(const v of ['順位','別症例の区間/症例','距離']){const th=document.createElement('th');th.textContent=v;tr.appendChild(th);}table.appendChild(tr);for(const n of rows){tr=document.createElement('tr');for(const v of [n.rank,n.neighbor_unit_id,fmt(n.distance)]){const td=document.createElement('td');td.textContent=v;tr.appendChild(td);}table.appendChild(tr);}draw();}
function hit(e){const c=el('scatter'),r=c.getBoundingClientRect(),x=(e.clientX-r.left)*c.width/r.width,y=(e.clientY-r.top)*c.height/r.height;let closest=null,best=15*15;for(const a of dots){const delta=(a.x-x)**2+(a.y-y)**2;if(delta<best){best=delta;closest=a.d;}}return closest;}
el('scatter').onmousemove=e=>{const d=hit(e);if(d)el('hover').textContent=`${d.id} / 作動 ${fmt(d.raw.active_s)}秒 / run ${fmt(d.raw.n_runs)}回 / duty ${fmt(d.raw.duty)}`;};
el('scatter').onclick=e=>{const d=hit(e);if(d)nearest(d);};for(const id of ['level','view'])el(id).onchange=()=>{selected=null;draw();};el('search').oninput=draw;draw();
"""


if __name__=='__main__':main()
