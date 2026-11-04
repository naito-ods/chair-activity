"""Scientific figures for semantics-corrected observed-only non-DTW models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from visualize_results import NAMES, NUMERIC, STATES, safe


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dpi',type=int,default=150)
    parser.add_argument('--html-only',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve()
    (out/'figures').mkdir(exist_ok=True)
    (out/'process').mkdir(exist_ok=True)
    def log(event,**details):
        row=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),event=event,**details)
        with (out/'process/visualization_log.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    def read(name):
        return pd.read_csv(out/name)
    manifest=read('case_manifest.csv')
    scope=read('cohort_scope.csv')
    selected=read('selected_models.csv')
    selected=selected.sort_values('valid',ascending=False,kind='stable').reset_index(drop=True)
    candidates=read('model_candidates.csv')
    assignment=read('assignments_long.csv')
    reps=read('representatives.csv')
    features=read('features.csv').set_index('case_id')
    profiles=read('coarse_profiles.csv')
    profiles.code=profiles.code.astype(str).str.replace(r'\.0$','',regex=True)
    embedding=read('embedding_mds.csv')
    stability=read('stability_repeats.csv')
    sensitivity=read('support_sensitivity.csv')
    comparison=read('comparison_with_v1.csv')
    setting=read('setting_time_audit.csv')
    audit=pd.read_csv(out.parent/'signal_audit.csv',dtype={'code':str})
    plan=json.loads((out/'model_plan.json').read_text())
    assert len(manifest)==manifest.case_id.nunique()==300
    assert not assignment.duplicated(['case_id','model_id']).any()
    log('start',n_models=len(selected),n_classified=assignment[assignment.cluster.notna()].case_id.nunique(),
        input_policy='compact feature, profile and NxN distance tables only; raw sources never loaded')
    if args.html_only:
        figure_index=read('figure_index.csv').fillna('').to_dict('records')
        report(out,manifest,scope,selected,assignment,embedding,figure_index,log)
        return
    figure_index=[]
    with tempfile.TemporaryDirectory(prefix='chair-v2-figures-') as tmp:
        os.environ['MPLCONFIGDIR']=tmp
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False,
            'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
        palette=plt.get_cmap('tab10')
        def save(fig,name,title,caption,model_id='',category='overview'):
            path=out/'figures'/name
            fig.savefig(path,dpi=args.dpi,bbox_inches='tight')
            plt.close(fig)
            figure_index.append(dict(path=str(path.relative_to(out)),title=title,caption=caption,
                model_id=model_id,category=category))

        counts=scope.v2_status.value_counts()
        translations={'CLASSIFIED':'共通支持条件を満たす','NO_TARGET_SIGNAL':'対象10コード未記録',
            'INSUFFICIENT_COMPARABLE_SUPPORT':'比較可能な観測支持が不足'}
        shared=pd.read_csv(out/'shared_code_counts.csv',index_col=0).to_numpy(float)
        numshared=pd.read_csv(out/'shared_numeric_code_counts.csv',index_col=0).to_numpy(float)
        off=np.triu_indices(len(shared),1)
        fig,axes=plt.subplots(1,3,figsize=(16,5),gridspec_kw={'width_ratios':[1.3,1,1]})
        for i,(status,n) in enumerate(counts.items()):
            axes[0].barh(i,n,color={'CLASSIFIED':'#437fa8','NO_TARGET_SIGNAL':'#aab3bb'}.get(status,'#d2a35b'))
            axes[0].text(n+2,i,str(n),va='center')
        axes[0].set_yticks(range(len(counts)),[translations.get(s,s) for s in counts.index])
        axes[0].set_xlim(0,330)
        axes[0].set_xlabel('症例数（監査対象300例）')
        axes[0].set_title('分類対象と非割当の内訳')
        axes[1].hist(shared[off],bins=np.arange(-.5,11.5,1),color='#437fa8',edgecolor='white')
        axes[1].set_xlabel('両症例で共通に観測されたコード数')
        axes[1].set_ylabel('症例ペア数')
        axes[1].set_xticks(range(11))
        axes[2].hist(numshared[off],bins=np.arange(-.5,7.5,1),color='#58a184',edgecolor='white')
        axes[2].set_xlabel('共通に観測された数値コード数')
        axes[2].set_ylabel('症例ペア数')
        axes[2].set_xticks(range(7))
        fig.suptitle('v2：300症例の監査と、比較に使える共通情報',fontsize=16)
        fig.tight_layout()
        zero=int(np.sum(numshared[off]==0))
        save(fig,'01_cohort_and_shared_support.png','対象症例と共通の観測支持',
            f'HP選択・作動の状態既知率20%以上と、支持のある数値1コード以上を満たす症例で比較しています。数値コードが共通しないペアは{zero}/{len(off[0])}組あり、そこでは共通の状態情報に基づきます。未記録や支持不足の症例を0埋めして割り当てていません。')

        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        for method,g in candidates.groupby('method'):
            g=g.sort_values('k')
            axes[0].plot(g.k,g.silhouette,marker='o',label=method)
            axes[1].plot(g.k,g.stability_median,marker='o',label=method)
            bad=g[~g.valid.astype(bool)]
            axes[0].scatter(bad.k,bad.silhouette,s=80,marker='x',color='#333333')
        axes[0].set_ylabel('Silhouette（観測共通項目の距離）')
        axes[1].set_ylabel('探索時サブサンプリングARI中央値')
        axes[1].set_ylim(-.05,1.05)
        for ax in axes:
            ax.set_xlabel('クラス数 k')
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle('v2：手法とクラス数の比較（×はサイズ・バランス等の条件不合格）',fontsize=15)
        fig.tight_layout()
        save(fig,'02_candidate_methods.png','2手法・クラス数の比較',
            '共通観測コードに基づく距離はユークリッド距離を前提としないため、k-medoidsと平均連結を比較しています。探索時ARIと、選定後60反復のARIは別途記録しています。')

        fig,axes=plt.subplots(1,3,figsize=(15,4.5),gridspec_kw={'width_ratios':[1.4,1,1]})
        order=selected.model_id.tolist()
        for i,row in selected.reset_index(drop=True).iterrows():
            trials=stability[stability.model_id==row.model_id]
            axes[0].boxplot([trials.ari],positions=[i],vert=False,widths=.55,patch_artist=True,
                boxprops={'facecolor':'#83b8d3'})
            axes[1].barh(i,row.silhouette,color='#58a184')
            axes[2].barh(i,row.get('quality_ari',np.nan),color='#d2a35b')
        for ax in axes:
            ax.set_yticks(range(len(order)),order if ax is axes[0] else ['']*len(order),fontsize=9)
            ax.grid(axis='x',alpha=.2)
        axes[0].set_xlabel('60反復の内部ARI分布')
        axes[0].set_xlim(-.05,1.05)
        axes[1].set_xlabel('Silhouette')
        axes[2].set_xlabel('観測状況だけの分類とのARI')
        axes[2].set_xlim(-.05,1.05)
        fig.suptitle('v2：代表候補の内部安定性・分離度・観測状況との関連',fontsize=15)
        fig.tight_layout()
        save(fig,'03_selected_models.png','候補モデルの評価',
            'ARIは症例単位で80%を再抽出した内部安定性です。未知症例や別装置への外部検証ではありません。高ARIのみで意味のある処置クラスと判断しません。')

        sens=sensitivity.pivot(index='model_id',columns='scenario',values='ari').reindex(order)
        ncommon=sensitivity.pivot(index='model_id',columns='scenario',values='n_common_cases').reindex(order)
        fig,ax=plt.subplots(figsize=(11,max(3.2,len(sens)*1.2)))
        im=ax.imshow(sens.to_numpy(float),cmap='RdYlGn',vmin=-.05,vmax=1,aspect='auto')
        for i in range(len(sens)):
            for j in range(len(sens.columns)):
                v=sens.iloc[i,j]
                ax.text(j,i,f'ARI {v:.3f}\nn={int(ncommon.iloc[i,j])}',ha='center',va='center',fontsize=12,
                    color='white' if v<.1 or v>.85 else '#25394a')
        labels={'numeric5_known20':'数値≧5観測bin\n状態既知≧20%',
            'numeric20_known20':'数値≧20観測bin\n状態既知≧20%',
            'numeric10_known50':'数値≧10観測bin\n状態既知≧50%'}
        ax.set_xticks(range(len(sens.columns)),[labels.get(c,c) for c in sens.columns])
        ax.set_yticks(range(len(sens)),sens.index,fontsize=10)
        ax.set_title('v2：観測支持条件を変えた場合の分類一致',fontsize=15)
        fig.colorbar(im,ax=ax,label='基準とのARI')
        fig.tight_layout()
        save(fig,'04_support_sensitivity.png','観測支持条件への感度',
            '各条件で比較できる共通症例だけに揃えて、基準の分類と再分類を比較しています。nの減少も同時に確認してください。緑は基準との一致を示し、正解や外部再現性を意味しません。')

        comp=comparison.pivot(index='v2_model',columns='v1_model',values='ari').reindex(order)
        ns=comparison.pivot(index='v2_model',columns='v1_model',values='n_common_cases').reindex(order)
        fig,ax=plt.subplots(figsize=(max(12,len(comp.columns)*.8),4.5))
        im=ax.imshow(comp.to_numpy(float),cmap='Blues',vmin=0,vmax=1,aspect='auto')
        for i in range(len(comp)):
            for j in range(len(comp.columns)):
                v=comp.iloc[i,j]
                ax.text(j,i,f'{v:.2f}\nn={int(ns.iloc[i,j])}',ha='center',va='center',fontsize=8,
                    color='white' if v>.65 else '#243b4b')
        ax.set_xticks(range(len(comp.columns)),comp.columns,rotation=60,ha='right',fontsize=8)
        ax.set_yticks(range(len(comp)),comp.index,fontsize=9)
        ax.set_title('v2と旧v1の分類一致：意味・欠測の扱いを変える影響',fontsize=15)
        fig.colorbar(im,ax=ax,label='共通症例のARI')
        fig.tight_layout()
        save(fig,'05_comparison_with_v1.png','旧v1との比較',
            'v2は記録意味・距離・支持条件を変更した別バージョンです。ARIが低い場合、旧v1の分類をそのまま再現するものではありません。旧v1を上書きしていません。')

        old=features['E20776__q50']
        weighted=features['E20776__active_setting_timeweighted_q50']
        both=old.notna()&weighted.notna()
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        axes[0].scatter(old[both],weighted[both],s=18,alpha=.6,color='#437fa8')
        bound=max(float(old[both].max()),float(weighted[both].max()),1)
        axes[0].plot([0,bound],[0,bound],color='#8b9aa5',linestyle='--')
        axes[0].set_xlabel('旧v1：設定変更が記録された秒binの中央値')
        axes[0].set_ylabel('v2：作動中の時間加重設定中央値')
        axes[0].set_title(f'同一症例の設定値比較（n={int(both.sum())}）')
        axes[1].hist((weighted-old)[both],bins=20,color='#58a184',edgecolor='white')
        axes[1].set_xlabel('時間加重中央値 − 変更記録点中央値')
        axes[1].set_ylabel('症例数')
        coverage=setting.setting_known_active_fraction.dropna()
        axes[2].hist(coverage*100,bins=20,color='#d2a35b',edgecolor='white')
        axes[2].set_xlabel('スケーラー作動中の設定既知率（%）')
        axes[2].set_ylabel('症例数')
        fig.suptitle('20776：変更イベントの頻度と、設定が持続した時間は別',fontsize=16)
        fig.tight_layout()
        save(fig,'06_setting_record_vs_duration.png','設定値の持続時間を反映する効果',
            '20776は次の変更まで持続します。20774=1の作動区間と設定既知区間の正確な交差時間で重み付けしました。初回設定記録前は未知のままで、未記録の設定を推定していません。')

        summarypath=out/'interpretable_class_summary.csv'
        if summarypath.exists():
            summary=pd.read_csv(summarypath)
            specs=[('E20742__q50','回転数：症例中央値のクラス中央値',1),
                ('HP_nonzero_fraction','作動HPの非0状態割合（%）',100),
                ('E20039__state3_fraction','作動HPの記録状態3割合（%）',100),
                ('E20101__pickup_fraction','シリンジ取り上げ割合（%）',100),
                ('E20774__active_fraction','スケーラー作動状態1割合（%）',100),
                ('E20776__active_setting_timeweighted_q50','作動中の時間加重出力設定',1)]
            fig,axes=plt.subplots(2,3,figsize=(15,8))
            for ax,(field,label,multiplier) in zip(axes.ravel(),specs):
                values=summary[field].to_numpy(float)*multiplier
                ax.bar(range(len(summary)),values,color=[palette((int(c)-1)%10) for c in summary.cluster],alpha=.8)
                for i,r in summary.iterrows():
                    n=int(r.get(field+'__n',r.n_cases))
                    ax.text(i,values[i],f'{values[i]:,.1f}\nn={n}/{int(r.n_cases)}',ha='center',va='bottom',fontsize=9)
                ax.set_xticks(range(len(summary)),[f'C{int(r.cluster)}\n全{int(r.n_cases)}例' for r in summary.itertuples()])
                ax.set_ylabel(label);ax.grid(axis='y',alpha=.15)
                ax.set_ylim(0,max(float(np.nanmax(values))*1.3,1))
            fig.suptitle('k-medoidsの3群候補：何が違うかを原単位で見る',fontsize=17)
            fig.tight_layout()
            save(fig,'07_interpretable_class_summary.png','3群候補の違いと観測例数',
                '棒は支持条件を満たす症例のraw特徴のクラス中央値、nはその特徴を観測できた症例数です。全症例に値があるわけではありません。分布の重なりはモデル別の点・箱ひげ図で確認できます。')

        # Support-masked distributions match the feature support used for modeling.
        masked=features.copy()
        analysisfile=out/'analysis_features.csv'
        if analysisfile.exists():
            supported=pd.read_csv(analysisfile).set_index('case_id').reindex(masked.index)
            for col in supported:
                masked[col]=supported[col]
            log('guarded_analysis_features_used',n_features=len(supported.columns))
        bin_count=audit.pivot(index='case_id',columns='code',values='n_observed_bins')
        for code in NUMERIC:
            col=f'E{code}__q50'
            if code!='20776' and col in masked:
                low=bin_count[code].reindex(masked.index)<plan['numeric_min_bins']
                masked.loc[low,col]=np.nan
        for code in STATES:
            low=features[f'E{code}__known_fraction']<plan['state_min_known_fraction']
            statecols=[c for c in masked if c.startswith(f'E{code}__state') and c.endswith('_fraction')]
            if code=='20101':statecols+=['E20101__pickup_fraction']
            masked.loc[low,statecols]=np.nan
        limits={}
        for code in NUMERIC:
            vals=profiles.loc[profiles.code==code,'raw_median'].dropna()
            if len(vals):
                lo=min(0,float(vals.min()));hi=float(vals.max());span=max(hi-lo,1)
                limits[code]=(lo,hi+.08*span)
        mi=manifest.set_index('case_id')
        for row in selected.itertuples():
            mid=str(row.model_id);slug=safe(mid)
            ass=assignment[(assignment.model_id==mid)&assignment.cluster.notna()].copy()
            ass.cluster=ass.cluster.astype(int)
            clusters=sorted(ass.cluster.unique())
            counts=ass.groupby('cluster').size()
            emb=embedding.merge(ass[['case_id','cluster']],on='case_id',validate='one_to_one')
            fig,axes=plt.subplots(1,2,figsize=(13,5),gridspec_kw={'width_ratios':[1.7,1]})
            for cl in clusters:
                g=emb[emb.cluster==cl]
                axes[0].scatter(g.mds1,g.mds2,color=palette((cl-1)%10),s=24,alpha=.75,label=f'C{cl} (n={len(g)})',linewidths=0)
                med=reps[(reps.model_id==mid)&(reps.cluster==cl)&reps.is_medoid.astype(bool)]
                mm=emb[emb.case_id.isin(med.case_id)]
                axes[0].scatter(mm.mds1,mm.mds2,color=palette((cl-1)%10),s=190,marker='*',edgecolor='black',linewidth=.8)
            axes[0].legend(fontsize=9)
            axes[0].set_xlabel('MDS1');axes[0].set_ylabel('MDS2')
            axes[0].set_title('共通観測項目の距離を2次元へ投影')
            axes[0].grid(alpha=.15)
            axes[1].bar([f'C{c}' for c in clusters],counts.reindex(clusters),color=[palette((c-1)%10) for c in clusters])
            for i,c in enumerate(clusters):axes[1].text(i,int(counts[c])+1,str(int(counts[c])),ha='center')
            axes[1].set_ylabel('独立した症例ID数')
            axes[1].set_title('クラスサイズ')
            fig.suptitle(mid,fontsize=16)
            fig.tight_layout()
            save(fig,f'{slug}_embedding_and_sizes.png','症例の分布とクラスサイズ',
                '星印は実観測症例のmedoidです。MDSは距離を表示するための近似投影で、分離の見た目だけで妥当性を判断しません。',mid,'model')

            fig,axes=plt.subplots(2,3,figsize=(15,8))
            for ax,code in zip(axes.ravel(),NUMERIC):
                col=f'E{code}__q50' if code!='20776' else 'E20776__active_setting_timeweighted_q50'
                for pos,cl in enumerate(clusters):
                    vals=masked.reindex(ass.loc[ass.cluster==cl,'case_id'])[col].dropna().to_numpy()
                    if len(vals):
                        ax.boxplot([vals],positions=[pos],widths=.5,showfliers=False,patch_artist=True,
                            boxprops={'facecolor':palette((cl-1)%10),'alpha':.25},medianprops={'color':'black'})
                        ax.scatter(pos+np.linspace(-.15,.15,len(vals)),vals,color=palette((cl-1)%10),s=11,alpha=.55,linewidths=0)
                    ax.text(pos,.99,f'{len(vals)}/{int(counts[cl])}',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
                ax.set_xticks(range(len(clusters)),[f'C{c}' for c in clusters])
                ax.set_title(f'{code} {NAMES[code]}')
                ax.set_ylabel('作動中設定の時間加重中央値' if code=='20776' else '観測秒binの症例中央値')
                ax.grid(axis='y',alpha=.15)
            fig.suptitle(mid+'：数値水準（点＝1症例、数字＝支持のある観測例/全例）',fontsize=15)
            fig.tight_layout()
            save(fig,f'{slug}_raw_levels.png','原単位で比べる数値水準',
                '20776は作動中の持続時間で重み付けした設定値です。他の数値は10観測秒bin以上ある症例の中央値です。未観測・支持不足は点を描かず、0として含めません。',mid,'model')

            fig,axes=plt.subplots(2,2,figsize=(13,7))
            for ax,code in zip(axes.ravel(),STATES):
                if code=='20101':
                    cols=['E20101__pickup_fraction'];labels=['取り上げ(0)'];colors=['#4d8ab7']
                else:
                    states=[0,1,2,3] if code in ['20011','20039'] else [0,1]
                    cols=[f'E{code}__state{s}_fraction' for s in states]
                    labels=[f'記録状態{s}' for s in states]
                    colors=['#d4dce2','#4d8ab7','#e6a258','#69a98b']
                bottom=np.zeros(len(clusters))
                for col,label,color in zip(cols,labels,colors):
                    vals=[]
                    for cl in clusters:
                        v=masked.reindex(ass.loc[ass.cluster==cl,'case_id'])[col].mean()
                        vals.append(v)
                    vals=np.asarray(vals,float);good=np.isfinite(vals)
                    ax.bar(np.flatnonzero(good),vals[good]*100,bottom=bottom[good],label=label,color=color)
                    bottom[good]+=vals[good]*100
                if code=='20101':
                    good=np.isfinite(vals)
                    ax.bar(np.flatnonzero(good),(1-vals[good])*100,bottom=vals[good]*100,label='格納(1)',color='#d4dce2')
                ax.set_xticks(range(len(clusters)),[f'C{c}' for c in clusters])
                ax.set_ylim(0,105);ax.set_ylabel('既知時間内の比率：症例平均（%）')
                ax.set_title(f'{code} {NAMES[code]}');ax.legend(fontsize=8)
            fig.suptitle(mid+'：状態構成（シリンジは取り上げと格納を分離）',fontsize=15)
            fig.tight_layout()
            save(fig,f'{slug}_state_composition.png','操作状態の構成',
                '各症例の状態既知時間が20%以上ある場合に集計しています。取り上げ状態は噴射を意味しません。HPの状態番号に処置名や未確認の機器番号を割り当てていません。',mid,'model')

            fig,axes=plt.subplots(1,3,figsize=(15,4.5))
            joined=ass[['case_id','cluster']].merge(manifest,on='case_id')
            joined['year']=joined.date.astype(str).str[:4]
            for i,cl in enumerate(clusters):
                g=joined[joined.cluster==cl]
                axes[0].boxplot([g.duration_s/60],positions=[i],widths=.5)
                axes[0].scatter(i+np.linspace(-.14,.14,len(g)),g.duration_s/60,s=10,alpha=.5,color=palette((cl-1)%10))
                cov=setting.set_index('case_id').reindex(g.case_id).setting_known_active_fraction.dropna()*100
                if len(cov):axes[2].boxplot([cov],positions=[i],widths=.5)
            years=sorted(joined.year.unique());bottom=np.zeros(len(clusters))
            for j,year in enumerate(years):
                vals=joined.assign(in_year=joined.year.eq(year)).groupby('cluster').in_year.mean().reindex(clusters).to_numpy()*100
                axes[1].bar(range(len(clusters)),vals,bottom=bottom,label=year,color=plt.get_cmap('viridis')(.15+.7*j/max(1,len(years)-1)))
                bottom+=vals
            axes[1].legend(fontsize=8)
            for ax in axes:ax.set_xticks(range(len(clusters)),[f'C{c}' for c in clusters]);ax.grid(axis='y',alpha=.15)
            axes[0].set_ylabel('解析区間（分）');axes[1].set_ylabel('記録年の構成比（%）')
            axes[2].set_ylabel('作動中の設定既知率（%）');axes[2].set_ylim(-3,103)
            fig.suptitle(mid+'：時間・記録年・設定既知率との関連',fontsize=15)
            fig.tight_layout()
            save(fig,f'{slug}_confounds.png','時間・年・設定既知率の確認',
                '分類が症例時間や記録年・設定が分かる時間だけで説明されないかを確認する図です。関連がないことを保証するものではありません。',mid,'model')

            for cl in clusters:
                members=reps[(reps.model_id==mid)&(reps.cluster==cl)].sort_values('rank')
                handles=[]
                fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
                for i,rep in enumerate(members.itertuples()):
                    label=rep.case_id.split('__')[-1].replace('Chair_log_seq_','#')+(' [medoid]' if rep.is_medoid else '')
                    color=palette(i%10);handles.append(Line2D([],[],marker='D' if rep.is_medoid else 'o',linestyle='None',color=color,label=label))
                    for ax,code in zip(axes.ravel(),NUMERIC):
                        g=profiles[(profiles.case_id==rep.case_id)&(profiles.code==code)].sort_values('progress_bin')
                        good=g.raw_median.notna()
                        ax.scatter((g.loc[good,'progress_bin']+.5)*10,g.loc[good,'raw_median'],color=color,
                            s=38 if rep.is_medoid else 22,alpha=.95 if rep.is_medoid else .65,marker='D' if rep.is_medoid else 'o')
                for ax,code in zip(axes.ravel(),NUMERIC):
                    ax.set_title(f'{code} {NAMES[code]}');ax.set_xlabel('症例全体の進捗（%）')
                    ax.set_ylabel('作動中設定の時間加重中央値' if code=='20776' else '区間内の観測値中央値')
                    ax.set_xlim(0,100);ax.grid(alpha=.2)
                    if code in limits:ax.set_ylim(*limits[code])
                    if not any(len(c.get_offsets()) for c in ax.collections):
                        ax.text(.5,.5,'この代表群では観測なし',transform=ax.transAxes,ha='center',color='#788995')
                fig.suptitle(f'{mid} / C{cl}：代表症例の10段階プロフィール',fontsize=15,y=.99)
                fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.953),ncol=max(1,len(handles)),fontsize=9)
                fig.tight_layout(rect=[0,0,1,.91])
                save(fig,f'{slug}_C{cl:02d}_numeric_representatives.png',f'C{cl} 代表症例：数値水準',
                    'medoidと同じクラスの別症例を表示。20776のみ、作動中の設定持続時間を重み付けしています。縦軸はコードごとに全モデル・全クラスで共通。未観測は空白で、補間・DTW・平滑化は行っていません。',mid,'class')
                fig,axes=plt.subplots(2,2,figsize=(13,7),sharex=True,sharey=True)
                for i,rep in enumerate(members.itertuples()):
                    for ax,code in zip(axes.ravel(),STATES):
                        g=profiles[(profiles.case_id==rep.case_id)&(profiles.code==code)].sort_values('progress_bin')
                        col='pickup_fraction' if code=='20101' else 'active_fraction'
                        good=g[col].notna()
                        ax.scatter((g.loc[good,'progress_bin']+.5)*10,g.loc[good,col]*100,color=palette(i%10),
                            s=38 if rep.is_medoid else 22,alpha=.95 if rep.is_medoid else .65,marker='D' if rep.is_medoid else 'o')
                for ax,code in zip(axes.ravel(),STATES):
                    ax.set_title(f'{code} {NAMES[code]}');ax.set_xlabel('症例全体の進捗（%）')
                    ax.set_ylabel('既知時間内の取り上げ割合（%）' if code=='20101' else '既知時間内の非0割合（%）')
                    ax.set_xlim(0,100);ax.set_ylim(-3,103);ax.grid(alpha=.2)
                    if not any(len(c.get_offsets()) for c in ax.collections):
                        ax.text(.5,.5,'この代表群では既知状態なし',transform=ax.transAxes,ha='center',color='#788995')
                fig.suptitle(f'{mid} / C{cl}：代表症例の10段階・状態構成',fontsize=15,y=.99)
                fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.953),ncol=max(1,len(handles)),fontsize=9)
                fig.tight_layout(rect=[0,0,1,.91])
                save(fig,f'{slug}_C{cl:02d}_state_representatives.png',f'C{cl} 代表症例：状態の構成',
                    '20101は0＝取り上げであり、噴射時間ではありません。他は記録保持した非0状態の比率。初回記録前や未観測区間をOFFにしていません。区間内の既知時間を分母にしています。',mid,'class')
            log('model_figures_complete',model_id=mid,n_cases=len(ass),n_classes=len(clusters),n_figures=len(figure_index))
        pd.DataFrame(figure_index).to_csv(out/'figure_index.csv',index=False)
        report(out,manifest,scope,selected,assignment,embedding,figure_index,log)
        log('complete',n_figures=len(figure_index))


def report(out,manifest,scope,selected,assignment,embedding,figures,log):
    h=html.escape
    scope_counts=scope.v2_status.value_counts()
    nfit=int(scope_counts.get('CLASSIFIED',0))
    def figure(f):
        return (f'<figure><h3>{h(f["title"])}</h3><a href="{h(f["path"])}" target="_blank">'
            f'<img loading="lazy" src="{h(f["path"])}" alt="{h(f["title"])}"></a>'
            f'<figcaption>{h(f["caption"])}</figcaption></figure>')
    cols=[c for c in ['model_id','n_cases','k','silhouette','min_cluster_n','largest_fraction','stability_median',
        'stability_q25','stability_q75','stability_min','quality_ari','duration_eta2','date_nmi','valid'] if c in selected]
    content=['<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>v2：記録の意味を反映した300症例の操作パターン比較</title>',
        '''<style>*{box-sizing:border-box}body{margin:0;background:#f3f7f9;color:#213b4b;font:15px/1.8 system-ui,sans-serif}header{background:#173f57;color:white;padding:40px max(24px,calc((100vw - 1350px)/2));}h1{font-size:30px;line-height:1.5}header p{max-width:1100px;color:#dce9f1}nav{position:sticky;top:0;background:white;padding:12px 24px;border-bottom:1px solid #d7e2e9;z-index:10;display:flex;gap:20px;flex-wrap:wrap}a{color:#226c96;text-decoration:none}a:hover{text-decoration:underline}main{max-width:1400px;margin:auto;padding:28px 24px}section{background:white;border:1px solid #d7e2e9;border-radius:12px;padding:24px;margin-bottom:26px;scroll-margin-top:80px}h2{font-size:24px}h3{font-size:18px}figure{margin:28px 0 38px;padding-top:14px;border-top:1px solid #d7e2e9}img{max-width:100%;height:auto;display:block;margin:auto}figcaption,.muted{font-size:13px;color:#5d7383}table{border-collapse:collapse;font-size:12px;width:100%}td,th{border:1px solid #d7e2e9;padding:8px;text-align:left;white-space:nowrap}th{background:#eef5f9}.scroll{overflow:auto}.note{padding:16px;background:#fff5dd;border-left:4px solid #c49435}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.metric{background:#f7fbfd;border:1px solid #d7e2e9;padding:16px;border-radius:8px}.metric strong{display:block;font-size:30px;color:#2b708e}.control{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:20px 0}select,input{font:inherit;border:1px solid #a9becb;border-radius:6px;padding:8px}input{min-width:280px}.model-panel[hidden]{display:none}#mds{width:100%;height:auto;background:#fcfdfe;border:1px solid #d7e2e9;border-radius:8px}#tip{display:none;position:fixed;background:#173f57;color:white;padding:10px;border-radius:6px;pointer-events:none;z-index:30;font-size:12px;white-space:pre-line}pre{white-space:pre-wrap;font:14px/1.8 system-ui,sans-serif}button{font:inherit;background:#2b708e;color:white;border:0;border-radius:6px;padding:8px 14px;cursor:pointer}@media(max-width:700px){h1{font-size:23px}main{padding:15px}section{padding:15px}}</style>''',
        '<header><h1>v2：記録の意味を反映した操作パターン比較<br>設定値の持続・シリンジの状態・観測共通項目を分けて扱う</h1>',
        '<p>90012の最初のON→最後のOFFを継続。DTWや欠測の中央値補完を使わず、実際に共通して観測された特徴で症例を比較します。</p></header>',
        '<nav><a href="#summary">結果と範囲</a><a href="#methods">方法の比較</a><a href="#explore">症例を探す</a><a href="#models">モデル・クラス別の図</a><a href="#audit">監査</a><a href="../report.html">旧v1との比較記録</a></nav>',
        '<main><section id="summary"><h2>記録の意味を確認してから比較する</h2><div class="metrics">',
        f'<div class="metric"><strong>300</strong>監査した症例</div><div class="metric"><strong>{nfit}</strong>支持条件を満たす分類対象</div>',
        f'<div class="metric"><strong>{300-nfit}</strong>対象コード未記録または支持不足</div><div class="metric"><strong>{len(figures)}</strong>比較用の図</div></div>',
        '<ul><li><strong>20776：</strong>変更時だけ記録される出力設定。次の変更まで保持し、20774=1の作動時間との交差時間で重み付け。</li>',
        '<li><strong>20101：</strong>0＝取り上げ、1＝格納。取り上げ時間は実際の噴射時間ではありません。</li>',
        '<li><strong>未観測：</strong>初回記録前は未知。欠測を0・OFF・全体中央値に置き換えて距離を計算しません。</li>',
        '<li><strong>比較対象：</strong>HP選択と作動の状態既知時間20%以上、支持のある数値コード1種類以上を満たす症例に限定。全300例は監査表に保持。</li></ul>',
        '<p class="note">クラスはログ上の操作パターン候補です。処置名や診断ではありません。内部安定性・分離度・支持条件への感度を合わせて判断し、未知の症例・別施設・別装置での外部再現性は未検証です。</p>']
    summaryfile=out/'evaluation_summary.json'
    if summaryfile.exists():
        evaluation=json.loads(summaryfile.read_text())
        content.append('<h2>現時点の判断</h2><p><strong>'+h(str(evaluation.get('conclusion','')))+'</strong></p>')
        if 'v2_median_ari' in evaluation:
            content.append(f'<p>主候補：k-medoids 3群、内部ARI中央値 {evaluation["v2_median_ari"]:.3f} '
                f'（IQR {evaluation["v2_ari_q25"]:.3f}–{evaluation["v2_ari_q75"]:.3f}）、'
                f'Silhouette {evaluation["v2_silhouette"]:.3f}。支持条件への感度が残る探索結果です。</p>')
    invalid=selected[~selected.valid.astype(bool)]
    if len(invalid):
        content.append('<p class="note"><strong>不採用の比較：</strong>'+h('、'.join(invalid.model_id.astype(str)))+
            ' はクラスサイズ・偏り等の条件を満たしません。比較資料として表示し、推奨する分類とは区別します。</p>')
    txt=out/'report.txt'
    if txt.exists():content.append('<details><summary>解析担当の評価と判断を読む（全文）</summary><pre>'+h(txt.read_text())+'</pre></details>')
    else:content.append('<p class="muted">このページの図と数値はモデルの比較資料です。科学的な結論は安定性・感度・観測支持の確認と合わせて評価します。</p>')
    for name in ['07_interpretable_class_summary.png','01_cohort_and_shared_support.png','06_setting_record_vs_duration.png']:
        content.extend(figure(f) for f in figures if f['path'].endswith(name))
    content.extend(['</section><section id="methods"><h2>手法・安定性・感度の比較</h2>',
        '<p>共通観測項目に基づく距離を使い、k-medoidsと平均連結を比較します。症例ペアにより共通項目が異なり、通常のユークリッド距離とは性質が異なります。2次元投影の見た目だけでクラスの正しさを判断しません。</p>',
        '<div class="scroll">'+selected[cols].to_html(index=False,border=0,float_format=lambda v:f'{v:.3f}')+'</div>'])
    content.extend(figure(f) for f in figures if not f['model_id'] and not any(f['path'].endswith(name) for name in ['01_cohort_and_shared_support.png','06_setting_record_vs_duration.png','07_interpretable_class_summary.png']))
    content.extend(['</section><section id="explore"><h2>同じ距離空間でモデルを切り替える</h2>',
        '<p>点＝1症例。カーソルで症例ID・記録日・クラスを確認できます。支持不足で分類していない症例は、ここに点を作らず監査一覧に残しています。</p>',
        '<div class="control"><label>モデル <select id="exploreModel"></select></label><label>症例検索 <input id="query" placeholder="例：2024-06、seq番号、症例ID"></label><button id="reset">検索を解除</button></div>',
        '<svg id="mds" viewBox="0 0 1100 600" aria-label="症例のMDS投影"></svg><p id="status" class="muted"></p>',
        '</section><section id="models"><h2>モデル・クラス別の原データ比較</h2><div class="control"><label>表示モデル <select id="modelSelect"></select></label></div>'])
    for model in selected.model_id:
        content.append(f'<div class="model-panel" data-model="{h(str(model))}"><h2>{h(str(model))}</h2>')
        row=selected[selected.model_id==model].iloc[0]
        if not row.valid:
            content.append('<p class="note">このモデルはサイズ・偏り等の条件不合格で、不採用の比較として保存しています。</p>')
        content.extend(figure(f) for f in figures if f['model_id']==model)
        content.append('</div>')
    content.extend(['</section><section id="audit"><h2>監査と再現用ファイル</h2><ul>'])
    for name in ['report.txt','cohort_scope.csv','model_plan.json','selected_models.csv','model_candidates.csv',
        'assignments_long.csv','cluster_sizes.csv','representatives.csv','stability_repeats.csv','support_sensitivity.csv',
        'comparison_with_v1.csv','setting_time_audit.csv','semantic_metadata.json','feature_catalog.csv','features.csv',
        'analysis_features.csv','analysis_feature_catalog.csv','class_profiles.csv','interpretable_class_summary.csv','evaluation_summary.json',
        'used_features.csv','observed_distances.csv','shared_code_counts.csv','shared_numeric_code_counts.csv','figure_index.csv']:
        if (out/name).exists():content.append(f'<li><a href="{name}">{name}</a></li>')
    for file in sorted((out/'process').glob('*')):
        if file.is_file():content.append(f'<li><a href="process/{h(file.name)}">process/{h(file.name)}</a></li>')
    content.append('</ul></section></main><div id="tip"></div>')
    data={}
    mi=manifest.set_index('case_id')
    for model in selected.model_id:
        a=assignment[(assignment.model_id==model)&assignment.cluster.notna()]
        df=embedding.merge(a[['case_id','cluster']],on='case_id',validate='one_to_one')
        data[str(model)]=[dict(id=r.case_id,x=float(r.mds1),y=float(r.mds2),c=int(r.cluster),
            date=str(mi.loc[r.case_id,'date']),minutes=round(float(mi.loc[r.case_id,'duration_s'])/60,2)) for r in df.itertuples()]
    content.append('<script>const points='+json.dumps(data,ensure_ascii=False).replace('</','<\/')+';')
    content.append('''
    const models=Object.keys(points),palette=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
    const svg=document.getElementById('mds'),tip=document.getElementById('tip'),query=document.getElementById('query');
    for(const id of ['exploreModel','modelSelect']){const el=document.getElementById(id);models.forEach(m=>{const o=document.createElement('option');o.value=m;o.textContent=m;el.appendChild(o);});}
    function node(tag,attrs,text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;svg.appendChild(e);return e;}
    function draw(){svg.replaceChildren();const model=document.getElementById('exploreModel').value,data=points[model],q=query.value.toLowerCase();if(!data||!data.length)return;const xs=data.map(p=>p.x),ys=data.map(p=>p.y),loX=Math.min(...xs),hiX=Math.max(...xs),loY=Math.min(...ys),hiY=Math.max(...ys),x=v=>70+(v-loX)/(hiX-loX||1)*850,y=v=>525-(v-loY)/(hiY-loY||1)*440;
      node('line',{x1:60,y1:540,x2:950,y2:540,stroke:'#8fa8b5'});node('line',{x1:60,y1:540,x2:60,y2:65,stroke:'#8fa8b5'});node('text',{x:470,y:580,fill:'#536e7e','font-size':15},'MDS1');node('text',{x:12,y:270,fill:'#536e7e','font-size':15},'MDS2');let found=0;
      data.forEach(p=>{const match=!q||p.id.toLowerCase().includes(q);if(match)found++;const e=node('circle',{cx:x(p.x),cy:y(p.y),r:q&&match?7:4.5,fill:palette[(p.c-1)%palette.length],opacity:match?0.85:0.07,stroke:q&&match?'#193e52':'none'});e.addEventListener('mouseenter',()=>{tip.textContent=p.id+'\\nC'+p.c+' / '+p.date+' / '+p.minutes+'分';tip.style.display='block';});e.addEventListener('mousemove',ev=>{tip.style.left=Math.max(8,Math.min(ev.clientX+12,window.innerWidth-440))+'px';tip.style.top=ev.clientY+12+'px';});e.addEventListener('mouseleave',()=>tip.style.display='none');});
      [...new Set(data.map(p=>p.c))].sort((a,b)=>a-b).forEach((c,i)=>{node('circle',{cx:980,cy:90+i*28,r:5,fill:palette[(c-1)%palette.length]});node('text',{x:994,y:95+i*28,fill:'#365063','font-size':13},'C'+c+' ('+data.filter(p=>p.c===c).length+')');});document.getElementById('status').textContent=q?'検索一致 '+found+' / '+data.length+' 症例':'表示 '+data.length+' 症例。分類対象外は監査一覧で確認できます。';}
    function panels(){const model=document.getElementById('modelSelect').value;document.querySelectorAll('.model-panel').forEach(e=>e.hidden=e.dataset.model!==model);}
    document.getElementById('exploreModel').addEventListener('change',draw);document.getElementById('modelSelect').addEventListener('change',panels);query.addEventListener('input',draw);document.getElementById('reset').addEventListener('click',()=>{query.value='';draw();});draw();panels();
    </script></html>''')
    (out/'report.html').write_text('\n'.join(content),encoding='utf-8')
    log('html_written',n_figures=len(figures),interactive_points=sum(map(len,data.values())),bytes=(out/'report.html').stat().st_size)


if __name__=='__main__':
    main()
