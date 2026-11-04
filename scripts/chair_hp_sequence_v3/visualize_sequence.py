"""Read-only scientific visualizations of HP order and observed HP2 regimes."""
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

PATTERNS = {
    'post_hp3_low': ('HP3後の低回転HP2', 'has_post_hp3_low'),
    'high_dominant': ('観測範囲で高回転主体', 'has_high_dominant'),
    'high_then_short_mid': ('高回転後、末尾に短い中回転', 'has_high_then_short_mid'),
}
HP_COLORS = {0: '#dce1e5', 1: '#a778b4', 2: '#3e85b8', 3: '#50a078'}
ZONE_COLORS = {'Z': '#8b959e', 'L': '#347fb5', 'M': '#e5b34c', 'H': '#d86a50', 'X': '#9b4d69'}
NAMES = {'20742': '回転数', '20743': 'トルク', '20108': 'バルク開量',
         '10014': 'ペダル1', '10015': 'ペダル2'}


def safe(text):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(text))


def boolcol(series):
    return series.astype(str).str.lower().isin(['true', '1', '1.0', 'yes'])


def firstcol(frame, *choices, default=None):
    return next((c for c in choices if c in frame.columns), default)


def cap(frame, n=3500):
    return frame.iloc[np.unique(np.linspace(0, len(frame)-1, n, dtype=int))] if len(frame) > n else frame


def shortcase(case):
    parts = str(case).split('__')
    return (parts[0] + ' #' + parts[-1].replace('Chair_log_seq_', '')) if len(parts) == 3 else str(case)


def pattern_key(value):
    value = str(value).lower()
    if 'post' in value and ('hp3' in value or 'scaler' in value):
        return 'post_hp3_low'
    if 'mid' in value and ('then' in value or 'tail' in value or 'short' in value):
        return 'high_then_short_mid'
    if 'then' in value or 'tail' in value or 'short_low' in value:
        return 'high_then_short_low'
    if 'high' in value:
        return 'high_dominant'
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dpi', type=int, default=145)
    parser.add_argument('--html-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    (out/'figures').mkdir(exist_ok=True)
    (out/'process').mkdir(exist_ok=True)

    def log(event, **details):
        rec = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), event=event, **details)
        with (out/'process/visualization_log.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False)+'\n')
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    def read(name, required=True):
        p = out/name
        if not p.exists():
            if required:
                raise FileNotFoundError(p)
            return pd.DataFrame()
        try:
            return pd.read_csv(p,low_memory=False)
        except pd.errors.EmptyDataError:
            if required:
                raise
            return pd.DataFrame()

    manifest = read('case_manifest.csv')
    intervals = read('hp_intervals.csv')
    runs = read('hp_runs.csv')
    audit = read('hp_status_audit.csv')
    transitions = read('transition_audit.csv')
    blocks = read('blocks.csv')
    hp2 = read('hp2_blocks.csv')
    points = read('hp2_points.csv')
    cases = read('case_patterns.csv')
    observations = read('observations.csv')
    observations['code'] = observations.code.astype(str).str.replace(r'\.0$', '', regex=True)
    reps = read('representatives.csv', required=False)
    reps_mid = read('representatives_mid.csv', required=False)
    if len(reps_mid):reps=pd.concat([reps,reps_mid],ignore_index=True)
    if len(reps):reps=reps.drop_duplicates(['pattern','case_id','block_id'],keep='first')
    conditional = read('conditional_numeric.csv', required=False)
    context_cases = read('context_case_comparisons.csv', required=False)
    prevalence = read('pattern_prevalence.csv', required=False)
    sensitivity = read('sensitivity.csv', required=False)
    for extra in ['sensitivity_mid.csv','mid_tail_sensitivity.csv','sensitivity_extended.csv','sensitivity_long_gaps.csv','scale_sensitivity.csv']:
        additional=read(extra,required=False)
        if len(additional):sensitivity=pd.concat([sensitivity,additional],ignore_index=True)
    if len(sensitivity) and set(['scenario','pattern']).issubset(sensitivity):
        sensitivity=sensitivity.drop_duplicates(['scenario','pattern'],keep='last')
    assert len(manifest) == manifest.case_id.nunique() == 300
    assert not cases.case_id.duplicated().any()
    for field in ['post_hp3', 'low_dominant', 'high_dominant', 'high_then_short_low', 'terminal_low_near_stop',
                  'high_then_short_mid','terminal_mid_near_stop']:
        if field in hp2:
            hp2[field] = boolcol(hp2[field])
    for key, (_, field) in PATTERNS.items():
        if field not in cases:
            cases[field] = False
        cases[field] = boolcol(cases[field])
    for field in ['hp', 'run_id', 'block_id']:
        for frame in [observations, hp2, points, blocks, intervals]:
            if field in frame:
                frame[field] = pd.to_numeric(frame[field], errors='coerce')
    if 'zone' not in points:
        points['zone'] = np.select([points.rpm == 0, (points.rpm > 0)&(points.rpm <= 5000),
            (points.rpm >= 36000)&(points.rpm <= 44000), points.rpm>44000], ['Z','L','H','X'], default='M')
    log('start', cases=len(manifest), hp2_blocks=len(hp2), hp2_numeric_bins=len(points),
        policy='Derived narrow tables only; no original/preprocessed wide case tables; no assignments changed')
    if args.html_only:
        figures = read('figure_index.csv').fillna('').to_dict('records')
        make_html(out, manifest, intervals, points, cases, hp2, figures, log)
        return

    figures = []
    with tempfile.TemporaryDirectory(prefix='chair-sequence-figures-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        plt.rcParams.update({'font.family':'Noto Sans CJK JP', 'font.size':10,
            'axes.unicode_minus':False, 'axes.spines.top':False, 'axes.spines.right':False,
            'figure.facecolor':'white', 'savefig.facecolor':'white'})

        def save(fig, name, title, caption, category='overview', pattern='', case_id='', block_id=''):
            fig.savefig(out/'figures'/name, dpi=args.dpi, bbox_inches='tight')
            plt.close(fig)
            figures.append(dict(path='figures/'+name, title=title, caption=caption, category=category,
                                pattern=pattern, case_id=case_id, block_id=block_id))

        # 1. Whole-cohort scope and confirmed HP composition, never equating unknown with OFF.
        fig, axes = plt.subplots(1, 3, figsize=(17,5))
        statuscol = firstcol(cases, 'status', 'analysis_status')
        status = cases[statuscol].value_counts(dropna=False) if statuscol else pd.Series({'監査症例':300})
        for i, (s,n) in enumerate(status.items()):
            axes[0].barh(i,n,color='#4d8bb3' if 'EVALUABLE' in str(s) or 'SUPPORTED' in str(s) else '#a9b4bb')
            axes[0].text(n+2,i,str(n),va='center')
        axes[0].set_yticks(range(len(status)), status.index.astype(str))
        axes[0].set_xlim(0,330);axes[0].set_title('全300症例の監査範囲');axes[0].set_xlabel('症例数')
        combo = cases.hp_combination.value_counts() if 'hp_combination' in cases else pd.Series(dtype=float)
        if len(combo):
            axes[1].barh(range(len(combo)),combo.to_numpy(),color='#57a183')
            axes[1].set_yticks(range(len(combo)),combo.index.astype(str))
            for i,n in enumerate(combo):axes[1].text(n+1,i,str(n),va='center')
        axes[1].set_title('作動を確認できたHPの組合せ');axes[1].set_xlabel('症例数')
        durations=[];labels=[];colors=[]
        for hp in [0,1,2,3]:
            col = f'hp{hp}_s'
            if col in audit:
                durations.append(audit[col].sum()/60);labels.append('OFF' if hp==0 else f'HP{hp}');colors.append(HP_COLORS[hp])
        unknown=firstcol(audit,'unknown_s','hp_unknown_s')
        if unknown:durations.append(audit[unknown].sum()/60);labels.append('不明');colors.append('#c9b2cb')
        axes[2].bar(labels,durations,color=colors);axes[2].set_title('記録状態別の総時間');axes[2].set_ylabel('分（症例間合計）')
        fig.tight_layout()
        save(fig,'01_cohort_and_hp_types.png','300症例：範囲とHP種類',
             '作動を確認できた種類の組合せ。未記録は未使用とは判断しない。HP時間は20039の記録値保持モデル。')

        # 2. All-case actual-time timeline, ordered by source cohort.
        order = manifest.case_id.tolist();rowmap={cid:i for i,cid in enumerate(order)}
        fig,ax=plt.subplots(figsize=(16,25))
        for hp in [0,1,2,3,np.nan]:
            g=intervals[intervals.hp.isna()] if pd.isna(hp) else intervals[intervals.hp==hp]
            seg=[[(r.start_s/60,rowmap[r.case_id]),(r.end_s/60,rowmap[r.case_id])] for r in g.itertuples()]
            if seg:ax.add_collection(LineCollection(seg,colors='#c9b2cb' if pd.isna(hp) else HP_COLORS[int(hp)],linewidths=3.5))
        ax.set_xlim(0,manifest.duration_s.max()/60);ax.set_ylim(len(order)-.5,-.5)
        tick=np.arange(0,len(order),15);ax.set_yticks(tick,[shortcase(order[i]) for i in tick],fontsize=8)
        ax.set_xlabel('最初の90012 ONからの時間（分）');ax.set_ylabel('症例（固定コホート順）')
        ax.set_title('全300症例のHP作動順序：OFFと不明を分離')
        handles=[Line2D([],[],color=HP_COLORS[h],lw=5,label='OFF' if h==0 else f'HP{h}') for h in range(4)]
        handles.append(Line2D([],[],color='#c9b2cb',lw=5,label='状態不明'))
        ax.legend(handles=handles,ncol=5,loc='lower right')
        save(fig,'02_all_case_hp_timelines.png','全症例の作動HPタイムライン',
             '各行は1症例、横軸は実時間。空白は症例終了後。初回以前/無効状態は不明色で表示し、OFFに変換しない。')

        # Signal support by actual operating HP prevents treating a pedal channel
        # recorded only for another HP as missing values to be filled in HP2.
        support=observations[observations.hp.isin([1,2,3])&observations.value.notna()].groupby(['hp','code']).case_id.nunique().unstack(fill_value=0)
        support=support.reindex(index=[1,2,3],columns=list(NAMES),fill_value=0)
        fig,axes=plt.subplots(1,2,figsize=(16,5),gridspec_kw={'width_ratios':[1.4,1]})
        im=axes[0].imshow(support.to_numpy(),cmap='Blues',aspect='auto')
        for i in range(3):
            for j in range(len(NAMES)):
                val=support.iloc[i,j]
                axes[0].text(j,i,str(int(val)),ha='center',va='center',color='white' if val>150 else '#263a46',fontsize=12)
        axes[0].set_yticks(range(3),['HP1','HP2','HP3']);axes[0].set_xticks(range(len(NAMES)),[f'{k}\n{v}' for k,v in NAMES.items()])
        axes[0].set_title('作動HPごとに観測できた数値信号：症例数')
        if 'hp3_scaler_comparable_s' in audit and 'conflict_s' in audit:
            valid=audit.hp3_scaler_comparable_s>0
            disagreement=100*audit.loc[valid,'conflict_s']/audit.loc[valid,'hp3_scaler_comparable_s']
            axes[1].hist(disagreement,bins=30,color='#58a184',edgecolor='white')
            axes[1].set_xlabel('20039=HP3 と 20774=1 の不一致時間割合 (%)')
            axes[1].set_ylabel('両信号が既知の症例数');axes[1].set_title(f'独立コードで整合確認：n={int(valid.sum())}')
        fig.tight_layout()
        save(fig,'02b_signal_support_and_hp3_consistency.png','HPの種類ごとの数値支持とHP3整合性',
             '数値は実時刻で作動HPへ対応付けた観測。別HPで記録されたペダルをHP2へ補完しない。不一致割合は両状態が既知の時間だけを分母とする。')

        # 3. RPM states are explicitly exploratory, not manufacturer-validated thresholds.
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        rpm=points.rpm.dropna()
        axes[0].hist(rpm,bins=90,color='#5b96bb',edgecolor='white')
        axes[0].axvline(5000,color=ZONE_COLORS['L'],ls='--',label='探索低回転上限 5000')
        axes[0].axvspan(36000,44000,color=ZONE_COLORS['H'],alpha=.12,label='40000付近：探索band ±10%')
        axes[0].axvline(40000,color=ZONE_COLORS['H'],ls='--')
        axes[0].set_xlabel('HP2作動中の回転数（記録値）');axes[0].set_ylabel('観測bin数');axes[0].legend(fontsize=8)
        axes[0].set_title('観測分布（bin単位、症例数とは別）')
        zones=points.groupby('zone').size().reindex(['Z','L','M','H','X'],fill_value=0)
        axes[1].bar(zones.index,zones.values,color=[ZONE_COLORS[z] for z in zones.index])
        axes[1].set_xticks(range(5),['0回転','低','中間','40000付近','上限超']);axes[1].set_ylabel('観測bin数');axes[1].set_title('0回転と44000超を別扱い')
        if 'rpm_observed_fraction' in hp2:
            axes[2].hist(hp2.rpm_observed_fraction.dropna()*100,bins=np.linspace(0,100,21),color='#7b9e84',edgecolor='white')
        axes[2].set_xlabel('回転数の観測bin相当時間 / HP2作動時間 (%)');axes[2].set_ylabel('HP2ブロック数');axes[2].set_title('「高回転が続く」を判断できる観測支持')
        fig.tight_layout()
        save(fig,'03_rpm_levels_and_support.png','HP2の回転数水準と観測支持',
             '利用者の目安は低5000以下、高40000付近。その具体化として0超〜5000、36000〜44000（±10%）を用いる。中間/0/44000超は別扱い。欠測の間を補間しない。')

        # 4. Raw starts/stops vs tool changes after consolidation.
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        a=firstcol(transitions,'from_hp');b=firstcol(transitions,'to_hp')
        rawmat=pd.crosstab(transitions[a],transitions[b]).reindex(index=[1,2,3],columns=[1,2,3],fill_value=0) if a and b else pd.DataFrame(np.zeros((3,3)),index=[1,2,3],columns=[1,2,3])
        previous=firstcol(blocks,'previous_hp','previous_block_hp')
        blockmat=pd.crosstab(blocks[previous],blocks.hp).reindex(index=[1,2,3],columns=[1,2,3],fill_value=0) if previous else rawmat*0
        for ax,mat,title in [(axes[0],rawmat,'個々の作動run間の遷移'),(axes[1],blockmat,'操作ブロック間の遷移')]:
            im=ax.imshow(mat.to_numpy(),cmap='Blues')
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{int(mat.iloc[i,j]):,}',ha='center',va='center',color='white' if mat.iloc[i,j]>mat.to_numpy().max()*.55 else '#263a46')
            ax.set_xticks(range(3),['HP1','HP2','HP3']);ax.set_yticks(range(3),['HP1','HP2','HP3']);ax.set_xlabel('次');ax.set_ylabel('前');ax.set_title(title)
        gapcol=firstcol(transitions,'gap_s','gap_seconds')
        if gapcol:
            gap=transitions[gapcol].dropna();axes[2].hist(np.log10(gap+1),bins=35,color='#5b96bb',edgecolor='white')
            axes[2].set_xlabel('停止間隔：log10(秒+1)');axes[2].set_ylabel('遷移数')
        axes[2].set_title('OFF間隔は元データで監査')
        fig.tight_layout()
        save(fig,'04_hp_transitions_and_gaps.png','HPの遷移と休止間隔',
             '短いOFFを挟む同HPの繰返しと、異なるHPへの切替を区別する。未知状態をまたぐ順序は別監査。')

        # 5. Case-level, within-case comparison; no pseudo replication by blocks.
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        comparable=hp2.copy()
        if 'post_hp3' in comparable and 'low_fraction' in comparable:
            comparable=comparable[comparable.low_fraction.notna()]
            supportfield=firstcol(comparable,'supported','evaluable','numeric_supported')
            if supportfield:comparable=comparable[boolcol(comparable[supportfield])]
            if len(context_cases) and 'gap_cap_s' in context_cases:
                cc=context_cases[np.isinf(context_cases.gap_cap_s)]
                med=cc.pivot(index='case_id',columns='context',values='low_fraction').rename(
                    columns={'OTHER_KNOWN_PREDECESSOR':False,'POST_HP3':True})
            else:
                if 'context' in comparable:
                    comparable=comparable[comparable.context.isin(['POST_HP3','OTHER_KNOWN_PREDECESSOR'])]
                med=comparable.groupby(['case_id','post_hp3']).low_fraction.median().unstack()
            for pos,condition in enumerate([False,True]):
                vals=med[condition].dropna().to_numpy()*100 if condition in med else np.array([])
                if len(vals):
                    axes[0].boxplot([vals],positions=[pos],widths=.5,showfliers=False,patch_artist=True,
                        boxprops={'facecolor':['#aebac3','#83bca0'][pos],'alpha':.6})
                    axes[0].scatter(pos+np.linspace(-.12,.12,len(vals)),vals,s=10,alpha=.45)
                axes[0].text(pos,103,f'n={len(vals)}',ha='center',fontsize=9)
            axes[0].set_xticks([0,1],['その他の直前HP','直前HP3']);axes[0].set_ylabel('低回転割合（観測bin重み、%）');axes[0].set_ylim(-4,110)
            paired=med.dropna(subset=[False,True]) if False in med and True in med else pd.DataFrame()
            if len(paired):
                for _,row in paired.iterrows():axes[1].plot([0,1],np.array([row[False],row[True]])*100,color='#7994a2',alpha=.25,lw=.8)
                axes[1].scatter(np.zeros(len(paired)),paired[False]*100,s=12,color='#8598a4')
                axes[1].scatter(np.ones(len(paired)),paired[True]*100,s=12,color='#4d9c7b')
                delta=(paired[True]-paired[False])*100
                axes[2].hist(delta,bins=25,color='#58a184',edgecolor='white');axes[2].axvline(0,color='#a45c50',ls='--')
                axes[2].set_xlabel('同症例：HP3後 − その他（ポイント）');axes[2].set_title(f'対照がある{len(paired)}症例の差')
            else:axes[1].text(.5,.5,'同症例内の両条件が不足',ha='center',transform=axes[1].transAxes)
            axes[1].set_xticks([0,1],['その他','HP3後']);axes[1].set_ylabel('低回転割合（%）');axes[1].set_title('同じ症例内で比較')
        axes[0].set_title('各点は1症例の条件別代表値')
        fig.tight_layout()
        save(fig,'05_after_hp3_case_comparison.png','HP3後のHP2は低回転か：症例単位の対照',
             '前のHPが既知である場合だけを対照にする。条件内では正の回転数観測bin相当時間で加重し、症例を解析単位とする。ここはgap上限なし。因果関係を示さない。')

        # 6. Explicit overlap rather than forced three-class assignment.
        fig,axes=plt.subplots(1,2,figsize=(15,5))
        fields=[v[1] for v in PATTERNS.values()]
        counts=[int(cases[f].sum()) for f in fields]
        axes[0].barh(range(3),counts,color=['#529b7a','#d86a50','#8974b5'])
        axes[0].set_yticks(range(3),[v[0] for v in PATTERNS.values()]);axes[0].set_xlabel('確認された症例数（重複可）')
        for i,n in enumerate(counts):
            n_eval=None
            if len(prevalence) and 'pattern' in prevalence:
                matched=prevalence[prevalence.pattern.map(pattern_key)==list(PATTERNS)[i]]
                if len(matched):n_eval=int(matched.iloc[0].n_evaluable_cases)
            text=f'{n}/{n_eval} 評価可能例' if n_eval is not None else str(n)
            axes[0].text(n+.5,i,text,va='center',fontsize=9)
        axes[0].set_xlim(0,max(counts+[1])*1.35)
        evaluable=cases[cases.status=='HP2_PATTERN_EVALUABLE'] if 'status' in cases else cases
        overlaps=evaluable.groupby(fields,dropna=False).size().sort_values(ascending=False)
        ticks=[]
        for i,(signature,n) in enumerate(overlaps.items()):
            axes[1].bar(i,n,color='#598ba8' if any(signature) else '#bdc5cb')
            ticks.append(' / '.join(str(j+1) for j,x in enumerate(signature) if x) or '評価可能・3特徴未検出')
            axes[1].text(i,n+1,str(n),ha='center',fontsize=9)
        axes[1].set_xticks(range(len(ticks)),ticks,rotation=25,ha='right');axes[1].set_ylabel('症例数');axes[1].set_title(f'評価可能{len(evaluable)}症例：特徴の重なり（非排他的）')
        fig.tight_layout()
        save(fig,'06_patterns_and_overlap.png','3種類の候補特徴は重複してよい',
             '1=HP3後低回転、2=観測高回転主体、3=高回転後末尾短中回転。右パネルは評価可能249症例のみ。未記録・支持不足など残る51症例は監査内訳で示し、陰性例に混ぜない。')

        # 7. Joint numeric values: only within the same run/bin, no forward fill.
        available=[c for c in ['torque','bulk','pedal1','pedal2'] if c in points]
        fig,axes=plt.subplots(2,2,figsize=(15,10));axes=axes.ravel()
        for ax,code in zip(axes,available):
            g=points[['rpm',code,'zone']].dropna();g=cap(g,16000)
            for zone in ['Z','L','M','H','X']:
                q=g[g.zone==zone];ax.scatter(q.rpm,q[code],s=5,color=ZONE_COLORS[zone],alpha=.25,label=zone,rasterized=True)
            ax.set_xlabel('HP2回転数（記録値）');ax.set_ylabel({'torque':'トルク','bulk':'バルク開量','pedal1':'ペダル1','pedal2':'ペダル2'}[code])
            paired_all=points.dropna(subset=['rpm',code])
            ax.set_title(f'共通run・1秒bin：{paired_all.case_id.nunique()}症例 / {len(paired_all):,}観測');ax.legend(title='回転数ゾーン',fontsize=8)
            if paired_all.case_id.nunique()<3:
                ax.text(.5,.52,'共観測症例が少ないため\n集団の関係は判断しない',ha='center',va='center',
                        transform=ax.transAxes,color='#735d3e',bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
        fig.tight_layout()
        save(fig,'07_rpm_torque_pedals_joint.png','HP2回転数とトルク・ペダルの組合せ',
             '同じ作動runの同じ1秒binに観測がある組合せのみ。片方の欠測を補完しない。密集時は図示点だけを決定的に間引く。')

        # 8. Paired case-level low/high summaries, avoiding pooled point dominance.
        fig,axes=plt.subplots(2,2,figsize=(15,9));axes=axes.ravel()
        for ax,code in zip(axes,available):
            if len(conditional) and set(['code','zone','raw_median']).issubset(conditional):
                q=conditional[(conditional.code==code)&conditional.zone.isin(['L','H'])]
                g=q.pivot(index='case_id',columns='zone',values='raw_median')
            else:
                g=points[points.zone.isin(['L','H'])&points[code].notna()].groupby(['case_id','zone'])[code].median().unstack()
            paired=g.dropna(subset=['L','H']) if 'L' in g and 'H' in g else pd.DataFrame()
            if len(paired):
                for _,row in paired.iterrows():ax.plot([0,1],row[['L','H']],color='#8aa3b1',alpha=.2,lw=.8)
                for j,zone in enumerate(['L','H']):
                    vals=paired[zone];ax.scatter(np.full(len(vals),j),vals,s=10,color=ZONE_COLORS[zone],alpha=.5)
                ax.plot([-.15,.15],[paired.L.median()]*2,color='#183c52',lw=3)
                ax.plot([.85,1.15],[paired.H.median()]*2,color='#77392a',lw=3)
            ax.set_xticks([0,1],['低回転','高回転']);ax.set_title(f'{code}: 同症例で両条件あり n={len(paired)}')
            ax.set_ylabel({'torque':'トルク','bulk':'バルク開量','pedal1':'ペダル1','pedal2':'ペダル2'}[code])
        fig.tight_layout()
        save(fig,'08_case_paired_numeric_comparison.png','低回転と高回転で数値の組合せは変わるか',
             '同じ症例で低/高回転の両方に共通binの観測がある場合のみ。各点は1症例の中央値、線は同一症例を結ぶ。')

        # 9. Block support and shutdown-near-tail audit.
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        if 'active_s' in hp2:
            axes[0].scatter(hp2.active_s,hp2.rpm_observed_fraction*100,s=10,alpha=.45,color='#5b8ba7')
            axes[0].set_xscale('log');axes[0].set_xlabel('HP2実作動時間（秒、log）');axes[0].set_ylabel('回転数観測支持（%）')
        if 'n_rpm_bins' in hp2:
            axes[1].hist(hp2.n_rpm_bins,bins=40,color='#5b8ba7',edgecolor='white');axes[1].set_xlabel('1ブロックの回転数観測bin数');axes[1].set_ylabel('ブロック数')
        tailflag='high_then_short_mid' if 'high_then_short_mid' in hp2 else 'high_then_short_low'
        nearflag='terminal_mid_near_stop' if 'terminal_mid_near_stop' in hp2 else 'terminal_low_near_stop'
        if tailflag in hp2:
            selected=hp2[hp2[tailflag]]
            ramp=selected[nearflag].value_counts().reindex([False,True],fill_value=0) if nearflag in selected else pd.Series([len(selected),0])
            axes[2].bar([0,1],ramp.to_numpy(),color=['#8c7ab2','#c69a5d']);axes[2].set_xticks([0,1],['停止直前だけではない','停止直前：要確認']);axes[2].set_ylabel('末尾中回転候補のブロック数')
        axes[0].set_title('短い区間と疎な観測を区別');axes[1].set_title('判断に使えた実測量');axes[2].set_title('減速・停止の可能性を別表示')
        fig.tight_layout()
        save(fig,'09_support_and_terminal_low_audit.png','観測支持と末尾中回転の監査',
             '末尾の中回転が停止直前の減速だけで生じた可能性を監査する。これだけで意図的中回転操作とは断定しない。')

        # Sensitivity uses exactly the measured table schema, with no invented repetitions.
        if len(sensitivity):
            numeric_cols=[c for c in sensitivity.select_dtypes(include=[np.number]) if c not in ['low_threshold','high_threshold','merge_gap_s']]
            scenario=firstcol(sensitivity,'scenario','setting','name')
            relevant=[c for c in numeric_cols if any(s in c for s in ['jaccard','n_cases','n_blocks','post_hp3','high_dominant','short_low'])]
            if scenario and relevant:
                if 'pattern' in sensitivity and 'case_jaccard' in sensitivity:
                    sensitivity_main=sensitivity[sensitivity.pattern.map(pattern_key).isin(PATTERNS)].copy()
                    counts_grid=sensitivity_main.pivot(index=scenario,columns='pattern',values='n_cases')
                    similarity_grid=sensitivity_main.pivot(index=scenario,columns='pattern',values='case_jaccard')
                    order=sensitivity_main[scenario].drop_duplicates().tolist()
                    counts_grid=counts_grid.reindex(order);similarity_grid=similarity_grid.reindex(order)
                else:
                    show=sensitivity.groupby(scenario,sort=False)[relevant].first()
                    counts_grid=show[[c for c in show if 'jaccard' not in c]]
                    similarity_grid=show[[c for c in show if 'jaccard' in c]]
                fig,axes=plt.subplots(1,2,figsize=(18,max(5,len(counts_grid)*.38)))
                for ax,grid,is_similarity,title in [(axes[0],counts_grid,False,'候補の症例数'),(axes[1],similarity_grid,True,'基準条件の候補症例集合とのJaccard')]:
                    if len(grid.columns):
                        data=grid.to_numpy(float);im=ax.imshow(data,aspect='auto',cmap='YlGnBu',vmin=0,vmax=1 if is_similarity else None)
                        for i in range(len(grid)):
                            for j in range(len(grid.columns)):
                                val=data[i,j];txt='—' if not np.isfinite(val) else f'{val:.2f}' if is_similarity else f'{val:.0f}'
                                ax.text(j,i,txt,ha='center',va='center',fontsize=8)
                        labels=[PATTERNS.get(pattern_key(c),(str(c),''))[0]+('（最終HP2）' if 'FINAL_' in str(c) else '') for c in grid.columns]
                        ax.set_xticks(range(len(grid.columns)),labels,rotation=25,ha='right');ax.set_yticks(range(len(grid)),grid.index,fontsize=8);ax.set_title(title)
                fig.tight_layout()
                save(fig,'10_threshold_and_gap_sensitivity.png','回転数・休止・支持条件への感度',
                     '候補が回転数帯・同HPブロック併合・支持条件に依存するかを比較する。—は当該条件未評価または候補空集合でJaccard未定義。最良条件だけを選んでいない。')

        mid_signals=read('mid_tail_signal_support.csv',required=False)
        if len(mid_signals):
            fig,axes=plt.subplots(2,2,figsize=(15,9));axes=axes.ravel()
            for ax,code in zip(axes,['rpm','torque','pedal1','pedal2']):
                g=mid_signals[mid_signals.code==code]
                samples=g.pivot(index=['case_id','block_id'],columns='phase',values='raw_median')
                nobs=g.pivot(index=['case_id','block_id'],columns='phase',values='n_pairs')
                phase_names=[c for c in samples if 'high' in c]+[c for c in samples if 'mid' in c]
                if len(phase_names)>=2:
                    pair=samples.dropna(subset=phase_names[:2])
                    for i,(idx,row) in enumerate(pair.iterrows()):
                        color=plt.get_cmap('tab10')(i)
                        values=row[phase_names[:2]].to_numpy(float)
                        counts=[int(nobs.loc[idx,phase]) for phase in phase_names[:2]]
                        label=(shortcase(idx[0])+f' | {values[0]:,.1f}→{values[1]:,.1f}'
                               +f' | n={counts[0]}→{counts[1]}')
                        ax.plot([0,1],values,color=color,alpha=.75,marker='o',label=label)
                    ax.set_title(f'{code}: 両phaseで観測あり {len(pair)}症例')
                else:
                    pair=pd.DataFrame()
                if len(pair)==0:ax.text(.5,.5,'対応する実測がないため比較不可',ha='center',transform=ax.transAxes)
                ax.set_xticks([0,1],['高回転部分','末尾中回転部分']);ax.set_xlim(-.25,1.4)
                ax.set_ylabel({'rpm':'回転数（記録値）','torque':'トルク','pedal1':'ペダル1','pedal2':'ペダル2'}[code]);ax.grid(axis='y',alpha=.15)
                if len(pair):ax.legend(fontsize=7,loc='best')
            fig.suptitle('高回転→末尾中回転：同じ症例・同じHP2ブロック内の数値比較',fontsize=15)
            fig.tight_layout(rect=[0,0,1,.95])
            save(fig,'11_high_to_mid_multisignal.png','高回転から末尾中回転へ：トルク・ペダルも比較',
                 '厳格候補における同ブロック内のphase別原単位中央値。nは共通run・1秒binの観測数。少数症例の記述であり、噴射・処置や因果関係を示さない。')

        # Representative evidence: a full case, an actual-time focus and active-progress scatter.
        if len(reps):
            reps=reps.copy();reps['pattern_key']=reps.pattern.map(pattern_key)
            all_limits={}
            for code in NAMES:
                vals=observations.loc[observations.code==code,'value'].dropna()
                all_limits[code]=(min(0,float(vals.min())) if len(vals) else 0,
                                  max(1,float(vals.max())*1.04) if len(vals) else 1)
            manifest_index=manifest.set_index('case_id')
            case_groups={case:g for case,g in observations.groupby('case_id',sort=False)}
            interval_groups={case:g for case,g in intervals.groupby('case_id',sort=False)}
            for row in reps.itertuples():
                case=row.case_id;bid=row.block_id;key=row.pattern_key
                blockmatch=blocks[(blocks.case_id==case)&(blocks.block_id==bid)]
                if blockmatch.empty:
                    log('representative_missing_block',case_id=case,block_id=bid);continue
                block=blockmatch.iloc[0];obs=case_groups.get(case,pd.DataFrame())
                interval=interval_groups.get(case,pd.DataFrame());duration=float(manifest_index.loc[case,'duration_s'])
                title=PATTERNS.get(key,(str(row.pattern),''))[0]
                casepoints=points[(points.case_id==case)&(points.block_id==bid)].copy()
                tag=f'{safe(key)}_{int(getattr(row,"rank",0)):02d}_{safe(case)}_B{int(bid):03d}'
                for mode in ['full','focus']:
                    lo,hi=(0,duration) if mode=='full' else (max(0,float(block.start_s)-20),min(duration,float(block.end_s)+20))
                    if mode=='focus' and bool(block.get('post_hp3',False)):
                        previous=interval[(interval.hp==3)&(interval.end_s<=block.start_s)]
                        if len(previous):lo=max(0,min(lo,float(previous.end_s.max())-10))
                    fig,axes=plt.subplots(6,1,figsize=(15,13),sharex=True,gridspec_kw={'height_ratios':[.65,2,1.3,1.3,1.3,1.3]})
                    for r in interval.itertuples():
                        if r.end_s<lo or r.start_s>hi:continue
                        hp=r.hp;color='#c9b2cb' if not np.isfinite(hp) else HP_COLORS.get(int(hp),'#c9b2cb')
                        axes[0].broken_barh([(max(r.start_s,lo)/60,(min(r.end_s,hi)-max(r.start_s,lo))/60)],(.1,.8),facecolors=color)
                    axes[0].set_ylim(0,1);axes[0].set_yticks([]);axes[0].set_ylabel('作動HP')
                    axes[0].legend(handles=[Patch(color=HP_COLORS[h],label='OFF' if h==0 else f'HP{h}') for h in range(4)]+[Patch(color='#c9b2cb',label='不明')],ncol=5,loc='upper right',fontsize=8)
                    for ax,code in zip(axes[1:],NAMES):
                        g=obs[(obs.code==code)&(obs.time_s>=lo)&(obs.time_s<=hi)]
                        g=cap(g,4500)
                        if code in ['20742','20743']:
                            g=g[g.hp==2]
                        for hp,h in g.groupby('hp',dropna=False):
                            color=HP_COLORS.get(int(hp),'#999') if np.isfinite(hp) else '#999'
                            ax.scatter(h.time_s/60,h.value,s=7 if mode=='full' else 13,color=color,alpha=.65,rasterized=True)
                        ax.set_ylabel(f'{code}\n{NAMES[code]}');ax.set_ylim(*all_limits[code]);ax.grid(axis='y',alpha=.15)
                        if len(g)==0:ax.text(.98,.75,'この範囲の観測なし',ha='right',transform=ax.transAxes,fontsize=9)
                        ax.axvspan(block.start_s/60,block.end_s/60,color='#f6d78b',alpha=.17)
                    if mode=='focus' and key=='post_hp3_low':
                        q=obs[(obs.code=='20742')&(obs.hp==2)&(obs.time_s>=block.start_s)&(obs.time_s<=block.end_s)]
                        zoom=axes[1].inset_axes([.65,.42,.32,.52])
                        zoom.scatter(q.time_s/60,q.value,s=5,color='#397fa9',alpha=.6)
                        zoom.set_ylim(0,5000);zoom.set_xlim(block.start_s/60,block.end_s/60)
                        zoom.tick_params(labelsize=7);zoom.set_title('注目区間：低回転0〜5000拡大',fontsize=8)
                    axes[-1].set_xlim(lo/60,hi/60);axes[-1].set_xlabel('症例開始からの実時間（分）')
                    predecessor=f' / 直前HP{int(block.previous_hp)}から{block.previous_gap_s:.1f}秒' if np.isfinite(block.get('previous_hp',np.nan)) else ''
                    fig.suptitle(f'{title} | {shortcase(case)} | 注目HP2ブロック {int(bid)}\n'
                        f'HP2実作動 {block.active_s:.1f}秒 / 経過 {float(block.end_s-block.start_s):.1f}秒'+predecessor,fontsize=14)
                    fig.tight_layout(rect=[0,0,1,.95])
                    save(fig,tag+f'_{mode}.png',title+'：'+('全症例' if mode=='full' else '注目区間の拡大'),
                        '点は実測1秒bin代表値。回転数/トルクはHP2作動に対応する観測だけ。黄色は注目HP2ブロック。線で欠測を補間しない。',
                        'representative',key,case,int(bid))
                fig,axes=plt.subplots(4,1,figsize=(13,10),sharex=True)
                for ax,field,label in zip(axes,['rpm','torque','pedal1','pedal2'],['回転数','トルク','ペダル1','ペダル2']):
                    if field not in casepoints:continue
                    for zone in ['Z','L','M','H','X']:
                        g=casepoints[casepoints.zone==zone].dropna(subset=[field])
                        x=g.active_progress.to_numpy(float)
                        if len(x) and np.nanmax(x)<=1.01:x=x*100
                        ax.scatter(x,g[field],s=14,color=ZONE_COLORS[zone],label=zone,alpha=.75)
                    if casepoints[field].notna().sum()==0:
                        ax.text(.5,.5,'同run・同1秒binでの共観測なし',ha='center',va='center',transform=ax.transAxes,color='#6a7882')
                    ax.set_ylabel(label);ax.grid(alpha=.15)
                    code={'rpm':'20742','torque':'20743','pedal1':'10014','pedal2':'10015'}[field]
                    ax.set_ylim(*all_limits[code])
                axes[0].axhline(5000,color=ZONE_COLORS['L'],ls='--',lw=.8)
                axes[0].axhspan(36000,44000,color=ZONE_COLORS['H'],alpha=.09)
                axes[0].legend(ncol=5,title='観測回転数ゾーン（H=40000±10%）',fontsize=8)
                axes[-1].set_xlim(-1,101);axes[-1].set_xlabel('HP2の累積実作動時間に対する進捗（%）：OFF時間は除外、DTWなし')
                fig.suptitle(f'{title} | {shortcase(case)} | ブロック {int(bid)}',fontsize=14)
                fig.tight_layout(rect=[0,0,1,.95])
                save(fig,tag+'_active_progress.png',title+'：累積実作動時間での比較',
                    'OFF時間を除いたHP2累積実作動進捗。DTWによる整列ではない。トルク/ペダルは同run・同1秒binで回転数と共観測の場合のみ。',
                    'representative',key,case,int(bid))
                log('representative_complete',pattern=key,case_id=case,block_id=int(bid),figures=len(figures))

        # Same-HP stages are a separate analysis unit. Never reuse primary IDs.
        stage_details=read('stage_mid_tail_blocks.csv',required=False)
        stage_reps=read('representatives_stage_mid.csv',required=False)
        if len(stage_details):
            for col in ['post_hp3_low','high_dominant','high_then_short_mid','final_hp2_high_then_short_mid']:
                if col in stage_details:stage_details[col]=boolcol(stage_details[col])
            fig,axes=plt.subplots(1,2,figsize=(16,5))
            labels=['HP3後低回転','観測高回転主体','高→末尾短中回転','症例最後のHP2で高→中']
            fields=['post_hp3_low','high_dominant','high_then_short_mid','final_hp2_high_then_short_mid']
            counts=[]
            for table in [hp2,stage_details]:
                row=[]
                for field in fields:
                    mask=boolcol(table[field]) if field in table else pd.Series(False,index=table.index)
                    row.append(table.loc[mask,'case_id'].nunique())
                counts.append(row)
            x=np.arange(4)
            for i,(label,row) in enumerate(zip(['主解析：同HP・OFF間隔30秒以内を併合','補助解析：異HPまで同じHP段階として統合'],counts)):
                bars=axes[0].bar(x+(i-.5)*.34,row,width=.34,label=label,color=['#5b8eae','#c2a16b'][i])
                axes[0].bar_label(bars,padding=3,fontsize=9)
            axes[0].set_xticks(x,labels,rotation=18,ha='right');axes[0].set_ylabel('特徴を確認した症例数');axes[0].legend(fontsize=8)
            axes[0].set_title('作業単位を変えると同じ特徴の判定も変わる')
            sets=[set(table.loc[boolcol(table.high_then_short_mid),'case_id']) for table in [hp2,stage_details]]
            subset=[len(sets[0]-sets[1]),len(sets[0]&sets[1]),len(sets[1]-sets[0])]
            axes[1].bar(range(3),subset,color=['#5b8eae','#7c9f8b','#c2a16b']);axes[1].set_xticks(range(3),['30秒ブロックのみ','両方','同HP段階のみ'])
            axes[1].set_ylabel('高→末尾中回転の候補症例数')
            for i,n in enumerate(subset):axes[1].text(i,n+.08,str(n),ha='center')
            axes[1].set_title('症例集合の違い：件数増加は性能改善を意味しない')
            fig.tight_layout()
            save(fig,'12_analysis_unit_comparison.png','30秒ブロックと大きな同HP段階を分けて比較',
                 '両方とも同じ300症例の監査から作るが、作業単位が異なる。既知OFFが長くても異なるHPが現れるまで統合する補助版を主解析へ上書きしない。')
        if len(stage_reps):
            stage_blocks=read('stage_blocks.csv')
            stage_points=read('stage_hp2_points.csv')
            if 'zone' not in stage_points:
                stage_points['zone']=np.select([stage_points.rpm==0,(stage_points.rpm>0)&(stage_points.rpm<=5000),
                    (stage_points.rpm>=36000)&(stage_points.rpm<=44000),stage_points.rpm>44000],['Z','L','H','X'],default='M')
            for row in stage_reps.itertuples():
                stage=stage_blocks[(stage_blocks.case_id==row.case_id)&(stage_blocks.block_id==row.block_id)].iloc[0]
                detail=stage_details[(stage_details.case_id==row.case_id)&(stage_details.block_id==row.block_id)].iloc[0]
                pp=stage_points[(stage_points.case_id==row.case_id)&(stage_points.block_id==row.block_id)]
                ii=intervals[(intervals.case_id==row.case_id)&(intervals.end_s>=stage.start_s)&(intervals.start_s<=stage.end_s)]
                fig,axes=plt.subplots(5,1,figsize=(15,12),sharex=True,gridspec_kw={'height_ratios':[.6,2,1.3,1.3,1.3]})
                for a in ii.itertuples():
                    hp=a.hp;color=HP_COLORS.get(int(hp),'#c9b2cb') if np.isfinite(hp) else '#c9b2cb'
                    left=max(a.start_s,stage.start_s);right=min(a.end_s,stage.end_s)
                    axes[0].broken_barh([(left/60,(right-left)/60)],(.1,.8),facecolors=color)
                axes[0].set_yticks([]);axes[0].set_ylabel('作動HP');axes[0].set_ylim(0,1)
                axes[0].legend(handles=[Patch(color=HP_COLORS[h],label='OFF' if h==0 else f'HP{h}') for h in range(4)],ncol=4,loc='upper right',fontsize=8)
                for ax,field,code in zip(axes[1:],['rpm','torque','pedal1','pedal2'],['20742','20743','10014','10015']):
                    for zone in ['Z','L','M','H','X']:
                        gg=pp[(pp.zone==zone)&pp[field].notna()];ax.scatter(gg.time_s/60,gg[field],color=ZONE_COLORS[zone],s=8,alpha=.7,label=zone)
                    if pp[field].notna().sum()==0:ax.text(.5,.5,'同run・同1秒binでの共観測なし',ha='center',transform=ax.transAxes)
                    ax.set_ylabel(f'{code}\n{NAMES[code]}');ax.set_ylim(*all_limits[code]);ax.grid(axis='y',alpha=.15)
                axes[1].axhspan(36000,44000,color=ZONE_COLORS['H'],alpha=.09);axes[1].axhline(5000,color=ZONE_COLORS['L'],ls='--',lw=.8)
                axes[1].legend(ncol=5,title='H=40000±10%、M=その下〜5000超',fontsize=8,loc='best')
                axes[-1].set_xlim(stage.start_s/60,stage.end_s/60);axes[-1].set_xlabel('症例開始からの実時間（分）：長いOFFもそのまま表示')
                is_final=boolcol(pd.Series([detail.final_hp2_high_then_short_mid])).iloc[0]
                fig.suptitle(f'補助解析：大きな同HP段階の高回転→末尾中回転 | {shortcase(row.case_id)}\n'
                    f'段階ID {int(row.block_id)}（主30秒ブロックIDとは別） / 実作動{stage.active_s:.1f}秒 / '
                    f'末尾中回転観測相当{detail.tail_mid_equivalent_s:.1f}秒 / 症例最後HP2段階={is_final}',fontsize=13)
                fig.tight_layout(rect=[0,0,1,.94])
                filename=f'stage_mid_{int(row.rank):02d}_{safe(row.case_id)}_S{int(row.block_id):03d}.png'
                save(fig,filename,'補助解析：同HP段階で確認する高回転→末尾中回転',
                     'stage_blocks/stage_hp2_points/stage_mid_tail_blocksを使用。主解析の同番号ブロックと混同しない。一過性の低回転も残し、連続した中回転と断定しない。',
                     'representative','high_then_short_mid',row.case_id,int(row.block_id))
    pd.DataFrame(figures).to_csv(out/'figure_index.csv',index=False)
    make_html(out,manifest,intervals,points,cases,hp2,figures,log)
    log('complete',n_figures=len(figures))


def make_html(out,manifest,intervals,points,cases,hp2,figures,log):
    esc=html.escape
    case_index=cases.set_index('case_id')
    interval_groups={case:g for case,g in intervals.groupby('case_id',sort=False)}
    point_groups={case:cap(g,500) for case,g in points.groupby('case_id',sort=False)}
    payload=[]
    def finite(value):
        try:return float(value) if np.isfinite(float(value)) else None
        except (ValueError,TypeError):return None
    for r in manifest.itertuples():
        row=case_index.loc[r.case_id]
        ig=interval_groups.get(r.case_id,pd.DataFrame());pg=point_groups.get(r.case_id,pd.DataFrame())
        states=[[finite(a.start_s),finite(a.end_s),finite(a.hp)] for a in ig.itertuples()]
        samples=[]
        for a in pg.itertuples():
            samples.append([finite(a.time_s),finite(a.rpm),str(getattr(a,'zone','')),finite(getattr(a,'torque',np.nan)),
                finite(getattr(a,'pedal1',np.nan)),finite(getattr(a,'pedal2',np.nan)),finite(getattr(a,'block_id',np.nan))])
        payload.append(dict(id=r.case_id,short=shortcase(r.case_id),duration=float(r.duration_s),
            status=str(row.get('status','')),combo=str(row.get('hp_combination','')),
            patterns=[key for key,(_,f) in PATTERNS.items() if bool(row.get(f,False))],states=states,points=samples))
    report=(out/'report.txt').read_text() if (out/'report.txt').exists() else '最終科学的評価文は集計完了後に追加します。'
    summary={}
    if (out/'evaluation_summary.json').exists():summary=json.loads((out/'evaluation_summary.json').read_text())
    overview=[g for g in figures if g.get('category','')!='representative']
    representative=[g for g in figures if g.get('category')=='representative']
    def figure(g):
        return (f'<figure class="figure" data-pattern="{esc(str(g.get("pattern","")))}" data-case="{esc(str(g.get("case_id","")))}">'
            f'<a href="{esc(g["path"])}"><img loading="lazy" src="{esc(g["path"])}" alt="{esc(g["title"])}"></a>'
            f'<figcaption><strong>{esc(g["title"])}</strong><br>{esc(g["caption"])}</figcaption></figure>')
    sections=['<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>HPの作動順序とHP2回転数パターン</title>',
        '<style>body{font:16px/1.7 system-ui,"Noto Sans CJK JP",sans-serif;color:#243440;margin:0;background:#f4f7fa}'
        'main{max-width:1500px;margin:auto;padding:30px 28px}h1{font-size:30px;line-height:1.4}h2{margin-top:44px}'
        '.lead,.notice,.viewer{background:white;border-radius:12px;padding:22px;margin:18px 0;box-shadow:0 2px 10px #2434400b}'
        '.notice{border-left:5px solid #c38c35}.figure{background:white;margin:24px 0;padding:16px;border-radius:12px}'
        '.figure img{display:block;width:100%;height:auto}.figure figcaption{font-size:14px;color:#536674;margin:12px 5px}'
        'select,input,button{padding:9px;border:1px solid #aebdc8;border-radius:6px;font-size:15px;margin:4px}'
        'canvas{display:block;background:#fff;max-width:100%;height:auto;border:1px solid #dde5eb;margin-top:12px}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:24px;border-radius:10px;font:14px/1.8 system-ui}'
        'a{color:#22658f}.links{columns:2}.legend span{display:inline-block;margin:3px 12px 3px 0}.dot{width:12px;height:12px;display:inline-block;margin-right:5px}'
        '.small{font-size:13px;color:#536674}.hidden{display:none}table{border-collapse:collapse;width:100%;font-size:13px;background:white}'
        'td,th{border-bottom:1px solid #d9e3ea;padding:8px;text-align:left}th{background:#eaf0f5}.tablewrap{overflow:auto;max-height:440px}'
        '</style><main>',
        '<h1>HPの種類 → 操作順序 → HP2の数値水準</h1>',
        '<div class="lead"><strong>300症例を固定した、DTWなしの段階的評価</strong><p>'
        'HP3の後の低回転HP2、高回転主体、長い高回転後の末尾中回転を、回転数・トルク・ペダルの実測と一緒に確認します。'
        '同一症例が複数の特徴を持つことを許し、全症例を3クラスへ押し込みません。</p></div>',
        '<div class="notice"><strong>読み方</strong><ul><li>高回転主体は観測範囲内の特徴であり、未観測区間まで高回転が続いたとは判断しません。</li>'
        '<li>利用者の訂正を反映し、3番目の主特徴は「高回転後の末尾中回転」です。旧「末尾低回転」結果は参考監査に残しています。</li>'
        '<li>「末尾」は注目するHP2ブロックの末尾です。症例全体で最後のHP2かどうかは別の指標として監査しています。</li>'
        '<li>末尾中回転は停止直前の減速の可能性を別監査します。意図的な操作・処置を確定しません。</li>'
        '<li>利用者の目安は低回転5000以下、高回転40000付近です。高回転の具体化36000〜44000（±10%）と同HPブロック併合条件は探索上の定義であり、閾値感度を併記します。</li>'
        '<li>20039=1/2/3をHP1/HP2/HP3、20774=1をスケーラー作動として利用者確認済み。'
        '20101=0は取り上げ、1は格納（取り上げ≠噴射）。90012は最初ON→最後OFFです。</li>'
        '<li>元データ・旧版分類は変更していません。内部反復・関連を評価し、外部/臨床的な検証は行っていません。</li></ul></div>',
        '<nav><a href="#overview">集団の比較図</a> · <a href="#viewer">300症例の順序・回転数</a> · '
        '<a href="#representatives">候補特徴の代表図</a> · <a href="#report">科学的評価</a> · <a href="#downloads">監査CSV・過程</a></nav>',
        '<h2 id="overview">集団の比較図</h2>']
    sections.extend(figure(g) for g in overview)
    sections.extend(['<h2 id="viewer">症例ごとの順序とHP2回転数</h2><div class="viewer">',
        '<label>候補特徴 <select id="patternFilter"><option value="">全症例</option>'+''.join(f'<option value="{k}">{esc(v[0])}</option>' for k,v in PATTERNS.items())+'</select></label>',
        '<label>症例検索 <input id="caseSearch" placeholder="日付・seq番号・case ID"></label>',
        '<select id="caseSelect" aria-label="症例"></select><p id="caseInfo"></p>',
        '<div class="legend"><span><i class="dot" style="background:#dce1e5"></i>OFF</span><span><i class="dot" style="background:#a778b4"></i>HP1</span>'
        '<span><i class="dot" style="background:#3e85b8"></i>HP2</span><span><i class="dot" style="background:#50a078"></i>HP3</span>'
        '<span><i class="dot" style="background:#c9b2cb"></i>不明</span></div>',
        '<canvas id="timeline" width="1300" height="360"></canvas><p id="hoverInfo" class="small">回転数の点へマウスを近づけると実測値を表示します。</p>',
        '<p class="small">横軸は症例内の実時間。上段はHP作動状態、下段はHP2作動中の回転数。'
        'ブラウザ表示は各症例最大500点を図示し、CSVと代表図には元の集計観測を残しています。欠測を線で結びません。</p></div>',
        '<div class="tablewrap"><table id="caseTable"><thead><tr><th>症例</th><th>監査状態</th><th>確認HP</th><th>候補特徴（重複可）</th></tr></thead><tbody></tbody></table></div>',
        '<h2 id="representatives">候補特徴の代表図</h2><p>上の候補特徴/症例検索で絞り込めます。全症例・拡大・累積作動進捗の3通りで同じ区間を確認してください。</p>'])
    sections.extend(figure(g) for g in representative)
    sections.extend(['<h2 id="report">科学的評価と判断過程</h2><pre>'+esc(report)+'</pre>',
        '<h2 id="downloads">監査データと過程</h2><ul class="links">'])
    downloads=['report.txt','evaluation_summary.json','analysis_plan.json','case_patterns.csv','hp2_blocks.csv','hp2_points.csv',
        'blocks.csv','hp_intervals.csv','hp_runs.csv','hp_status_audit.csv','transition_audit.csv','block_numeric_summary.csv',
        'pattern_prevalence.csv','transition_effects.csv','context_case_comparisons.csv','conditional_numeric.csv','sensitivity.csv',
        'sensitivity_mid.csv','sensitivity_extended.csv','sensitivity_long_gaps.csv','scale_sensitivity.csv',
        'mid_tail_sensitivity.csv','representatives.csv','representatives_mid.csv',
        'mid_tail_blocks.csv','mid_tail_candidates.csv','mid_tail_signal_support.csv','mid_tail_plan.json',
        'stage_mid_tail_blocks.csv','stage_hp2_blocks.csv','stage_hp2_points.csv','stage_blocks.csv','representatives_stage_mid.csv',
        'tail_signal_support.csv','case_manifest.csv',
        'extraction_metadata.json','figure_index.csv']
    for filename in downloads:
        if (out/filename).is_file():sections.append(f'<li><a href="{filename}">{esc(filename)}</a></li>')
    for p in sorted((out/'process').glob('*')):
        if p.is_file():sections.append(f'<li><a href="process/{esc(p.name)}">process/{esc(p.name)}</a></li>')
    sections.append('</ul><p class="small">図は原単位を保持し、同じコードは代表図間で共通の縦軸を使用。未観測と0は別状態です。</p></main>')
    data=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\/')
    labels=json.dumps({k:v[0] for k,v in PATTERNS.items()},ensure_ascii=False)
    sections.append('<script>const DATA='+data+';const LABELS='+labels+';'+JS+'</script></html>')
    (out/'report.html').write_text('\n'.join(sections),encoding='utf-8')
    log('html_written',figures=len(figures),cases=len(payload),interactive_observations=sum(len(r['points']) for r in payload),
        scientific_report_embedded=(out/'report.txt').exists(),bytes=(out/'report.html').stat().st_size)


JS = r'''
const byId=id=>document.getElementById(id);let active=null,rendered=[];
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const hpColors={0:'#dce1e5',1:'#a778b4',2:'#3e85b8',3:'#50a078'};
const zoneColors={Z:'#8b959e',L:'#347fb5',M:'#e5b34c',H:'#d86a50',X:'#9b4d69'};
const patternText=d=>d.patterns.map(p=>LABELS[p]||p).join(' / ')||(d.status==='HP2_PATTERN_EVALUABLE'?'3特徴未検出（評価可能）':'未評価／情報不足');
function eligible(){const p=byId('patternFilter').value,q=byId('caseSearch').value.toLowerCase();return DATA.filter(d=>(!p||d.patterns.includes(p))&&(!q||d.id.toLowerCase().includes(q)||d.short.toLowerCase().includes(q)));}
function refresh(){const list=eligible(),current=byId('caseSelect').value;byId('caseSelect').innerHTML=list.map(d=>`<option value="${esc(d.id)}">${esc(d.short)}</option>`).join('');if(list.some(d=>d.id===current))byId('caseSelect').value=current;
byId('caseTable').querySelector('tbody').innerHTML=list.map(d=>`<tr data-case="${esc(d.id)}"><td><button type="button" data-select="${esc(d.id)}">${esc(d.short)}</button></td><td>${esc(d.status)}</td><td>${esc(d.combo)}</td><td>${esc(patternText(d))}</td></tr>`).join('');
document.querySelectorAll('[data-select]').forEach(b=>b.onclick=()=>{byId('caseSelect').value=b.dataset.select;draw();});const p=byId('patternFilter').value,q=byId('caseSearch').value.toLowerCase();document.querySelectorAll('figure[data-case]').forEach(f=>{if(!f.dataset.case)return;f.classList.toggle('hidden',!!((p&&f.dataset.pattern!==p)||(q&&!f.dataset.case.toLowerCase().includes(q))));});draw();}
function draw(){active=DATA.find(d=>d.id===byId('caseSelect').value);const canvas=byId('timeline'),ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;ctx.clearRect(0,0,W,H);rendered=[];if(!active){byId('caseInfo').textContent='該当症例なし';return;}
byId('caseInfo').textContent=`${active.id} | ${active.status} | 確認HP ${active.combo} | ${(active.duration/60).toFixed(1)}分 | 候補: ${patternText(active)}`;
const left=70,right=W-25,top=100,bottom=H-45;const sx=t=>left+t/active.duration*(right-left),sy=v=>bottom-v/45000*(bottom-top);ctx.font='14px sans-serif';ctx.fillStyle='#304756';ctx.fillText('作動HP',8,39);ctx.fillText('HP2回転数',5,85);
for(const s of active.states){ctx.fillStyle=s[2]===null?'#c9b2cb':hpColors[s[2]]||'#c9b2cb';ctx.fillRect(sx(s[0]),20,Math.max(.4,sx(s[1])-sx(s[0])),37);}
for(const value of [0,5000,36000,40000,44000]){ctx.strokeStyle=value===5000?'#adc5d8':value>=36000?'#e0bbb0':'#e3e9ee';ctx.beginPath();ctx.moveTo(left,sy(value));ctx.lineTo(right,sy(value));ctx.stroke();ctx.fillStyle='#536674';ctx.fillText(String(value),8,sy(value)+5);}
for(let m=0;m<=Math.ceil(active.duration/60);m+=5){const x=sx(m*60);if(x>right)continue;ctx.strokeStyle='#dfe7ed';ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.stroke();ctx.fillStyle='#536674';ctx.fillText(m+'分',x-12,H-18);}
for(const p of active.points){if(p[0]===null||p[1]===null)continue;const x=sx(p[0]),y=sy(p[1]);ctx.fillStyle=zoneColors[p[2]]||'#5686a5';ctx.globalAlpha=.65;ctx.beginPath();ctx.arc(x,y,3,0,2*Math.PI);ctx.fill();rendered.push({x,y,p});}ctx.globalAlpha=1;ctx.strokeStyle='#899ba8';ctx.strokeRect(left,top,right-left,bottom-top);}
byId('patternFilter').onchange=refresh;byId('caseSearch').oninput=refresh;byId('caseSelect').onchange=draw;
byId('timeline').onmousemove=e=>{const canvas=e.currentTarget,r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)*canvas.width/r.width,y=(e.clientY-r.top)*canvas.height/r.height;let best=null,distance=13*13;for(const a of rendered){const d=(a.x-x)**2+(a.y-y)**2;if(d<distance){distance=d;best=a;}}if(best){const p=best.p,fmt=v=>v===null?'未観測':Number(v).toLocaleString();byId('hoverInfo').textContent=`${(p[0]/60).toFixed(2)}分 / RPM ${fmt(p[1])} (${p[2]}) / トルク ${fmt(p[3])} / ペダル1 ${fmt(p[4])} / ペダル2 ${fmt(p[5])} / block ${fmt(p[6])}`;}};
refresh();
'''


if __name__ == '__main__':
    main()
