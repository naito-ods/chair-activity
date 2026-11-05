"""Local scientific figures for the supervised interval/window study.

Requires frozen final evaluation before reading predictions. Figures retain
raw observations and exact HP states. Unknown/unannotated time is never shown
as a clinical negative. Clinical figures remain local, outside public code.
"""
from pathlib import Path
import argparse,hashlib,json,datetime,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix
from PIL import Image
from features import state_segments
from extract import load_case_events

SEED=20261105

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def configure_font():
    candidates=['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc']
    for p in candidates:
        if Path(p).exists():
            font_manager.fontManager.addfont(p)
            prop=font_manager.FontProperties(fname=p)
            plt.rcParams['font.family']=prop.get_name()
            plt.rcParams['axes.unicode_minus']=False
            return p
    available={f.name for f in font_manager.fontManager.ttflist}
    for family in ['Noto Sans CJK JP','Noto Serif CJK JP','DejaVu Sans']:
        if family in available:
            plt.rcParams['font.family']=family;return family
    return 'matplotlib default'

def read_optional(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def add_figure(fig,out,name,rows,kind,description,dpi):
    path=out/'figures'/name
    fig.savefig(path,dpi=dpi,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    rows.append(dict(figure=name,path='figures/'+name,kind=kind,description=description))

def support_figure(out,meta,rows,dpi):
    q=meta[meta.confirmed.astype(bool)].copy()
    treatment=q.groupby('treatment_normalized').agg(n_cases=('seq','nunique'),n_intervals=('id','size')).sort_values('n_cases')
    instrument=[]
    for r in q.itertuples():
        for item in json.loads(r.instrument_labels):instrument.append(dict(seq=r.seq,label=item,id=r.id))
    instruments=pd.DataFrame(instrument).groupby('label').agg(n_cases=('seq','nunique'),n_intervals=('id','size')).sort_values('n_cases')
    fig,axs=plt.subplots(1,2,figsize=(14,8),gridspec_kw={'width_ratios':[1.5,1]})
    for ax,tab,title in [(axs[0],treatment,'Treatment labels: independent case groups'),(axs[1],instruments,'Instrument components: independent case groups')]:
        y=np.arange(len(tab));ax.barh(y,tab.n_cases,color='#397795')
        ax.set_yticks(y,tab.index);ax.set_xlabel('Distinct seq cases (not patients verified)');ax.set_title(title)
        for i,r in enumerate(tab.itertuples()):ax.text(r.n_cases+.15,i,f'{r.n_cases} cases / {r.n_intervals} intervals',va='center',fontsize=8)
        ax.set_xlim(0,max(tab.n_cases.max()*1.7,4));ax.spines[['top','right']].set_visible(False)
    fig.suptitle('All confirmed annotation support; missing instruments are not negative labels')
    add_figure(fig,out,'01_annotation_support.png',rows,'support','Confirmed label counts by distinct case and interval; all labels retained.',dpi)

def comparison_figure(out,rows,dpi):
    dev=pd.read_csv(out/'development_comparison.csv')
    final=json.loads((out/'final_summary.json').read_text())
    fig,axs=plt.subplots(2,2,figsize=(16,10))
    for j,task in enumerate(['instrument','treatment']):
        q=dev[dev.task.eq(task)].sort_values('supported_macro_f1',ascending=False).groupby('view',sort=False).head(1)
        q=q.sort_values('supported_macro_f1')
        ax=axs[0,j];y=np.arange(len(q));ax.barh(y,q.supported_macro_f1,color='#407d9b')
        ax.set_yticks(y,q.view);ax.set(xlim=(0,1),xlabel='Case-weighted macro F1 (supported development labels)',title=task+': best development model per feature view')
        for i,r in enumerate(q.itertuples()):ax.text(r.supported_macro_f1+.012,i,f'{r.supported_macro_f1:.3f}',va='center',fontsize=9)
        ax=axs[1,j];info=final['targets'][task]
        names=['Selected','Prior'];values=[info['metrics']['supported_macro_f1'],info['baseline']['supported_macro_f1']]
        for key,metric in info.get('controls',{}).items():names.append(key);values.append(metric['supported_macro_f1'])
        ax.bar(np.arange(len(names)),values,color=['#b44f43','#a0a7ad']+['#7d9f77']*(len(names)-2))
        ax.set_xticks(np.arange(len(names)),names,rotation=20,ha='right');ax.set(ylim=(0,1),ylabel='Case-weighted supported macro F1',title=task+': frozen chronological final cases')
        for i,v in enumerate(values):ax.text(i,v+.025,f'{v:.3f}',ha='center',fontsize=9)
    for ax in axs.ravel():ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Oracle annotation interval recognition; final cases never used to choose models')
    fig.tight_layout()
    add_figure(fig,out,'02_model_comparison.png',rows,'comparison','Best development model per feature view selected within development; frozen final candidate and controls.',dpi)

def confusion_figures(out,meta,rows,dpi):
    pred=pd.read_csv(out/'final_predictions.csv').fillna('')
    treatment=pred[pred.task.eq('treatment')]
    labels=sorted(set(meta.treatment_normalized.dropna().astype(str)))
    cm=confusion_matrix(treatment.actual,treatment.predicted,labels=labels)
    fig,ax=plt.subplots(figsize=(12,10));im=ax.imshow(cm,cmap='Blues',vmin=0)
    ax.set_xticks(np.arange(len(labels)),labels,rotation=70,ha='right',fontsize=8)
    ax.set_yticks(np.arange(len(labels)),labels,fontsize=8)
    for a in range(len(labels)):
        for b in range(len(labels)):
            if cm[a,b]:ax.text(b,a,str(cm[a,b]),ha='center',va='center',fontsize=9)
    ax.set(xlabel='Predicted annotation label',ylabel='User annotation label',title='Treatment: all labels retained, including labels unseen in development\nCounts are intervals; no treatment validation outside this cohort')
    fig.colorbar(im,ax=ax,shrink=.7,label='Interval count')
    add_figure(fig,out,'03_treatment_confusion.png',rows,'confusion','Raw interval-count confusion; rare and unseen labels remain visible, not silently excluded.',dpi)
    q=pred[pred.task.eq('instrument')];instruments=sorted({v for s in meta.instrument_labels for v in json.loads(s)})
    fig,axs=plt.subplots(2,3,figsize=(13,8));matrices={}
    for ax,label in zip(axs.ravel(),instruments):
        actual=q.actual.map(lambda s:label in s.split('|')).astype(int)
        predicted=q.predicted.map(lambda s:label in s.split('|')).astype(int)
        c=confusion_matrix(actual,predicted,labels=[0,1]);matrices[label]=c.tolist()
        ax.imshow(c,cmap='Blues',vmin=0)
        for a in range(2):
            for b in range(2):ax.text(b,a,str(c[a,b]),ha='center',va='center',fontsize=14)
        ax.set_xticks([0,1],['Not annotated','Annotated']);ax.set_yticks([0,1],['Not annotated','Annotated'])
        ax.set(xlabel='Predicted',ylabel='User label',title=label)
    fig.suptitle('Instrument multi-label components, fixed score threshold 0.5\nA novel combination can contain already-seen instrument components')
    fig.tight_layout()
    add_figure(fig,out,'04_instrument_confusions.png',rows,'confusion','Component-wise instrument confusion at 0.5; full-set exact match is reported separately.',dpi)
    (out/'process/instrument_confusion_counts.json').write_text(json.dumps(matrices,ensure_ascii=False,indent=2))

def window_summary_figures(out,rows,dpi):
    path=out/'windows30/final_summary.json'
    if not path.exists():return
    info=json.loads(path.read_text());fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,task in zip(axs,['instrument','treatment']):
        q=info['targets'][task];names=['Selected','Prior'];values=[q['metrics']['supported_macro_f1'],q['baseline']['supported_macro_f1']]
        for key,v in q.get('controls',{}).items():names.append(key);values.append(v['supported_macro_f1'])
        ax.bar(range(len(names)),values,color=['#b44f43','#a0a7ad']+['#7d9f77']*(len(names)-2));ax.set_xticks(range(len(names)),names,rotation=20,ha='right')
        ax.set(ylim=(0,1),ylabel='Case/annotation-weighted supported macro F1',title=task+': frozen weak-window evaluation')
        for i,v in enumerate(values):ax.text(i,v+.025,f'{v:.3f}',ha='center')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Window labels inherited from annotated intervals; background specificity is NOT validated')
    fig.tight_layout()
    add_figure(fig,out,'05_weak_window_comparison.png',rows,'window_comparison','Frozen 30-second weak-window recognition; inherited labels are not point-level verification or background negatives.',dpi)
    application_figure(out,rows,dpi)

def application_figure(out,rows,dpi):
    application=out/'windows30/application_summary.json'
    if not application.exists():return
    s=json.loads(application.read_text());atoms=pd.read_csv(out/'windows30/candidate_segments.csv');q=atoms[atoms.status.eq('PROVISIONAL_CANDIDATE')]
    has_raw='hp3_observed_active_s' in q
    fig,axs=plt.subplots(1,3 if has_raw else 2,figsize=(16 if has_raw else 12,5.1))
    names=['UNKNOWN','Provisional'];values=[s['candidate_window_status_counts'].get('UNKNOWN',0),s['candidate_window_status_counts'].get('PROVISIONAL_CANDIDATE',0)]
    axs[0].bar(names,values,color=['#a2aab3','#b69155']);axs[0].set_ylabel('30-second windows (overlap; not independent cases)')
    for i,v in enumerate(values):axs[0].text(i,v+max(values)*.02,str(v),ha='center')
    times=[s['unannotated_s']-s['provisional_treatment_candidate_s'],s['provisional_treatment_candidate_s']]
    axs[1].barh(['All unannotated time'],[times[0]],color='#a2aab3',label='UNKNOWN')
    axs[1].barh(['All unannotated time'],[times[1]],left=[times[0]],color='#b69155',label='Provisional window-agreement span')
    axs[1].set_xlabel('Seconds, nonoverlapping atomic union');axs[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.17))
    axs[1].set_title(f'{s["provisional_treatment_candidate_s"]:.0f} s agreement span; NOT active duration')
    if has_raw:
        keys=['hp3_observed_active_s','hp_known_off_s','hp_unknown_s','otherhp_active_s']
        parts=[float(q[k].sum()) for k in keys];labels=['Observed HP3','Known OFF','Unknown HP','Other HP'];colors=['#397c54','#a6acb4','#d7d9de','#d99c4b']
        left=0
        for value,label,color in zip(parts,labels,colors):
            axs[2].barh(['Agreement span'],[value],left=[left],color=color,label=f'{label}: {value:.2f}s');left+=value
        axs[2].set_xlabel('Raw-state overlap seconds inside agreement span');axs[2].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.17))
        axs[2].set_title('Raw HP support shown separately; no clinical-use claim')
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('候補帯（OFFを含む窓の合意範囲；実作動時間ではない）\nProvisional scaler-associated hypothesis; NOT confirmed clinical treatment',fontsize=12)
    fig.tight_layout()
    add_figure(fig,out,'06_selective_application_coverage.png',rows,'application','Candidate band is an OFF/unknown-containing window-agreement span, not actual operation duration. Raw HP3 state overlap is shown separately; all other/unscanned time remains UNKNOWN.',dpi)

def draw_band(ax,start,end,label,y,color,hatched=False,text=True):
    if not np.isfinite(start+end) or end<=start:return
    ax.broken_barh([(start/60,(end-start)/60)],(y-.35,.7),facecolors=color,alpha=.72,hatch='//' if hatched else None,edgecolors='white',linewidth=.5)
    if text and end-start>=20:ax.text((start+end)/120,y,label,ha='center',va='center',fontsize=6,clip_on=True)

def case_figures(out,meta,manifest,rows,dpi,only_seqs=None,write_audit=True):
    seqs=np.sort(meta.seq.unique());random=np.sort(np.random.default_rng(SEED).choice(seqs,size=min(6,len(seqs)),replace=False))
    final=np.sort(meta.loc[meta.split.eq('final'),'seq'].unique())
    predictions=pd.read_csv(out/'final_predictions.csv').fillna('')
    candidates=read_optional(out/'windows30/candidate_predictions.csv')
    atoms=read_optional(out/'windows30/candidate_segments.csv')
    rawruns=read_optional(out/'windows30/candidate_observed_hp3_runs.csv')
    accepted=candidates.loc[candidates.status.eq('PROVISIONAL_CANDIDATE'),'seq'].unique() if len(candidates) else np.array([],int)
    selected=np.unique(np.r_[random,final,accepted])
    if only_seqs is not None:selected=np.intersect1d(selected,np.asarray(only_seqs,int))
    # Optional standalone window output invocation can find root metadata/cache.
    cache_root=out if (out/'cases').exists() else out.parent
    case_sources=pd.read_csv(cache_root/'cases_manifest.csv').set_index('seq')
    labels=sorted(meta.treatment_normalized.dropna().astype(str).unique());palette=plt.get_cmap('tab20')
    colors={label:palette(i%20) for i,label in enumerate(labels)}
    inst=sorted({x for s in meta.instrument_labels for x in json.loads(s)});ic={v:plt.get_cmap('Set2')(i%8) for i,v in enumerate(inst)}
    audits=[]
    for seq in selected:
        events=load_case_events(cache_root/'cases'/f'{seq}.csv.gz');lo=float(events.time_s.min());hi=float(events.time_s.max())+1e-6
        hp=state_segments(events,'20039',lo,hi);ann=meta[meta.seq.eq(seq)]
        fig,axs=plt.subplots(8,1,figsize=(16,14),sharex=True,gridspec_kw={'height_ratios':[1.1,1.4,1.1,1.1,1.0,1.0,1.0,1.0]})
        for a,b,value in hp:
            color='#b9bec5' if not np.isfinite(value) else ['#d7dce1','#72a4c3','#d8994a','#75a96f'][int(value)]
            y=-1 if not np.isfinite(value) else value
            axs[0].broken_barh([(a/60,(b-a)/60)],(y-.35,.7),facecolors=color)
        axs[0].set_yticks([-1,0,1,2,3],['Unknown','OFF','HP1','HP2','HP3']);axs[0].set_ylabel('HP state')
        for ax,code,label in zip(axs[1:4],['20742','20743','10015'],['RPM (raw)','Torque (raw)','Pedal2 (raw)']):
            q=events[events.code.eq(code)&np.isfinite(events.value)]
            original=len(q)
            if len(q)>3500:q=q.iloc[np.linspace(0,len(q)-1,3500).astype(int)]
            ax.scatter(q.time_s/60,q.value,s=3,color='#225d85',alpha=.6,rasterized=True)
            ax.set_ylabel(label);ax.text(.99,.93,f'{original} actual observations',ha='right',va='top',transform=ax.transAxes,fontsize=8)
            if not len(q):ax.text(.5,.5,'No observed values',ha='center',transform=ax.transAxes,color='#777')
        for ax,title in zip(axs[4:],['Annotated treatment','Annotated instrument(s)','Model treatment','Model instrument(s)']):
            ax.set_ylim(-.65,.65);ax.set_yticks([]);ax.set_ylabel(title,fontsize=9)
            ax.axhspan(-.35,.35,color='#eef0f3');ax.text(.005,.9,'Blank/grey = unannotated or UNKNOWN',transform=ax.transAxes,fontsize=6,color='#777')
        for ax in axs[6:]:
            ax.text(.995,.9,'候補帯（OFFを含む窓の合意範囲；実作動時間ではない）・用途未判別',ha='right',transform=ax.transAxes,fontsize=6,color='#93542c')
        for r in ann.itertuples():
            label=str(r.treatment_normalized);draw_band(axs[4],r.start_s,r.end_s,label,0,colors.get(label,'#aaa'),not bool(r.confirmed))
            items=json.loads(r.instrument_labels)
            draw_band(axs[5],r.start_s,r.end_s,' / '.join(items) if items else 'Instrument unknown',0,ic.get(items[0],'#aaa') if items else '#aaa',not bool(r.confirmed))
        pred=predictions[predictions.seq.eq(seq)]
        annotation_index=ann.set_index('id')
        for r in pred.itertuples():
            if r.id not in annotation_index.index:continue
            row=annotation_index.loc[r.id];label=str(r.predicted)
            if not label:continue
            if r.task=='treatment':draw_band(axs[6],row.start_s,row.end_s,label,0,colors.get(label,'#aaa'),True)
            else:
                first=label.split('|')[0];draw_band(axs[7],row.start_s,row.end_s,label.replace('|',' / '),0,ic.get(first,'#aaa'),True)
        nc=0
        if len(atoms):
            cq=atoms[atoms.seq.eq(seq)&atoms.status.eq('PROVISIONAL_CANDIDATE')]
            for r in cq.itertuples():
                for field,ax,color_map in [('treatment_candidate',axs[6],colors),('instrument_candidate',axs[7],ic)]:
                    label=str(getattr(r,field,''))
                    if label in ['', 'nan','UNKNOWN','[]']:continue
                    color=color_map.get(label.split('|')[0],'#916f9c')
                    if hasattr(r,'hp3_observed_active_s') and r.hp3_observed_active_s<=1e-9:color='#a6acb4'
                    draw_band(ax,r.start_s,r.end_s,label,0,color,True,text=False)
                nc+=1
        if len(rawruns):
            for r in rawruns[rawruns.seq.eq(seq)].itertuples():
                for ax in axs[6:]:ax.plot([r.start_s/60,r.end_s/60],[-.23,-.23],color='#174e2b',linewidth=3,solid_capstyle='butt')
            for ax in axs[6:]:ax.text(.005,.08,'Dark line = observed HP3 overlap only; grey hatching = no observed HP3 within candidate atom',transform=ax.transAxes,fontsize=6,color='#174e2b')
        axs[-1].set_xlabel('Minutes from first original raw timestamp; no DTW or numeric interpolation')
        axs[-1].set_xlim(lo/60,hi/60)
        for ax in axs:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
        origin=str(case_sources.loc[seq,'raw_start_datetime']);split=ann.split.iloc[0]
        fig.suptitle(f'seq {seq} | {split} | raw origin {origin}\nSolid annotation bands = supplied labels; hatched bands = model estimates/candidates (not confirmed labels)',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.965])
        name=f'case_{int(seq)}_raw_and_labels.png'
        add_figure(fig,out,name,rows,'case_replay','Exact HP states/raw observations; candidate hatching is window-agreement span (includes OFF), not activity. Dark line separately shows observed HP3 state overlap; clinical use remains unverified.',dpi)
        audits.append(dict(seq=int(seq),selection='random6+all_final8+all_provisional_candidate_cases',selected_random=bool(seq in random),selected_final=bool(seq in final),selected_candidate=bool(seq in accepted),n_annotation_rows=len(ann),n_final_prediction_rows=len(pred),n_candidate_atomic_segments=nc,plot_numeric_downsample_max=3500))
    if write_audit:
        pd.DataFrame(audits).to_csv(out/'plotted_cases.csv',index=False)
        (out/'process/replay_selection.json').write_text(json.dumps(dict(seed=SEED,random_cases=random.tolist(),all_final_cases=final.tolist(),all_candidate_cases=accepted.tolist(),selected_cases=selected.tolist(),selection_uses_model_errors=False,candidate_bands='nonoverlapping unanimous atomic window-context agreement, includes OFF and is not operating duration'),indent=2))
    return audits

def write_gallery(out,rows):
    gallery=['<!doctype html><html lang="ja"><meta charset="utf-8"><title>Supervised study figures</title>',
             '<style>body{font:16px/1.7 sans-serif;max-width:1500px;margin:auto;padding:24px;color:#203047}img{max-width:100%;height:auto}figure{margin:35px 0}figcaption{color:#536375}a{color:#185d93}.note{padding:16px;background:#fff3df;border-left:4px solid #b58b4c}</style>',
             '<h1>アノテーション学習：比較図と実測Replay</h1>',
             '<p>供給された区間ラベルと、モデルの推定・未確認候補を別々の帯で表示します。未記載は負例ではなくUNKNOWNです。数値は原観測点で、補間・前方補完を行っていません。</p>']
    note='候補帯（OFFを含む窓の合意範囲；実作動時間ではない）。スケーラー関連操作の候補であり、スケーリング等の臨床用途は未判別です。'
    support=out/'windows30/candidate_raw_support_summary.json'
    if support.exists():
        s=json.loads(support.read_text())
        note+=f' 候補文脈範囲{s["candidate_context_s"]:.2f}秒には、実測HP3作動{s["candidate_hp3_observed_active_s"]:.2f}秒、既知OFF{s["candidate_hp_known_off_s"]:.2f}秒、HP不明{s["candidate_hp_unknown_s"]:.2f}秒を含みます。別信号20774のON時間は重なるため、HP3秒数と加算しません。灰色ハッチの候補原子区間には、その範囲内のHP3作動が観測されていません。'
    gallery.append('<p class="note">'+html.escape(note)+'</p>')
    for r in rows:
        gallery.append(f'<figure><h2>{html.escape(r["figure"])}</h2><a href="{r["path"]}"><img loading="lazy" src="{r["path"]}"></a><figcaption>{html.escape(r["description"])}</figcaption></figure>')
    gallery.append('</html>');(out/'figures.html').write_text('\n'.join(gallery))

def visualize(out,dpi=110):
    if not (out/'final_summary.json').exists():raise RuntimeError('Frozen final evaluation must finish before visualization')
    if (out/'figures_index.csv').exists():raise FileExistsError('Use a new output or preserve existing visualizations')
    (out/'figures').mkdir(exist_ok=True);(out/'process').mkdir(exist_ok=True)
    font=configure_font();meta=pd.read_csv(out/'annotations_aligned.csv')
    root=out if (out/'cases_manifest.csv').exists() else out.parent
    manifest=pd.read_csv(root/'cases_manifest.csv');original={str(p):sha(p) for p in [root/'annotations_aligned.csv',root/'features_basic.csv',root/'features_extended.csv']}
    sourcebefore={str(p):sha(p) for p in manifest.source_path}
    integrity=pd.read_csv(root/'source_integrity.csv')
    metadata_before={str(p):sha(p) for p in integrity.metadata_path}
    preflight=json.loads((root/'process/preflight.json').read_text())
    annotation_path=Path(preflight['annotation_path']);annotation_before=sha(annotation_path)
    rows=[];support_figure(out,meta,rows,dpi);comparison_figure(out,rows,dpi);confusion_figures(out,meta,rows,dpi);window_summary_figures(out,rows,dpi);case_figures(out,meta,manifest,rows,dpi)
    pd.DataFrame(rows).to_csv(out/'figures_index.csv',index=False)
    write_gallery(out,rows)
    for p in (out/'figures').glob('*.png'):
        with Image.open(p) as im:im.verify()
    assert original=={p:sha(p) for p in original}
    sourceafter={str(p):sha(p) for p in manifest.source_path};assert sourcebefore==sourceafter
    expected=dict(zip(manifest.source_path,manifest.source_sha256));assert all(sourceafter[p]==h for p,h in expected.items())
    assert metadata_before=={p:sha(p) for p in metadata_before}
    assert all(metadata_before[str(r.metadata_path)]==r.metadata_sha256 for r in integrity.itertuples())
    assert sha(annotation_path)==annotation_before==preflight['annotation_sha256']
    checks=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),figures=len(rows),all_pngs_decodable=True,dpi=dpi,font=font,
        source_case_count=len(manifest),all_raw_source_hashes_match_initial=True,all_metadata_hashes_match_initial=True,original_annotation_hash_match=True,annotation_features_unchanged=True,
        numeric_policy='Raw observed scatter, no interpolation/ffill. Drawing downsampling only.',
        interpretation='Annotation interval labels are user-provided; model/candidate bands are separate. Unannotated time is UNKNOWN, not a clinical negative.')
    (out/'process/visualization_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    print(json.dumps(checks,ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--dpi',type=int,default=110);a=ap.parse_args();visualize(a.output,a.dpi)
