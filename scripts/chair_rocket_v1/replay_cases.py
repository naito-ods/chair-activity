"""Apply frozen ROCKET models and render raw timelines for explicit query cases."""
import argparse,os,sys,json,pickle,hashlib,datetime
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/chair_rocket_replay_mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from threadpoolctl import threadpool_limits
import rocket_core as core
from features import state_segments
from extract import read_raw

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class TrustedModelLoader(pickle.Unpickler):
    """Support the original local pilot's module names. Trusted local models only."""
    def find_class(self,module,name):
        if module in ['run_rocket','__main__'] and name in ['InputEncoding','WeightedRocket']:
            return getattr(core,name)
        return super().find_class(module,name)

def load_model(path):
    with Path(path).open('rb') as f:return TrustedModelLoader(f).load()

def build_figure(events,case,annotations,pred,selection_reason):
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    fig,axs=plt.subplots(8,1,figsize=(16,14),sharex=True,
        gridspec_kw={'height_ratios':[.8,1.2,.8,1,1.2,1.2,1.2,1.2]})
    end=float(case.raw_duration_s)
    def band(ax,a,b,y,label,color,h=.65):
        ax.broken_barh([(a/60,(b-a)/60)],(y-h/2,h),facecolors=color,edgecolors='white',linewidth=.65)
        if label:ax.text((a+b)/120,y,label.replace('|','＋'),ha='center',va='center',fontsize=5.5,clip_on=True)
    for a,b,v in state_segments(events,'20039',0,end):
        y=-1 if not np.isfinite(v) else v
        band(axs[0],a,b,y,'','#AFB7BE' if y<0 else ['#DDE2E7','#78A4CE','#E7A04A','#77A777'][int(y)])
    axs[0].set_yticks([-1,0,1,2,3],['不明','0','1','2','3']);axs[0].set_ylabel('20039\n状態')
    q=events[events.code.eq('20742')&events.value.notna()]
    axs[1].scatter(q.time_s/60,q.value,s=5,color='#2A719A',rasterized=True);axs[1].set_ylabel('20742\n原観測')
    for a,b,v in state_segments(events,'20101',0,end):
        y=-1 if not np.isfinite(v) else v
        band(axs[2],a,b,y,'',{-1:'#AFB7BE',0:'#77A4CD',1:'#DDE2E7'}[y])
    axs[2].set_yticks([-1,0,1],['不明','取上げ','収納']);axs[2].set_ylabel('20101\n取上げ≠噴射')
    for r in annotations.itertuples():
        band(axs[3],r.start_s,r.end_s,0,r.instrument_normalized,'#72ACD0')
        band(axs[3],r.start_s,r.end_s,1,r.treatment_normalized,'#DEA258')
    axs[3].set_yticks([0,1],['器具','治療']);axs[3].set_ylabel('人の記載');axs[3].set_ylim(-.5,1.5)
    panels=[('time_series','instrument','ROCKET\n器具候補'),('time_series','treatment','ROCKET\n治療候補'),
            ('mask_only','instrument','マスク対照\n器具候補'),('mask_only','treatment','マスク対照\n治療候補')]
    for ax,(variant,task,title) in zip(axs[4:],panels):
        q=pred[pred.variant.eq(variant)&pred.task.eq(task)].sort_values('start_s')
        labels=sorted(set(q.candidate)-{'UNKNOWN'});colors={label:plt.cm.tab20(i%20) for i,label in enumerate(labels)}
        for i,r in enumerate(q.itertuples()):band(ax,r.start_s,r.end_s,i%2,r.candidate,colors.get(r.candidate,'#D8DFE4'))
        ax.set_yticks([]);ax.set_ylim(-.5,1.5);ax.set_ylabel(title)
    for ax in axs:
        ax.set_xlim(0,end/60);ax.grid(axis='x',alpha=.15)
        for r in annotations.itertuples():
            ax.axvline(r.start_s/60,color='#666',ls=':',lw=.8);ax.axvline(r.end_s/60,color='#666',ls=':',lw=.8)
    axs[-1].set_xlabel(f'原ログ先頭からの経過時間（分）｜{case.raw_start_datetime}')
    fig.suptitle(f'seq{int(case.seq)}：原観測・アノテーション・固定ROCKET予測',x=.12,y=.97,ha='left',fontsize=17,weight='bold')
    fig.text(.12,.937,'選定理由：'+selection_reason,fontsize=11)
    fig.text(.12,.911,'120秒窓・60秒間隔｜学習に使っていない症例｜モデル・閾値は変更せず適用',fontsize=10)
    fig.subplots_adjust(left=.12,right=.98,top=.88,bottom=.11,hspace=.30)
    fig.text(.12,.041,'1本＝120秒窓。上下交互に重複窓を表示。候補は未確認で、作動時間・治療の確定ではありません。\n'
             '数値は実観測のみ。点のない部分を0と解釈しません。未記載部を負例にしていません。',fontsize=10,linespacing=1.6)
    return fig

def run(a):
    out=a.output.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    annotations=pd.read_csv(a.annotations);cases=pd.read_csv(a.case_manifest)
    assert annotations.id.is_unique and cases.seq.is_unique
    requested=[int(v) for v in a.cases.split(',')]
    if len(requested)!=len(set(requested)):raise ValueError('Duplicate requested cases')
    if set(requested)-set(cases.seq):raise ValueError('Requested cases missing from manifest')
    if set(requested)-set(annotations.seq):raise ValueError('Requested cases lack annotation records')
    reasons=json.loads(a.reasons) if a.reasons else {}
    selection={'requested_cases':requested,'requested_count':len(requested),
        'selected_before_model_inference':True,'basis':'Annotation pattern diversity, not prediction scores',
        'reasons':reasons,'models_refitted':False,'seed':core.SEED,'width_s':120,'step_s':60,
        'statistical_representativeness_claimed':False}
    (out/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
    paths={v:a.model_root/f'{v}_model.pkl' for v in ['time_series','mask_only']}
    modelhashes={v:sha(p) for v,p in paths.items()};models={v:load_model(p) for v,p in paths.items()}
    for variant,model in models.items():
        for task,traincases in model.train_cases.items():
            if set(requested)&set(traincases):raise ValueError(f'Query/training overlap: {variant}/{task}')
    hashes={str(a.annotations):sha(a.annotations),str(a.case_manifest):sha(a.case_manifest)}
    allrows=[];summaries=[];audits=[];coverage=[]
    with PdfPages(out/'three_cases_rocket_replay.pdf') as pdf:
        for seq in requested:
            case=cases[cases.seq.eq(seq)].iloc[0];ann=annotations[annotations.seq.eq(seq)]
            raw=a.c31_root/'chair_export'/case.date/case.chair/f'Chair_log_seq_{seq}.dat'
            rawhash=sha(raw)
            if rawhash!=case.source_sha256:raise ValueError(f'Raw source hash mismatch seq{seq}')
            events,origin,end,audit=read_raw(raw)
            length=float(case.raw_duration_s)
            starts=np.arange(int(np.floor((length-120)/60))+1,dtype=float)*60 if length>=120 else np.array([])
            if not len(starts):raise ValueError(f'No full120s query window seq{seq}')
            xx=np.stack([core.tensor(events,float(t)) for t in starts])
            np.savez_compressed(out/f'case_{seq}_raw_tensor.npz',query=xx,channels=np.array(core.CHANNELS),starts=starts)
            rows=[]
            for variant,model in models.items():
                outputs=model.predict(xx)
                for task,(labels,margins) in outputs.items():
                    np.savez_compressed(out/f'case_{seq}_{variant}_{task}_margins.npz',
                        margins=margins,classes=np.array(model.classes[task]))
                    for i,(start,label) in enumerate(zip(starts,labels)):
                        own=events[events.time_s.ge(start)&events.time_s.lt(start+120)]
                        candidate=label if label and len(own) else 'UNKNOWN'
                        inside=ann[ann.start_s.le(start)&ann.end_s.ge(start+120)]
                        truth='';parent='';correct=None
                        overlaps=ann[(ann.start_s<start+120)&(ann.end_s>start)]
                        scope='BOUNDARY_OR_MIXED' if len(overlaps) else 'UNANNOTATED_UNKNOWN'
                        if len(inside)==1 and bool(inside.iloc[0].confirmed):
                            r=inside.iloc[0];parent=r.id
                            truth=('|'.join(sorted(json.loads(r.instrument_labels))) if task=='instrument'
                                   else (str(r.treatment_normalized) if pd.notna(r.treatment_normalized) else ''))
                            if truth:
                                scope='FULLY_INSIDE_PARENT_WEAK_LABEL';correct=set(truth.split('|'))==set(candidate.split('|'))
                        rows.append(dict(variant=variant,task=task,seq=seq,case_id=case.case_id,
                            id=f'{seq}__scan120_{i:04d}',start_s=start,end_s=start+120,
                            predicted=label,candidate=candidate,reference=truth,annotation_id=parent,
                            reference_scope=scope,correct=correct,human_confirmed=False))
            pred=pd.DataFrame(rows)
            assert len(pred)==4*len(starts) and not pred.duplicated(['variant','task','id']).any()
            pred.to_csv(out/f'case_{seq}_predictions.csv',index=False)
            ann.to_csv(out/f'case_{seq}_annotations.csv',index=False)
            for (variant,task),g in pred.groupby(['variant','task']):
                scored=g[g.reference_scope.eq('FULLY_INSIDE_PARENT_WEAK_LABEL')]
                summaries.append(dict(seq=seq,variant=variant,task=task,n_scanned_windows=len(starts),
                    n_scored_windows=len(scored),n_correct=int(scored.correct.sum()),
                    n_scored_parent_intervals=int(scored.annotation_id.nunique()),
                    unweighted_window_agreement=float(scored.correct.mean()) if len(scored) else None))
            fig=build_figure(events,case,ann,pred,reasons.get(str(seq),'Annotation pattern diversity'))
            fig.savefig(out/f'case_{seq}_rocket_replay.png',dpi=130,facecolor='white')
            pdf.savefig(fig,facecolor='white')
            # Also supply a readable view around the existing human intervals.
            lo=max(0.,float(ann.start_s.min())-60.)
            hi=min(length,float(ann.end_s.max())+60.)
            fig.axes[-1].set_xlim(lo/60,hi/60)
            fig.savefig(out/f'case_{seq}_rocket_replay_focus.png',dpi=130,facecolor='white')
            plt.close(fig)
            allrows.extend(rows)
            coverage.append(dict(seq=seq,raw_duration_s=length,n_windows=len(starts),
                scanned_union_s=float(starts[-1]+120),unscanned_tail_s=float(length-starts[-1]-120),
                annotation_intervals=len(ann),parent_intervals_with_no_fully_contained_window=[r.id for r in ann.itertuples() if not np.any((starts>=r.start_s)&(starts+120<=r.end_s))]))
            assert sha(raw)==rawhash
            audits.append(dict(seq=seq,raw_path=str(raw),source_sha256=rawhash,source_unchanged=True,**audit))
            print(json.dumps({'case_completed':seq,'query_windows':len(starts),'annotation_intervals':len(ann)},ensure_ascii=False),flush=True)
    combined=pd.DataFrame(allrows);assert combined.seq.nunique()==len(requested)
    combined.to_csv(out/'all_predictions.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'window_agreement_summary.csv',index=False)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    assert all(sha(paths[v])==h for v,h in modelhashes.items())
    result=dict(selection,actual_cases=len(requested),assignment_rows=len(combined),models_sha256=modelhashes,
        input_hashes=hashes,raw_audits=audits,coverage=coverage,window_agreement=summaries,
        no_duplicate_variant_task_window_keys=True,interpretation='Descriptive weak-parent agreement on selected cases; overlapping windows are not independent. Background accuracy unknown.')
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['c31-root','model-root','annotations','case-manifest','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cases',required=True,help='Comma-separated explicit seq IDs')
    p.add_argument('--reasons',default='',help='JSON map of seq to reason; fixed before inference')
    with threadpool_limits(limits=4):run(p.parse_args())
