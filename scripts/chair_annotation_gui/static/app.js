/* Local annotation UI; no external services, libraries, or model predictions. */
'use strict';
const $ = id => document.getElementById(id);
const clone = value => JSON.parse(JSON.stringify(value));
const S = {config:null,cases:[],case:null,doc:null,selected:null,formDirty:false,dirty:false,history:[],view:[0,1],cursor:0,signals:{},signalRequest:0,change:0,saveChain:Promise.resolve(),mediaUrl:null};
const clinicalNames = {treatment:'処置名あり',procedure_unspecified:'処置あり・内容不明',no_procedure:'確認された処置なし',unknown:'不明'};
const codeNames = {'20039':'HP作動','20011':'HP選択','20101':'シリンジ保持','20742':'RPM','20743':'トルク','20108':'開度','20774':'スケーラー作動','20776':'スケーラー設定','10014':'ペダル1','10015':'ペダル2','90012':'ランプ'};
const colors = {treatment:'#23978d',procedure_unspecified:'#c19c50',no_procedure:'#7e91a9',unknown:'#a7afb4'};
const esc = s => String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sec = n => Number(n).toFixed(3).replace(/\.?0+$/,'');
const clock = n => `${Math.floor(n/60)}:${(n%60).toFixed(1).padStart(4,'0')}`;
const qs = obj => new URLSearchParams(obj).toString();
let messageTimer;
function message(text,error=false){clearTimeout(messageTimer);$('message').textContent=text;$('message').className=error?'error':'';$('message').hidden=false;messageTimer=setTimeout(()=>$('message').hidden=true,error?16000:6000);}
async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Annotation-Token':S.config.token},body:JSON.stringify(body)});const v=await r.json();if(!r.ok)throw new Error(v.error||`HTTP ${r.status}`);return v;}
const guarded = fn => async event => {try{await fn(event);}catch(e){message(e.message,true);}};
function backupKey(){return `chair-annotation:${S.doc.case_id}:${S.doc.source_sha256}`;}
function backup(){if(!S.doc)return;try{localStorage.setItem(backupKey(),JSON.stringify(S.doc));}catch{message('ブラウザの下書き保存ができません。ディスク保存状態を確認してください。',true);}}
function changed(){S.change++;S.dirty=true;backup();$('saveStatus').textContent='未保存';}
function pushHistory(){S.history.push({intervals:clone(S.doc.intervals),scope:clone(S.doc.scope)});if(S.history.length>30)S.history.shift();$('undoButton').disabled=false;}
async function persist(){
  const cid=S.doc?.case_id;
  const operation=async()=>{
    if(!S.doc||S.doc.case_id!==cid||!S.dirty)return;
    $('saveStatus').textContent='保存中…';const generation=S.change;const snapshot=clone(S.doc);
    const response=await api('/api/save',snapshot);
    if(S.doc.case_id!==cid)return;
    S.doc.revision=response.document.revision;S.doc.updated_at=response.document.updated_at;S.doc.created_at=response.document.created_at;
    if(generation===S.change)S.dirty=false;
    $('saveStatus').textContent=S.dirty?'未保存':`保存済み · r${S.doc.revision}`;backup();renderQuality(response.quality);
  };
  S.saveChain=S.saveChain.catch(()=>{}).then(operation);
  try{await S.saveChain;}catch(e){$('saveStatus').textContent='保存できません';backup();throw e;}
}
function gapInfo(){
  if(!S.doc)return {gaps:[],overlaps:[],coverage_fraction:0};
  const d=S.doc;let cursor=d.scope.start_s;const gaps=[],overlaps=[],active=[];
  for(const r of [...d.intervals].sort((a,b)=>a.start_s-b.start_s||a.end_s-b.end_s)){
    if(r.start_s>cursor+1e-6)gaps.push({start_s:cursor,end_s:r.start_s});
    for(const a of active)if(a.end_s>r.start_s+1e-6)overlaps.push([a.id,r.id]);
    active.push(r);cursor=Math.max(cursor,r.end_s);
  }
  if(cursor<d.scope.end_s-1e-6)gaps.push({start_s:cursor,end_s:d.scope.end_s});
  return {gaps,overlaps,coverage_fraction:1-gaps.reduce((t,g)=>t+g.end_s-g.start_s,0)/(d.scope.end_s-d.scope.start_s)};
}
function renderQuality(q){
  const g=q||gapInfo();$('coverageText').textContent=`記入済み ${(100*g.coverage_fraction).toFixed(1)}% · 未記入 ${g.gaps.length}区間 · 重複 ${g.overlaps.length}件`;
  $('coverageBar').style.width=`${Math.max(0,Math.min(100,g.coverage_fraction*100))}%`;
  if(!q)return;
  $('qualitySummary').textContent=q.final_export_allowed?'品質チェック：最終出力できます':`品質チェック：最終出力まで ${q.blockers.length}項目`;
  $('qualityList').innerHTML=[...q.blockers.map(x=>`<div class="qc-error">• ${esc(x.reason)}${x.id?` <button class="quiet" data-qcid="${esc(x.id)}">区間へ</button>`:''}</div>`),...q.warnings.map(x=>`<div class="qc-warning">• ${esc(x.reason)}${x.id?` <button class="quiet" data-qcid="${esc(x.id)}">区間へ</button>`:''}</div>`),...q.gaps.map(g=>`<div>未記入 ${sec(g.start_s)}–${sec(g.end_s)} 秒 <button class="quiet" data-gap="${g.start_s},${g.end_s}">記入</button></div>`) ].join('')||'<div>未記入・重複・未確認はありません。不明として確認した区間も出力に含まれます。</div>';
}
function renderCases(){const term=$('caseSearch').value.trim().toLowerCase();const found=S.cases.filter(c=>`${c.seq} ${c.date} ${c.chair}`.toLowerCase().includes(term));const visible=found.slice(0,120);const selected=found.find(c=>c.case_id===S.case?.case_id);if(selected&&!visible.includes(selected)){visible.pop();visible.unshift(selected);}$('caseCount').textContent=`${found.length.toLocaleString()}件${found.length>120?' · 先頭120件を表示':''}`;$('caseList').innerHTML=visible.map(c=>`<button class="case-item ${S.case?.case_id===c.case_id?'active':''}" data-case="${esc(c.case_id)}"><strong>seq ${c.seq}</strong><span>${esc(c.date)} · ${esc(c.chair)}</span></button>`).join('');}
function renderRows(){
  const rows=[...S.doc.intervals].sort((a,b)=>a.start_s-b.start_s||a.end_s-b.end_s);$('intervalCount').textContent=rows.length;$('emptyIntervals').hidden=!!rows.length;
  $('intervalRows').innerHTML=rows.map(r=>`<tr data-interval="${esc(r.id)}" class="${r.id===S.selected?'selected':''}"><td>${sec(r.start_s)} – ${sec(r.end_s)}<div class="subtle">${sec(r.end_s-r.start_s)} 秒</div></td><td>${esc(r.treatment_jp||clinicalNames[r.clinical_status])}</td><td><div class="instrument-text">${esc(r.instruments.join(' / ')||(r.instrument_status==='none'?'器具なしを確認':'器具不明'))}</div></td><td><span class="pill ${r.reviewed?'ok':''}">${r.reviewed?'確認済':'未確認'}</span></td></tr>`).join('');renderQuality();draw();
}
function renderLegacy(){
  $('legacyCount').textContent=S.case.legacy.length;
  $('legacyRows').innerHTML=S.case.legacy.map((r,i)=>`<div class="legacy-row"><div>${sec(r.start_s)}–${sec(r.end_s)}秒 · ${esc(r.treatment||'処置名なし')}<span>旧器具名：${esc(r.instrument||'記載なし')} · ${esc(r.status||'状態不明')}</span></div><button data-legacy="${i}">下書きへ追加</button></div>`).join('')+S.case.legacy_errors.map(e=>`<p class="qc-error">参照できない注釈 ${esc(e.id)}：${esc(e.reason)}</p>`).join('');
  const labels=[...new Set(S.case.legacy.map(r=>r.treatment).filter(Boolean))];$('treatmentSuggestions').innerHTML=labels.map(x=>`<option value="${esc(x)}"></option>`).join('');
}
async function loadCase(cid){
  if(S.formDirty&&!confirm('編集中の区間を反映せずに症例を切り替えますか？'))return;
  if(S.doc)await persist();
  const response=await api('/api/case?'+qs({id:cid}));S.case=response.case;S.doc=response.document;S.history=[];S.change=0;S.dirty=false;S.view=[0,S.case.duration_s];S.cursor=0;
  const stored=localStorage.getItem(backupKey());
  if(stored){try{const candidate=JSON.parse(stored);if(JSON.stringify(candidate)!==JSON.stringify(S.doc)&&candidate.revision>=S.doc.revision&&confirm('この症例にブラウザ内の下書きがあります。復元しますか？')){candidate.revision=S.doc.revision;await api('/api/validate',candidate);S.doc=candidate;changed();}}catch(e){message('下書きの復元を中止しました: '+e.message,true);}}
  $('caseTitle').textContent=`seq ${S.case.seq}`;$('caseMeta').textContent=`${S.case.time_origin.replace('T',' ')} · ${clock(S.case.duration_s)} · 原記録 ${S.case.raw_rows.toLocaleString()}行`;
  $('annotator').value=S.doc.annotator;$('studyRole').value=S.doc.study_role;$('scopeStart').value=S.doc.scope.start_s;$('scopeEnd').value=S.doc.scope.end_s;
  $('videoOffset').value=S.doc.media.offset_s;$('videoName').textContent=S.doc.media.filename?`参照ファイル：${S.doc.media.filename}（同じ動画を選び直してください）`:'';
  $('video').pause();$('video').removeAttribute('src');$('video').load();$('videoFile').value='';if(S.mediaUrl){URL.revokeObjectURL(S.mediaUrl);S.mediaUrl=null;}
  $('saveButton').disabled=false;$('saveStatus').textContent=S.dirty?'復元した下書き':S.doc.revision?`保存済み · r${S.doc.revision}`:'新規';$('undoButton').disabled=true;
  const codes=[...new Set(['20039','20101','20742',...Object.keys(S.case.codes)])];for(const [i,id] of ['signal1','signal2','signal3'].entries()){$(id).innerHTML=codes.map(c=>`<option value="${esc(c)}">${esc(c)} ${esc(codeNames[c]||'')} (${S.case.codes[c]||0})</option>`).join('');$(id).value=['20039','20101','20742'][i];}
  cancelEditor();renderCases();renderLegacy();renderRows();renderQuality(response.quality);await refreshSignals();
  if(S.case.invalid_timestamp_rows.length)message(`時刻不正の原記録 ${S.case.invalid_timestamp_rows.length}行。行番号は出力のmanifestに記録します。`,true);
  const url=new URL(location.href);url.searchParams.set('case',cid);window.history.replaceState(null,'',url);
}
function fresh(start,end){return {id:crypto.randomUUID(),start_s:start,end_s:end,clinical_status:'unknown',treatment_jp:'',instruments:[],instrument_status:'unknown',instrument_relation:'uncertain',observation_status:'unreviewed',certainty:'uncertain',evidence_sources:['unknown'],evidence_detail:'',boundary_certainty:'unknown',start_uncertainty_s:null,end_uncertainty_s:null,reviewed:false,notes:''};}
let editorRecord=null;
function cancelEditor(){S.selected=null;S.formDirty=false;editorRecord=null;$('editorFields').disabled=true;$('intervalForm').reset();$('formError').textContent='';$('editorTitle').textContent='区間を記録';if(S.doc)renderRows();}
function edit(r,isNew=false){
  if(S.formDirty&&!confirm('編集中の区間を反映せずに別の区間を開きますか？'))return false;
  editorRecord=clone(r);S.selected=isNew?null:r.id;S.formDirty=isNew;$('editorFields').disabled=false;$('formError').textContent='';$('editorTitle').textContent=isNew?'新しい区間':'区間を編集';
  const fields={start:r.start_s,end:r.end_s,clinicalStatus:r.clinical_status,treatment:r.treatment_jp,instrumentStatus:r.instrument_status,instrumentRelation:r.instrument_relation,observationStatus:r.observation_status,certainty:r.certainty,evidenceDetail:r.evidence_detail,boundaryCertainty:r.boundary_certainty,startUncertainty:r.start_uncertainty_s??'',endUncertainty:r.end_uncertainty_s??'',notes:r.notes};
  for(const [k,v]of Object.entries(fields))$(k).value=v;$('reviewed').checked=r.reviewed;
  for(const checkbox of $('evidenceOptions').querySelectorAll('input'))checkbox.checked=r.evidence_sources.includes(checkbox.value);
  selectedInstruments=new Set(r.instruments);$('instrumentSearch').value='';renderInstruments();updateConditional();$('deleteInterval').disabled=isNew;$('splitInterval').disabled=isNew;renderRows();return true;
}
function startNew(a,b){if(!S.doc)return;const start=a??Math.max(S.doc.scope.start_s,Math.min(S.cursor,S.doc.scope.end_s-.001));const end=b??Math.min(start+30,S.doc.scope.end_s);if(end<=start)throw new Error('対象範囲内の区間を指定してください。');edit(fresh(start,end),true);}
let selectedInstruments=new Set();
function renderInstruments(){const term=$('instrumentSearch').value.toLowerCase();$('instrumentOptions').innerHTML=S.config.instruments.filter(x=>x.toLowerCase().includes(term)).map(x=>`<label><input type="checkbox" value="${x}" ${selectedInstruments.has(x)?'checked':''}>${x}</label>`).join('');$('selectedInstruments').innerHTML=[...selectedInstruments].map(x=>`<span class="chip">${x}</span>`).join('');}
function updateConditional(){$('instrumentChooser').hidden=$('instrumentStatus').value!=='known';$('treatment').disabled=$('clinicalStatus').value!=='treatment';}
function collect(){if(!editorRecord)return null;const r=clone(editorRecord);Object.assign(r,{start_s:$('start').value===''?null:Number($('start').value),end_s:$('end').value===''?null:Number($('end').value),clinical_status:$('clinicalStatus').value,treatment_jp:$('clinicalStatus').value==='treatment'?$('treatment').value.trim():'',instrument_status:$('instrumentStatus').value,instruments:$('instrumentStatus').value==='known'?S.config.instruments.filter(x=>selectedInstruments.has(x)):[],instrument_relation:$('instrumentRelation').value,observation_status:$('observationStatus').value,certainty:$('certainty').value,evidence_sources:[...$('evidenceOptions').querySelectorAll('input:checked')].map(x=>x.value),evidence_detail:$('evidenceDetail').value.trim(),boundary_certainty:$('boundaryCertainty').value,start_uncertainty_s:$('startUncertainty').value===''?null:Number($('startUncertainty').value),end_uncertainty_s:$('endUncertainty').value===''?null:Number($('endUncertainty').value),reviewed:$('reviewed').checked,notes:$('notes').value});return r;}
async function applyEditor(){
  if(!editorRecord||!S.formDirty)return;
  const row=collect(),candidate=clone(S.doc);const index=candidate.intervals.findIndex(x=>x.id===row.id);if(index<0)candidate.intervals.push(row);else candidate.intervals[index]=row;
  try{await api('/api/validate',candidate);}catch(e){$('formError').textContent=e.message;throw e;}
  pushHistory();S.doc.intervals=candidate.intervals;S.selected=row.id;S.formDirty=false;editorRecord=clone(row);changed();$('editorTitle').textContent='区間を編集';$('formError').textContent='';$('deleteInterval').disabled=false;$('splitInterval').disabled=false;renderRows();await persist();
}
async function saveAll(){await applyEditor();if(S.doc&&!S.doc.revision&&!S.dirty)changed();await persist();}
async function exportDoc(mode){if(!S.doc)return;await saveAll();const batch=$('exportAll').checked;const result=await api(batch?'/api/export_all':'/api/export',{case_id:S.doc.case_id,revision:S.doc.revision,mode});if(result.quality)renderQuality(result.quality);const link=document.createElement('a');link.href=result.url;link.download=result.filename;document.body.append(link);link.click();link.remove();message(`${mode==='final'?'最終':'下書き'}ZIPを出力しました。${batch?result.case_count+'症例・'+result.interval_count+'区間。':''}Outputにも保存しています。`);}
function downloadDraft(){if(!S.doc)return;const draft=clone(S.doc);if(S.formDirty&&editorRecord){const r=collect();draft.intervals=draft.intervals.filter(x=>x.id!==r.id).concat([r]);}const url=URL.createObjectURL(new Blob([JSON.stringify(draft,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`local_draft_seq${S.case.seq}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
async function refreshSignals(){if(!S.case)return;const request=++S.signalRequest;const result=await api('/api/signal?'+qs({id:S.case.case_id,start:S.view[0],end:S.view[1],codes:['signal1','signal2','signal3'].map(id=>$(id).value).join(',')}));if(request!==S.signalRequest)return;S.signals=result;const total=Object.values(result).reduce((a,s)=>a+s.total,0),shown=Object.values(result).reduce((a,s)=>a+s.displayed,0);$('plotCounts').textContent=shown===total?`${shown.toLocaleString()}点`:`${total.toLocaleString()}点中 ${shown.toLocaleString()}点を表示（拡大で詳細）`;draw();}
function view(a,b){if(!S.case)return;const duration=S.case.duration_s;const width=Math.min(duration,Math.max(.01,b-a));const start=Math.max(0,Math.min(duration-width,a));S.view=[start,start+width];draw();return refreshSignals();}
function atX(clientX){const rect=$('timeline').getBoundingClientRect();const x=Math.max(100,Math.min(rect.width-18,clientX-rect.left));return S.view[0]+(x-100)/(rect.width-118)*(S.view[1]-S.view[0]);}
function draw(){
  const canvas=$('timeline'),width=canvas.clientWidth,height=365,dpr=window.devicePixelRatio||1;canvas.width=width*dpr;canvas.height=height*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,width,height);if(!S.case)return;
  const left=100,right=width-18,a=S.view[0],b=S.view[1],x=t=>left+(t-a)/(b-a)*(right-left);ctx.font='10px sans-serif';
  for(let i=0;i<=5;i++){const t=a+(b-a)*i/5;const xx=x(t);ctx.strokeStyle='#edf0f1';ctx.beginPath();ctx.moveTo(xx,18);ctx.lineTo(xx,337);ctx.stroke();ctx.fillStyle='#7b8c91';ctx.textAlign='center';ctx.fillText(clock(t),xx,356);}ctx.textAlign='left';
  const labels=[['全イベント',35],['対象コード',62],[$('signal1').value,108],[$('signal2').value,169],[$('signal3').value,230],['既存の注釈',282],['今回の注釈',319]];for(const [text,y]of labels){ctx.fillStyle='#587078';ctx.fillText(text,13,y);}
  const bins=S.case.event_bins,bw=S.case.bin_width_s;const visible=bins.map((v,i)=>({v,t:i*bw})).filter(z=>z.t+bw>=a&&z.t<=b);for(let j=0;j<2;j++){const max=Math.max(1,...visible.map(z=>z.v[j]));ctx.fillStyle=j?'#76bdb0':'#b0bfc6';for(const z of visible){const xx=Math.max(left,x(z.t)),ww=Math.min(right,x(z.t+bw))-xx;if(ww>0&&z.v[j])ctx.fillRect(xx,43+j*27-19*Math.log1p(z.v[j])/Math.log1p(max),Math.max(1,ww),19*Math.log1p(z.v[j])/Math.log1p(max));}}
  for(const [i,id]of ['signal1','signal2','signal3'].entries()){
    const data=S.signals[$(id).value]?.points||[],finite=data.filter(p=>p[1]!==null&&p[0]>=a&&p[0]<=b);const vals=finite.map(p=>p[1]);let lo=Math.min(...vals),hi=Math.max(...vals);if(!vals.length){ctx.fillStyle='#9ba8ac';ctx.fillText('この表示範囲に数値観測なし',left+8,108+i*61);continue;}
    if(hi===lo){lo-=.5;hi+=.5;}const top=85+i*61,bottom=125+i*61;ctx.font='8px sans-serif';ctx.fillStyle='#8b999e';ctx.fillText(sec(Math.max(...vals)),50,top+3);ctx.fillText(sec(Math.min(...vals)),50,bottom);ctx.font='10px sans-serif';ctx.fillStyle=['#347c98','#b59143','#128e82'][i];for(const [t,v]of finite){ctx.beginPath();ctx.arc(x(t),bottom-(v-lo)/(hi-lo)*(bottom-top),1.7,0,Math.PI*2);ctx.fill();}
  }
  function band(r,y,h,color,text){if(r.end_s<=a||r.start_s>=b)return;const start=Math.max(left,x(r.start_s)),end=Math.min(right,x(r.end_s));ctx.fillStyle=color;ctx.fillRect(start,y,end-start,h);if(end-start>33){ctx.save();ctx.beginPath();ctx.rect(start+3,y,end-start-6,h);ctx.clip();ctx.fillStyle='#fff';ctx.font='9px sans-serif';ctx.fillText(text,start+5,y+14);ctx.restore();}if(r.id===S.selected){ctx.strokeStyle='#16323b';ctx.lineWidth=2;ctx.strokeRect(start,y,end-start,h);ctx.lineWidth=1;}}
  for(const r of S.case.legacy)band(r,268,20,'#93a8b9',r.treatment||'既存注釈');
  const scope=S.doc.scope;ctx.fillStyle='#f0f2f3';ctx.fillRect(Math.max(left,x(scope.start_s)),301,Math.max(0,Math.min(right,x(scope.end_s))-Math.max(left,x(scope.start_s))),26);
  for(const r of S.doc.intervals)band(r,301,26,colors[r.clinical_status],r.treatment_jp||clinicalNames[r.clinical_status]);
  if(editorRecord){const s=Number($('start').value),e=Number($('end').value);if(e>s){ctx.fillStyle='#0c8c7918';ctx.fillRect(Math.max(left,x(s)),18,Math.max(0,Math.min(right,x(e))-Math.max(left,x(s))),313);ctx.strokeStyle='#108c7d';ctx.setLineDash([4,3]);for(const t of [s,e])if(t>=a&&t<=b){ctx.beginPath();ctx.moveTo(x(t),18);ctx.lineTo(x(t),335);ctx.stroke();}ctx.setLineDash([]);}}
  if(S.cursor>=a&&S.cursor<=b){ctx.strokeStyle='#2b515c';ctx.beginPath();ctx.moveTo(x(S.cursor),14);ctx.lineTo(x(S.cursor),335);ctx.stroke();}
}
let drag=null;
$('timeline').addEventListener('pointerdown',e=>{if(!S.doc)return;const rect=$('timeline').getBoundingClientRect();const yy=e.clientY-rect.top,t=atX(e.clientX);S.cursor=t;if(yy>=301&&yy<=327){const r=S.doc.intervals.find(r=>r.start_s<=t&&t<r.end_s);if(r){edit(r);return;}}drag={x:e.clientX,t};$('timeline').setPointerCapture(e.pointerId);draw();});
$('timeline').addEventListener('pointermove',e=>{if(!S.doc)return;S.cursor=atX(e.clientX);$('cursorTime').textContent=`${sec(Math.round(S.cursor*1000)/1000)} 秒 · ${clock(S.cursor)}`;draw();});
$('timeline').addEventListener('pointerup',guarded(async e=>{if(!drag)return;const from=drag;drag=null;if(Math.abs(e.clientX-from.x)>5){const t=atX(e.clientX),a=Math.max(S.doc.scope.start_s,Math.min(t,from.t)),b=Math.min(S.doc.scope.end_s,Math.max(t,from.t));if(b>a)startNew(Math.round(a*1000)/1000,Math.round(b*1000)/1000);}draw();}));
$('timeline').addEventListener('pointercancel',()=>drag=null);
$('caseSearch').addEventListener('input',renderCases);
$('caseList').addEventListener('click',guarded(async e=>{const item=e.target.closest('[data-case]');if(item)await loadCase(item.dataset.case);}));
$('intervalRows').addEventListener('click',e=>{const row=e.target.closest('[data-interval]');if(row)edit(S.doc.intervals.find(r=>r.id===row.dataset.interval));});
$('qualityList').addEventListener('click',guarded(async e=>{const row=e.target.closest('[data-qcid]');if(row)edit(S.doc.intervals.find(r=>r.id===row.dataset.qcid));const gap=e.target.closest('[data-gap]');if(gap)startNew(...gap.dataset.gap.split(',').map(Number));}));
$('legacyRows').addEventListener('click',e=>{const button=e.target.closest('[data-legacy]');if(!button)return;const old=S.case.legacy[Number(button.dataset.legacy)];const r=fresh(old.start_s,old.end_s);Object.assign(r,{clinical_status:old.treatment?'treatment':'unknown',treatment_jp:old.treatment||'',evidence_sources:['legacy_annotation'],evidence_detail:old.evidence||'',notes:`旧器具名: ${old.instrument||'記載なし'}\n旧確認状態: ${old.status||'不明'}\n${old.notes||''}`,legacy_reference:{id:old.id,source_sha256:S.case.legacy_source_sha256}});edit(r,true);});
$('intervalForm').addEventListener('input',e=>{if(!editorRecord)return;S.formDirty=true;if(e.target.id!=='reviewed')$('reviewed').checked=false;draw();});
$('intervalForm').addEventListener('change',e=>{if(!editorRecord)return;S.formDirty=true;if(e.target.id!=='reviewed')$('reviewed').checked=false;updateConditional();if(e.target.id==='boundaryCertainty'&&e.target.value==='exact'){$('startUncertainty').value=0;$('endUncertainty').value=0;}if(e.target.closest('#evidenceOptions')){if(e.target.value==='unknown'&&e.target.checked){for(const c of $('evidenceOptions').querySelectorAll('input'))if(c!==e.target)c.checked=false;}else if(e.target.checked)$('evidenceOptions').querySelector('[value="unknown"]').checked=false;}draw();});
$('instrumentSearch').addEventListener('input',e=>{e.stopPropagation();renderInstruments();});
$('instrumentOptions').addEventListener('change',e=>{if(e.target.type!=='checkbox')return;if(e.target.checked)selectedInstruments.add(e.target.value);else selectedInstruments.delete(e.target.value);S.formDirty=true;$('reviewed').checked=false;renderInstruments();});
$('intervalForm').addEventListener('submit',guarded(async e=>{e.preventDefault();await applyEditor();message('区間を保存しました。');}));
$('newInterval').addEventListener('click',guarded(()=>startNew()));$('cancelEdit').addEventListener('click',()=>{if(!S.formDirty||confirm('区間の未反映の編集を取り消しますか？'))cancelEditor();});
$('saveButton').addEventListener('click',guarded(saveAll));
$('fillGaps').addEventListener('click',guarded(async()=>{if(!S.doc)return;await applyEditor();const g=gapInfo();if(!g.gaps.length){message('未記入区間はありません。');return;}pushHistory();for(const gap of g.gaps)S.doc.intervals.push(fresh(gap.start_s,gap.end_s));changed();renderRows();await persist();message(`${g.gaps.length}区間を「不明・未確認」として追加しました。各区間を確認してください。`);}));
$('undoButton').addEventListener('click',guarded(async()=>{if(!S.history.length)return;if(S.formDirty&&!confirm('未反映の編集を取り消し、直前の区間変更を戻しますか？'))return;Object.assign(S.doc,S.history.pop());$('scopeStart').value=S.doc.scope.start_s;$('scopeEnd').value=S.doc.scope.end_s;cancelEditor();changed();$('undoButton').disabled=!S.history.length;renderRows();await persist();}));
$('deleteInterval').addEventListener('click',guarded(async()=>{if(!S.selected||!confirm('この区間を削除しますか？ 保存履歴には残ります。'))return;pushHistory();S.doc.intervals=S.doc.intervals.filter(r=>r.id!==S.selected);cancelEditor();changed();renderRows();await persist();}));
$('splitInterval').addEventListener('click',guarded(async()=>{await applyEditor();const r=S.doc.intervals.find(x=>x.id===S.selected),t=Math.round(S.cursor*1000)/1000;if(!r||t<=r.start_s||t>=r.end_s)throw new Error('区間内の時刻をタイムラインで指定してください。');pushHistory();const left=clone(r),right=clone(r);left.end_s=t;right.start_s=t;left.reviewed=right.reviewed=false;left.end_uncertainty_s=right.start_uncertainty_s=null;left.boundary_certainty=right.boundary_certainty='unknown';right.id=crypto.randomUUID();left.split_from_id=right.split_from_id=r.id;S.doc.intervals=S.doc.intervals.filter(x=>x.id!==r.id).concat([left,right]);changed();edit(left);renderRows();await persist();}));
for(const id of ['annotator','studyRole'])$(id).addEventListener('change',guarded(async()=>{if(!S.doc)return;S.doc.annotator=$('annotator').value.trim();S.doc.study_role=$('studyRole').value;changed();await persist();}));
$('scopeApply').addEventListener('click',guarded(async()=>{if(!S.doc)return;await applyEditor();const candidate=clone(S.doc);candidate.scope={start_s:Number($('scopeStart').value),end_s:Number($('scopeEnd').value)};await api('/api/validate',candidate);pushHistory();S.doc.scope=candidate.scope;changed();renderRows();await persist();}));
for(const id of ['signal1','signal2','signal3'])$(id).addEventListener('change',guarded(refreshSignals));
$('resetView').addEventListener('click',guarded(()=>S.case&&view(0,S.case.duration_s)));
for(const [id,factor]of [['zoomIn',.5],['zoomOut',2]])$(id).addEventListener('click',guarded(()=>{const mid=(S.view[0]+S.view[1])/2,half=(S.view[1]-S.view[0])*factor/2;return view(mid-half,mid+half);}));
$('zoomSelection').addEventListener('click',guarded(()=>{if(!editorRecord)throw new Error('まず区間を選択してください。');const a=Number($('start').value),b=Number($('end').value);if(!(b>a))throw new Error('開始・終了を確認してください。');const padding=(b-a)*.1;return view(a-padding,b+padding);}));
$('draftExport').addEventListener('click',guarded(async()=>{try{await exportDoc('draft');}catch(e){downloadDraft();throw new Error(e.message+' ブラウザの編集内容をJSONとして退避しました。');}}));
$('finalExport').addEventListener('click',guarded(()=>exportDoc('final')));
$('transferExport').addEventListener('click',guarded(async()=>{
  if(!S.doc)return;const button=$('transferExport');button.disabled=true;button.textContent='移行ZIPを作成中…';
  try{await saveAll();const result=await api('/api/transfer',{case_id:S.doc.case_id,revision:S.doc.revision,all_saved:$('exportAll').checked});
    const link=document.createElement('a');link.href=result.url;link.download=result.filename;document.body.append(link);link.click();link.remove();
    message(`${result.case_count}症例の移行ZIPを出力しました。別PCで展開し、Windows/Macの起動ファイルを開いてください。動画は別にコピーします。`);
  }finally{button.disabled=false;button.textContent='別PCへ移す ZIP ↓';}
}));
$('importJson').addEventListener('change',guarded(async e=>{try{if(!S.doc)throw new Error('先に症例を選択してください。');const file=e.target.files[0];if(!file)return;const doc=JSON.parse(await file.text());await api('/api/validate',doc);if(doc.case_id!==S.doc.case_id)throw new Error('現在の症例とJSONの症例が異なります。対応する症例を選択してください。');if(!confirm('現在の編集内容を、このJSONの内容に置き換えますか？ 既存の保存履歴は残ります。'))return;await S.saveChain;const rev=S.doc.revision;pushHistory();S.doc=doc;S.doc.revision=rev;changed();cancelEditor();$('annotator').value=doc.annotator;$('studyRole').value=doc.study_role;$('scopeStart').value=doc.scope.start_s;$('scopeEnd').value=doc.scope.end_s;$('videoOffset').value=doc.media.offset_s;renderRows();await persist();message('JSONを読み込み、保存しました。');}finally{e.target.value='';}}));
$('videoFile').addEventListener('change',guarded(async e=>{if(!S.doc)return;const file=e.target.files[0];if(!file)return;if(S.mediaUrl)URL.revokeObjectURL(S.mediaUrl);S.mediaUrl=URL.createObjectURL(file);$('video').src=S.mediaUrl;S.doc.media.filename=file.name;$('videoName').textContent=file.name;changed();await persist();}));
$('videoOffset').addEventListener('change',guarded(async()=>{if(!S.doc)return;const n=Number($('videoOffset').value);if(!Number.isFinite(n))throw new Error('動画の対応時刻を確認してください。');S.doc.media.offset_s=n;changed();await persist();}));
$('video').addEventListener('timeupdate',()=>{if(S.doc)$('videoTime').textContent=`ログ時刻 ${sec(Math.round(($('video').currentTime+S.doc.media.offset_s)*1000)/1000)} 秒`;});
$('videoToCursor').addEventListener('click',guarded(()=>{const t=S.cursor-(S.doc?.media.offset_s||0);if(!S.mediaUrl||t<0||t>$('video').duration)throw new Error('指定時刻に対応する動画がありません。動画と対応時刻を確認してください。');$('video').currentTime=t;}));
for(const [id,field]of [['startFromVideo','start'],['endFromVideo','end']])$(id).addEventListener('click',guarded(()=>{if(!S.mediaUrl||!editorRecord)throw new Error('動画と編集区間を開いてください。');$(field).value=Math.round(($('video').currentTime+S.doc.media.offset_s)*1000)/1000;S.formDirty=true;$('reviewed').checked=false;draw();}));
$('helpButton').addEventListener('click',()=>$('helpDialog').showModal());$('closeHelp').addEventListener('click',()=>$('helpDialog').close());
document.addEventListener('keydown',guarded(async e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();if(S.doc)await saveAll();return;}if(e.target.closest('input,textarea,select')||$('helpDialog').open)return;if(e.key.toLowerCase()==='n')startNew();if((e.key==='['||e.key===']')&&editorRecord){$(e.key==='['?'start':'end').value=Math.round(S.cursor*1000)/1000;S.formDirty=true;$('reviewed').checked=false;draw();}}));
window.addEventListener('beforeunload',e=>{if(S.formDirty||S.dirty){e.preventDefault();e.returnValue='';}});
new ResizeObserver(draw).observe($('timeline'));
async function init(){S.config=await api('/api/config');$('outputPath').textContent=S.config.output;S.cases=await api('/api/cases');renderCases();const requested=new URL(location.href).searchParams.get('case');if(requested&&S.cases.some(c=>c.case_id===requested))await loadCase(requested);else if(S.cases.length)await loadCase(S.cases[0].case_id);}
init().catch(e=>message(e.message,true));
