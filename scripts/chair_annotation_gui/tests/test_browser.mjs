/* Chrome DevTools integration tests. Node >=22; no npm packages required. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
const base=process.argv[2]||'http://127.0.0.1:8766';
const output=process.argv[3]||'/tmp/chair-annotation-browser-results';
const realUrl=process.argv[4];
fs.mkdirSync(output,{recursive:true});
const target=await (await fetch('http://127.0.0.1:9225/json/new?about:blank',{method:'PUT'})).json();
const ws=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
let seq=0;const pending=new Map(),exceptions=[],dialogs=[];
ws.onmessage=event=>{const msg=JSON.parse(event.data);if(msg.id){const p=pending.get(msg.id);if(!p)return;pending.delete(msg.id);msg.error?p.reject(new Error(JSON.stringify(msg.error))):p.resolve(msg.result);}
  if(msg.method==='Runtime.exceptionThrown')exceptions.push(msg.params.exceptionDetails);
  if(msg.method==='Page.javascriptDialogOpening'){dialogs.push(msg.params.message);command('Page.handleJavaScriptDialog',{accept:true}).catch(()=>{});}
};
function command(method,params={}){return new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await command('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true,userGesture:true});if(r.exceptionDetails)throw new Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value;}
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function wait(expression){let last;for(let i=0;i<160;i++){try{if(await evaluate(expression))return;}catch(e){last=e;}await sleep(50);}throw new Error('Timeout: '+expression+(last?' '+last.message:''));}
async function fill(id,value,type='input'){await evaluate(`{const e=document.getElementById(${JSON.stringify(id)});e.value=${JSON.stringify(value)};e.dispatchEvent(new Event(${JSON.stringify(type)},{bubbles:true}));}`);}
async function click(id){await evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);}
async function check(selector,value=true){await evaluate(`{const e=document.querySelector(${JSON.stringify(selector)});if(!e)throw new Error('Missing checkbox');e.checked=${value};e.dispatchEvent(new Event('change',{bubbles:true}));}`);}
async function screenshot(name){const r=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(output,name),Buffer.from(r.data,'base64'));}
const checks=[];
try{
  await command('Page.enable');await command('Runtime.enable');await command('DOM.enable');
  await command('Emulation.setDeviceMetricsOverride',{width:1600,height:1100,deviceScaleFactor:1,mobile:false});
  await command('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:output});
  await command('Page.navigate',{url:base});
  await wait("typeof S !== 'undefined' && S.doc && Object.keys(S.signals).length>0");
  await sleep(100);
  assert.ok(!(await evaluate("document.getElementById('message').classList.contains('error') && document.getElementById('message').textContent.includes('is not a function')")), 'No startup application errors');
  assert.equal(await evaluate('S.case.seq'),1);
  assert.equal(await evaluate('S.config.instruments.length'),40);
  assert.equal(await evaluate('S.doc.case_id'),'2000-01-01__TEST__Chair_log_seq_1');
  if(await evaluate('S.doc.intervals.length'))await evaluate('S.doc.intervals=[];changed();persist()');
  const canvas=await evaluate("{const r=document.getElementById('timeline').getBoundingClientRect();({x:r.left,y:r.top,w:r.width})}");
  const px=t=>canvas.x+100+(canvas.w-118)*t/30;
  await command('Input.dispatchMouseEvent',{type:'mousePressed',x:px(2),y:canvas.y+150,button:'left',clickCount:1});
  await command('Input.dispatchMouseEvent',{type:'mouseMoved',x:px(10),y:canvas.y+150,button:'left',buttons:1});
  await command('Input.dispatchMouseEvent',{type:'mouseReleased',x:px(10),y:canvas.y+150,button:'left',clickCount:1});
  assert.equal(await evaluate("Number(document.getElementById('start').value)"),2);
  assert.equal(await evaluate("Number(document.getElementById('end').value)"),10);
  await click('cancelEdit');checks.push('Mouse drag selects exact time interval on raw timeline');
  await fill('annotator','GUI TEST','change');await evaluate('S.saveChain');
  await evaluate(`(async()=>{
    const canvas=document.createElement('canvas');canvas.width=100;canvas.height=60;const ctx=canvas.getContext('2d');ctx.fillStyle='teal';ctx.fillRect(0,0,100,60);
    const stream=canvas.captureStream(10),chunks=[];const recorder=new MediaRecorder(stream,{mimeType:'video/webm'});
    const stopped=new Promise(resolve=>recorder.onstop=resolve);recorder.ondataavailable=e=>chunks.push(e.data);recorder.start();await new Promise(r=>setTimeout(r,300));recorder.stop();await stopped;stream.getTracks().forEach(t=>t.stop());
    const transfer=new DataTransfer();transfer.items.add(new File(chunks,'synthetic.webm',{type:'video/webm'}));const input=document.getElementById('videoFile');input.files=transfer.files;input.dispatchEvent(new Event('change',{bubbles:true}));
  })()`);
  await wait("document.getElementById('video').readyState>=1 && S.doc.media.filename==='synthetic.webm' && !S.dirty");
  await fill('videoOffset','5','change');await evaluate('S.saveChain');
  await click('newInterval');await fill('start','0');await fill('end','12');
  await click('startFromVideo');assert.equal(await evaluate("Number(document.getElementById('start').value)"),5);await fill('start','0');checks.push('Local video opens and offset maps playback time to log seconds');
  await fill('clinicalStatus','treatment','change');await fill('treatment','う蝕除去');
  await fill('instrumentStatus','known','change');
  await check('#instrumentOptions input[value="contra-angle_handpiece"]');
  await check('#instrumentOptions input[value="dental_mirror"]');
  await fill('instrumentRelation','sequential','change');
  await fill('observationStatus','sparse','change');
  await check('#evidenceOptions input[value="video"]');await fill('evidenceDetail','synthetic.mp4 00:00–00:12');
  await fill('certainty','confirmed','change');await fill('boundaryCertainty','approximate','change');
  await fill('startUncertainty','1.5');await fill('endUncertainty','2');await fill('notes','日本語テスト・倍率は別途確認');
  await check('#reviewed');await click('applyInterval');
  await wait('S.doc.intervals.length===1 && !S.dirty && !S.formDirty');
  const first=await evaluate('S.doc.intervals[0]');assert.equal(first.treatment_jp,'う蝕除去');assert.deepEqual(first.instruments,['contra-angle_handpiece','dental_mirror']);assert.equal(first.reviewed,true);assert.equal(first.start_uncertainty_s,1.5);checks.push('GUI edit: Japanese treatment, exact English instruments, evidence and uncertainty');
  await click('finalExport');await wait("document.getElementById('message').classList.contains('error') && document.getElementById('message').textContent.includes('未記入')");checks.push('Final export blocks incomplete timeline');
  await click('fillGaps');await wait('S.doc.intervals.length===2 && !S.dirty');
  const gap=await evaluate('S.doc.intervals.find(x=>x.start_s===12)');assert.equal(gap.clinical_status,'unknown');assert.equal(gap.reviewed,false);assert.equal(gap.instrument_status,'unknown');
  await evaluate(`document.querySelector('[data-interval="${gap.id}"]').click()`);
  await fill('observationStatus','no_target_events','change');await fill('notes','臨床内容は確認できず不明として保持');await check('#reviewed');await click('applyInterval');await wait('!S.dirty && S.doc.intervals.every(x=>x.reviewed)');
  checks.push('Gap remains explicit unknown, not an idle/negative label');
  await click('finalExport');await wait("document.getElementById('message').textContent.includes('最終ZIPを出力')");
  let zip;
  for(let i=0;i<80;i++){zip=fs.readdirSync(output).find(x=>x.endsWith('.zip'));if(zip)break;await sleep(50);}
  assert.ok(zip,'ZIP downloaded');
  const doc=JSON.parse(execFileSync('python3',['-c','import sys,zipfile; print(zipfile.ZipFile(sys.argv[1]).read("annotations.json").decode())',path.join(output,zip)],{encoding:'utf8'}));
  assert.equal(doc.intervals.length,2);assert.equal(doc.intervals[0].treatment_jp,'う蝕除去');assert.deepEqual(doc.intervals[0].instruments,first.instruments);checks.push('Final ZIP download preserves labels, fields and unknown intervals');
  await screenshot('synthetic_annotation.png');
  const dialogsBefore=dialogs.length,oldOrigin=await evaluate('performance.timeOrigin');await command('Page.reload');await wait(`performance.timeOrigin>${oldOrigin} && typeof S !== 'undefined' && S.doc && S.doc.intervals.length===2 && Object.keys(S.signals).length>0`);assert.equal(dialogs.length,dialogsBefore,'Saved documents should reload without a spurious recovery prompt');assert.deepEqual(await evaluate('S.doc.intervals[0].instruments'),first.instruments);checks.push('Reload restores persisted document without false recovery prompt');
  // Round-trip through the actual file input and save a new revision.
  const importPath=path.join(output,'roundtrip.json');fs.writeFileSync(importPath,JSON.stringify(doc));const revision=await evaluate('S.doc.revision');
  const root=await command('DOM.getDocument');const fileNode=await command('DOM.querySelector',{nodeId:root.root.nodeId,selector:'#importJson'});await command('DOM.setFileInputFiles',{nodeId:fileNode.nodeId,files:[importPath]});
  await wait(`S.doc.revision>${revision} && !S.dirty`);assert.deepEqual(await evaluate('S.doc.intervals'),doc.intervals);checks.push('JSON file import round-trip through GUI');
  // Server must reject stale revisions and foreign/missing write tokens.
  const cfg=await (await fetch(base+'/api/config')).json();let stale=clone(doc);
  let response=await fetch(base+'/api/save',{method:'POST',headers:{'Content-Type':'application/json','X-Annotation-Token':cfg.token},body:JSON.stringify(stale)});assert.equal(response.status,409);
  response=await fetch(base+'/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(stale)});assert.equal(response.status,403);checks.push('Stale write rejected; missing CSRF token rejected');
  // Test split and undo with explicit boundaries, without changing final export above.
  await evaluate(`document.querySelector('[data-interval="${gap.id}"]').click();S.cursor=20;`);await click('splitInterval');await wait('S.doc.intervals.length===3 && !S.dirty');assert.equal(await evaluate('S.doc.intervals.filter(x=>!x.reviewed).length'),2);await click('undoButton');await wait('S.doc.intervals.length===2 && !S.dirty');checks.push('Split resets review state; undo restores original intervals');
  await check('#exportAll');await click('finalExport');await wait("document.getElementById('message').textContent.includes('症例・')");
  let collection;
  for(let i=0;i<80;i++){collection=fs.readdirSync(output).find(x=>x.startsWith('annotations_collection_')&&x.endsWith('.zip'));if(collection)break;await sleep(50);}
  assert.ok(collection,'Collection ZIP downloaded');
  const manifest=JSON.parse(execFileSync('python3',['-c','import sys,zipfile; print(zipfile.ZipFile(sys.argv[1]).read("collection_manifest.json").decode())',path.join(output,collection)],{encoding:'utf8'}));
  assert.equal(manifest.actual_cases,1);assert.equal(manifest.actual_intervals,2);checks.push('All saved cases export with combined CSV and case manifest');
  // Optional real case: visualization only. No annotation changes or saves.
  if(realUrl){const old=await evaluate('performance.timeOrigin');await command('Page.navigate',{url:realUrl});await wait(`performance.timeOrigin>${old} && typeof S !== 'undefined' && S.doc && Object.keys(S.signals).length>0`);await sleep(100);assert.equal(await evaluate('S.doc.revision'),0);assert.equal(await evaluate("document.getElementById('message').classList.contains('error')"),false);await screenshot('real_case_readonly.png');checks.push('Real case loaded read-only with raw observations and legacy reference');}
  assert.deepEqual(exceptions,[]);
  fs.writeFileSync(path.join(output,'browser_test_results.json'),JSON.stringify({passed:true,checks,exceptions,dialog_count:dialogs.length},null,2));
  console.log(JSON.stringify({passed:true,checks,output},null,2));
}catch(e){await screenshot('failure.png').catch(()=>{});fs.writeFileSync(path.join(output,'browser_test_results.json'),JSON.stringify({passed:false,checks,error:e.stack,exceptions},null,2));console.error(e);process.exitCode=1;}
finally{ws.close();}
function clone(x){return JSON.parse(JSON.stringify(x));}
