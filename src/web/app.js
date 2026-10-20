'use strict';
const $ = id => document.getElementById(id);
const state = {token:'', datasets:[], result:null, run:'', selected:null, image:null, view:null, playing:false, lastCompleted:null, frame:7, activeJob:null};
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = (v, digits=2) => v == null || !Number.isFinite(Number(v)) ? '—' : Number(v).toLocaleString('en-US', {maximumFractionDigits:digits, minimumFractionDigits:digits});
const percent = v => v == null ? '—' : `${number(v*100, 1)}%`;
const elapsed = s => `${String(Math.floor(s/60)).padStart(2,'0')}:${String(Math.floor(s%60)).padStart(2,'0')}`;
const artifact = (name, download=false) => `/api/artifact?run=${encodeURIComponent(state.run)}&name=${encodeURIComponent(name)}${download?'&download=1':''}`;
function notify(message) { $('notice').textContent=message; $('notice').hidden=!message; }
async function api(url, body) {
  const response=await fetch(url, body ? {method:'POST',headers:{'Content-Type':'application/json','X-Console-Token':state.token},body:JSON.stringify(body)} : {});
  const value=await response.json();
  if(!response.ok) throw new Error(value.error || '请求失败');
  return value;
}
function panel(name) {
  $('observationPanel').hidden=name!=='observation'; $('validationPanel').hidden=name!=='validation';
  $('observeTab').classList.toggle('active',name==='observation'); $('validationTab').classList.toggle('active',name==='validation');
  if(name==='observation') requestAnimationFrame(draw);
}
$('observeTab').onclick=()=>panel('observation'); $('validationTab').onclick=()=>panel('validation');
function datasetChanged() {
  const d=state.datasets.find(d=>d.id===$('dataset').value);
  $('reference').replaceChildren();
  if(!d) { $('startButton').disabled=true; return; }
  d.frames.forEach((f,i)=>$('reference').add(new Option(`f${i} · 第 ${i+1} 张`, i)));
  $('reference').value=String(Math.min(7,d.frames.length-1));
  $('frameCount').textContent=d.frames.length;
  $('catalog').checked=d.id==='data/rst19';
  referenceChanged();
}
function referenceChanged() {
  const d=state.datasets.find(d=>d.id===$('dataset').value);
  $('referenceName').textContent=d?.frames[Number($('reference').value)] || '未发现 FITS 图像。将数据放入项目 data 目录后刷新页面。';
}
$('dataset').onchange=datasetChanged; $('reference').onchange=referenceChanged;
async function history() {
  const runs=await api('/api/runs');
  const select=$('history'); select.replaceChildren(new Option('选择运行记录',''));
  runs.forEach(r=>select.add(new Option(`${r.kind==='benchmark'?'[验证]':'[识别]'} ${r.id}`,r.id)));
  const chosen=state.historyRun||state.run;
  if(runs.some(r=>r.id===chosen))select.value=chosen;
  return runs;
}
$('refreshHistory').onclick=()=>history().catch(e=>notify(e.message));
$('history').onchange=()=>{ if($('history').value)loadResult($('history').value,false).catch(e=>notify(e.message)); };
async function start(kind) {
  notify('');
  try {
    const job=await api('/api/start',{kind,dataset:$('dataset').value,reference_frame:Number($('reference').value),catalog:$('catalog').checked,workers:Number($('workers').value),executor:$('executor').value});
    showJob(job);
  } catch(e) { notify(e.message); }
}
$('runForm').onsubmit=e=>{e.preventDefault();panel('observation');start('analysis');};
$('benchmarkButton').onclick=()=>start('benchmark');
$('stopButton').onclick=async()=>{try {showJob(await api('/api/stop',{}));}catch(e){notify(e.message);}};
function showJob(job) {
  if(!job)return;
  state.activeJob=job;
  const busy=['running','stopping'].includes(job.state);
  $('startButton').disabled=busy || !state.datasets.length; $('benchmarkButton').disabled=busy;
  $('dataset').disabled=busy; $('reference').disabled=busy; $('catalog').disabled=busy;
  $('workers').disabled=busy; $('executor').disabled=busy;
  $('stopButton').hidden=!busy; $('stopButton').disabled=job.state==='stopping';
  $('runState').textContent=({running:job.kind==='benchmark'?'验证实验运行中':'正在分析',stopping:'正在停止',succeeded:'运行完成',failed:'运行失败',cancelled:'任务已停止'})[job.state];
  $('elapsed').textContent=elapsed(job.elapsed_seconds);
  const lines=job.logs||[];
  const stages=lines.filter(line=>/阶段：|天基分析|分析进度|CPU 执行|基准实验/.test(line));
  $('runStage').textContent=busy?(stages.at(-1)||'启动计算环境，准备输入数据…'):(job.state==='succeeded'?(job.kind==='benchmark'?'本次验证实验已完成，计时包含环境启动。':'本次识别已完成，计时包含环境启动、识别与导出。'):'查看运行日志后可以重新启动。');
  $('logs').textContent=lines.join('\n')||'进程启动中…';
  $('logs').scrollTop=$('logs').scrollHeight;
  if(busy && state.result?.kind==='analysis')$('resultBadge').textContent='上次结果 · 新任务运行中';
  if(!busy && state.result?.kind==='analysis' && state.run!==job.run_id)$('resultBadge').textContent='已保存结果';
}
async function poll() {
  try {
    const {job}=await api('/api/status'); showJob(job);
    if(job && ['succeeded','failed','cancelled'].includes(job.state) && state.lastCompleted!==job.run_id) {
      state.lastCompleted=job.run_id;
      await history();
      if(job.state==='succeeded')await loadResult(job.run_id,true);
      else notify(job.state==='failed'?'运行未完成。请展开左侧日志查看原因，当前展示的历史结果不会被替换。':'任务已停止；已有结果仍可查看。');
    }
  } catch(e) { notify(`工作台连接中断：${e.message}`); }
  setTimeout(poll,1000);
}
async function loadResult(id, fresh) {
  const sequence=(state.loadSequence||0)+1;state.loadSequence=sequence;
  const data=await api(`/api/result?id=${encodeURIComponent(id)}`);
  if(sequence!==state.loadSequence)return;
  state.historyRun=id;
  $('history').value=id;
  notify('');
  if(data.kind==='benchmark') {renderBenchmark(data.report,id);panel('validation');return;}
  state.run=id; state.result=data; state.selected=null; state.image=null; state.view=null;
  panel('observation');
  const r=data.report, s=r.star_catalog;
  $('resultBadge').textContent=fresh?'本次运行结果':'已保存结果';
  $('resultSource').textContent=`${r.dataset_id} / ${r.reference_file}`;
  $('starCount').textContent=number(s.n_confirmed_stars,0);
  $('starDescription').textContent=`f${s.reference_frame} · 至少 ${s.min_hits}/${s.verification_frames.length} 帧核验`;
  $('targetCount').textContent=number(r.target_count,0);
  $('magnitude').textContent=number(s.faintest_confirmed_star?.mag,2);
  $('magnitudeDescription').textContent=s.calibration ? (/Gaia/.test(s.magnitude_system)?'Gaia G 参照估计':'真值参照估计 · 波段未指定') : '未完成绝对定标';
  $('registrationRms').textContent=number(r.registration.rms_median_px,4);
  $('skyLabel').textContent=`IMAGE / f${s.reference_frame}`;
  state.frame=s.reference_frame;
  $('timeline').min=r.analysis_frames[0]; $('timeline').max=r.analysis_frames.at(-1); $('timeline').value=state.frame;
  $('timelineLabel').textContent=`f${state.frame}`;
  $('canvasCaption').textContent=`背景固定为 f${s.reference_frame}；滑块仅切换轨迹坐标。像素 x=列、y=行，0 起算。滚轮缩放 · 拖动平移。`;
  $('tableCount').textContent=`/ ${r.targets.length}`;
  $('tracksBody').innerHTML=r.targets.length?r.targets.map(t=>`<tr data-id="${t.target_id}"><td><button aria-label="查看候选 T${String(t.target_id).padStart(2,'0')}">T${String(t.target_id).padStart(2,'0')}</button></td><td>${t.n_frames}</td><td>${number(t.speed_sky_px_per_s)}</td><td>${number(t.fit_rms_px,3)}</td><td>${percent(t.trail_evidence?.consistent_fraction)}</td><td>f${t.frames[0]} → f${t.frames[1]}</td></tr>`).join(''):'<tr><td colspan="6" class="table-empty">当前数据未检出通过判据的运动候选。</td></tr>';
  $('tracksBody').querySelectorAll('[data-id]').forEach(row=>row.onclick=()=>selectTarget(Number(row.dataset.id)));
  $('downloads').innerHTML=['targets.csv','stars.csv','report.json'].map(name=>`<a href="${artifact(name,true)}">↓ ${name==='targets.csv'?'轨迹 CSV':name==='stars.csv'?'恒星 CSV':'完整报告'}</a>`).join('');
  const seconds=data.receipt?.elapsed_seconds;
  $('scienceNotes').innerHTML=`<p>${escape(r.algorithm_scope)}</p><p>${escape(s.note)}</p><p>${escape(s.calibration_note)}</p><p>分析：${number(r.analysis_seconds,1)} s；含出图：${number(r.total_seconds,1)} s；容器启动至导出完成：${number(seconds,1)} s。未记录的计时显示为 —。</p><p>本次输入 ${r.input_frame_count} 帧；识别使用目标真值：${r.truth_used_for_detection?'是':'否'}。逐文件 SHA-256、参数与定标配对见完整报告。</p><p>运行记录：${escape(id)}</p>`;
  $('imageLinks').innerHTML=['faintest-star.png',...(r.diagnostics?.mode==='space'?['motion-tracks.png']:[])].map(name=>`<a href="${artifact(name)}" target="_blank" rel="noopener">${name==='faintest-star.png'?'查看最暗恒星标注图':'查看全序列轨迹成果图'} ↗</a>`).join('');
  if(r.viewer) {
    const image=new Image();
    image.onload=()=>{if(state.run!==id)return;state.image=image;resetView();$('emptyState').hidden=true;};
    image.onerror=()=>{if(state.run===id)notify('原图预览加载失败，可使用下方成果图与导出文件。');};
    image.src=artifact(r.viewer.preview);
  } else {
    $('emptyState').hidden=false;
    $('emptyState').querySelector('h3').textContent='历史报告未包含交互预览';
    $('emptyState').querySelector('p').textContent='可查看下方成果图，或重新运行生成交互结果。';
  }
  if(r.targets.length)selectTarget(r.targets[0].target_id);
  else {$('selectedTitle').textContent='暂无候选';$('targetSummary').textContent='当前运行没有通过判据的运动轨迹。';$('evidence').replaceChildren();$('cutouts').replaceChildren();$('fluxChart').replaceChildren();}
  draw();
}
function selectTarget(id) {
  const data=state.result;if(data?.kind!=='analysis')return;
  const t=data.report.targets.find(t=>t.target_id===id);if(!t)return;
  state.selected=id;
  $('selectedTitle').textContent=`T${String(id).padStart(2,'0')}`;
  $('targetSummary').textContent=`f${t.frames[0]} → f${t.frames[1]} · ${number(t.time_span_s,2)} s 观测弧段`;
  const rows=[['有效观测',`${t.n_frames} 帧`],['拟合残差',`${number(t.fit_rms_px,3)} px`],['相对运动速度',`${number(t.speed_sky_px_per_s)} px/s`],['拖尾相容比例',percent(t.trail_evidence?.consistent_fraction)],['区间观测覆盖',percent(t.temporal_evidence?.coverage_fraction)],['最长连续漏帧',t.temporal_evidence?`${t.temporal_evidence.max_consecutive_missing} 帧`:'—']];
  $('evidence').innerHTML=rows.map(([label,value])=>`<div><dt>${label}</dt><dd>${escape(value)}</dd></div>`).join('');
  const crops=data.report.viewer?.target_cutouts?.[id]||[];
  $('cutouts').innerHTML=crops.length?crops.map(c=>`<figure><a href="${artifact(c.file)}" target="_blank" rel="noopener"><img src="${artifact(c.file)}" alt="T${id} 在 f${c.frame} 的原始图像切片"></a><figcaption>f${c.frame}</figcaption></figure>`).join(''):'<p class="muted">此历史报告没有导出候选切片。</p>';
  const observations=data.tracks[id]||[];
  const flux=observations.map(p=>p.flux);
  if(flux.length){
    const low=Math.min(...flux),high=Math.max(...flux),f0=observations[0].frame,f1=observations.at(-1).frame;
    const points=observations.map(p=>`${10+260*(p.frame-f0)/Math.max(f1-f0,1)},${70-50*(p.flux-low)/Math.max(high-low,1)}`).join(' ');
    $('fluxChart').innerHTML=`<polyline fill="none" stroke="#387362" stroke-width="1.8" points="${points}"/><text x="10" y="12" fill="#77817b" font-size="9">${number(high,0)}</text><text x="10" y="87" fill="#77817b" font-size="9">f${f0}</text><text x="250" y="87" fill="#77817b" font-size="9">f${f1}</text>`;
  }
  $('tracksBody').querySelectorAll('[data-id]').forEach(row=>row.classList.toggle('selected',Number(row.dataset.id)===id));
  draw();
}
function resetView(){
  const viewer=state.result?.report?.viewer;
  if(viewer)state.view={x:0,y:0,w:viewer.preview_width||viewer.width,h:viewer.preview_height||viewer.height};
  draw();
}
function transform() {
  const v=state.view,c=$('sky');if(!v)return null;
  const w=c.clientWidth,h=c.clientHeight,scale=Math.min(w/v.w,h/v.h);
  return {scale,ox:(w-v.w*scale)/2-v.x*scale,oy:(h-v.h*scale)/2-v.y*scale};
}
function draw() {
  const c=$('sky'),ctx=c.getContext('2d'),dpr=window.devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;
  if(!w||!h)return;
  c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);ctx.scale(dpr,dpr);ctx.fillStyle='#080e13';ctx.fillRect(0,0,w,h);
  if(!state.image||!state.view||state.result?.kind!=='analysis')return;
  const {scale:s,ox,oy}=transform(),data=state.result,viewer=data.report.viewer;
  ctx.drawImage(state.image,ox,oy,(viewer.preview_width||viewer.width)*s,(viewer.preview_height||viewer.height)*s);
  ctx.strokeStyle='#b7d77c77';ctx.lineWidth=.7;
  if($('showStars').checked){ctx.beginPath();for(const [x,y]of data.stars){ctx.moveTo(ox+(x+.5)*s+2,oy+(y+.5)*s);ctx.arc(ox+(x+.5)*s,oy+(y+.5)*s,2,0,Math.PI*2);}ctx.stroke();}
  if($('showTracks').checked)for(const [id,points]of Object.entries(data.tracks)){
    const chosen=Number(id)===state.selected;ctx.strokeStyle=chosen?'#d9f36a':'#7abca6b0';ctx.lineWidth=chosen?2:1;ctx.beginPath();
    points.forEach((p,i)=>{const x=ox+(p.x+.5)*s,y=oy+(p.y+.5)*s;if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();
    const point=points.find(p=>p.frame===state.frame);
    if(point){ctx.beginPath();ctx.arc(ox+(point.x+.5)*s,oy+(point.y+.5)*s,chosen?5:3,0,Math.PI*2);ctx.stroke();}
    const anchor=point||points[0];
    if(chosen){ctx.fillStyle='#e1f8a7';ctx.font='10px Consolas';ctx.fillText(`T${String(id).padStart(2,'0')}`,ox+anchor.x*s+8,oy+anchor.y*s-8);}
  }
  const faint=data.report.star_catalog.faintest_confirmed_star;
  if(faint){const x=ox+(faint.x+.5)*s,y=oy+(faint.y+.5)*s;ctx.strokeStyle='#f4b270';ctx.lineWidth=1;ctx.strokeRect(x-5,y-5,10,10);ctx.fillStyle='#f4b270';ctx.font='9px Consolas';ctx.fillText(faint.mag==null?'FAINTEST':`m≈${number(faint.mag)}`,x+9,y+3);}
  $('canvasTag').textContent=`f${viewer.reference_frame} BACKGROUND / ${(viewer.width/state.view.w).toFixed(1)}× / DETECTOR PIXELS`;
}
function zoom(factor){if(!state.view)return;const v=state.view,base=state.result.report.viewer.width;const nw=Math.max(base/20,Math.min(base*1.5,v.w*factor));const nh=v.h*nw/v.w;state.view={x:v.x+(v.w-nw)/2,y:v.y+(v.h-nh)/2,w:nw,h:nh};draw();}
$('zoomIn').onclick=()=>zoom(.75);$('zoomOut').onclick=()=>zoom(1.33);$('resetView').onclick=resetView;
$('showStars').onchange=draw;$('showTracks').onchange=draw;
$('sky').addEventListener('wheel',e=>{if(state.view){e.preventDefault();zoom(e.deltaY>0?1.12:.89);}},{passive:false});
let drag=null;
$('sky').onpointerdown=e=>{if(state.view){drag={x:e.clientX,y:e.clientY,v:{...state.view},moved:false};$('sky').setPointerCapture(e.pointerId);}};
$('sky').onpointermove=e=>{if(!drag)return;const s=transform().scale,dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.moved ||= Math.hypot(dx,dy)>4;state.view={...drag.v,x:drag.v.x-dx/s,y:drag.v.y-dy/s};draw();};
$('sky').onpointerup=e=>{
  if(drag&&!drag.moved&&state.result?.tracks){const rect=$('sky').getBoundingClientRect(),tr=transform();let nearest=null,distance=20;
    for(const [id,points]of Object.entries(state.result.tracks))for(const p of points){const d=Math.hypot(e.clientX-rect.left-(tr.ox+p.x*tr.scale),e.clientY-rect.top-(tr.oy+p.y*tr.scale));if(d<distance){distance=d;nearest=Number(id);}}
    if(nearest)selectTarget(nearest);
  }drag=null;
};
$('sky').onpointercancel=()=>{drag=null;};
$('sky').onkeydown=e=>{if(e.key==='+'||e.key==='=')zoom(.8);if(e.key==='-')zoom(1.25);if(e.key==='0')resetView();};
$('timeline').oninput=()=>{state.frame=Number($('timeline').value);$('timelineLabel').textContent=`f${state.frame}`;draw();};
$('playButton').onclick=()=>{state.playing=!state.playing;$('playButton').textContent=state.playing?'Ⅱ':'▶';};
setInterval(()=>{if(!state.playing||!state.result||state.result.kind!=='analysis')return;const slider=$('timeline');slider.value=Number(slider.value)>=Number(slider.max)?slider.min:Number(slider.value)+1;slider.oninput();},500);
new ResizeObserver(draw).observe($('sky'));
function renderBenchmark(r, identifier) {
  const names={full:'完整方法',without_trail:'移除拖尾核验',without_gaps:'移除漏帧容忍'},scenes={clean:'常规亮目标',gaps:'间歇漏帧',faint:'弱源与高噪声'};
  const download=`/api/artifact?run=${encodeURIComponent(identifier)}&name=benchmark.json&download=1`;
  $('benchmarkResults').innerHTML=`<div class="table-heading"><h3>配对消融对照</h3><div class="downloads"><a href="${download}">↓ 完整实验 JSON</a></div></div><div class="table-scroll"><table><thead><tr><th>实验方法</th><th>命中 TP</th><th>误报 FP</th><th>漏检 FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>${r.aggregate.map(a=>`<tr><td>${names[a.variant]}</td><td>${a.tp}</td><td>${a.fp}</td><td>${a.fn}</td><td>${percent(a.precision)}</td><td>${percent(a.recall)}</td><td>${number(a.f1,3)}</td></tr>`).join('')}</tbody></table></div><p>${escape(r.scope)}</p><p>随机种子：${r.seeds.join(' / ')} · ${r.image_shape.join(' × ')} px · 每组 ${r.frames_per_case} 帧 · 完整实验 ${number(r.total_seconds,1)} s</p><h3>逐场景结果</h3><div class="table-scroll"><table><thead><tr><th>种子</th><th>场景</th><th>方法</th><th>TP / FP / FN</th><th>定位 RMSE px</th><th>匹配轨迹可见覆盖</th></tr></thead><tbody>${r.cases.map(c=>`<tr><td>${c.seed}</td><td>${scenes[c.scenario]}</td><td>${names[c.variant]}</td><td>${c.tp} / ${c.fp} / ${c.fn}</td><td>${number(c.localization_rmse_px,3)}</td><td>${percent(c.matched_visible_coverage)}</td></tr>`).join('')}</tbody></table></div><h3>指标口径与限制</h3><p>${escape(r.metrics_policy)}</p><p>${escape(r.controls)}</p>${r.limitations.map(l=>`<p>· ${escape(l)}</p>`).join('')}<p>逐文件源码校验值、完整参数、随机种子与运行时间见实验 JSON。</p>`;
}
async function boot(){
  try {
    const b=await api('/api/bootstrap');state.token=b.token;state.datasets=b.datasets;
    b.datasets.forEach(d=>$('dataset').add(new Option(d.name,d.id)));
    if(b.datasets.some(d=>d.id==='data/rst19'))$('dataset').value='data/rst19';
    datasetChanged();$('runtime').textContent=b.runtime==='docker'?'CPU · OFFLINE':'LOCAL PYTHON';
    const runs=await history();
    if(b.job){showJob(b.job);if(!['running','stopping'].includes(b.job.state))state.lastCompleted=b.job.run_id;}
    const latest=runs.find(r=>r.kind==='analysis');
    if(latest)await loadResult(latest.id,false);
  }catch(e){notify(e.message);}poll();
}
boot();
