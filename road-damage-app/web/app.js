const root = document.querySelector('#app');
const q = s => root.querySelector(s);
const media = '/samples/media/';
const paths = {
 photo:'<rect x="2" y="3" width="20" height="17" rx="3"/><circle cx="8" cy="9" r="1.4"/><path d="m3 18 6-6 4 4 3-3 5 5"/>',
 video:'<rect x="2" y="4" width="20" height="16" rx="3"/><path d="m10 9 5 3-5 3z"/>',
 list:'<path d="M8 6h12M8 12h12M8 18h12"/><circle cx="4" cy="6" r=".6"/><circle cx="4" cy="12" r=".6"/><circle cx="4" cy="18" r=".6"/>',
 download:'<path d="M12 3v12m-5-5 5 5 5-5M3 16v4h18v-4"/>',
 slider:'<path d="M3 7h8m5 0h5M3 17h3m5 0h10"/><circle cx="13" cy="7" r="2.5"/><circle cx="8" cy="17" r="2.5"/>',
 back:'<path d="m10 5-7 7 7 7M3 12h18"/>'
};
const icon = n => '<svg viewBox="0 0 24 24" aria-hidden="true">'+paths[n]+'</svg>';
const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const bar = (samples=false) => '<nav class="top"><a href="'+(samples?'/samples':'/')+'">'+icon('back')+(samples?'Back to samples':'Back to home')+'</a><span class="brand">Road Damage Detection</span></nav>';
const page = (body,cls='') => {root.innerHTML='<div class="wrap '+cls+'">'+body+'</div>';};
const title = s => document.title = s ? s+' · Road Damage Detection':'Road Damage Detection';
const btn = (id,i,t) => '<button class="glass" id="'+id+'">'+icon(i)+t+'</button>';
const time = t => Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0');
function home(){
 title('');root.innerHTML='<div class="home"><div class="hero"><section><i class="rule"></i><h1>Road Damage<br>Detection</h1><p class="intro">I built this project to explore how computer vision could help with road inspections. Upload a road photo or dashcam clip to see how the model identifies possible cracks and potholes and marks their locations.</p><div class="buttons">'+btn('pickPhoto','photo','Upload photo')+btn('pickVideo','video','Upload video')+'</div><div class="home-links"><a href="/samples">Try a sample</a><a href="/model">Model &amp; Results</a></div><p class="home-status" id="msg" role="status"></p><input id="filePhoto" type="file" accept="image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif" hidden><input id="fileVideo" type="file" accept="video/mp4,video/quicktime,.mp4,.mov" hidden></section><figure class="print-stack"><img src="'+media+'cover.jpg" alt="A stack of road photographs with an illustrative pothole marked on the front print"></figure></div></div>';
 for(const [label,kind] of [['Photo','photo'],['Video','video']]){
  const file=q('#file'+label); q('#pick'+label).onclick=()=>file.click();
  file.onchange=async()=>{const f=file.files[0];if(!f)return;
   const msg=q('#msg');if(f.size>50*1024*1024){msg.textContent='Choose a file smaller than 50 MB.';file.value='';return}
   for(const b of root.querySelectorAll('button'))b.disabled=true;
   msg.textContent='Uploading '+(kind==='photo'?'photo':'clip')+'…';
   try{const res=await fetch('/api/uploads/'+kind,{method:'POST',headers:{'X-Road-Upload':'1','Content-Type':'application/octet-stream'},body:f});const data=await res.json();if(!res.ok)throw Error(data.detail||'Could not upload this file.');location.assign('/'+kind+'/'+data.key)}
   catch(e){msg.textContent=e.message;for(const b of root.querySelectorAll('button'))b.disabled=false;file.value=''}
  };
 }
}
function samples(){
 title('Try a sample');page(bar()+'<header><i class="rule"></i><h1>Try a sample</h1><p class="lead">Choose a road photo or dashcam clip to explore the model’s results.</p></header><div class="sample-grid">'+[['photo','Road photo','huelva-road.jpg','An unmarked road near Huelva, photographed through a windshield'],['video','Dashcam clip','dashcam.jpg','A frame from the selected rural dashcam clip']].map(([k,h,file,alt])=>'<article class="sample-card"><p class="eyebrow">'+k+'</p><h2>'+h+'</h2><a tabindex="-1" aria-hidden="true" class="preview" href="/'+k+'/sample"><img alt="'+alt+'" src="'+media+file+'">'+(k==='video'?'<span class="duration">0:27</span>':'')+'</a><a class="glass" href="/'+k+'/sample">View '+k+' result</a></article>').join('')+'</div><p class="note">No upload needed.</p>');
}
function model(){
 title('Model & Results');page(bar()+'<header class="doc-header"><h1>Model &amp; Results</h1><i class="rule"></i><p class="lead">I trained and evaluated models on RDD2022 to see how well they could locate cracks and potholes in road images. The final demo uses YOLO26s to mark possible damage in uploaded photos and dashcam clips.</p></header><div class="evidence-grid"><article class="evidence"><p class="eyebrow">Model choice</p><h2>Why YOLO26s</h2><p>I chose YOLO26s because it ran faster than YOLO26m with a small drop in validation performance. At an input size of 640, the small model reached <strong>31.55% mAP50–95</strong>, compared with <strong>32.04%</strong> for the medium model. On the same Tesla T4, model inference averaged <strong>8.7 ms per image</strong> for YOLO26s and <strong>24.1 ms</strong> for YOLO26m. I chose the smaller model because that speed difference mattered for a demo that processes video frame by frame.</p></article><article class="evidence"><p class="eyebrow">Held-out results</p><h2>Held-out test results</h2><div class="stats"><div class="stat"><b>60.06%</b> mAP50</div><div class="stat"><b>31.42%</b> mAP50–95</div><div class="stat"><b>3,846</b> test images</div></div></article></div><section class="classes"><h2>What the model looks for</h2><p>The model looks for four types of road damage: <strong>longitudinal cracks</strong> that run along the road, <strong>transverse cracks</strong> that run across it, <strong>alligator cracks</strong> that form a web-like pattern, and <strong>potholes</strong> where part of the road surface has broken away.</p><p>It draws a box around each area it flags so you know where to look. Some damage may go unmarked, and some boxes may be incorrect.</p></section>');
}
function photo(d,sample){
 page(bar(sample)+'<h1 class="sr">Photo results</h1><div class="media-frame photo-frame"><canvas aria-label="Road photo with model predictions"></canvas></div><div class="photo-actions">'+btn('summary','list','Interactive photo summary')+btn('download','download','Download result')+btn('confidence','slider','Confidence control')+'</div><section class="panel" id="summaryPanel"><h2>Interactive photo summary</h2><div id="findings"></div></section><section class="panel" id="confidencePanel" hidden><div class="scale"><label for="threshold">Minimum confidence</label><output for="threshold" id="value">0.25</output></div><input type="range" id="threshold" min="0.25" max="1" step="0.01" value="0.25"><p class="small">Raise the threshold to show only higher-scoring predictions.</p></section><p id="feedback" class="small" role="status"></p>','result');
 const canvas=q('canvas'),image=new Image(),rows=d.detections.map((b,i)=>({...b,i}));let t=.25,selected=null;
 const visible=()=>rows.filter(r=>r.score>=t);
 function draw(target,highlight,display){
  if(!image.naturalWidth)return;
  const ctx=target.getContext('2d'),W=target.width,H=target.height;
  ctx.drawImage(image,0,0,W,H);
  const scale=display?Math.max(W/Math.max(1,target.getBoundingClientRect().width),1):Math.max(1,W/1050);
  const fs=Math.max(13*scale,Math.min(W,H)*.018);ctx.font='600 '+fs+'px Georgia';ctx.textBaseline='top';
  for(const b of visible()){
   const [x,y,r,bot]=b.box; const label=b.label+' '+b.score.toFixed(2);
   ctx.globalAlpha=(highlight!==null&&highlight!==b.i)?.28:1;
   ctx.lineWidth=Math.max(2,scale*(highlight===b.i?3:1.7));ctx.strokeStyle='#ffd29c';ctx.strokeRect(x,y,r-x,bot-y);
   const pad=3*scale, width=Math.min(W,ctx.measureText(label).width+pad*2),height=fs+pad*2;
   const left=Math.max(0,Math.min(x,W-width)),top=Math.max(0,y-height);
   ctx.fillStyle='#f7d1a7';ctx.fillRect(left,top,width,height);ctx.fillStyle='#261a13';ctx.fillText(label,left+pad,top+pad,width-pad*2);
  }ctx.globalAlpha=1;
 }
 function update(){
  const seen=visible();if(!seen.some(b=>b.i===selected))selected=null;
  const findings=q('#findings');findings.replaceChildren();
  for(const b of seen){const row=document.createElement('button');row.className='finding';row.setAttribute('aria-pressed',String(b.i===selected));row.innerHTML=esc(b.label)+'<span class="score">'+b.score.toFixed(2)+'</span>';row.onclick=()=>{selected=selected===b.i?null:b.i;update();q('#findings').children[seen.indexOf(b)].focus()};findings.append(row)}
  if(!seen.length)findings.textContent='No detections at this confidence.';
  canvas.setAttribute('aria-label','Road photo. '+(seen.length?seen.map(b=>b.label+' '+b.score.toFixed(2)).join(', '):'No detections at this confidence.'));
  draw(canvas,selected,true);
 }
 for(const id of ['summary','confidence']){
  const button=q('#'+id),panel=q('#'+id+'Panel');button.setAttribute('aria-controls',panel.id);button.setAttribute('aria-expanded',String(!panel.hidden));
  button.onclick=()=>{panel.hidden=!panel.hidden;button.setAttribute('aria-expanded',String(!panel.hidden))};
 }
 q('#threshold').oninput=e=>{t=Number(e.target.value);q('#value').textContent=t.toFixed(2);update()};
 q('#download').disabled=true;
 q('#download').onclick=()=>{const out=document.createElement('canvas');out.width=d.width;out.height=d.height;draw(out,null,false);out.toBlob(b=>{if(!b){q('#feedback').textContent='Could not export this image.';return}const url=URL.createObjectURL(b),a=document.createElement('a');a.href=url;a.download='road-damage-result.jpg';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60000)},'image/jpeg',.95)};
 image.onload=()=>{canvas.width=d.width;canvas.height=d.height;update();q('#download').disabled=false;new ResizeObserver(()=>draw(canvas,selected,true)).observe(canvas)};
 image.onerror=()=>q('#feedback').textContent='Could not load the photo. Reopen the sample or upload the file again.';
 image.src=d.media;
}
function video(d,sample){
 page(bar(sample)+'<header class="video-heading"><i class="rule"></i><h1>Video Results</h1></header><div class="media-frame video-frame"><video controls playsinline preload="metadata" aria-label="Annotated road video"></video></div><div class="timeline" id="timeline" aria-label="Moments with detections"><div class="track"></div><div class="times"><span>0:00</span><span>'+time(d.duration)+'</span></div></div><div class="buttons video-actions"><button class="toggle" role="switch" aria-checked="true" aria-controls="timeline">Video detection markers <span class="switch" aria-hidden="true"></span></button><a class="glass" id="saveVideo" download="road-damage-result.mp4">'+icon('download')+'Download result</a></div><p class="small" id="feedback" role="status"></p>','result');
 const player=q('video');player.src=d.media;q('#saveVideo').href=d.media;
 const track=q('.track');
 d.markers.forEach(([start,end])=>{const b=document.createElement('button');b.className='mark';b.style.left=(start/d.duration*100)+'%';b.style.width=((end-start)/d.duration*100)+'%';b.title=b.ariaLabel='Jump to detection at '+start.toFixed(2)+' seconds';b.onclick=()=>{const seek=()=>{player.currentTime=start;q('#feedback').textContent='Showing detection at '+start.toFixed(2)+' seconds.'};if(player.readyState>=1)seek();else player.addEventListener('loadedmetadata',seek,{once:true})};track.append(b)});
 if(!d.markers.length){track.replaceChildren();track.className='small';track.textContent='No moments with detections in this clip.'}
 q('.toggle').onclick=e=>{const b=e.currentTarget,on=b.getAttribute('aria-checked')==='true';b.setAttribute('aria-checked',String(!on));q('#timeline').hidden=on};
 player.onerror=()=>{q('#feedback').textContent='This browser could not play the video. Use Download result to open the MP4.'};
}
async function result(kind,key){
 const sample=key==='sample',label=kind==='photo'?'Photo':'Video';title(label+' Results');
 page(bar(sample)+'<section class="state"><div class="spinner"></div><h1>Preparing your '+kind+'</h1><p role="status">Opening result…</p></section>');
 try{while(true){
   const res=await fetch('/api/jobs/'+kind+'/'+encodeURIComponent(key),{cache:'no-store'}),j=await res.json();
   if(!res.ok||j.status==='error')throw Error(j.detail||j.message||'Could not open this result.');
   if(j.status==='ready'){(kind==='photo'?photo:video)(j.result,sample);return}
   q('[role=status]').textContent=j.message;await new Promise(r=>setTimeout(r,650));
  }}catch(e){page(bar(sample)+'<section class="state"><i class="rule"></i><h1>Result unavailable</h1><p role="alert">'+esc(e.message)+'</p><a class="glass" href="/">Choose another file</a></section>')}
}
const [route,key]=location.pathname.split('/').filter(Boolean);
if(!route)home();else if(route==='samples')samples();else if(route==='model')model();else if(['photo','video'].includes(route)&&key)result(route,key);else home();
