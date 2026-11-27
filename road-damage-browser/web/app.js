import {detect,loadDetector} from './detector.js';
import {scan,step} from './scan.js';

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
let cleanup = () => {};
const swap = () => {cleanup();cleanup=()=>{};};
const page = (body,cls='') => {swap();root.innerHTML='<div class="wrap '+cls+'">'+body+'</div>';};
const title = s => document.title = s ? s+' · Road Damage Detection':'Road Damage Detection';
const btn = (id,i,t) => '<button class="glass" id="'+id+'">'+icon(i)+t+'</button>';
const time = t => Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0');
function home(){
 swap();title('');root.innerHTML='<div class="home"><div class="hero"><section><i class="rule"></i><h1>Road Damage<br>Detection</h1><p class="intro">I built this project to explore how computer vision could help with road inspections. Upload a road photo or dashcam clip to see how the model identifies possible cracks and potholes and marks their locations.</p><div class="buttons">'+btn('pickPhoto','photo','Upload photo')+btn('pickVideo','video','Upload video')+'</div><div class="home-links"><a href="/samples">Try a sample</a><a href="/model">Model &amp; Results</a></div><p class="home-status" id="msg" role="status"></p><input id="filePhoto" type="file" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" hidden><input id="fileVideo" type="file" accept="video/mp4,.mp4" hidden></section><figure class="print-stack"><img src="'+media+'cover.jpg" alt="A stack of road photographs with an illustrative pothole marked on the front print"></figure></div></div>';
 for(const [label,kind] of [['Photo','photo'],['Video','video']]){
  const file=q('#file'+label); q('#pick'+label).onclick=()=>file.click();
  file.onchange=async()=>{const f=file.files[0];if(!f)return;
   const msg=q('#msg');if(kind==='photo' && f.size>50*1024*1024){msg.textContent='Choose a photo smaller than 50 MB.';file.value='';return}
   history.pushState({},'', '/'+kind+'/');
   window.scrollTo(0,0);
   inspect(kind,f);
  };
 }
}
function samples(){
 title('Try a sample');page(bar()+'<header><i class="rule"></i><h1>Try a sample</h1><p class="lead">Choose a road photo or dashcam clip to explore the model’s results.</p></header><div class="sample-grid">'+[['photo','Road photo','huelva-road.jpg','An unmarked road near Huelva, photographed through a windshield'],['video','Dashcam clip','dashcam.jpg','A frame from the selected rural dashcam clip']].map(([k,h,file,alt])=>'<article class="sample-card"><p class="eyebrow">'+k+'</p><h2>'+h+'</h2><a tabindex="-1" aria-hidden="true" class="preview" href="/'+k+'/?sample=1"><img alt="'+alt+'" src="'+media+file+'">'+(k==='video'?'<span class="duration">0:27</span>':'')+'</a><a class="glass" href="/'+k+'/?sample=1">View '+k+' result</a></article>').join('')+'</div><p class="note">No upload needed.</p>');
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
 let observer;
 cleanup=()=>{observer?.disconnect();image.onload=null;image.onerror=null;if(!sample)URL.revokeObjectURL(d.media);};
 q('#threshold').oninput=e=>{t=Number(e.target.value);q('#value').textContent=t.toFixed(2);update()};
 q('#download').disabled=true;
 q('#download').onclick=()=>{const out=document.createElement('canvas');out.width=d.width;out.height=d.height;draw(out,null,false);out.toBlob(b=>{if(!b){q('#feedback').textContent='Could not export this image.';return}const url=URL.createObjectURL(b),a=document.createElement('a');a.href=url;a.download='road-damage-result.jpg';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60000)},'image/jpeg',.95)};
 image.onload=()=>{canvas.width=d.width;canvas.height=d.height;update();q('#download').disabled=false;observer=new ResizeObserver(()=>draw(canvas,selected,true));observer.observe(canvas)};
 image.onerror=()=>q('#feedback').textContent='Could not load the photo. Reopen the sample or upload the file again.';
 image.src=d.media;
}
function drawBoxes(ctx,W,H,detections,scale=1) {
  const font=Math.max(12*scale,H*.025);
  ctx.font='600 '+font+'px Georgia';ctx.textBaseline='top';
  for(const {label,score,box:[x,y,right,bottom]} of detections) {
    const text=label+' '+score.toFixed(2), pad=3*scale, width=Math.min(W,ctx.measureText(text).width+2*pad);
    const left=Math.max(0,Math.min(x,W-width)),top=Math.max(0,y-font-2*pad);
    ctx.lineWidth=2*scale;ctx.strokeStyle='#ffd29c';ctx.strokeRect(x,y,right-x,bottom-y);
    ctx.fillStyle='#f7d1a7';ctx.fillRect(left,top,width,font+2*pad);
    ctx.fillStyle='#261a13';ctx.fillText(text,left+pad,top+pad,width-2*pad);
  }
}
function ranges(frames,duration) {
  const result=[];
  for(let i=0;i<frames.length;i++){
    if(!frames[i].length)continue;
    const start=i;
    while(i+1<frames.length && frames[i+1].length)i++;
    result.push([start*step,Math.min(duration,(i+1)*step)]);
  }
  return result;
}
function video(d,sample){
  page(bar(sample)+'<header class="video-heading"><i class="rule"></i><h1>Video Results</h1></header><div class="media-frame video-frame"><video controls playsinline preload="metadata" aria-label="Road video with predicted damage"></video>'+(sample?'':'<canvas aria-hidden="true"></canvas>')+'</div><div class="timeline" id="timeline" aria-label="Moments with detections"><div class="track"></div><div class="times"><span>0:00</span><span>'+time(d.duration)+'</span></div></div><div class="buttons video-actions"><button class="toggle" role="switch" aria-checked="true" aria-controls="timeline">Video detection markers <span class="switch" aria-hidden="true"></span></button>'+(sample?'<a class="glass" id="saveVideo" download="road-damage-result.mp4">'+icon('download')+'Download result</a>':btn('saveVideo','download','Download result'))+'</div><p class="small" id="feedback" role="status">'+(sample?'':'Detections are sampled at 4 frames per second. The download is silent.')+'</p>','result');
  const player=q('video'),track=q('.track'),save=q('#saveVideo'),feedback=q('#feedback');
  player.src=d.media;
  const markers=sample?d.markers:ranges(d.frames,d.duration);
  for(const [start,end] of markers){
    const b=document.createElement('button');b.className='mark';
    b.style.left=(start/d.duration*100)+'%';b.style.width=((end-start)/d.duration*100)+'%';
    const jump=sample?start:(start+Math.min(end,start+step))/2;
    b.title='Jump to detection at '+jump.toFixed(2)+' seconds';b.setAttribute('aria-label',b.title);
    b.onclick=()=>{const seek=()=>{player.currentTime=jump;};if(player.readyState>=1)seek();else player.addEventListener('loadedmetadata',seek,{once:true});};
    track.append(b);
  }
  if(!markers.length){track.className='small';track.textContent='No moments with detections in this clip.';}
  q('.toggle').onclick=e=>{const b=e.currentTarget,on=b.getAttribute('aria-checked')==='true';b.setAttribute('aria-checked',String(!on));q('#timeline').hidden=on;};
  player.onerror=()=>{feedback.textContent='This browser could not play the video.';};
  if(sample){save.href=d.media;return;}
  const overlay=q('canvas'),ctx=overlay.getContext('2d'),controller=new AbortController();
  const backingScale=Math.min(1,1920/Math.max(d.width,d.height));
  overlay.width=Math.max(1,Math.round(d.width*backingScale));
  overlay.height=Math.max(1,Math.round(d.height*backingScale));
  let frame=0,exportUrl,resizeObserver,releaseExport,downloadStarted=false;
  const paint=(at)=>{
    ctx.clearRect(0,0,overlay.width,overlay.height);
    const index=Math.max(0,Math.min(d.frames.length-1,Math.floor(at/step)));
    ctx.save();
    ctx.scale(overlay.width/d.width,overlay.height/d.height);
    drawBoxes(ctx,d.width,d.height,d.frames[index],d.width/Math.max(1,overlay.clientWidth));
    ctx.restore();
  };
  const sync=()=>{
    const scale=Math.min(player.clientWidth/d.width,player.clientHeight/d.height);
    if(scale<=0)return;
    overlay.style.width=d.width*scale+'px';overlay.style.height=d.height*scale+'px';
    paint(player.currentTime);
  };
  for(const event of ['loadeddata','seeked','timeupdate'])player.addEventListener(event,sync);
  resizeObserver=new ResizeObserver(sync);resizeObserver.observe(player);
  if('requestVideoFrameCallback' in player){
    const onFrame=(_,metadata)=>{if(!player.seeking)paint(metadata.mediaTime);frame=player.requestVideoFrameCallback(onFrame);};
    frame=player.requestVideoFrameCallback(onFrame);
  }
  cleanup=()=>{
    controller.abort();
    if(frame)player.cancelVideoFrameCallback(frame);
    resizeObserver.disconnect();player.pause();URL.revokeObjectURL(d.media);
    if(exportUrl)URL.revokeObjectURL(exportUrl);
    // Let the browser keep reading the export after the user starts a download.
    if(!downloadStarted)void releaseExport?.();
  };
  save.onclick=async()=>{
    if(!isSecureContext || !('VideoEncoder' in window) || !('VideoDecoder' in window)){
      feedback.textContent='To save an annotated MP4, open this app over HTTPS in a browser that supports video export.';
      return;
    }
    save.disabled=true;
    feedback.textContent='Preparing MP4…';
    try {
      const {makeMp4}=await import('./video-export.js');
      const {file,release}=await makeMp4(d,drawBoxes,message=>{feedback.textContent=message;},controller.signal);
      releaseExport=release;
      if(controller.signal.aborted)await release();
      controller.signal.throwIfAborted();
      exportUrl=URL.createObjectURL(file);
      const link=document.createElement('a');
      link.href=exportUrl;link.download='road-damage-result.mp4';
      link.onclick=()=>{downloadStarted=true;};
      link.className='glass';link.innerHTML=icon('download')+'Save annotated MP4';
      save.replaceWith(link);link.focus();
      feedback.textContent='Your annotated MP4 is ready to save.';
    } catch(error) {if(!controller.signal.aborted)feedback.textContent=error.name==='QuotaExceededError'?'This device ran out of browser storage during export. Free some space or try a shorter clip.':error.message||'Could not create the MP4.';}
    finally{save.disabled=false;}
  };
}
function state(kind,sample){
  title((kind==='photo'?'Photo':'Video')+' Results');
  page(bar(sample)+'<section class="state"><div class="spinner" aria-hidden="true"></div><h1>Preparing '+(sample?'sample':'your')+' '+kind+'</h1><p role="status">Opening '+kind+'…</p><progress max="1" aria-label="Video scan progress" hidden></progress></section>');
  const message=q('[role=status]'),progress=q('progress'),controller=new AbortController();
  cleanup=()=>controller.abort();
  return {signal:controller.signal,report:(text,fraction)=>{
    if(controller.signal.aborted)return;
    message.textContent=text;
    if(fraction!==undefined){progress.hidden=false;progress.value=fraction;}
  }};
}
function problem(message,sample=false){
  page(bar(sample)+'<section class="state"><i class="rule"></i><h1>Result unavailable</h1><p role="alert">'+esc(message)+'</p><a class="glass" href="'+(sample?'/samples/':'/')+'">'+(sample?'Back to samples':'Choose another file')+'</a></section>');
}
async function inspect(kind,file){
  const {signal,report}=state(kind,false);
  let url;
  try{
    if(kind==='photo'){
      const image=new Image();url=URL.createObjectURL(file);image.src=url;
      try{await image.decode();}catch{throw new Error('Choose a photo saved as JPG, PNG, or WebP.');}
      const {naturalWidth:width,naturalHeight:height}=image;
      if(width*height>25000000)throw new Error('Choose a photo under 25 megapixels.');
      report('Loading detector…');await loadDetector();signal.throwIfAborted();
      report('Checking your photo…');
      const detections=await detect(image,width,height);signal.throwIfAborted();
      photo({media:url,width,height,detections},false);
    }else{
      const result=await scan(file,report,signal);
      url=result.media;signal.throwIfAborted();video(result,false);
    }
  }catch(error){if(url)URL.revokeObjectURL(url);if(!signal.aborted)problem(error.message);}
}
async function showSample(kind){
  const {signal}=state(kind,true);
  try{
    const response=await fetch(media+kind+'.json',{signal});
    if(!response.ok)throw new Error('Could not open this sample.');
    const data=await response.json();signal.throwIfAborted();
    if(kind==='photo')photo({...data,media:media+'huelva-road.jpg'},true);
    else video({...data,media:media+'road-clip-result.mp4'},true);
  }catch(error){if(!signal.aborted)problem(error.message,true);}
}
function route(){
  const [name]=location.pathname.split('/').filter(Boolean);
  if(name==='samples')samples();
  else if(name==='model')model();
  else if(name==='photo'||name==='video'){
    if(new URLSearchParams(location.search).get('sample')==='1')showSample(name);
    else {title((name==='photo'?'Photo':'Video')+' Results');problem('Choose a file from the homepage to see its results.');}
  }else home();
}
window.addEventListener('popstate',route);
window.addEventListener('pagehide',event=>{if(!event.persisted)swap();});
route();
