import {detect,loadDetector} from './detector.js';
export const step=0.25;

function waitFor(video, events, ready, signal, trigger) {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => done(new Error('The browser timed out while reading a frame. Try another MP4.')), 12000);
    const check = () => {if (ready()) done();};
    const failed = () => done(new Error('The browser could not read this clip. Try an H.264 MP4.'));
    const canceled = () => done(signal.reason);
    function done(error) {
      clearTimeout(timer);
      for (const name of events) video.removeEventListener(name, check);
      video.removeEventListener('error', failed);
      signal.removeEventListener('abort', canceled);
      if (error) reject(error); else resolve();
    }
    for (const name of events) video.addEventListener(name, check);
    video.addEventListener('error', failed);
    signal.addEventListener('abort', canceled, {once:true});
    try {trigger(); check();} catch (error) {done(error);}
  });
}
const nextPaint = () => new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve,0)));
export async function scan(data,report,signal){
  const decoder = document.createElement('video');
  decoder.muted = true;
  decoder.playsInline = true;
  decoder.preload = 'auto';
  decoder.className = 'scan-decoder';
  decoder.setAttribute('aria-hidden','true');
  document.body.append(decoder);
  let url;
  try {
    signal.throwIfAborted();
    url = URL.createObjectURL(data);
    await waitFor(decoder,['loadedmetadata'],() => decoder.readyState >= 1,signal,() => {decoder.src = url;});
    const {duration,videoWidth:width,videoHeight:height} = decoder;
    if (!Number.isFinite(duration) || duration <= 0 || !width || !height) {
      throw new Error('Could not read the video length or dimensions. Try a saved MP4 file.');
    }
    // Read large frames through a reusable canvas before passing them to the model.
    const factor = Math.min(1,1920/Math.max(width,height),1080/Math.min(width,height));
    let scratch, ctx;
    if (factor < 1) {
      scratch = document.createElement('canvas');
      scratch.width = Math.max(1,Math.round(width*factor));
      scratch.height = Math.max(1,Math.round(height*factor));
      ctx = scratch.getContext('2d',{alpha:false});
    }
    report('Loading detector…');
    await loadDetector();
    const count = Math.ceil(duration/step), found = [];
    for (let i = 0; i < count; i++) {
      signal.throwIfAborted();
      report('Checking frame '+(i+1)+' of '+count+'…',i/count);
      await nextPaint();
      const at = (i*step+Math.min(duration,(i+1)*step))/2;
      await waitFor(decoder,['seeked','loadeddata','canplay'],() => !decoder.seeking && decoder.readyState >= 2,signal,() => {decoder.currentTime = at;});
      if (scratch) {
        ctx.drawImage(decoder,0,0,scratch.width,scratch.height);
        const boxes = await detect(scratch,scratch.width,scratch.height);
        const sx = width/scratch.width, sy = height/scratch.height;
        for (const b of boxes) b.box = b.box.map((v,index)=>v*(index%2===0?sx:sy));
        found.push(boxes);
      } else found.push(await detect(decoder,width,height));
    }
    signal.throwIfAborted();
    return {media:url,width,height,duration,frames:found,file:data};
  } catch(error) {
    if(url)URL.revokeObjectURL(url);
    throw error;
  } finally {
    decoder.removeAttribute('src');decoder.load();decoder.remove();
  }
}
