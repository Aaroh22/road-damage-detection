import {
  ALL_FORMATS, Input, BlobSource, CanvasSink, canEncodeVideo, QUALITY_HIGH,
  Output, Mp4OutputFormat, CanvasSource
} from './mediabunny.min.mjs';
import {step} from './scan.js';
import {openExportTarget} from './export-target.js';

export async function makeMp4(result,draw,report,signal) {
  const input = new Input({source:new BlobSource(result.file),formats:ALL_FORMATS});
  let output, destination;
  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track || !(await track.canDecode())) throw new Error('Your browser cannot export this clip’s video format. Try an H.264 MP4.');
    const canvas = document.createElement('canvas');
    const even = size => Math.max(2,2*Math.round(size/2));
    const cap = Math.min(1,1920/Math.max(result.width,result.height),1080/Math.min(result.width,result.height));
    let width = even(result.width), height = even(result.height);
    const canEncode = (w,h) => canEncodeVideo('avc',{width:w,height:h,bitrate:QUALITY_HIGH});
    if (!(await canEncode(width,height))) {
      width = even(result.width*cap);
      height = even(result.height*cap);
      if (!(await canEncode(width,height))) {
        throw new Error('This browser can show detections but cannot save an annotated MP4.');
      }
      report('Preparing MP4 at '+width+' × '+height+'…');
    }
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d',{alpha:false});
    signal.throwIfAborted();
    destination = await openExportTarget();
    signal.throwIfAborted();
    output = new Output({format:new Mp4OutputFormat(destination.format),target:destination.target});
    const source = new CanvasSource(canvas,{codec:'avc',bitrate:QUALITY_HIGH});
    output.addVideoTrack(source);
    await output.start();
    const sink = new CanvasSink(track,{width,height,fit:'fill',poolSize:2});
    let count=0;
    const first = await input.getFirstTimestamp();
    for await (const sample of sink.canvases()) {
      signal.throwIfAborted();
      const at=Math.max(0,sample.timestamp-first);
      ctx.drawImage(sample.canvas,0,0,canvas.width,canvas.height);
      const detections = result.frames[Math.min(result.frames.length-1,Math.floor(at/step))];
      ctx.save();ctx.scale(canvas.width/result.width,canvas.height/result.height);
      draw(ctx,result.width,result.height,detections);
      ctx.restore();
      await source.add(at, sample.duration);
      if (++count % 15===0) {
        report('Preparing MP4 · '+Math.min(99,Math.round(at/result.duration*100))+'%');
        await new Promise(resolve=>setTimeout(resolve,0));
      }
    }
    if(!count)throw new Error('The video had no frames to export.');
    source.close();
    await output.finalize();
    signal.throwIfAborted();
    return {file:await destination.file(),release:destination.release};
  } catch(error) {
    if(output)await output.cancel().catch(()=>{});
    if(destination)await destination.discard();
    throw error;
  } finally {input.dispose();}
}
