import {AppendOnlyStreamTarget, StreamTarget} from './mediabunny.min.mjs';

// Prefer a temporary file so a long MP4 need not fit in browser memory.
export async function openExportTarget() {
  let directory, name, writable;
  const release = async () => {
    if (directory && name) await directory.removeEntry(name).catch(()=>{});
  };
  try {
    const root = await navigator.storage?.getDirectory?.();
    if (root) {
      directory = await root.getDirectoryHandle('rdd-exports',{create:true});
      // Clear older temporary files, including ones left behind after a download.
      const stale = Date.now()-86400000;
      for await (const entry of directory.values()) {
        if (entry.kind!=='file' || !entry.name.startsWith('result-')) continue;
        try {
          if ((await entry.getFile()).lastModified<stale) await directory.removeEntry(entry.name);
        } catch {}
      }
      name = 'result-'+crypto.randomUUID()+'.mp4';
      const handle=await directory.getFileHandle(name,{create:true});
      writable=await handle.createWritable();
      return {
        target:new StreamTarget(writable,{chunked:true,chunkSize:2*1024*1024}),
        format:{fastStart:false},
        file:()=>handle.getFile(),
        discard:async()=>{await writable.abort().catch(()=>{});await release();},
        release
      };
    }
  } catch (error) {
    if (writable) await writable.abort().catch(()=>{});
    await release();
    if (error.name==='QuotaExceededError') throw error;
  }

  const parts=[];
  const stream=new WritableStream({write(chunk){parts.push(chunk.slice());}});
  const clear=()=>{parts.length=0;};
  return {
    target:new AppendOnlyStreamTarget(stream),
    format:{fastStart:'fragmented'},
    file:()=>{const file=new Blob(parts,{type:'video/mp4'});clear();return file;},
    discard:clear,
    release:clear
  };
}
