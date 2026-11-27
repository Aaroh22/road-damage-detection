"""Start with python app.py; pass --lan to use the same app from a phone."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
from pathlib import Path
import shutil
import socket
import tempfile
from threading import Lock
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import engine

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets'
LIMIT=50*1024*1024
TMP=tempfile.TemporaryDirectory(prefix='road-inspections-')
JOBS={}; guard=Lock(); executor=ThreadPoolExecutor(max_workers=1)
TTL=3600

def expire():
    with guard:
        keys=[k for k,v in JOBS.items() if v['status'] in ('ready','error') and time.time()-v['time']>TTL]
        for key in keys:
            JOBS.pop(key);shutil.rmtree(Path(TMP.name)/key,ignore_errors=True)

@asynccontextmanager
async def life(app):
    async def housekeep():
        while True:
            await asyncio.sleep(60);expire()
    task=asyncio.create_task(housekeep())
    try:yield
    finally:
        task.cancel()
        executor.shutdown(wait=True,cancel_futures=True);TMP.cleanup()

app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None,lifespan=life)
app.mount('/web',StaticFiles(directory=ROOT/'web'),name='web')
app.mount('/samples/media',StaticFiles(directory=ASSETS),name='samples')
@app.get('/')
@app.get('/samples')
@app.get('/model')
@app.get('/photo/{key}')
@app.get('/video/{key}')
def page(key:str=''):
    return FileResponse(ROOT/'web/index.html',headers={'Cache-Control':'no-cache'})

def update(key,**fields):
    with guard:JOBS[key].update(time=time.time(),**fields)

def run(kind,key,folder):
    source=folder/'input'
    try:
        update(key,status='processing',message='Looking for road damage…')
        if kind=='photo':
            result=engine.photo(source,folder); media='photo.jpg'
        else:
            result=engine.video(source,folder,lambda s:update(key,message=s));media='video.mp4'
        update(key,status='ready',result={**result,'media':f'/api/files/{key}/{media}'})
    except ImportError:
        update(key,status='error',message='Model dependencies are missing. On the Mac running this app, install requirements.txt, then try again.')
    except Exception as e:
        import logging
        logging.getLogger('road-damage').exception('Could not complete inspection')
        message=str(e) if isinstance(e,ValueError) else 'Could not process this file. Try a supported road photo or a short MP4 or MOV.'
        update(key,status='error',message=message)
    finally:source.unlink(missing_ok=True)

@app.post('/api/uploads/{kind}',status_code=202)
async def upload(kind:str, request:Request):
    if kind not in ('photo','video'):raise HTTPException(404)
    # Browser uploads use a custom header; cross-origin forms cannot enqueue jobs.
    if request.headers.get('x-road-upload')!='1':raise HTTPException(400,'Use the upload button on the homepage.')
    if int(request.headers.get('content-length') or 0)>LIMIT:raise HTTPException(413,'Choose a file smaller than 50 MB.')
    expire();key=uuid4().hex; folder=Path(TMP.name)/key
    with guard:
        if sum(j['status'] not in ('error','ready') for j in JOBS.values())>=4:raise HTTPException(429,'The detector is busy. Try again in a moment.')
        JOBS[key]=dict(kind=kind,status='queued',time=time.time(),message='Waiting for the detector…')
    try:
        folder.mkdir();total=0
        with (folder/'input').open('wb') as f:
            async for chunk in request.stream():
                total+=len(chunk)
                if total>LIMIT:raise HTTPException(413,'Choose a file smaller than 50 MB.')
                f.write(chunk)
        if not total:raise HTTPException(400,'This file is empty.')
        executor.submit(run,kind,key,folder)
    except (Exception,asyncio.CancelledError):
        with guard:JOBS.pop(key,None)
        shutil.rmtree(folder,ignore_errors=True);raise
    return {'key':key}

@app.get('/api/jobs/{kind}/{key}')
def status(kind:str,key:str):
    if key=='sample' and kind in ('photo','video'):
        d=json.loads((ASSETS/(kind+'.json')).read_text())
        name='huelva-road.jpg' if kind=='photo' else 'road-clip-result.mp4'
        return dict(status='ready',result={**d,'media':'/samples/media/'+name})
    with guard:
        v=JOBS.get(key)
        if not v or v['kind']!=kind:raise HTTPException(404,'This result has expired. Upload the file again.')
        v['time']=time.time();return {k:v[k] for k in ('status','message','result') if k in v}

@app.get('/api/files/{key}/{name}')
def media(key:str,name:str):
    with guard:
        v=JOBS.get(key)
        if not v or v['status']!='ready' or name!=('photo.jpg' if v['kind']=='photo' else 'video.mp4'):raise HTTPException(404)
        v['time']=time.time()
    return FileResponse(Path(TMP.name)/key/name,headers={'Cache-Control':'private, no-store'})

def main():
    import uvicorn
    cli=argparse.ArgumentParser();cli.add_argument('--lan',action='store_true');cli.add_argument('--port',type=int,default=8501)
    cli.add_argument('--host',default=None);args=cli.parse_args()
    if args.lan:
        try:
            with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
                sock.connect(('192.0.2.1',80)); ip=sock.getsockname()[0]
            print(f'Phone or iPad on the same Wi-Fi: http://{ip}:{args.port}',flush=True)
        except OSError:print('Open your Mac’s Wi-Fi IP address on the same network.')
    uvicorn.run(app,host=args.host or ('0.0.0.0' if args.lan else '127.0.0.1'),port=args.port)
if __name__=='__main__':main()
