"""YOLO26s inference; model imports are lazy so the recorded samples open instantly."""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
from threading import Lock

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
WEIGHT_HASH = '2304871205864929cfbb86739fe0867021e56de456d33b30f99c0f1fa6ab8f27'
MODEL_VERSION = '8.4.204'
_model = None
_lock = Lock()
MAX_SECONDS = 90
FPS = 25

def predict(image):
    global _model
    with _lock:
        if _model is None:
            from ultralytics import YOLO, __version__
            if __version__ != MODEL_VERSION:
                raise ValueError('Install the versions in requirements.txt to use the selected model.')
            weights = ROOT / 'weights/road_damage_yolo26s.pt'
            if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest() != WEIGHT_HASH:
                raise ValueError('The selected YOLO26s checkpoint is missing or does not match the test report.')
            _model = YOLO(str(weights))
        r = _model.predict(image, imgsz=640, conf=.25, device='cpu', verbose=False)[0]
        return [dict(box=list(map(float, row[:4])), score=float(row[4]), label=r.names[int(row[5])])
                for row in r.boxes.data.cpu().tolist()]

def photo(source, dest):
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    with Image.open(source) as original:
        if original.format not in {'JPEG','PNG','WEBP','HEIF','HEIC'}:
            raise ValueError('Choose a JPG, PNG, WebP, or HEIC photo.')
        if original.width*original.height > 50_000_000:
            raise ValueError('Choose a photo under 50 megapixels.')
        img = ImageOps.exif_transpose(original).convert('RGB')
    img.thumbnail((4096,4096),Image.Resampling.LANCZOS)
    boxes = predict(img)
    img.save(dest/'photo.jpg', quality=94)
    return {'detections':boxes,'width':img.width,'height':img.height}

def ffmpeg():
    from_system = shutil.which('ffmpeg')
    if from_system:
        return from_system
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def paint(image, boxes):
    if not boxes: return image
    draw=ImageDraw.Draw(image)
    size=max(14,round(max(image.size)*.018))
    font=ImageFont.load_default(size=size)
    for item in boxes:
        x,y,r,b=item['box']; label=f"{item['label']} {item['score']:.2f}"
        draw.rectangle((x,y,r,b),outline='#ffc18c',width=max(2,size//8))
        w=draw.textlength(label,font=font)+10
        left=max(0,min(x,image.width-w)); top=max(0,y-size-9)
        draw.rectangle((left,top,left+w,top+size+8),fill='#f6cea6')
        draw.text((left+5,top+2),label,font=font,fill='#261c17')
    return image

def add_moment(moments, frame):
    start=frame/FPS; end=(frame+1)/FPS
    if moments and start-moments[-1][1] < .48:
        moments[-1][1]=end
    else:
        moments.append([start,end])

def stop(p):
    if p.poll() is None: p.kill()
    p.wait()

def video(source, dest, progress):
    with source.open('rb') as f:
        header=f.read(12)
    if len(header)<12 or header[4:8] not in (b'ftyp',b'moov',b'mdat',b'wide',b'free'):
        raise ValueError('Choose an MP4 or MOV video.')
    exe=ffmpeg()
    # A constant output timebase keeps marker times aligned, even for phone VFR footage.
    probe=subprocess.run([exe,'-hide_banner','-protocol_whitelist','file,pipe','-f','mov','-i',str(source)],capture_output=True,timeout=15)
    match=re.search(r'Duration: (\d+):(\d+):([\d.]+)',probe.stderr.decode(errors='replace'))
    if match and int(match[1])*3600+int(match[2])*60+float(match[3]) > MAX_SECONDS+.04:
        raise ValueError('Choose a clip no longer than 90 seconds.')
    scale="scale=w='min(1280,iw)':h='min(1280,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1,fps=25"
    moments=[]; n=0; writer=None
    with ExitStack() as stack:
        reader=subprocess.Popen([exe,'-nostdin','-v','error','-protocol_whitelist','file,pipe','-f','mov','-i',str(source),'-map','0:v:0','-an','-t','91','-vf',scale,'-c:v','ppm','-f','image2pipe','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        stack.callback(stop,reader); stack.callback(reader.stdout.close)
        while True:
            magic=reader.stdout.readline()
            if not magic:break
            dims=reader.stdout.readline()
            if magic.strip()!=b'P6' or reader.stdout.readline().strip()!=b'255':
                raise ValueError('This clip could not be decoded.')
            w,h=map(int,dims.split()); data=reader.stdout.read(w*h*3)
            if len(data)!=w*h*3 or n>=MAX_SECONDS*FPS:
                raise ValueError('Choose a complete clip no longer than 90 seconds.')
            img=Image.frombytes('RGB',(w,h),data)
            if writer is None:
                writer=subprocess.Popen([exe,'-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'video.mp4')],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
                stack.callback(stop,writer)
            boxes=predict(img)
            if boxes:add_moment(moments,n)
            writer.stdin.write(paint(img,boxes).tobytes())
            n+=1
            if n%10==0:progress(f'Checking footage · {n/FPS:.0f} seconds processed')
        if not n or reader.wait()!=0:raise ValueError('Could not read this clip. Try an MP4 or MOV.')
        writer.stdin.close()
        if writer.wait()!=0:raise ValueError('Could not save the annotated video.')
    return {'duration':n/FPS,'markers':moments}
