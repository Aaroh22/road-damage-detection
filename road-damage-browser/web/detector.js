const names=['Longitudinal crack','Transverse crack','Alligator crack','Pothole'];
let model;
const N=640, S=N*N;
const c=document.createElement('canvas'), orig=document.createElement('canvas');
c.width=c.height=N;
const ctx=c.getContext('2d',{willReadFrequently:true});
const o=orig.getContext('2d',{willReadFrequently:true});
export async function loadDetector(){
  ort.env.wasm.numThreads=1;
  ort.env.wasm.wasmPaths=new URL('/',location.href).href;
  model ??= ort.InferenceSession.create('/model.onnx',{executionProviders:['wasm']});
  try{return await model}catch(error){model=undefined;throw error}
}
export async function detect(source,width,height){
let g=Math.min(N/width,N/height),w=Math.round(width*g),h=Math.round(height*g),x=Math.floor((N-w)/2),y=Math.floor((N-h)/2);
ctx.fillStyle='#727272';ctx.fillRect(0,0,N,N);if(orig.width!==width)orig.width=width;if(orig.height!==height)orig.height=height;o.clearRect(0,0,width,height);o.drawImage(source,0,0);let raw=o.getImageData(0,0,width,height).data, res=ctx.createImageData(w,h);for(let v=0;v<h;v++)for(let u=0;u<w;u++){let X=(u+.5)*width/w-.5,Y=(v+.5)*height/h-.5,i=Math.max(0,Math.min(width-1,Math.floor(X))),j=Math.max(0,Math.min(height-1,Math.floor(Y))),dx=Math.max(0,X-i),dy=Math.max(0,Y-j);for(let k=0;k<3;k++){let n=4*(j*width+i)+k;res.data[4*(v*w+u)+k]=(1-dy)*((1-dx)*raw[n]+dx*raw[4*(j*width+Math.min(width-1,i+1))+k])+dy*((1-dx)*raw[4*(Math.min(height-1,j+1)*width+i)+k]+dx*raw[4*(Math.min(height-1,j+1)*width+Math.min(width-1,i+1))+k])}res.data[4*(v*w+u)+3]=255}ctx.putImageData(res,x,y);let pixels=ctx.getImageData(0,0,N,N).data,arr=new Float32Array(S*3);
for(let i=0;i<S;i++)for(let j=0;j<3;j++)arr[i+j*S]=pixels[i*4+j]/255;
const model=await loadDetector();
let r=await model.run({[model.inputNames[0]]:new ort.Tensor('float32',arr,[1,3,N,N])}),out=r[model.outputNames[0]],a=out.data,[,k,n]=out.dims,boxes=[];
if(k!==8)throw Error('Unexpected output shape '+out.dims);
for(let i=0;i<n;i++){let cls=0;for(let j=1;j<4;j++)if(a[(4+j)*n+i]>a[(4+cls)*n+i])cls=j;
let score=a[(4+cls)*n+i];if(score<.25)continue;let X=a[i],Y=a[n+i],W=a[2*n+i],H=a[3*n+i];boxes.push({cls,score,l:X-W/2,t:Y-H/2,r:X+W/2,b:Y+H/2})}
let keep=[];for(let q of boxes.sort((a,b)=>b.score-a.score)){if(keep.some(p=>{if(q.cls!==p.cls)return false;let s=Math.max(0,Math.min(q.r,p.r)-Math.max(q.l,p.l))*Math.max(0,Math.min(q.b,p.b)-Math.max(q.t,p.t));return s/((q.r-q.l)*(q.b-q.t)+(p.r-p.l)*(p.b-p.t)-s)>.7}))continue;keep.push(q)}

return keep.map(q=>({label:names[q.cls],score:q.score,box:[(q.l-x)/g,(q.t-y)/g,(q.r-x)/g,(q.b-y)/g]}));
}
