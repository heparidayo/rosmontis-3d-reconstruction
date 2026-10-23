"""COLMAP Sparse 점군을 원래 색상으로 자동 회전하는 오프라인 뷰어 생성."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

from diagnose_sfm import read_model, rotation, unpack


TEMPLATE = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rosmontis · 자동 회전 3D 점군</title>
<style>
:root{color-scheme:dark;font-family:"Malgun Gothic",system-ui,sans-serif;color:#e7edf5;background:#10161f}
*{box-sizing:border-box}body{margin:0;overflow:hidden}main{height:100dvh;display:flex;flex-direction:column}
header{padding:17px 24px 12px;display:flex;justify-content:space-between;gap:12px;align-items:center;border-bottom:1px solid #293342}
h1{font-size:21px;font-weight:600;margin:0 0 5px}header p{margin:0;color:#a9b6c6;font-size:13px}.badge{font-size:12px;color:#bad6d4;border:1px solid #3e6462;padding:6px 10px;border-radius:5px;white-space:nowrap}
.tools{display:flex;align-items:center;gap:12px 19px;flex-wrap:wrap;padding:12px 24px;border-bottom:1px solid #293342;background:#151d28;font-size:13px}
button{font:inherit;cursor:pointer;border:1px solid #445367;border-radius:5px;background:#202c3b;color:#edf4fa;padding:8px 12px;min-height:36px}button:hover{background:#304255}button:focus-visible,input:focus-visible{outline:2px solid #7cc9c0;outline-offset:3px}
button.primary{background:#275852;border-color:#49847c}button[aria-pressed="true"]{border-color:#84c7be;background:#254d49}
label{display:flex;align-items:center;gap:8px;white-space:nowrap}input[type=range]{width:104px;accent-color:#8acdc2}input[type=checkbox]{accent-color:#8acdc2;width:16px;height:16px}
output{color:#b9c8d9;font-variant-numeric:tabular-nums;min-width:35px}.views{display:flex;gap:6px}
#stage{position:relative;flex:1;min-height:0;background:#0c121b}canvas{width:100%;height:100%;display:block;cursor:grab;touch-action:none}canvas.dragging{cursor:grabbing}
.hint{position:absolute;left:24px;bottom:18px;color:#9bacc0;font-size:12px;pointer-events:none;line-height:1.7}.status{position:absolute;right:24px;bottom:18px;color:#bac8d7;font-size:12px;pointer-events:none;text-align:right;font-variant-numeric:tabular-nums}
.legend{position:absolute;right:24px;top:18px;color:#91a4b9;font-size:12px;display:none}.legend span{display:block;line-height:1.9}.legend i{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:7px}
@media(max-width:760px){header{padding:12px 15px}h1{font-size:17px}.tools{padding:9px 13px;gap:9px 12px}input[type=range]{width:75px}.hint{left:13px;bottom:12px;font-size:11px}.status{right:13px;bottom:12px;font-size:11px}.badge{display:none}}
</style></head><body><main>
<header><div><h1>Rosmontis · 3D 점군</h1><p>71장으로 복원한 원래 점군 · <span id="pointCount"></span>개 점 · 마우스로 자유롭게 살펴보세요</p></div><span class="badge">Sparse 복원 · 표면 생성 전</span></header>
<div class="tools" aria-label="3D 보기 설정">
<button class="primary" id="play" aria-pressed="true">회전 일시정지</button>
<label>회전 속도 <input id="speed" aria-label="회전 속도" type="range" min="2" max="30" value="12"><output id="speedValue">12°/초</output></label>
<label>점 크기 <input id="size" aria-label="점 크기" type="range" min="1" max="9" step="0.5" value="3.5"><output id="sizeValue">3.5</output></label>
<label><input id="cameras" type="checkbox">카메라 표시</label>
<div class="views"><button id="fit">피규어 크게</button><button id="front">정면</button><button id="back">후면</button><button id="top">위에서</button></div>
<button id="fullscreen">전체 화면</button>
</div>
<div id="stage"><canvas id="view" aria-label="자동 회전하는 Rosmontis 3D 점군. 드래그로 회전하고 휠로 확대할 수 있습니다."></canvas>
<div class="hint">드래그: 회전 · 오른쪽 드래그: 이동 · 휠: 확대<br>Space: 회전 멈춤/재개 · 더블클릭: 피규어 맞춤</div>
<div class="status"><span id="viewState">피규어 확대</span><br><span id="angle">회전 0°</span></div>
<div class="legend" id="legend"><span><i style="background:#72c6cf"></i>낮은 각도 24장</span><span><i style="background:#dfaa73"></i>중간 각도 23장</span><span><i style="background:#b8a1da"></i>높은 각도 24장</span></div>
</div></main>
<script id="cloud-data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('cloud-data').textContent);
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d',{alpha:false}),stage=document.getElementById('stage');
const el=id=>document.getElementById(id);
const points=data.points.map(p=>({p:p.slice(0,3),color:`rgb(${p[3]},${p[4]},${p[5]})`}));
const cameras=data.cameras, cameraColors=['#72c6cf','#dfaa73','#b8a1da'];
el('pointCount').textContent=points.length.toLocaleString('ko-KR');
const state={yaw:0,pitch:0.2,distance:4,target:[0,0,0],playing:true,speed:12,size:3.5,cameras:false,drag:null,lastTime:null,resumeAt:0};
const FOV=38*Math.PI/180, sinHalf=Math.sin(FOV/2), tanHalf=Math.tan(FOV/2);
let width=1,height=1,dpr=1,lastStatus=0;
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const mul=(a,s)=>a.map(v=>v*s);
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>Math.sqrt(dot(a,a));
const unit=a=>mul(a,1/Math.max(norm(a),1e-10));
function resize(){
  const r=stage.getBoundingClientRect();width=Math.max(r.width,1);height=Math.max(r.height,1);dpr=Math.min(window.devicePixelRatio||1,2);
  canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);
}
new ResizeObserver(resize).observe(stage);resize();
function fit(includeCameras=false){
  const positions=points.map(p=>p.p).concat(includeCameras?cameras.map(c=>c.center):[]);
  const lo=[0,1,2].map(i=>Math.min(...positions.map(p=>p[i]))),hi=[0,1,2].map(i=>Math.max(...positions.map(p=>p[i])));
  state.target=lo.map((v,i)=>(v+hi[i])/2);
  const radius=Math.max(...positions.map(p=>norm(sub(p,state.target))));
  const aspect=Math.min(width/height,1),effectiveSin=Math.sin(Math.atan(tanHalf*aspect));
  state.distance=Math.max(radius/Math.max(effectiveSin,.05)*1.08,.3);
  el('viewState').textContent=includeCameras?'촬영 위치 포함 전체 보기':'피규어 확대';
}
function reset(){state.cameras=false;el('cameras').checked=false;el('legend').style.display='none';fit(false);}
function setPlaying(value){state.playing=value;el('play').textContent=value?'회전 일시정지':'자동 회전 시작';el('play').setAttribute('aria-pressed',String(value));}
el('play').addEventListener('click',()=>setPlaying(!state.playing));
el('speed').addEventListener('input',e=>{state.speed=Number(e.target.value);el('speedValue').textContent=state.speed+'°/초';});
el('size').addEventListener('input',e=>{state.size=Number(e.target.value);el('sizeValue').textContent=state.size.toFixed(1);});
el('cameras').addEventListener('change',e=>{state.cameras=e.target.checked;el('legend').style.display=state.cameras?'block':'none';fit(state.cameras);});
el('fit').addEventListener('click',reset);
el('front').addEventListener('click',()=>{state.yaw=0;state.pitch=.15;reset();});
el('back').addEventListener('click',()=>{state.yaw=Math.PI;state.pitch=.15;reset();});
el('top').addEventListener('click',()=>{state.pitch=85*Math.PI/180;reset();});
el('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{el('viewState').textContent='이 브라우저에서는 전체 화면을 사용할 수 없습니다';}});
document.addEventListener('fullscreenchange',()=>{el('fullscreen').textContent=document.fullscreenElement?'전체 화면 종료':'전체 화면';resize();});
document.addEventListener('keydown',e=>{if(e.code==='Space'&&!['INPUT','BUTTON','TEXTAREA'].includes(document.activeElement.tagName)){e.preventDefault();setPlaying(!state.playing);}});
function basis(){
  const cp=Math.cos(state.pitch),eye=add(state.target,mul([cp*Math.cos(state.yaw),cp*Math.sin(state.yaw),Math.sin(state.pitch)],state.distance));
  const forward=unit(sub(state.target,eye)),right=unit(cross(forward,[0,0,1])),up=cross(right,forward);
  return {eye,forward,right,up};
}
canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);state.drag={x:e.clientX,y:e.clientY,button:e.button};canvas.classList.add('dragging');});
canvas.addEventListener('pointermove',e=>{
  if(!state.drag)return;
  const dx=e.clientX-state.drag.x,dy=e.clientY-state.drag.y;state.drag.x=e.clientX;state.drag.y=e.clientY;
  if(state.drag.button===2){const b=basis(),factor=state.distance*tanHalf*2/height;state.target=add(state.target,add(mul(b.right,-dx*factor),mul(b.up,dy*factor)));}
  else{state.yaw-=dx*.007;state.pitch=Math.max(-1.48,Math.min(1.48,state.pitch+dy*.006));}
});
function endDrag(){state.drag=null;state.resumeAt=performance.now()+650;canvas.classList.remove('dragging');}
canvas.addEventListener('pointerup',endDrag);canvas.addEventListener('pointercancel',endDrag);canvas.addEventListener('lostpointercapture',endDrag);
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('wheel',e=>{e.preventDefault();state.distance=Math.max(.15,Math.min(150,state.distance*Math.exp(e.deltaY*.001)));},{passive:false});
canvas.addEventListener('dblclick',reset);
function draw(time){
  const delta=state.lastTime===null?0:Math.min((time-state.lastTime)/1000,.1);state.lastTime=time;
  if(state.playing&&!state.drag&&time>state.resumeAt)state.yaw+=delta*state.speed*Math.PI/180;
  const b=basis(),focal=height/(2*tanHalf);
  function project(p){const q=sub(p,b.eye),z=dot(q,b.forward);if(z<=.005)return null;return {x:width/2+dot(q,b.right)*focal/z,y:height/2-dot(q,b.up)*focal/z,z};}
  ctx.fillStyle='#0c121b';ctx.fillRect(0,0,width,height);
  if(state.cameras){
    ctx.lineWidth=.9;ctx.globalAlpha=.65;
    for(const c of cameras){
      const a=project(c.center);if(!a)continue;
      const corners=c.corners.map(project);ctx.strokeStyle=cameraColors[c.group];
      for(let i=0;i<4;i++){const p=corners[i],q=corners[(i+1)%4];if(!p||!q)continue;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();}
      ctx.fillStyle=cameraColors[c.group];ctx.beginPath();ctx.arc(a.x,a.y,2.4,0,Math.PI*2);ctx.fill();
    }
    ctx.globalAlpha=1;
  }
  const visible=[];
  for(const point of points){const q=project(point.p);if(q&&q.x>-20&&q.x<width+20&&q.y>-20&&q.y<height+20)visible.push({...q,color:point.color});}
  visible.sort((a,b)=>b.z-a.z);
  const radius=state.size/2;
  for(const p of visible){ctx.fillStyle=p.color;ctx.beginPath();ctx.arc(p.x,p.y,radius,0,Math.PI*2);ctx.fill();}
  if(time-lastStatus>160){const angle=((state.yaw*180/Math.PI)%360+360)%360;el('angle').textContent=`회전 ${angle.toFixed(1)}° · 표시 ${visible.length.toLocaleString('ko-KR')}점`;lastStatus=time;}
  requestAnimationFrame(draw);
}
fit(false);requestAnimationFrame(draw);
</script></body></html>'''


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("runs/full360_71_20261018_02/sparse/0"))
    parser.add_argument("--validation", type=Path, default=Path("results/diagnostics/full360_71_20261018_02/validation_360.json"))
    parser.add_argument("--output", type=Path, required=True, help="새 뷰어 폴더")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    model, output = args.model.resolve(strict=True), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to(project/"results"):
        raise ValueError("뷰어 출력은 results 아래의 새로운 폴더여야 합니다")
    sources = [p for p in model.iterdir() if p.is_file()]+[args.validation.resolve(strict=True)]
    before = {str(p): sha(p) for p in sources}
    cameras, images, points = read_model(model)
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    transform = np.array(validation["analysis_transform"])
    # 앞선 분석용 축의 거울 반사를 제거하여 실제 모델의 좌우 관계를 유지한다.
    if np.linalg.det(transform) < 0:
        transform[1] *= -1
    assert np.allclose(transform@transform.T, np.eye(3)) and np.linalg.det(transform) > .999
    origin = np.array(validation["analysis_focus"])
    raw = np.array([transform@(p["xyz"]-origin) for p in points.values()])
    center = (raw.min(0)+raw.max(0))/2
    scale = 2/np.max(np.ptp(raw, axis=0))
    def local(xyz):
        return ((transform@(xyz-origin)-center)*scale).tolist()
    colors = {}
    with (model/"points3D.bin").open("rb") as stream:
        for _ in range(unpack(stream, "Q")[0]):
            row = unpack(stream, "QdddBBBd")
            count = unpack(stream, "Q")[0]
            stream.seek(count*8, 1)
            colors[row[0]] = list(row[4:7])
        assert not stream.read(1)
    cloud = [local(p["xyz"])+colors[pid] for pid, p in points.items()]
    camera_payload = []
    for image in images.values():
        cam = cameras[image["camera_id"]]
        R = rotation(image["q"])
        camera_center = -R.T@image["t"]
        f, cx, cy, k = cam["params"]
        assert k == 0
        # 표시용 프러스텀만 작은 크기로 만든다. 복원된 카메라 자세는 그대로다.
        depth = .035
        corners = [local(camera_center+R.T@(np.array([(x-cx)/f, (y-cy)/f, 1.])*depth))
                   for x, y in ((0, 0), (cam["width"], 0), (cam["width"], cam["height"]), (0, cam["height"]))]
        group = next(i for i, g in enumerate(validation["coverage"].values()) if image["name"] in g["names"])
        camera_payload.append(dict(name=image["name"], center=local(camera_center), corners=corners, group=group))
    payload = dict(points=cloud, cameras=camera_payload)
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", data)
    output.mkdir(parents=True)
    with (output/"index.html").open("x", encoding="utf-8") as file:
        file.write(html)
    after = {str(p): sha(p) for p in sources}
    assert before == after
    manifest = dict(source_model=str(model), points3D=len(points), images=len(images),
                    original_rgb=True, point_filtering=False, source_unchanged=True,
                    source_sha256=before, display_transform=transform.tolist(), display_scale=scale)
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(output/"index.html"), points=len(cloud), cameras=len(camera_payload), source_unchanged=True)))


if __name__ == "__main__":
    main()
