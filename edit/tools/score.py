"""Score every shot of one or more catalogued scene packs, to shortlist clips before looking.

  python score.py dexter_s2_a dexter_s4 ...   (run from the repo root; needs scenepacks/catalog/<pack>.json)

For 3 frames per shot: largest face height (fraction of frame), face centre x, face-region
sharpness (Laplacian variance at a fixed size: higher = more real detail), brightness, motion,
glow (fraction of hot yellow/orange pixels: lightning, fire) and red (fraction of saturated red: e.g. a red suit).
Writes media/scores.json. Good candidates: face > 0.2 (a close-up) and sharp > ~150.
"""
import json, os, sys, numpy as np, cv2
from multiprocessing import Pool
MODEL=os.path.expanduser('~/.cache/rea-video/models/face_detection_yunet_2023mar.onnx')
def score(pack):
    cv2.setNumThreads(1)
    shots=json.load(open('scenepacks/catalog/%s.json'%pack)); cap=cv2.VideoCapture('media/scenepacks/%s.mp4'%pack)
    W,H=int(cap.get(3)),int(cap.get(4)); det=cv2.FaceDetectorYN.create(MODEL,'',(W,H),0.6,0.3,20); out=[]
    for i,(a,b) in enumerate(shots):
        if b-a<0.9: out.append(None); continue
        fh=[];sh=[];br=[];fx=[];prev=None;mot=[];gl=[];rd=[]
        for q in (0.2,0.5,0.8):
            cap.set(cv2.CAP_PROP_POS_MSEC,(a+(b-a)*q)*1000); ok,f=cap.read()
            if not ok: continue
            g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY); br.append(g.mean())
            hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV); hh,sv,vv=hsv[...,0],hsv[...,1],hsv[...,2]
            gl.append(float(((hh>=8)&(hh<=35)&(sv>90)&(vv>200)).mean())); rd.append(float((((hh<=6)|(hh>=172))&(sv>120)&(vv>70)).mean()))
            if prev is not None: mot.append(np.abs(cv2.resize(g,(64,36)).astype(float)-prev).mean())
            prev=cv2.resize(g,(64,36)).astype(float)
            _,faces=det.detect(f)
            if faces is not None and len(faces):
                k=np.argmax(faces[:,3]); x,y,w,h=faces[k,:4]; fh.append(h/H); fx.append((x+w/2)/W)
                x0,y0=int(max(0,x)),int(max(0,y)); roi=g[y0:int(y+h),x0:int(x+w)]
                if roi.size>100: sh.append(cv2.Laplacian(cv2.resize(roi,(128,128)),cv2.CV_64F).var())
        out.append(dict(i=i,a=round(a,2),b=round(b,2),face=round(float(np.mean(fh)),3) if fh else 0,nf=len(fh),fx=round(float(np.mean(fx)),2) if fx else None,
                        sharp=round(float(np.mean(sh)),1) if sh else 0,bright=round(float(np.mean(br)),1),motion=round(float(np.mean(mot)),1) if mot else 0,glow=round(float(np.mean(gl)),4) if gl else 0,red=round(float(np.mean(rd)),4) if rd else 0))
    return pack,out
if __name__=='__main__':
    packs=sys.argv[1:] or sys.exit(__doc__)
    with Pool(4) as p: res=dict(p.map(score,packs))
    json.dump(res,open('media/scores.json','w'))
    for k,v in res.items():
        v=[s for s in v if s]; good=[s for s in v if s['face']>0.22 and s['nf']>=2]
        print('%-18s %3d shots, %3d with a clear close-up face, median sharpness %.0f'%(k,len(v),len(good),np.median([s['sharp'] for s in good]) if good else 0))
