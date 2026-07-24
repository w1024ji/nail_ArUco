import cv2, numpy as np
SC=4080/1200.
crops={'sh_middle':(500,505,110,140),'sh_thumb':(460,590,130,150)}
def fit_robust(pts,deg=2,iters=6):
    if len(pts)<8: return None
    p=np.array(pts,float); xs,ys=p[:,0],p[:,1]; keep=np.ones(len(p),bool)
    for _ in range(iters):
        cf=np.polyfit(ys[keep],xs[keep],deg)
        res=np.abs(xs-np.polyval(cf,ys)); s=max(np.median(res)*2.5,2.0); nk=res<s
        if nk.sum()<8: break
        keep=nk
    return cf,keep,int(keep.sum())
for n,(x,y,w,h) in crops.items():
    im=cv2.imread('sh_images/%s.jpg'%n)
    X,Y,W,H=[int(v*SC) for v in (x,y,w,h)]
    c=im[Y:Y+H,X:X+W]
    lab=cv2.cvtColor(c,cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv=cv2.cvtColor(c,cv2.COLOR_BGR2HSV).astype(np.float32)
    L,A=lab[:,:,0],lab[:,:,1]; Sa=hsv[:,:,1]
    skin=((A>136)&(L>110))
    fing=cv2.morphologyEx(skin.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    fing=cv2.morphologyEx(fing,cv2.MORPH_OPEN,np.ones((9,9),np.uint8))
    k=61; th=lambda v,s: s*(v-cv2.blur(v,(k,1)))
    g=cv2.GaussianBlur(th(Sa,1)/12.+th(A,1)/4.+th(L,-1)/12.,(1,15),0)
    g[cv2.erode(fing,np.ones((3,3),np.uint8))==0]=-99
    core=((Sa<95)&(L>170)&(fing>0)).astype(np.uint8)
    core=cv2.morphologyEx(core,cv2.MORPH_OPEN,np.ones((15,15),np.uint8))
    ys,xs=np.nonzero(core)
    cx=int(np.median(xs)); y0,y1=int(np.percentile(ys,3)),int(np.percentile(ys,97))
    def scan(r,sgn):
        row=g[r]; best=None;bv=2.5
        for d in range(int(W*0.05),int(W*0.50)):
            i=cx+sgn*d
            if i<2 or i>=W-2: break
            if row[i]<=bv or not(row[i]>=row[i-1] and row[i]>=row[i+1]): continue
            # validate: skin must continue OUTWARD past this peak (else it's the silhouette)
            o0,o1=(i+4,i+14) if sgn>0 else (i-14,i-4)
            o0=max(0,o0);o1=min(W,o1)
            if o1<=o0: continue
            if skin[r,o0:o1].mean()<0.75: continue
            best=i;bv=row[i]
        return best
    pl=[];pr=[]
    for r in range(y0,y1):
        i=scan(r,-1)
        if i is not None: pl.append((i,r))
        i=scan(r,1)
        if i is not None: pr.append((i,r))
    vis=c.copy()
    for p in pl: cv2.circle(vis,p,1,(0,255,0),-1)
    for p in pr: cv2.circle(vis,p,1,(0,128,255),-1)
    out={}
    for side,pts,col in (('L',pl,(0,0,255)),('R',pr,(255,0,255))):
        f=fit_robust(pts)
        if not f: print(n,side,'FAIL',len(pts)); continue
        cf,keep,ni=f; out[side]=cf
        iy=np.array(pts)[keep][:,1]
        print(n,side,'pts',len(pts),'inl',ni,'span rows %d-%d'%(iy.min(),iy.max()))
        for r in range(int(iy.min()),int(iy.max())):
            xx=int(np.polyval(cf,r))
            if 0<=xx<W: vis[r,xx]=col
    if len(out)==2:
        ws=[np.polyval(out['R'],r)-np.polyval(out['L'],r) for r in range(y0,y1)]
        print('   nail width px: max %.1f'%max(ws))
    cv2.imwrite('sh_images/%s_e5.jpg'%n, cv2.resize(vis,(W*2,H*2),interpolation=cv2.INTER_CUBIC))
