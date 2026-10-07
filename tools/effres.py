import os,json,numpy as np
from PIL import Image
S='../site/data.json'; D=json.load(open(S)); R={r['base']:r for r in D['images']}
for f in sorted(os.listdir('../raw')):
    b=f.rsplit('.',1)[0]; r=R[b]
    if 'eff' in r: continue
    im=Image.open('../raw/'+f).convert('L'); W,H=im.size; c=min(1536,H//2*2,W//2*2)
    im=im.crop((W//2-c//2,H//2-c//2,W//2+c//2,H//2+c//2)); g=np.asarray(im,np.float32)
    errs={}
    for s in (1.25,1.5,2,2.5,3,4):
        d=im.resize((round(c/s),round(c/s)),Image.LANCZOS).resize((c,c),Image.LANCZOS)
        errs[s]=float(np.sqrt(((np.asarray(d,np.float32)-g)[8:-8,8:-8]**2).mean()))
    best=1.0
    for s in (1.25,1.5,2,2.5,3,4):
        if errs[s]<=1.2: best=s
    r['eff_scale']=best; r['eff']=round(W*H/best/best/1e6,2); r['rt_err']={str(k):round(v,2) for k,v in errs.items()}
    print(b,best,r['eff'],'MP',[round(v,2) for v in errs.values()])
json.dump(D,open(S,'w'))
