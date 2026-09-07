"""Causal probe: give the existing refiner a padded, valid undistorted canvas.
Keeps checkpoint/poses/weights fixed. Full output and corner-only composite are
scored externally. No test pixels are read. Changes are process-local adapters.
"""
import argparse,gc,json,os,sys
from pathlib import Path
import cv2,numpy as np,torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa
from dataset import SceneData
import fuse_test as FT
import refiner_data as RD
import refiner_train as RT
from gsplat.rendering import rasterization

def main():
    ap=argparse.ArgumentParser()
    for k in ('scene','model','refiner','pin','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--baseline',default='')  # diagnostics only (corner_only composite)
    ap.add_argument('--names',default='',help='comma list of view stems; empty = every test pose');ap.add_argument('--K',default='2');a=ap.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(2)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    s=SceneData(a.scene,load_images=False,distorted=False,holdout_every=0)
    tp=s.test_poses[0];mx,my=s.redistort_map(tp.K,tp.width,tp.height)
    pad=int(np.ceil((max(-mx.min(),mx.max()-s.width,-my.min(),my.max()-s.height)+32)/64)*64)
    oldK=s.K.copy();W,H=s.width,s.height;K=oldK.copy();K[:2,2]+=pad;WP,HP=W+2*pad,H+2*pad
    ux,uy=cv2.initUndistortRectifyMap(oldK,np.asarray(s.dist),None,K,(WP,HP),cv2.CV_32FC1)
    valid=((ux>=1)&(ux<=W-2)&(uy>=1)&(uy<=H-2)).astype(np.float32)
    if a.names.strip():
        want=[n for n in a.names.split(',') if n]
        selected=[t for t in s.test_poses if any('_'+n+'_' in t.image_name or Path(t.image_name).stem==n for n in want)]
        assert len(selected)==len(want),f'matched {len(selected)} of {len(want)} names'
    else:
        selected=list(s.test_poses)
    print('VIEWS',len(selected),flush=True)
    rd=out/'renders';rd.mkdir(exist_ok=True)
    print('PAD',pad,'canvas',WP,HP,'valid_fraction',valid.mean(),flush=True)
    spl,sh=FT.load_splats(str(Path(a.model)/'ckpt.pt'),'cuda');compare=[]
    with torch.inference_mode():
        for tp in selected:
            name=Path(tp.image_name).stem
            r,alpha,_=rasterization(means=spl['means'],quats=spl['quats'],scales=spl['scales'],opacities=spl['opacities'],colors=spl['colors'],viewmats=torch.tensor(tp.w2c,dtype=torch.float32,device='cuda')[None],Ks=torch.tensor(K,dtype=torch.float32,device='cuda')[None],width=WP,height=HP,sh_degree=sh,render_mode='RGB',rasterize_mode='antialiased',near_plane=.01,far_plane=1e10,packed=False)
            im=cv2.cvtColor((r[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8),cv2.COLOR_RGB2BGR)
            ref=cv2.imread(str(Path(a.pin)/(name+'.png')));assert ref is not None,name
            diff=im[pad:pad+H,pad:pad+W].astype(float)-ref.astype(float)
            compare.append(dict(name=name,central_raw_mse=float(np.mean(diff**2)),max_abs=float(np.max(abs(diff)))))
            print('CENTRAL_RENDER_CONTROL',compare[-1],flush=True)
            cv2.imwrite(str(rd/(name+'.png')),im)
    del spl,r,alpha;gc.collect();torch.cuda.empty_cache()
    def factory(*args,**kwargs):
        scene=SceneData(*args,**kwargs);scene.width,scene.height=WP,HP;scene.K=K.copy();scene.test_poses=selected
        return scene
    RD.SceneData=factory
    # refiner_data._rd_und normally calls cv2.undistort; remap the same original
    # source image into the expanded camera without changing original intrinsics.
    original_und=cv2.undistort
    cv2.undistort=lambda im,*args,**kwargs:cv2.remap(im,ux,uy,cv2.INTER_LINEAR)
    warp=FT.warp_source;v=torch.from_numpy(valid).cuda()[None,None]
    def valid_warp(img,*args,**kwargs):
        wr,inb,occ=warp(torch.cat([img,v],1),*args,**kwargs)
        return wr[:,:3],inb&(wr[0,3]>.999),occ
    FT.warp_source=valid_warp
    dump=out/'dump';sys.argv=['refiner_data','--result_dir',a.model,'--scene_dir',a.scene,'--dump',str(dump),'--targets','test','--render_dir',str(rd),'--K',a.K,'--depth_fix','1']
    RD.main();cv2.undistort=original_und;FT.warp_source=warp
    gc.collect();torch.cuda.empty_cache()
    import subprocess
    subprocess.run([sys.executable,os.path.join(os.path.dirname(os.path.abspath(__file__)),'refine','refiner_train.py'),'apply','--data',str(dump),'--ckpt',a.refiner,'--out',str(out/'refined'),'--K',a.K,'--extra','1','--fp16','1','--ch','48,96,192,384'],check=True)
    for sub in ('redistort','corner_only'):(out/sub).mkdir(exist_ok=True)
    inside=((mx>=1)&(mx<=W-2)&(my>=1)&(my<=H-2)).astype(np.float32)
    mask=cv2.GaussianBlur(inside,(0,0),3)[...,None]
    for tp in selected:
        n=Path(tp.image_name).stem;im=cv2.imread(str(out/'refined'/(n+'.png')));assert im is not None,n
        red=cv2.remap(im,mx+pad,my+pad,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(str(out/'redistort'/(n+'.png')),red)
        base=cv2.imread(str(Path(a.baseline)/(n+'.png'))) if a.baseline else None  # 07/09: baseline chỉ để dựng corner_only (chẩn đoán) — thiếu thì bỏ qua
        if base is not None:
            comp=(base.astype(float)*mask+red.astype(float)*(1-mask)).round().clip(0,255).astype(np.uint8)
            cv2.imwrite(str(out/'corner_only'/(n+'.png')),comp)
    json.dump(dict(pad=pad,n=len(selected),central_raw_control=compare),open(out/'metadata.json','w'),indent=2)
    print('PADDED_DONE',len(selected),flush=True)
if __name__=='__main__':main()
