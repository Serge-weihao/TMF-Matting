from mmedit.apis import init_model, matting_inference
from PIL import Image
import numpy as np
import cv2
import argparse
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="alpha,ot,comp"
        )
    parser.add_argument(
        "--img",
        help="path to mask png",
        type=str,
    )
    parser.add_argument(
        "--out",
        help="where2output",
        type=str,
    )
    parser.add_argument(
    "--trimap",
        default= None,
        type=str,
    )
    parser.add_argument(
    "--config",
        default= None,
        type=str,
    )
    parser.add_argument(
    "--ckpt",
        default= None,
        type=str,
    )
    args = parser.parse_args()
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    model = init_model(args.config,args.ckpt)
    imlist_o = os.listdir(args.img)
    imlist = []
    for i in imlist_o:
        if i.endswith('.png') or i.endswith('.jpg'):
            imlist.append(i)
    imlist.sort()
    trilist_o = os.listdir(args.trimap)
    trilist=[]
    for i in trilist_o:
        if i.endswith('.png') or i.endswith('.jpg'):
            trilist.append(i)
    trilist.sort()
    assert len(imlist)==len(trilist)
    for im,trimap in zip(imlist,trilist):
        h,w=cv2.imread(os.path.join(args.trimap,trimap),0).shape
        if h*w>4100*4000: 
            print(os.path.join(args.img,im))
            continue
        try:
            t1 = matting_inference(model, img=os.path.join(args.img,im), trimap=os.path.join(args.trimap,trimap))*255
            cv2.imwrite(os.path.join(args.out,trimap),t1.astype(np.uint8))
        except:
            print(os.path.join(args.img,im))