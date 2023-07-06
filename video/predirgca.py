from mmedit.apis import init_model, matting_inference
from PIL import Image
import numpy as np
import cv2
import argparse
import os
import torch
from mmcv.parallel import collate, scatter

from mmedit.datasets.pipelines import Compose
def matting_inference(model, img, trimap):
    """Inference image(s) with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): Image file path.
        trimap (str): Trimap file path.

    Returns:
        np.ndarray: The predicted alpha matte.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # remove alpha from test_pipeline
    keys_to_remove = ['alpha', 'ori_alpha']
    for key in keys_to_remove:
        for pipeline in list(cfg.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                cfg.test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    cfg.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)
    # prepare data
    data = dict(merged_path=img, trimap_path=trimap)
    data = test_pipeline(data)
    #print('before',data)
    data = collate([data], samples_per_gpu=1)#scatter(collate([data], samples_per_gpu=1), [device])[0]
    data['meta'] = data.copy()['meta'].data[0]
    #print('after',data)
    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)

    return result['pred_alpha']
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
    
    model = init_model(args.config,args.ckpt).cpu()
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
            print(os.path.join(args.img,im),'big')
            continue
        t1 = matting_inference(model, img=os.path.join(args.img,im), trimap=os.path.join(args.trimap,trimap))*255
        cv2.imwrite(os.path.join(args.out,trimap),t1.astype(np.uint8))
