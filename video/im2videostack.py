import os
import cv2
import argparse
import numpy as np
'''
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4
videoWriter = cv2.VideoWriter(file_path,fourcc,fps,size)
for item in filelist:
    if item.endswith('.jpg'):   #判断图片后缀是否是.jpg
        item = path  + item 
        img = cv2.imread(item) #使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
        # print(type(img))  # numpy.ndarray类型  
        videoWriter.write(img)        #把图片写进视频



videoWriter.release()
'''
def imgs2video(img_root, video_path, fps, size, s, e): # 仅将s到e的图转为视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(video_path, fourcc, fps, size)
    for i in range(s, e):  # 有多少张图片，从编号s到编号e
        img = cv2.imread(img_root + "%04d"%i + '.png')
        # cropped_img = img[0:960, 0:720] # 裁剪图片
        videoWriter.write(img)
    videoWriter.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="alpha,ot,comp"
        )
    parser.add_argument(
        "--vp",
        help="path to mask png",
        type=str,
    )
    parser.add_argument(
        "--ip",
        help="where2output",
        type=str,
    )
    parser.add_argument(
        "--ap",
        help="where2output",
        type=str,
    )
    parser.add_argument(
        "--shape",
        help="where2output",
        type=int,
        nargs='+',
    )
    args = parser.parse_args()

    img_root = args.ip
    alpha_root = args.ap
    video_save_path =args.vp #+ "test.avi" # 视频保存路径
    #img_len = len(os.listdir(img_root)) # 图片数量
    #imgs2video(img_root, video_save_path, 10, tuple(args.shape), 0, img_len)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4
    videoWriter = cv2.VideoWriter(video_save_path,fourcc,10,args.shape)
    imfilelist_o=os.listdir(img_root)
    aplhafilelist_o=os.listdir(alpha_root)
    imfilelist=[]
    aplhafilelist=[]
    for item in imfilelist_o:
        if item.endswith('.jpg') or item.endswith('.png'):
            imfilelist.append(item)
    for item in aplhafilelist_o:
        if item.endswith('.jpg') or item.endswith('.png'):
            aplhafilelist.append(item)
    aplhafilelist.sort()
    imfilelist.sort()
    for im,alpha in zip(imfilelist,aplhafilelist):
           #判断图片后缀是否是.jpg
            im =  os.path.join(args.ip, im)
            img = cv2.imread(im)
            alpha =  os.path.join(args.ap, alpha)
            alpha = cv2.imread(alpha)
            res = np.hstack([img, alpha])
            # print(type(img))  # numpy.ndarray类型  
            videoWriter.write(res)        #把图片写进视频
    videoWriter.release()
    print("imgs转video成功！")