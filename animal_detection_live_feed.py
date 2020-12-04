from ctypes import *
import cv2
import numpy as np
import runner
import os
import math
import threading
import time
import argparse
import json
import xir.graph 
import xir.subgraph
import pathlib
import imutils
import urllib.request

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def get_subgraph (g):
    sub = []
    root = g.get_root_subgraph()
    sub = [ s for s in root.children
    if s.metadata.get_attr_str ("device") == "DPU"]
    return sub

def runApp(threads, camera,model):
    

    listImage=os.listdir(image_dir)

    runTotal = len(listImage)
    
    global out_q
    out_q = [None] * runTotal
    g = xir.graph.Graph.deserialize(pathlib.Path(model))
    subgraphs = get_subgraph(g)
    assert len(subgraphs) == 1
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(runner.Runner(subgraphs[0],"run"))
    

    """ initialize camera """
    cam = cv2.VideoCapture(camera)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,400)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,400)
    if not (cam.isOpened()):
        print("[ERROR] failed to initialize camera",camera)
        exit()
    fps = FPS().start
    while True:
        
        ret,frame = cam.read()
        frame = imutils.resize(frame,width=200,height=200)
        greyScaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inputFrame = greyScaleFrame.reshape(-1,200,200,1).astype('float32')
        inputFrame = inputFrame/255.0
        name = runDPU(all_dpu_runners[i],inputFrame)
        print(name)
        cv2.imshow('frame',frame)
    img = []
    print(listImage)
    for i in range(len(listImage)):
        image = cv2.imread(os.path.join(image_dir,listImage[i]), cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # image = cv2.resize(image,200,200)
        image = image.reshape(-1,200,200,1).astype('float32')
        image = image/255.0
        img.append(image)


    """run with batch """
    threadAll = []
   
    start = 0
    
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start + (len(img)//threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i],in_q,listImage))
        threadAll.append(t1)
        start = end
    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1
    
    fps = float(runTotal/timetotal)
    print("FPS=%.2f, total frames = %.0f , time = %.4f seconds" %(fps,runTotal,timetotal))


def main():
    
    # command line arguments
    ap = argparse.ArgumentParser()
    
    ap.add_argument('-m','--model', type=str,
                    default='/home/root/animalRecognition/xillinx_animal_detector/dpu_densenetx_0.elf'
                    )
    
    ap.add_argument('-c', '--camera',
                    type=int,
                    default='0,
  	                help='camera number')
    ap.add_argument('-t', '--threads',
                    type=int,
                    default=1,
  	                help='Number of threads. Default is 1')
    args = ap.parse_args()


    runApp(args.threads, args.camera, args.model)

if __name__ == '__main__':
    main()