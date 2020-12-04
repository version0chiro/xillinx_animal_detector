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

def runDPU(dpu,img):
    
    """get tensor"""
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    outputHeight = outputTensors[0].dims[1]
    outputWidth = outputTensors[0].dims[2]
    outputChannel = outputTensors[0].dims[3]

    outputSize = outputHeight*outputWidth*outputChannel
    #softmax = np.empty(outputSize)

    batchSize = inputTensors[0].dims[0]

    shapeIn = (runSize,) + tuple([inputTensors[0].dims[i] for i in range(inputTensors[0].ndim)][1:])
    
    """ prepare batch input/output """
    outputData = []
    inputData = []
    outputData.append(np.empty((runSize,outputHeight,outputWidth,outputChannel), dtype = np.float32, order = 'C'))
    inputData.append(np.empty((shapeIn), dtype = np.float32, order = 'C'))
    
    """ init input image to input buffer """
    imageRun = inputData[0]
    imageRun[j,...] = img

    """ run with batch """
    job_id = dpu.execute_async(inputData,outputData)
    
    dpu.wait(job_id)

    predictions = outputData[0][0]
    predictions = predictions[0][0]
    predictions = softmax(predictions)
    # print("predictions shape: ",predictions.shape)
    y = np.argmax(predictions)
    animal = {
        0 : 'cat',
        1 : 'dog',
        2 : 'monkey',
        3 : 'cow',
        4 : 'elephant',
        5 : 'horse' ,
        6 : 'squirrel',
        7 : 'chicken' ,
        8 : 'spider' ,
        9 : 'sheep'
    }
    print("detected animal is : "+str(animal[y]))
    count = count + runSize
    
        
    return str(animal[y])


def runApp(threads, camera,model):
    

    
    g = xir.graph.Graph.deserialize(pathlib.Path(model))
    subgraphs = get_subgraph(g)
    assert len(subgraphs) == 1
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(runner.Runner(subgraphs[0],"run"))
    

    """ initialize camera """
    cam = cv2.VideoCapture(camera)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
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
        frame = cv2.cvtColor(frame,name,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA,False)
        cv2.imshow('frame',frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        fps.update()

        if key == ord("q"):
            break

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] elapsed FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()

    

def main():
    
    # command line arguments
    ap = argparse.ArgumentParser()
    
    ap.add_argument('-m','--model', type=str,
                    default='/home/root/animalRecognition/xillinx_animal_detector/dpu_densenetx_0.elf'
                    )
    
    ap.add_argument('-c', '--camera',
                    type=int,
                    default=0,
  	                help='camera number')
    ap.add_argument('-t', '--threads',
                    type=int,
                    default=1,
  	                help='Number of threads. Default is 1')
    args = ap.parse_args()


    runApp(args.threads, args.camera, args.model)

if __name__ == '__main__':
    main()