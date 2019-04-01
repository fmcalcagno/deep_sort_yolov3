import numpy as np 
import cv2

def make_mask(mask_file,i=0):
    mask= cv2.imread(mask_file,i)
    if i == 0:
        ret1,th1 = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
        return th1
    else:
        return mask


def detect_path(frame,mask):
    #h2=h//2
    #frame [:h2,:,:] = 255
    frame[mask==0,:] = 255

    return frame

def check_input_output(point):
    x,y=point
    out=""
    if  y>350:
        out= 'Down'
    elif y<350:
        if x>350:
            out =  'Right'
        else:
            out= 'Left'

    return out