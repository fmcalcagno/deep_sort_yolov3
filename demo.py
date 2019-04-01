#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from tools import detectpath as dpath
from tools import detectpath as dpath
from deep_sort.detection import Detection as ddet
import os
import glob
import traceback
import copy
import json
warnings.filterwarnings('ignore')



def main(yolo,input_video,exit_video,json_file_people,json_file_cars):
    try:
    # Definition of the parameters
        max_cosine_distance = 1.5
        max_cosine_distance_car = 3.5
        nn_budget = None
        nms_max_overlap = 3
        
    # deep_sort 
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename,batch_size=1)
        
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        metric2 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance_car, nn_budget)

        tracker = Tracker(metric,max_iou_distance=1.5, max_age=5, n_init=1)

        tracker_cars = Tracker(metric2,max_iou_distance=5, max_age=5, n_init=0)

        writeVideo_flag = True 
        
        video_capture = cv2.VideoCapture(input_video)

        mask = dpath.make_mask("mask.png",0)
        color_mask = dpath.make_mask("mask.png")
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_TC89_KCOS)
        border = 75
        w = int(video_capture.get(3))
        h2 = int(video_capture.get(4))
        if writeVideo_flag:
        # Define the codec and create VideoWriter object
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))-border*2
            print("weight: {}, height: {}".format(w,h))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(exit_video , fourcc, 15, (w, h))
            list_file = open('detection.txt', 'w')
            frame_index = -1 
            
        fps = 0.0
        counted_people= dict()
        counted_cars= dict()
        positions_people= dict()
        positions_cars= dict()
        while True:

            ret, frame2 = video_capture.read()  
            if ret != True:
                break
            frame2 = frame2[border:h2-border, :]
            frame_original= frame2.copy()

        
            
            t1 = time.time()
            frame = dpath.detect_path (frame2,mask)
            image = Image.fromarray(frame[...,::-1]) #bgr to rgb

            boxs,objects = yolo.detect_image(image)
            features = encoder(frame,boxs)
            
            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            

          
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            detections_cars = copy.deepcopy(detections)
            detections_per = copy.deepcopy(detections)
            objects = [objects[i] for i in indices]
            # Call the tracker
            
            
            cv2.drawContours(frame_original, contours, -1, (255,255,0),2)

            for det,obj in zip(detections,objects):
                bbox = det.to_tlbr()
                if  obj in  ['car','truck','motorbike','boat','bicycle','bus']:
                    cv2.rectangle(frame_original,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,255,255), 1)
                    #cv2.putText(frame_original,obj ,(int(bbox[0]), int(bbox[1])),0, 5e-4 * 100, (0,255,0),1)
                elif obj in ['person']:
                    cv2.rectangle(frame_original,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,255), 1)
                else:
                    cv2.rectangle(frame_original,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,0), 1)
            
            detections_per2=detections_per.copy()
            for det,obj in zip(detections_per,objects):
                if obj not in ['person']:
                    detections_per.remove(det)

            detections_cars2=detections_cars.copy()

            for det,obj in zip(detections_cars2,objects):
                if obj not in ['car','truck','motorbike','boat','bicycle','bus']:
                    detections_cars.remove(det)
            
            tracker_cars.predict()
            tracker_cars.update(detections_cars)

            tracker.predict()
            tracker.update(detections_per)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 2:
                    continue 
                bbox = track.to_tlbr()
                center = track.to_center()
                cv2.rectangle(frame_original, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(frame_original, "p: " + str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 100, (0,255,0),2)
                path=dpath.check_input_output(center)
                if track.track_id in positions_people:
                    positions_people[track.track_id]["frames"]+=1
                    positions_people[track.track_id]["position"]["end"]=path
                else:
                    
                    positions_people[track.track_id]= dict()
                    positions_people[track.track_id]["frames"]=1
                    positions_people[track.track_id]["position"]= dict()
                    positions_people[track.track_id]["position"]["start"]=path
            


            for track in tracker_cars.tracks:
                if not track.is_confirmed() or track.time_since_update > 2:
                    continue 
                bbox = track.to_tlbr()
                cv2.rectangle(frame_original, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,100), 2)
                cv2.putText(frame_original, "c: " + str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 100, (0,255,100),2)
                center = track.to_center()
                path=dpath.check_input_output(center)
                if track.track_id in positions_cars:
                    positions_cars[track.track_id]["frames"]+=1
                    positions_cars[track.track_id]["position"]["end"]=path
                else:
                    
                    positions_cars[track.track_id]= dict()
                    positions_cars[track.track_id]["frames"]=1
                    positions_cars[track.track_id]["position"]= dict()
                    positions_cars[track.track_id]["position"]["start"]=path

         
           
            font = cv2.FONT_HERSHEY_SIMPLEX

            total_people= len(positions_people)
            total_cars= len(positions_cars)
            cv2.putText(frame_original,'Detected People: ' + str(total_people) ,
                (10, 35),font,0.8,( 0, 0, 0),2, cv2.FONT_HERSHEY_SIMPLEX,) 
            cv2.putText(frame_original,'Detected Cars: ' + str(total_cars),
                (10, 80),font,0.8,( 0, 0, 100),2, cv2.FONT_HERSHEY_SIMPLEX,)     
            cv2.imshow('', frame_original)
            
            if writeVideo_flag:
                # save a frame
                out.write(frame_original)
                frame_index = frame_index + 1
                list_file.write(str(frame_index)+' ')
                if len(boxs) != 0:
                    for i in range(0,len(boxs)):
                        list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
                list_file.write('\n')
                
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %f"%(fps))
            
            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        if writeVideo_flag:
            out.release()
            list_file.close()
        
        json_string_p = json.dumps(positions_people, ensure_ascii=False) 
        json_string_c = json.dumps(positions_cars, ensure_ascii=False) 
        datastore_p = json.loads(json_string_p)
        datastore_c = json.loads(json_string_c)
        with open(json_file_people, 'w') as f:
            json.dump(datastore_p, f)
        with open(json_file_cars, 'w') as f:
            json.dump(datastore_c, f)

        cv2.destroyAllWindows()
    except Exception as e:
    
        video_capture.release()
        if writeVideo_flag:
            out.release()
            list_file.close()
        
        json_string_p = json.dumps(positions_people, ensure_ascii=False) 
        json_string_c = json.dumps(positions_cars, ensure_ascii=False) 
        datastore_p = json.loads(json_string_p)
        datastore_c = json.loads(json_string_c)
        with open(json_file_people, 'w') as f:
            json.dump(datastore_p, f)
        with open(json_file_cars, 'w') as f:
            json.dump(datastore_c, f)
        cv2.destroyAllWindows()
        print("An exception occurred: ",e)
        traceback.print_exc()
if __name__ == '__main__':
    for filename in glob.glob("./videos/*.mkv"):
        
        output = "./output/output"+ os.path.splitext(os.path.basename(filename))[0] + '.avi'
        output_json_people= "./json/positions_people_" + os.path.splitext(os.path.basename(filename))[0] + '.json'
        output_json_cars= "./json/positions_cars_" + os.path.splitext(os.path.basename(filename))[0] + '.json'
        main(YOLO(),filename,output,output_json_people,output_json_cars)
        print("input: {} output:{}".format(filename,output))

    
