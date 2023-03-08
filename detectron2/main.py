from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import detectron2

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow
import sys
import json
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import time
from nanonets_object_tracking.deepsort import *


global minmaxavg_dict
minmaxavg_dict = {"max":[],"min":[],"avg":[],"num_of_bbox":0,"numof_poses":0, "num_of_ppl_tracked":0}
class detector():
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("hieve", {}, "label/train.json", "images/")
    def  __init__(self):
        self.cfg=get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS=r"C:\Users\Aditya\Documents\Aditya\independent study\keypointdetector\detectron2\weights\finetuned.pth"
        #self.cfg.MODEL.WEIGHTS='model_final.pth'
        #self.cfg.MODEL.WEIGHTS = r"C:\Users\Aditya\Documents\Aditya\model_final.pth"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE="cuda"
        print(self.cfg.MODEL.DEVICE)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # hand
        self.cfg.MODEL.RETINANET.NUM_CLASSES = 1
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 14
        self.cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((14, 1), dtype=float).tolist()

        MetadataCatalog.get("hieve").thing_classes=["person"]
        self.keypoint_names = ["Nose", "Chest", "Right_shoulder", "Right-elbow",
                          "Right_wrist","Left_shoulder","Left_elbow","Left_wrist",
                          "Right_hip","Right_knee","Right_ankle","Left_hip","Left_knee",
                          "Left_ankle"]
        self.color=[(250,0,0),(250,255,0),(0,255,0),(250,0,255),(0,20,100),(100,200,42),(250,40,209),(0,70,190),(250,20,129)]
        keypoint_flip_map = []
        MetadataCatalog.get("hieve").keypoint_names = self.keypoint_names
        MetadataCatalog.get("hieve").keypoint_flip_map = keypoint_flip_map
        self.hands_metadata = MetadataCatalog.get("hieve")
        self.predictor = DefaultPredictor(self.cfg)


    def onvid(self,vidpath,vid_,num_of_bbox,numof_poses,num_of_ppl_tracked,all_tracks):
    #def onvid(self, vidpath):
        dict_to_write = {'annolist': []}
        tracklets={}
        cap=cv2.VideoCapture(vidpath)
        deepsort = deepsort_rbc('nanonets_object_tracking/ckpts/model640.pt')
        frame_count=1
        while(True):
            to_append = {'ignore_regions': [], 'image': [{"name": frame_count}]}
            frame_count+=1
            ret, image=cap.read()
            if(not ret):
                break
            #k+=1

           # image=cv2.resize(image,(640,480))
            predictions = self.predictor(image)
            bbox_=predictions["instances"].pred_boxes
            key=predictions["instances"].pred_keypoints

           # num_of_bbox+=len(bbox_)
            minmaxavg_dict["num_of_bbox"]+=len(bbox_)
            detections=[]
            scores_list=[]
            all_key_list = []
            bbox=[]
            for ind,score,points in zip(bbox_,predictions["instances"].scores,key):
                    x,y,w,h=int(ind[0]),int(ind[1]),int(ind[2])-int(ind[0]),int(ind[3])-int(ind[1])
                    detections.append([x,y,w,h])
                    scores_list.append(round(float(score),2))
                    key_list=[]
                    for kp_id,kp in enumerate(points):
                        key_list.append([int(kp_id+1),int(kp[0]),int(kp[1]),float(kp[2])])
                    all_key_list.append(key_list)
                    cv2.rectangle(image,(int(ind[0]),int(ind[1])),(int(ind[2]),int(ind[3])),(0,255,5),2)
            #print(detections,scores_list)
            detections = np.array(detections)
            out_scores = np.array(scores_list)
            if(len(detections)>0):
                tracker, detections_class,kp = deepsort.run_deep_sort(image, out_scores, detections,all_key_list)
            else:
                 dict_to_write['annolist'].append(to_append)
                 continue

            for track,kps in zip(tracker.tracks,kp):
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bounding_box = track.to_tlbr()

                key_list = []
                for  actual_points in kps:
                    key_list.append( {"id": [actual_points[0]-1], "x": [actual_points[1]], "y": [actual_points[2]], "score": [actual_points[3]]})


                # Get the corrected/predicted bounding box
                id_num = str(track.track_id)  # Get the ID for the particular track.
                if(id_num not in tracklets):
                    tracklets[id_num]=0
                else:
                    tracklets[id_num]+=1
                #num_of_ppl_tracked+=1
                minmaxavg_dict["num_of_ppl_tracked"]+=1
                features = track.features  # Get the feature vector corresponding to the detection.
                bbox.append(
                    {"x1": [int(bounding_box[0])], "y1": [int(bounding_box[1])], "x2": [int(bounding_box[2])], "y2": [int(bounding_box[3])],
                     "score": [0.9],
                     "track_id": [int(id_num)], "annopoints": [{"point":key_list}]})


                # Draw bbox from tracker.
                #cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(image, str(id_num), (int(bounding_box[0]), int(bounding_box[1])), 0, 5e-3 * 200, (0, 0, 255), 2)

            to_append["annorect"] = bbox
            dict_to_write['annolist'].append(to_append)

            #print(dict_to_write, "\n")
           # numof_poses+=len(kp)
            minmaxavg_dict["numof_poses"]+=len(kp)
            print(len(kp))
            for j in kp:

                for a, m in enumerate(j):
                    image = cv2.circle(image, (int(m[1]), int(m[2])), 3, (255,255,0), -1)

            cv2.imshow("frame", image)
            k=cv2.waitKey(1)
            if(k==ord('q')):
                break

        with open("jsons/"+vid_+".json", "w") as outfile:
            json.dump(dict_to_write, outfile)

        if (len(tracklets) > 0):
            minmaxavg_dict["min"].append(min(i for i in list(tracklets.values()) if i > 0))
            minmaxavg_dict["max"].append(max(list(tracklets.values())))
            minmaxavg_dict["avg"].append(sum(list(tracklets.values()))/len(tracklets))

       # print(minmaxavg_dict)
det=detector()

#"C:\Users\Aditya\Documents\Aditya\independent study\New folder\HIE20\HIE20\videos\1.mp4"
#C:\Users\Aditya\Documents\Aditya\independent study\keypointdetector\detectron2\HIE20test\HIE20test\videos\22.mp4

num_of_bbox=0
numof_poses=0

num_of_ppl_tracked=0

all_tracks=[]
for vid in os.listdir(r"C:\Users\Aditya\Documents\Aditya\independent study\New folder\HIE20\HIE20\test"):
    det.onvid(r"C:\Users\Aditya\Documents\Aditya\independent study\New folder\HIE20\HIE20\test/"+vid,vid.split(".")[0],num_of_bbox,numof_poses,num_of_ppl_tracked,all_tracks)
print(sum(minmaxavg_dict['avg'])/len(minmaxavg_dict['avg']),max(minmaxavg_dict["max"]),min(minmaxavg_dict["min"]),"numof_poses ",
      minmaxavg_dict["numof_poses"],"num_of_ppl_tracked ",minmaxavg_dict["num_of_bbox"],"num_of_ppl_tracked ",minmaxavg_dict["num_of_ppl_tracked"])
#det.onvid(r"C:\Users\Aditya\Documents\Aditya\independent study\keypointdetector\unity_videos\RGB1bb09c4c-5aeb-4b9b-b42a-989c5943e6cb.avi")
