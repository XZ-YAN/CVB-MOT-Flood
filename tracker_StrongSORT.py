# coding=UTF-8
# Code Author: Xuzhong Yan
import sys, argparse, os, platform, shutil, time, cv2, glob, math, turtle, random, warnings, logging
import numpy as np
from pathlib import Path
import pandas as pd
from collections import Counter
from collections import deque
warnings.filterwarnings('ignore')
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from io import StringIO
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from tqdm import tqdm
from functools import reduce
from scipy.optimize import fsolve
from matplotlib import rc
rc('animation', html='html5')
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from MOT_StrongSORT_box.models.common import DetectMultiBackend
from MOT_StrongSORT_box.models.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from MOT_StrongSORT_box.models.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                                     check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from MOT_StrongSORT_box.models.utils.torch_utils import select_device, time_sync
from MOT_StrongSORT_box.models.utils.plots import Annotator, colors, save_one_box
from MOT_StrongSORT_box.strong_sort.utils.parser import get_config
from MOT_StrongSORT_box.strong_sort.strong_sort import StrongSORT

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

# Calculate the area of IOU. 1 means truck or mixer, 2 means their parts
def calculate_iou(xmin1,xmax1,ymin1,ymax1,xmin2,xmax2,ymin2,ymax2):
    if xmin1>xmax2 or xmax1<xmin2 or ymin1>ymax2 or ymax1<ymin2:   
        return 0.0
    else:
        left_x = np.max([xmin1, xmin2])
        right_x = np.min([xmax1, xmax2])
        top_y = np.min([ymax1, ymax2])
        bottom_y = np.max([ymin1, ymin2])
        inter_area = (right_x - left_x) * (top_y - bottom_y)
        box_area_1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        box_area_2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        iou = inter_area / (box_area_1 + box_area_2 - inter_area)
        return iou
    
def detect(video, subset, config, weight_path, label):
    ''' initialize StrongSORT '''
    device ='0'
    device = select_device(device)
    
    strong_sort_weights='./MOT_StrongSORT_box/weights/osnet_x0_25_msmt17.pt' 
    config_strongsort='./MOT_StrongSORT_box/strong_sort/configs/strong_sort.yaml'
    cfg = get_config()
    cfg.merge_from_file(config_strongsort)
    strongsort = StrongSORT(strong_sort_weights,
                            device,
                            max_dist=cfg.STRONGSORT.MAX_DIST,
                            max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.STRONGSORT.MAX_AGE,
                            n_init=cfg.STRONGSORT.N_INIT,
                            nn_budget=cfg.STRONGSORT.NN_BUDGET,
                            mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                            ema_alpha=cfg.STRONGSORT.EMA_ALPHA)

    ''' Object Detection '''
    input_video = subset+'Video-'+video+'/'+video+'.mp4'
    output_video = subset+'Video-'+video+'/'+video+'_'+label+'_mot_result.avi'
    frame_path = subset+'/Video-'+video+'/saveframe/'

    # load trained CNN model
    cfg_detect = get_cfg()
    cfg_detect.MODEL.DEVICE='cuda'
    cfg_detect.merge_from_file(model_zoo.get_config_file(config))
    cfg_detect.MODEL.WEIGHTS = os.path.join(weight_path, 'model_final.pth') 
    cfg_detect.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # change the number of classes here
    cfg_detect.MODEL.ROI_HEADS.NUM_CLASSES = 2
    predictor = DefaultPredictor(cfg_detect)

    # load video
    cap = cv2.VideoCapture(input_video)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # save the processed video
    ret, init_image = cap.read()
    height, width = init_image.shape[0], init_image.shape[1]
    # path to save processed frame
    if not os.path.exists(frame_path): os.makedirs(frame_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, FPS, (width,height))
    object_category = ['building','vehicle']
    
    frame = 1
    results_for_evaluation = [] # store all the hypothesis for evaluation
    while (cap.isOpened()):
        ret, image = cap.read()
        if ret == True:
            print('\n------------ Video '+video+' Frame '+str(frame)+'/'+ str(total_frame)+' ------------')
            outputs = predictor(image)
            
            v = Visualizer(image[:,:,::-1], scale=1)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))   
            processed_image = v.get_image()[:, :, ::-1] 
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)  

            # get all boxes, scores, and classes from outputs['instances']
            all_boxes = outputs['instances'].__dict__['_fields']['pred_boxes'].__dict__['tensor'].tolist()
            all_scores = outputs['instances'].__dict__['_fields']['scores'].tolist()
            all_classes = outputs['instances'].__dict__['_fields']['pred_classes'].tolist()
            # number of all the detected objects
            n = len(all_boxes)

            if n>0:
                detections = np.zeros((n,6)) # [xmin, ymin, xmax, ymax, score, class]
                for i in range(n):
                    xmin, ymin, xmax, ymax = all_boxes[i][0], all_boxes[i][1], all_boxes[i][2], all_boxes[i][3]
                    detections[i] = np.array([xmin, ymin, xmax, ymax, all_scores[i], all_classes[i]])

                ''' StrongSORT update '''
                pred = [detections]
                for i, det in enumerate(pred):
                    det = torch.from_numpy(det) # convert numpy array to torch tensor
                    xywhs = xyxy2xywh(det[:, 0:4]) # x_center, y_center, width, height
                    confs = det[:, 4] # confidence score
                    clss = det[:, 5] # class
                    
                    # pass detections to strongsort
                    bbox_ids = strongsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), image) # [xmin, ymin, xmax, ymax, id, class, score]
                    print(bbox_ids)
                    
                    # draw boxes for visualization
                    if len(bbox_ids) != 0:
                        for bbox_id in bbox_ids:
                            ID, xmin, ymin = str(int(bbox_id[4])), max(0,int(bbox_id[0])), max(0,int(bbox_id[1]))
                            xmax, ymax = int(bbox_id[2]), int(bbox_id[3])
                            width, height = xmax-xmin, ymax-ymin
                            score = round(bbox_id[6], 2)
                            category = object_category[int(bbox_id[5])]

                            # draw bbox and features
                            cv2.rectangle(image, (xmin,ymin), (xmin+width,ymin+height), (0,255,0), 2)
                            # draw category and ID
                            cv2.putText(image,category,(xmin+10,ymin+20),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,0),2)
                            cv2.putText(image,'ID-'+ID,(xmin+10,ymin+40),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,0),2)

                            results_for_evaluation.append(str(frame)
                                              + ', ' + str(ID)       
                                              + ', ' + str(xmin)     
                                              + ', ' + str(ymin)     
                                              + ', ' + str(width)    
                                              + ', ' + str(height)   
                                              + ', ' + str(score)   
                                              + ', ' + str(-1)
                                              + ', ' + str(-1)
                                              + ', ' + str(-1)) 
                    #else: results_for_evaluation.append(str(frame) + ' none')
            else:
                strongsort.increment_ages()
                #results_for_evaluation.append(str(frame) + ' none')

            # draw frame number in each frame
            cv2.putText(image, 'Frame: '+str(int(frame)), (50,73), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            # save the video
            out.write(image)#cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
            # save the frame
            cv2.imwrite(frame_path+'f_'+str(frame)+'.jpg', image)
            # display the video
            #cv2.imshow('Enter Detection', cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            frame += 1
            gt_detections = []
            
        else:
            break
            
    return results_for_evaluation        
    out.release()
    cap.release()
    cv2.destroyAllWindows               
                
if __name__ == "__main__":
    
    # Mask-RCNN + ResNet101 + FPN:
    #label = 'Mask_RCNN_101'
    #config = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
    #weight_path = 'output/MaskRCNN'

    # Maks-RCNN + ResNet50 + Deformable Conv
    #label = 'Deformable_Mask_RCNN'
    #config = 'Misc/mask_rcnn_R_50_FPN_3x_dconv_c3-c5.yaml'
    #weight_path = 'output/DeformCNN'

    # Cascade Maks-RCNN + ResNet50 + FPN
    label = 'Cascade_Mask_RCNN'
    config = 'Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml'
    weight_path = 'output/CascadeRCNN'
    
    for subset in ['test/']:
        for i in [subset]:
            all_video_folder = [name for name in os.listdir(subset) if os.path.isdir(os.path.join(subset, name))]
        for video_folder in all_video_folder:
            video = video_folder[-4:]
            results_for_evaluation = detect(video, subset, config, weight_path, label)
            
