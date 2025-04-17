import cv2
import numpy as np 
from KLT_Mask import getFeatures, estimateAllTranslation, applyGeometricTransformation
    
def objectTracking(rawVideo, init_mask, draw_bb=False, play_realtime=False, save_to_file=False):
    # initilize
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = np.empty((n_frame,),dtype=np.ndarray)
    frames_draw = np.empty((n_frame,),dtype=np.ndarray)
    masks = []
    
    for frame_idx in range(n_frame):
        _, frames[frame_idx] = rawVideo.read()
    
    # draw rectangle roi for target objects, or use default objects initilization
    masks.append(init_mask)
    n_object = len(masks[0])
    #if save_to_file:
    #    out = cv2.VideoWriter('output.mp4',0,cv2.VideoWriter_fourcc('X','V','I','D'),20.0,(frames[i].shape[1],frames[i].shape[0]))

    # Start from the first frame, do optical flow for every two consecutive frames.
    startXs, startYs = getFeatures(cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY), masks[0])
    for i in range(1, n_frame):
        print('Processing Frame', i)
        
        newXs, newYs = estimateAllTranslation(startXs, startYs, frames[i-1], frames[i])
        Xs, Ys , new_masks = applyGeometricTransformation(startXs, startYs, newXs, newYs, masks[i-1])
        
        # update feature points as required
        n_features_left = np.sum(Xs!=-1)
        print('number of Features: %d'%n_features_left)
        if n_features_left < 20:
            print('Generate New Features')
            startXs, startYs = getFeatures(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY), new_masks)

        # draw mask and visualize feature point for each object
        frames_draw[i] = frames[i].copy()
        for j in range(n_object):       
            #cv2.drawContours(frames_draw[i], np.array([masks[i-1][j]]), -1, (0,255,0), thickness=2)
            cv2.drawContours(frames_draw[i], np.array([new_masks[j]]), -1, (0,0,0), thickness=2) 
            for k in range(startXs.shape[0]):
                #cv2.circle(frames_draw[i], (int(startXs[k,j]), int(startYs[k,j])), 3, (0,255,0), thickness=1)
                cv2.circle(frames_draw[i], (int(Xs[k,j]), int(Ys[k,j])), 3, (0,0,0), thickness=1)          

        # update coordinates
        #startXs, startYs = getFeatures(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY), new_masks)
        startXs, startYs = Xs, Ys    
        masks.append(new_masks)
        
        # imshow if to play the result in real time
        if play_realtime:
            cv2.imshow("win", frames_draw[i])
            cv2.waitKey(10)
        if save_to_file:
            out.write(frames_draw[i])
