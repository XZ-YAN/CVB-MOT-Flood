# Code Author: Xuzhong Yan
import sys
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from skimage.feature import corner_harris, corner_shi_tomasi, peak_local_max
from skimage import transform as tf
from numpy.linalg import inv,det
import cv2
from scipy import signal

###################################################################### utils
def GaussianPDF_1D(mu, sigma, length):
    # create an array
    half_len = length / 2

    if np.remainder(length, 2) == 0:
        ax = np.arange(-half_len, half_len, 1)
    else:
        ax = np.arange(-half_len, half_len + 1, 1)

    ax = ax.reshape([-1, ax.size])
    denominator = sigma * np.sqrt(2 * np.pi)
    nominator = np.exp( -np.square(ax - mu) / (2 * sigma * sigma) )

    return nominator / denominator

def GaussianPDF_2D(mu, sigma, row, col):
    # create row vector as 1D Gaussian pdf
    g_row = GaussianPDF_1D(mu, sigma, row)
    # create column vector as 1D Gaussian pdf
    g_col = GaussianPDF_1D(mu, sigma, col).transpose()

    return signal.convolve2d(g_row, g_col, 'full')

def rgb2gray(I_rgb):
    r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
    I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return I_gray

def interp2(v, xq, yq):
    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()

    h = v.shape[0]
    w = v.shape[1]
    if xq.shape != yq.shape:
        raise 'query coordinates Xq Yq should have same shape'

    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)

    x_floor[x_floor<0] = 0
    y_floor[y_floor<0] = 0
    x_ceil[x_ceil<0] = 0
    y_ceil[y_ceil<0] = 0

    x_floor[x_floor>=w-1] = w-1
    y_floor[y_floor>=h-1] = h-1
    x_ceil[x_ceil>=w-1] = w-1
    y_ceil[y_ceil>=h-1] = h-1

    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

    if dim_input == 2:
        return interp_val.reshape(q_h,q_w)
    return interp_val
######################################################################
    
# choose point coordinates for mask contour in the initial video frame
def SetPoints(windowname, img):
    points = []
    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(temp_img, (x, y), 5, (0, 0, 255), -1)
            points.append([x, y])
            cv2.imshow(windowname, temp_img)

    temp_img = img.copy()
    cv2.namedWindow(windowname)
    cv2.imshow(windowname, temp_img)
    cv2.setMouseCallback(windowname, onMouse)
    key = cv2.waitKey(0)
    if key == 13:  # Enter
        #print('draw point', points)
        del temp_img
        cv2.destroyAllWindows()
        return points
    elif key == 27:  # ESC
        print('skip this image')
        del temp_img
        cv2.destroyAllWindows()
        return
    else:
        print('retry!')
        return SetPoints(windowname, img)

def getFeatures(img, masks, use_shi=False):
    n_object = len(masks)
    N = 0
    temp = np.empty((n_object,),dtype=np.ndarray) # temporary storage of x,y coordinates

    for i in range(n_object):
        xmin, xmax = min((np.array(masks[i]).T)[0]), max((np.array(masks[i]).T)[0])
        ymin, ymax = min((np.array(masks[i]).T)[1]), max((np.array(masks[i]).T)[1])
        if xmin == xmax: xmin = xmin-1
        if ymin == ymax: ymin = ymin-1
        box_roi = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        #cv2.imshow("box_roi", box_roi)

        h, w = box_roi.shape
        mask_new = np.zeros((h, w), dtype=np.uint8)
        pts = np.vstack((np.array(masks[i]).T[0]-xmin, np.array(masks[i]).T[1]-ymin)).astype(np.int32).T
        cv2.fillPoly(mask_new, [pts], (255), 8, 0)
        #cv2.imshow("mask", mask_new)
        # roi is a greyscale rectangle where pixels inside mask != 0 and pixles outside the mask = 0
        roi = cv2.bitwise_and(box_roi, box_roi, mask=mask_new)
        #cv2.imshow("roi",roi)

        if np.shape(roi)[0] == 1:
            roi = np.concatenate((roi, roi), axis = 0)
        if np.shape(roi)[1] == 1:
            roi = np.concatenate((roi, roi), axis = 1)
        

        # extract corner features
        if use_shi:
            corner_response = corner_shi_tomasi(roi)
        else:
            corner_response = corner_harris(roi)

        coordinates = peak_local_max(corner_response, num_peaks=100, exclude_border=2)
        coordinates[:,1] += xmin
        coordinates[:,0] += ymin
        temp[i] = coordinates
        if coordinates.shape[0] > N:
            N = coordinates.shape[0]        
    feature_x = np.full((N,n_object),-1)
    feature_y = np.full((N,n_object),-1)

    for i in range(n_object):
        n_feature = temp[i].shape[0]
        feature_x[0:n_feature,i] = temp[i][:,1]
        feature_y[0:n_feature,i] = temp[i][:,0]
    
    return feature_x, feature_y    

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
    WINDOW_SIZE = 25
    X = startX
    Y = startY
    mesh_x, mesh_y = np.meshgrid(np.arange(WINDOW_SIZE), np.arange(WINDOW_SIZE))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    mesh_x_flat_fix = mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
    mesh_y_flat_fix = mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
    coor_fix = np.vstack((mesh_x_flat_fix, mesh_y_flat_fix))
    I1_value = interp2(img1_gray, coor_fix[[0], :], coor_fix[[1], :])
    Ix_value = interp2(Ix, coor_fix[[0], :], coor_fix[[1], :])
    Iy_value = interp2(Iy, coor_fix[[0], :], coor_fix[[1], :])
    I = np.vstack((Ix_value, Iy_value))
    A = I.dot(I.T)
    
    for _ in range(15):
        mesh_x_flat = mesh_x.flatten() + X - np.floor(WINDOW_SIZE / 2)
        mesh_y_flat = mesh_y.flatten() + Y - np.floor(WINDOW_SIZE / 2)
        coor = np.vstack((mesh_x_flat,mesh_y_flat))
        I2_value = interp2(img2_gray, coor[[0],:], coor[[1],:])
        Ip = (I2_value-I1_value).reshape((-1,1))
        b = -I.dot(Ip)
        if det(A) == 0:
            solution = A.dot(b)
        else:
            solution = inv(A).dot(b)
        # solution = np.linalg.solve(A, b)
        X += solution[0,0]
        Y += solution[1,0]
    return X, Y

def estimateAllTranslation(startXs, startYs, img1, img2):
    I = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    I = cv2.GaussianBlur(I,(5,5),0.2)
    Iy, Ix = np.gradient(I.astype(float))

    startXs_flat = startXs.flatten()
    startYs_flat = startYs.flatten()
    newXs = np.full(startXs_flat.shape,-1,dtype=float)
    newYs = np.full(startYs_flat.shape,-1,dtype=float)
    for i in range(np.size(startXs)):
        if startXs_flat[i] != -1:
            newXs[i], newYs[i] = estimateFeatureTranslation(startXs_flat[i], startYs_flat[i], Ix, Iy, img1, img2)
    newXs = np.reshape(newXs, startXs.shape)
    newYs = np.reshape(newYs, startYs.shape)
    return newXs, newYs

def applyGeometricTransformation(startXs, startYs, newXs, newYs, masks):
    n_object = len(masks)
    Xs = newXs.copy()
    Ys = newYs.copy()

    new_masks = []
    for obj_idx in range(n_object):
        len_mask = len(masks[obj_idx])
        mask = np.array(masks[obj_idx])
        newmask = np.zeros_like(mask)

        startXs_obj = startXs[:,[obj_idx]]
        startYs_obj = startYs[:,[obj_idx]]
        newXs_obj = newXs[:,[obj_idx]]
        newYs_obj = newYs[:,[obj_idx]]
        desired_points = np.hstack((startXs_obj,startYs_obj))
        actual_points = np.hstack((newXs_obj,newYs_obj))
        t = tf.SimilarityTransform()#EuclideanTransform()
        t.estimate(dst=actual_points, src=desired_points)
        mat = t.params
        
        # estimate the new mask with only the inliners
        THRES = 5
        projected = mat.dot(np.vstack((desired_points.T.astype(float), np.ones([1,np.shape(desired_points)[0]]))))
        distance = np.square(projected[0:2,:].T - actual_points).sum(axis = 1)

        actual_inliers = actual_points[distance < THRES]
        desired_inliers = desired_points[distance < THRES]

        if np.shape(desired_inliers)[0]<4:
            print('too few points')
            actual_inliers = actual_points
            desired_inliers = desired_points

        t.estimate(dst=actual_inliers, src=desired_inliers)
        mat = t.params
        
        coords = np.vstack((mask[:,:].T, np.ones((len_mask))))
        new_coords = mat.dot(coords)
        newmask[:,:] = new_coords[0:2,:].T
        newmask = newmask.tolist()
        
        Xs[distance >= THRES,obj_idx] = -1
        Ys[distance >= THRES,obj_idx] = -1
        new_masks.append(newmask)
    return Xs, Ys, new_masks 

correct_n = 0 
pred_n_lastframe = 0 
appear_frame = 0 
KLT_features = [] 
Images = [] 
Scores = [] 
Classes = [] 
Polygons = [] 
def KLT(pred_n, all_scores, all_classes, all_boxes, all_polygons, image):

    global correct_n, pred_n_lastframe, appear_frame, KLT_features, Images, Scores, Classes, Polygons
    start_frame, end_frame = 5, 5 
    pred_results = {} 
    if pred_n != 0:
        for i in range(len(all_classes)):
            key = 1000 * (i + 1) + int(all_classes[i])
            pred_results[key] = all_polygons[i]
            

    ## 1 no objects predicted 
    if pred_n == 0 and correct_n == 0:
        print('1 no object detected')
                

    ## 2 objects firstly predicted 
    elif pred_n > 0 and correct_n == 0:
     
        pred_n_lastframe = pred_n
        if pred_n == pred_n_lastframe: appear_frame += 1
        else: appear_frame = 0

        
        if appear_frame <= start_frame:
            print('2.1 not sure predicted objects correct or not')
        
        else:
            print('2.2 objects first correctly predicted')
            appear_frame = 0
            correct_n = pred_n 
            
            feature_x, feature_y = getFeatures(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), all_polygons, use_shi=True)
            for i in range(correct_n):
                for j in range(len(feature_x)):
                    cv2.circle(image, (feature_x[j][i], feature_y[j][i]), radius=5, color=(0, 255, 0), thickness=2)

            KLT_features = [feature_x, feature_y]
            Images = image
            Scores = all_scores
            Classes = all_classes
            Polygons = all_polygons


    ## 3 objects missed or left 
    elif pred_n < correct_n and correct_n != 0:
        
        if pred_n == pred_n_lastframe: appear_frame += 1
        else: appear_frame = 0
        pred_n_lastframe = pred_n

        
        if appear_frame < end_frame:
            print('3.1 objects missed, not sure left or not, so update KLT bbox')
            
                      
            newX, newY = estimateAllTranslation(KLT_features[0], KLT_features[1], Images, image)
            
            if len(KLT_features[0])==0 or len(KLT_features[1])==0 or len(newX)==0 or len(newY)==0: KLT_mask = Polygons
            else: X, Y, KLT_mask = applyGeometricTransformation(KLT_features[0], KLT_features[1], newX, newY, Polygons)
            
            MASK = []
            MASK_ARRAY = []
            for mask in KLT_mask:
                mask = np.array(mask)
                mask[mask < 0] = 1
                MASK_ARRAY.append(mask)
                MASK.append(list(mask))
            KLT_mask = MASK

            
            index = []
            for i in range(len(KLT_mask)):
                test = []
                for coord in KLT_mask[i]:
                    if coord[0] == 1 and coord[1] == 1:
                        test.append(1)
                test = list(set(test))
                if test == [1]:
                    KLT_mask = Polygons
                    MASK_ARRAY = Polygons
            
            
            feature_x, feature_y = getFeatures(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),KLT_mask,use_shi=True)
            for i in range(len(KLT_mask)):
                for j in range(len(feature_x)):
                    cv2.circle(image, (feature_x[j][i], feature_y[j][i]), radius=5, color=(0, 255, 0), thickness=2)
            
            KLT_boxes = np.zeros((0,4))
            for i in range(len(KLT_mask)):
                xmin, xmax = min((np.array(KLT_mask[i]).T)[0]), max((np.array(KLT_mask[i]).T)[0])
                ymin, ymax = min((np.array(KLT_mask[i]).T)[1]), max((np.array(KLT_mask[i]).T)[1])
                KLT_boxes = np.row_stack((KLT_boxes, np.array([xmin, ymin, xmax, ymax])))
            all_boxes = KLT_boxes
            all_scores = Scores
            all_classes = Classes
            if MASK_ARRAY[0].all() != Polygons[0].all(): Polygons = MASK_ARRAY

        
        else:
            print('3.2 objects left')
            appear_frame = 0
            correct_n = pred_n 
            if pred_n != 0:

                
                all_polygons = list(all_polygons)
                for k in range(len(all_polygons)):
                    x_row, y_row = list(set(all_polygons[k].T[0])), list(set(all_polygons[k].T[1]))
                    if len(all_polygons[k])<4 or len(x_row)<3 or len(y_row)<3:
                        all_polygons[k] = np.array([[int(all_boxes[k][0]),int(all_boxes[k][1])],
                                                    [int(all_boxes[k][2]),int(all_boxes[k][1])],
                                                    [int(all_boxes[k][0]),int(all_boxes[k][3])],
                                                    [int(all_boxes[k][2]),int(all_boxes[k][3])]])
                all_polygons = np.array(all_polygons)

                
                feature_x, feature_y = getFeatures(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), all_polygons, use_shi=True)
                for i in range(len(all_polygons)):
                    for j in range(len(feature_x)):
                        cv2.circle(image, (feature_x[j][i], feature_y[j][i]), radius=5, color=(0, 255, 0), thickness=2)
                KLT_features = [feature_x, feature_y]
                
            Images = image
            Scores = all_scores
            Classes = all_classes
            Polygons = all_polygons


    ## 4 new objects come in or old objects return 
    elif pred_n > correct_n and correct_n != 0:
        
        if pred_n == pred_n_lastframe: appear_frame += 1
        else: appear_frame = 0
        pred_n_lastframe = pred_n

        
        if appear_frame < start_frame:
            print('4.1 objects occur, not sure correct or not')
            
            
            newX, newY = estimateAllTranslation(KLT_features[0], KLT_features[1], Images, image)
            
            if len(KLT_features[0])==0 or len(KLT_features[1])==0 or len(newX)==0 or len(newY)==0: KLT_mask = Polygons
            else: X, Y, KLT_mask = applyGeometricTransformation(KLT_features[0], KLT_features[1], newX, newY, Polygons)
            
            MASK = []
            MASK_ARRAY = []
            for mask in KLT_mask:
                mask = np.array(mask)
                mask[mask < 0] = 1
                MASK_ARRAY.append(mask)
                MASK.append(list(mask))
            KLT_mask = MASK
            
            
            index = [] 
            for i in range(len(KLT_mask)):
                test = []
                for coord in KLT_mask[i]:
                    if coord[0] == 1 and coord[1] == 1:
                        test.append(1)
                test = list(set(test))
                if test == [1]:
                    KLT_mask = Polygons
                    MASK_ARRAY = Polygons
            
            
            
            feature_x, feature_y = getFeatures(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),KLT_mask,use_shi=True)
            for i in range(len(KLT_mask)):
                for j in range(len(feature_x)):
                    cv2.circle(image, (feature_x[j][i], feature_y[j][i]), radius=5, color=(0, 255, 0), thickness=2)
            
            KLT_boxes = np.zeros((0,4))
            for i in range(len(KLT_mask)):
                xmin, xmax = min((np.array(KLT_mask[i]).T)[0]), max((np.array(KLT_mask[i]).T)[0])
                ymin, ymax = min((np.array(KLT_mask[i]).T)[1]), max((np.array(KLT_mask[i]).T)[1])
                KLT_boxes = np.row_stack((KLT_boxes, np.array([xmin, ymin, xmax, ymax])))
            all_boxes = KLT_boxes
            all_scores = Scores
            all_classes = Classes
            if MASK_ARRAY[0].all() != Polygons[0].all(): Polygons = MASK_ARRAY
            
        
        else:
            print('4.2 correct objects come in')
            appear_frame = 0
            correct_n = pred_n 

            
            all_polygons = list(all_polygons)
            for k in range(len(all_polygons)):
                x_row, y_row = list(set(all_polygons[k].T[0])), list(set(all_polygons[k].T[1]))
                if len(all_polygons[k])<4 or len(x_row)<3 or len(y_row)<3:
                    all_polygons[k] = np.array([[int(all_boxes[k][0]),int(all_boxes[k][1])],
                                                [int(all_boxes[k][2]),int(all_boxes[k][1])],
                                                [int(all_boxes[k][0]),int(all_boxes[k][3])],
                                                [int(all_boxes[k][2]),int(all_boxes[k][3])]])
            all_polygons = np.array(all_polygons)
            
            
            feature_x, feature_y = getFeatures(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), all_polygons, use_shi=True)

            for i in range(len(all_polygons)):
                for j in range(len(feature_x)):
                    cv2.circle(image, (feature_x[j][i], feature_y[j][i]), radius=5, color=(0, 255, 0), thickness=2)
            Polygons = all_polygons
            
        KLT_features = [feature_x, feature_y]
        Images = image
        Scores = all_scores
        Classes = all_classes        


    ## 5 object correctly detected, use both detector & KLT to generate weighted mask 
    elif pred_n == correct_n and correct_n != 0:
        print('5 objects corrrectly detected and tracked')
        appear_frame = 0
        correct_n = pred_n 

        
        all_polygons = list(all_polygons)
        for k in range(len(all_polygons)):
            x_row, y_row = list(set(all_polygons[k].T[0])), list(set(all_polygons[k].T[1]))
            if len(all_polygons[k])<4 or len(x_row)<3 or len(y_row)<3:
                all_polygons[k] = np.array([[int(all_boxes[k][0]),int(all_boxes[k][1])],
                                            [int(all_boxes[k][2]),int(all_boxes[k][1])],
                                            [int(all_boxes[k][0]),int(all_boxes[k][3])],
                                            [int(all_boxes[k][2]),int(all_boxes[k][3])]])
        all_polygons = np.array(all_polygons)
                        
        
        feature_x, feature_y = getFeatures(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), all_polygons, use_shi=True)
        
        if len(feature_x) == 0 or len(feature_y) == 0:
            feature_x = KLT_features[0]
            feature_y = KLT_features[1]
            
        for i in range(correct_n):
            for j in range(len(feature_x)):
                cv2.circle(image, (feature_x[j][i], feature_y[j][i]), radius=5, color=(0, 255, 0), thickness=2)
                
        KLT_features = [feature_x, feature_y]
        Images = image
        Scores = all_scores
        Classes = all_classes
        Polygons = all_polygons
               
    return correct_n, all_scores, all_classes, all_boxes, image
