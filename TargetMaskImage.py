import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument(
                '-i'
                ,'--image'
                , required = False
                , help = 'path to the image'
                )
ap.add_argument(
                '-v'
                ,'--video'
                , required = False
                , help = 'path to the video'
                )                
args = vars(ap.parse_args())

pts = []
roi = None
mask = None
circles = None

def selectROI(event,x,y,flags,params):
  global pts,roi
  
  if event == cv2.EVENT_LBUTTONDOWN:
    pts = [(x,y)]
  elif event == cv2.EVENT_LBUTTONUP:
    pts.append((x,y))
    roi = image[pts[0][1]:pts[1][1],pts[0][0]:pts[1][0]]
    print 'Region of interest selected.'
    return roi

def boundInliers(x):
  q1 = np.percentile(x,25)
  q3 = np.percentile(x,75)
  iqr = 1.5 * (q3 - q1)
  min = int(q1 - iqr)
  max = int(q3 + iqr)
  return min,max
  
# read image
image = cv2.imread(args['image'])
orig = np.copy(image)

while True:

  key = cv2.waitKey(1) & 0xFF
  
  cv2.imshow('image',image)
  
  # show roi
  if roi is not None:
    cv2.imshow('roi',roi)
    
  # show mask
  if mask is not None:
    cv2.imshow('mask',mask)
  
  # select rectangular region of interest
  if key == ord('s'):
    print 'Select region of interest.'
    roi = cv2.setMouseCallback('image',selectROI)
  
  # find hsv threshold for image mask based on hsv distribution of roi
  if key == ord('m'):
    if roi is None:
      print 'Region of interest not selected.'
    else:
      # convert to hsv color representation
      roi_hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
      # separate attributes
      roi_h = roi_hsv[:,:,0].flatten()
      roi_s = roi_hsv[:,:,1].flatten()
      roi_v = roi_hsv[:,:,2].flatten()
      # define threshold values and clip
      h_min,h_max = np.clip(boundInliers(np.unique(roi_h)),0,180)
      s_min,s_max = np.clip(boundInliers(roi_s),0,255)
      v_min,v_max = np.clip(boundInliers(roi_v),0,255)
      hsv_min = np.array([h_min,s_min,v_min],dtype = np.uint8)
      hsv_max = np.array([h_max,s_max,v_max],dtype = np.uint8)
      print 'Found hsv threshold.'
      print 'hsv min: %i,%i,%i' % (hsv_min[0],hsv_min[1],hsv_min[2])
      print 'hsv max: %i,%i,%i' % (hsv_max[0],hsv_max[1],hsv_max[2])
  
  # show statistics
  if key == ord('z'):
    if roi_h is None or roi_s is None or roi_v is None:
      print 'Region of interest not selected or hsv threshold not defined.'
    else:
      fig = plt.figure()
      ax = plt.subplot(111)
      data = [np.unique(roi_h),roi_s,roi_v]
      plt.boxplot(data)
      plt.xticks([1,2,3],['Unique Hue (0 to 180)','Saturation (0 to 255)','Value (0 to 255)'])
      # plt.xlabel('HSV')
      plt.ylabel('Measurement')
      plt.title('Boxplot of HSV Color Space from Sampled ROI')
      plt.show()
      
      plt.hist(roi_h,180,[0,180],color = 'b')
      plt.xlabel('Unique Hue (0 to 180)')
      plt.ylabel('Frequency')
      plt.title('Histogram of Unique Hue Measurements from Sampled ROI')
      plt.xlim([0,180])
      plt.show()
      
      plt.hist(roi_s,255,[0,255],color = 'b')
      plt.xlabel('Saturation (0 to 255)')
      plt.ylabel('Frequency')
      plt.title('Histogram of Saturation Measurements from Sampled ROI')
      plt.xlim([0,255])
      plt.show()
      
      plt.hist(roi_v,255,[0,255],color = 'b')
      plt.xlabel('Value (0 to 255)')
      plt.ylabel('Frequency')
      plt.title('Histogram of Value Measurements from Sampled ROI')
      plt.xlim([0,255])
      plt.show()
      
      cv2.imwrite('roi.jpg',roi)
      
  # detect circle
  if key == ord('d'):
    if roi is None or hsv_min is None or hsv_max is None:
      print 'Region of interest not selected or hsv threshold not defined.'
    else:
      # convert to hsv color representation
      image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
      mask = cv2.inRange(image_hsv,hsv_min,hsv_max)
      mask_raw = np.copy(mask)
      white = np.copy(mask)
      white[:,:] = 255
      
      # repeatedly smooth image
      for i in range(3):
        mask = cv2.medianBlur(mask,21)
        
      # find circle with hough transform
      circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,5,mask.shape[1])
      if circles is None:
        print 'No circles found.'
      else:
        print 'Circles found.'
      
  # show detected circle
  if key == ord('c'):
    if circles is not None:
      circles = np.round(circles[0,:]).astype('int')
      for (x,y,r) in circles:        
        cv2.circle(image,(x,y),r,(0,255,0),4)
        cv2.rectangle(image,(x - 4,y - 4),(x + 4,y + 4),(0,255,0),-1)
        

        mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        mask_raw = cv2.cvtColor(mask_raw,cv2.COLOR_GRAY2BGR) 
        cv2.circle(mask,(x,y),r,(0,255,0),4)
        cv2.rectangle(mask,(x - 4,y - 4),(x + 4,y + 4),(0,255,0),-1)        
        white = cv2.cvtColor(white,cv2.COLOR_GRAY2BGR)    
        cv2.imshow("output", np.vstack([np.hstack([orig,image_hsv,white]),np.hstack([mask_raw,mask,image])]))
        cv2.waitKey(0)
        
        cv2.imwrite('ball_track_pipeline.jpg',np.vstack([np.hstack([orig,image_hsv,white]),np.hstack([mask_raw,mask,image])]))
        cv2.imwrite('ball_hsv.jpg',cv2.cvtColor(orig,cv2.COLOR_BGR2HSV))
        cv2.imwrite('mask.jpg',mask)
        cv2.imwrite('mask_raw.jpg',mask_raw)
        cv2.imwrite('output.jpg',image)
    
  # quit program
  if key == ord('q'):
    break

cv2.destroyAllWindows()
