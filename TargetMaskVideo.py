import argparse
import numpy as np
import cv2

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
selectMode = False
iter = 0

def selectROI(event,x,y,flags,params):
  global pts,roi
  
  if event == cv2.EVENT_LBUTTONDOWN:
    pts = [(x,y)]
  elif event == cv2.EVENT_LBUTTONUP:
    pts.append((x,y))
    roi = frame[pts[0][1]:pts[1][1],pts[0][0]:pts[1][0]]
    print 'Region of interest selected.'
    return roi

def boundInliers(x):
  q1 = np.percentile(x,25)
  q3 = np.percentile(x,75)
  iqr = 1.5 * (q3 - q1)
  min = int(q1 - iqr)
  max = int(q3 + iqr)
  return min,max
  
# capture video
video = cv2.VideoCapture(args['video'])

# set frame size
# video.set(cv2.CAP_PROP_FRAME_WIDTH,816)
# video.set(cv2.CAP_PROP_FRAME_WIDTH,612)

while True:
  # retrieve next frame
  grab,frame = video.read()
  
  if not grab:
    break
    
  key = cv2.waitKey(1) & 0xFF
  
  # enter selection mode
  if key == ord('j'):
    selectMode = True
    while selectMode:
      cv2.imshow('frame',frame)
      key = cv2.waitKey(1) & 0xFF
      if key == ord('j'):
        selectMode = False
      
      # show roi
      if roi is not None:
        cv2.imshow('roi',roi)
        
      # show mask
      if mask is not None:
        cv2.imshow('mask',mask)
  
      # select rectangular region of interest
      if key == ord('s'):
        print 'Select region of interest.'
        roi = cv2.setMouseCallback('frame',selectROI)
      
      # find hsv threshold for frame mask based on hsv distribution of roi
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
      
  # detect circle
  if roi is not None and hsv_min is not None and hsv_max is not None:
    # convert to hsv color representation
    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv,hsv_min,hsv_max)
    
    # repeatedly smooth frame
    for i in range(3):
      mask = cv2.medianBlur(mask,21)
    
    # find circle with hough transform
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,5,mask.shape[1])
      
  # show detected circle
  if circles is not None:
    circles = np.round(circles[0,:]).astype('int')
    for (x,y,r) in circles:
      cv2.circle(frame,(x,y),r,(0,255,0),6)
      cv2.rectangle(frame,(x - 6,y - 6),(x + 6,y + 6),(0,255,0),-1)
    
  # quit program
  if key == ord('q'):
    break
  
  cv2.imshow('frame',frame)
  cv2.imwrite('G:/Temp Output/frame' + str(iter) + '.jpg',frame)
  iter += 1

video.release()
cv2.destroyAllWindows()
