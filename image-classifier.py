import cv2
import numpy as np
import matplotlib.pyplot as plt

#specify the image name taken for image processing
img = cv2.imread('mdb058.jpg')

#double thresholding
ret,img2 = cv2.threshold(img,177,255,cv2.THRESH_BINARY)
kernel = np.ones((2,2),np.uint8)
img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
sure_bg = cv2.dilate(img2,kernel,iterations=3)

#edge detection
edge = cv2.Canny(sure_bg,100,200)
edg = np.asarray(edge,dtype="uint8")

h,w,d = sure_bg.shape
#img2 = cv2.morphologyEx(img2,cv2.MORPH_OPEN,kernel, iterations = 2)

#mask function
mask = np.zeros(img.shape, dtype = "uint8")
cv2.rectangle(mask, (220,200), (800,1024), (255, 255, 255), -1)

#superimposition of images/ combining the thresholding and mask images
img4 = cv2.bitwise_or(img,sure_bg)
img3 = cv2.bitwise_and(img4,mask)

#combining edge and original image
#img5 = cv2.bitwise_or(img3,edg)

#print(sh)
#shows image
plt.figure(1)
plt.subplot(111)
plt.imshow(img3)
plt.show()

#cv2.imshow('result',edge)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
