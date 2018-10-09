import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
TRAIN_DIR = '/home/samanyu/Documents/kalpana/test'
for img in (os.listdir(TRAIN_DIR)):
	path = os.path.join(TRAIN_DIR,img)
	img_name = os.path.split(path)[1]
	img = cv2.imread(path,0)
	#extracting image number
	splt_path = os.path.split(path)
	split_image_name = splt_path[1]
	image_number = split_image_name[5]
	#if image number is even or odd
	#if odd
	if(int(image_number)%2!=0):
		img = cv2.flip(img,1)
	#elif(int(image_number)%2!=0):
		#img = cv2.flip(img,1)
	
	ret,img2 = cv2.threshold(img,177,255,cv2.THRESH_BINARY)
	kernel = np.ones((2,2),np.uint8)
	sure_bg = cv2.dilate(img2,kernel,iterations=3)
	#h,w,d = sure_bg.shape
	#img2 = cv2.morphologyEx(img2,cv2.MORPH_OPEN,kernel, iterations = 2)
	mask = np.zeros(img.shape, dtype = "uint8")
	cv2.rectangle(mask, (260,250), (750,1000), (255, 255, 255),-1)
	#print(h , w , d)
	img4 = cv2.bitwise_or(img,sure_bg)
	img3 = cv2.bitwise_and(img4,mask)
	row,col= img.shape
	#help(np.zeros)
	#output = np.zeros((row,col),np.uint8)
	for i in range(row):
		for j in range(col):
			if img3[i][j]==255:
				img3[i][j]=img[i][j]
			else:
				img3[i][j] =0

#cv2.imshow('result',proc_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


	cv2.imwrite('/home/samanyu/Documents/kalpana/ptest/' +img_name,img3)
	
#cv2.imshow('result',img3)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

