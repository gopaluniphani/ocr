import os
import glob
import cv2
from WordSegmentation import wordSegmentation, prepareImg


def segment():
	"""reads images from data/ and outputs the word-segmentation to out/"""

	# read input images from 'in' directory
	img = "data/in.jpg"

	# read image, prepare it by resizing it to fixed height and converting it to grayscale
	img = prepareImg(cv2.imread(img), 50)
	
	# execute segmentation with given parameters
	# -kernelSize: size of filter kernel (odd integer)
	# -sigma: standard deviation of Gaussian function used for filter kernel
	# -theta: approximated width/height ratio of words, filter function is distorted by this factor
	# - minArea: ignore word candidates smaller than specified area
	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
	
	#delete all files in segmeted directory
	files = glob.glob('segmented/*')
	for f in files:
		os.remove(f)

	#delete all files in contrast directory
	files = glob.glob('contrast/*')
	for f in files:
		os.remove(f)

	# iterate over all segmented words
	print('Segmented into %d words'%len(res))
	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		(x, y, w, h) = wordBox
		cv2.imwrite('segmented/%d.png'%(j), wordImg) # save word
		cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
	
	# output summary image with bounding boxes around words
	cv2.imwrite('summary/summary.png', img)