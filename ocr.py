import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
import cv2
import imutils
from imutils.object_detection import non_max_suppression
from collections import defaultdict, deque
from Levenshtein  import distance

pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'


def main(image, threshold, text_finds):

	#read image
	img = cv2.imread(image)
	h, w = img.shape[:2]
	
	#convert image to gray
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#create threshold for image with binary kind
	_,thresh_image = cv2.threshold(img_gray, threshold,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

	#kernel length
	kernel_length = w//50

	#vertical kernel
	ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

	#horizon kernel
	hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

	#kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


	#vertical lines
	vertical_temp = cv2.erode(thresh_image, ver_kernel, (-1,-1))
	vertical_lines = cv2.dilate(vertical_temp, ver_kernel, (-1,-1))	


	#horizon lines
	horizon_temp = cv2.erode(thresh_image, hor_kernel, (-1,-1))
	horizon_lines = cv2.dilate(horizon_temp, hor_kernel, (-1,-1))	


	#add vertical and horizon lines
	img_final = cv2.addWeighted(vertical_lines, 0.5, horizon_lines, 0.5, 0.0)
	# img_final = cv2.erode(~img_final, kernel, (-1,-1))
	_, img_final = cv2.threshold(img_final, threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	

	#remove vertical and horizon lines
	horizontal_inv = cv2.bitwise_not(img_final)	
	masked_img = cv2.bitwise_not(thresh_image, thresh_image, mask=horizontal_inv)	
	
	# minAreaRect on the nozeros
	pts = cv2.findNonZero(masked_img)
	ret = cv2.minAreaRect(pts)

	(cx,cy), (w_,h_), ang = ret
	if ang < -70:
		w_,h_ = h_,w_
		ang += 90

	## Find rotated matrix, do rotation
	M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)	
	image_rotation = cv2.warpAffine(masked_img, M, (img.shape[1], img.shape[0]))
	
	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 3))
	image_text = cv2.dilate(image_rotation, rect_kernel, iterations = 1)

	#find counter
	_, counters, _ = cv2.findContours(image_text, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	counters = sorted(counters, key=lambda x: ((cv2.boundingRect(x)[1]+0.5*cv2.boundingRect(x)[3]), 
												(cv2.boundingRect(x)[0]+0.5*cv2.boundingRect(x)[2])))


	result = deque()
	try:
		for idx, counter in enumerate(counters):
			x, y, w, h = cv2.boundingRect(counter)
			if w>h and 5<h<30 and 21<w<500:
				dict_value = dict()
				dict_value['position'] = (x, y, w, h)
				image_crop = img[y:y+h, x:x+w]			
				text = pytesseract.image_to_string(image_crop, lang='vietnamese')
				dict_value['text'] = text

			if dict_value not in result:
				result.append(dict_value)

		min_scores = [100]*len(text_finds)
		index_choices = np.zeros_like(text_finds, dtype=int)
		for index, value in enumerate(result):
			for index_find, text_find in enumerate(text_finds):
				if text_find.lower() in value['text'].lower():
					index_choices[index_find] = index
					min_scores[index_find] = 0
					break
				score = distance(value['text'], text_find)
				if score < min_scores[index_find]:
					index_choices[index_find] = index
					min_scores[index_find] = score
		
		for index, value in enumerate(index_choices):
			x, y, w, h = result[value]['position']
			image_crop = img[y:y+h, x:x+w]			
			text = pytesseract.image_to_string(image_crop, lang='vietnamese')
			if text_finds[index].lower() in text.lower() and len(text) - len(text_finds[index]) > 3:
				text = text.replace(':', '').replace(text_finds[index], '').strip()
				cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,0), 2)
				print(f'{text_finds[index]}: {text}')
			else:
				x, y, w, h = result[value+1]['position']
				cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,0), 2)
				image_crop = img[y:y+h, x:x+w]			
				text = pytesseract.image_to_string(image_crop, lang='vietnamese')

				print(f'{text_finds[index]}: {text}')

		cv2.imshow('image_rotation', img)
		cv2.waitKey()
	except:
		print('Không thể detect')	

	
if __name__ == '__main__':
	main('./anh/test_ocr.jpg', 180, ['văn bản ủy quyền', 'Chức vụ'])


