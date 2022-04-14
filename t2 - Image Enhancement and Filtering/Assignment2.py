####################################
# SCC0251 - Digital Image Processing (2022.1)
# Assingment 2 - Image Enhancement and Filtering


# Maria Fernanda Lucio de Mello
# nUSP: 11320860
# BCC 019
#####################################



####### Importing important libraries #######
import numpy as np
import imageio as im
import matplotlib.pyplot as plt
import math
import random
#############################################


def find_T (img, initial_T):
	t1 = initial_T
	x, y = img.shape

	# Dividing regions
	g1 = []
	g2 = []

	while 1:
		T = t1
		g1.clear()
		g2.clear()

		# Binarizing the image
		for i in range(x):
			for j in range(y):
				if img[i][j] > T:
					g1.append(img[i][j])
				else:
					g2.append(img[i][j])

		# Average intensity of each region
		average_intensity_1 = sum(g1)/len(g1)
		average_intensity_2 = sum(g2)/len(g2)
	
		# Updating t1
		t1 = (average_intensity_1 + average_intensity_2)/2

		# Break condition --> found the best T
		if (T - t1) < 0.5:
			break

	return T



def limiarization (img, t):
	x, y = img.shape

	# Limiarization process
	for i in range(x):
		for j in range (y):
			if(img[i][j] > t):
				img[i][j] = 1
			else:
				img[i][j] = 0
      
	return img



def filtering_1d (img, s, w):
	sz = img.shape
	sz_img = sz[0]*sz[1]

	# Collapse array in one dimension (vector)
	img_1d = img.flatten() 
	# Defines the output position
	center = int(s/2)
	output_img = np.zeros(sz_img)

	# Calculating the values for each position
	for i in range(sz_img):
		sum = 0
		for j in range(s):
			sum += w[j]*img_1d[(i+j - center) % sz_img] # Circular list
		output_img[i] = sum

	# Reshaping into a 2d array
	output_img = output_img.reshape(sz)
	return output_img



def filtering_2d (img, s, w, t):
	sz = img.shape
	center = int(s/2)
	output_img = np.zeros(sz)

	# Padding image reflecting the edges (symmetric matrix)
	img = np.pad(img, pad_width=center, mode='symmetric')

	# Filtering
	for i in range (center, sz[0] + center):
		for j in range (center, sz[1] + center):
			aux_img = img[i-center:i+center+1, j-center:j+center+1]
			output_img[i-center, j-center] = np.sum(aux_img*w)

	# Limiarization
	T = find_T(output_img, t)
	output_img = limiarization(output_img, T)
	return output_img

	

def median_filter (img, s):
	sz = img.shape
	center = int(s/2)
	output_img = np.zeros(sz)

	# Padding the image adding 0's on the edges
	img = np.pad(img,pad_width=center,mode='constant', constant_values=(0))

	for i in range(center, sz[0]+center):
		for j in range(center, sz[1]+center):
			aux = img[i-center: i+center+1, j-center: j+center+1]
			vec = aux.flatten()
			vec.sort()
			output_img[i-center, j-center] = vec[int((s*s)/2)+1] # Median

	return output_img

	

def Normalize (matrix, max_v):
	maxx = np.max(matrix)
	minn = np.min(matrix)
	matrixNormalized = ((matrix-minn)/(maxx-minn))
	matrixNormalized = (matrixNormalized*max_v).astype(np.uint8)

	return matrixNormalized
#########################



def RMSE (original_img, output_img):
	x, y = original_img.shape
	return math.sqrt((np.sum(np.square(original_img - output_img)))/(x*y))



def main():
############### Reading Inputs #################
	filename = str(input()).rstrip()
	method = int(input())

	# Getting the images
	original_img = im.imread(filename)
	output_img = im.imread(filename)

	
	# Limiarization
	if (method == 1):
		initial_threshold = float(input())
		# Finding the best value for T:
		T = find_T (output_img, initial_threshold)
		# Limiarization with T
		output_img = limiarization (output_img, T)
	

	# Filtering 1d
	elif (method == 2):
		size_filter = int(input())
		weights_str = str(input()).rstrip()
		weights = [int(i) for i in weights_str.split(' ')] 
		output_img = filtering_1d (output_img, size_filter, weights)
	

	# Filtering 2d
	elif (method == 3):
		size_filter = int(input())
		weights = []

		# Reading the weights of the filters
		for i in range(size_filter):
			w_str = str(input()).rstrip()
			w = [int(i) for i in w_str.split(' ')]
			weights.append(w)

		initial_threshold = float(input())	
		output_img = filtering_2d (output_img, size_filter, weights, initial_threshold)

	
	# Median Filter
	elif (method == 4):
		size_filter = int(input())
		output_img = median_filter (output_img, size_filter)
	#################################################



	# Normalizing the output image
	normalized_output = Normalize(output_img, 255)


	# Comparing original and output image
	print("%.4f" % RMSE(original_img, normalized_output))


if __name__ == "__main__":
	main()
