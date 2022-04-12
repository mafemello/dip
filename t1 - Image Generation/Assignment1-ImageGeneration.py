####################################
# SCC0251 - Digital Image Processing (2022.1)
# Assingment 1 - Image Generation

# Maria Fernanda Lucio de Mello
# nUSP: 11320860
# BCC 019
#####################################



####### importing important libraries #######
import numpy as np
import imageio as im
import matplotlib.pyplot as plt
import math
import random
#############################################



##### Scene Image Functions #####
def Synthesizing(sz, fun, s, img, q):

	if (fun == 1):
	    for x in range (sz):
	        for y in range (sz):
	            img[x,y] = function1(x,y)

	elif (fun == 2):
	    for x in range (sz):
	        for y in range (sz):
	            img[x,y] = function2(x,y,q)

	elif (fun == 3):
	    for x in range (sz):
	        for y in range (sz):
	            img[x,y] = function3(x,y,q)

	elif (fun == 4):
	    random.seed(s)
	    for x in range (sz):
	        for y in range (sz):
	            img[y, x] = function4(y,x)

	elif (fun == 5):
	    random.seed(s)
	    img = function5(sz,img)

	return
############################


##### Sampling the image #####
def Sampling(c, n, img):
	step = int(c/n)
	aux = int(n)
	newIMG = np.zeros((n, n), dtype=float)

	for i in range(aux):
		for j in range(aux):
			newIMG[i, j] = img[i*step, j*step]

	return newIMG
############################


# 𝑓(𝑥, 𝑦) = (𝑥𝑦 + 2𝑦)
def function1 (x,y):
	return (x*y + 2*y)
#########################


# 𝑓(𝑥, 𝑦) = | 𝑐𝑜𝑠(x/Q) + 2 𝑠𝑖𝑛(y/Q) |
def function2 (x,y,q):
	return (math.fabs(math.cos(x/q) + 2*math.sin(y/q)))
#########################


# 𝑓(𝑥, 𝑦) = | 3*(x/Q) - cbrt(y/Q) |
def function3 (x,y,q):
	return (math.fabs(3*(x/q) - np.cbrt(y/q)))
#########################


# 𝑓(𝑥, 𝑦) = 𝑟𝑎𝑛𝑑(0, 1, 𝑆)
def function4 (x,y):
	return (random.random())
#########################


# 𝑓(𝑥, 𝑦) = 𝑟𝑎𝑛𝑑𝑜𝑚𝑤𝑎𝑙𝑘(𝑆)
def function5 (sz,img):
	x = 0
	y = 0
	img[x,y] = 1
	for i in range (1 + sz*sz):
		dx = random.randint(-1,1)
		dy = random.randint(-1,1)
		x = (x+dx) % sz
		y = (y+dy) % sz
		img[x,y] = 1
	return img
#########################


##### Normalize img #####
def Normalize (matrix, max_v):
	maxx = np.max(matrix)
	minn = np.min(matrix)
	
	matrixNormalized = ((matrix-minn)/(maxx-minn))
	matrixNormalized = (matrixNormalized*max_v)

	return matrixNormalized
#########################


##### Biwise shift operation #####
def BitwiseShift(matrix, bits):
	matrix = np.right_shift(matrix, bits)
	return matrix
##################################


##### Root Squared Error #####
def RSE(g,r):
	return np.sqrt(sum(sum((g-r)**2)))
##############################


def main():
	####### Reading Inputs #######
	filename = str(input()).rstrip()
	sizeC = int(input())
	functionChoosed = int(input())
	paramQ = int(input())
	sizeN = int(input())
	bitsPerPixel = int(input())
	seedS = int(input())
	#############################


	######## Synthesizing Image ########
	f = np.zeros((sizeC, sizeC), dtype=float) # Scene Image
	Synthesizing(sizeC, functionChoosed, seedS, f, paramQ)
	f_norm = Normalize(f, 65535) # Normalizing image
	####################################


	######## Sampling Image ########
	g = Sampling(sizeC, sizeN, f_norm) # Digital Image
	################################


	######## Quantizing ########
	g = Normalize(g, 255)
	g = g.astype(np.uint8)
	g = BitwiseShift(g, 8-bitsPerPixel)
	############################


	###### Comparing Images ######
	refIMG = np.load(filename)	
	print("%.4f" % RSE(g, refIMG))


if __name__ == "__main__":
	main()