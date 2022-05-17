####################################
# SCC0251 - Digital Image Processing (2022.1)
# Assingment 3 - Filtering in Spatial and Frequency Domain

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
	input_image_filename = str(input()).rstrip()
	filter_filename = str(input()).rstrip()
	reference_image_filename = str(input()).rstrip()

	input_image = im.imread(input_image_filename)
	filter_ = im.imread(filter_filename)
	reference_image = im.imread(reference_image_filename)

	# Generate the Fourier Spectrum (F(I)) for the input image I.
	# img = DFT2D(input_image)
	img = np.fft.fft2(input_image)
	F1 = np.fft.fftshift(img)

	# Normalize the input filters from the [0, 255] interval to [0, 1] 
	normalized_filter = Normalize(filter_,1)

	# Filter F(I) multiplying it by the input filter M.
	img_filtered = np.multiply(F1, normalized_filter)

	# Generate the filtered image (Î) back in the space domain
	output_img_ifftshift = np.fft.ifftshift(img_filtered)
	output_img = np.fft.ifft2(output_img_ifftshift)
	
	# Compare the output image (Î) with the reference image ( G )
	# Normalizing the output image
	normalized_output = Normalize(np.real(output_img), 255)

	# Comparing original and output image
	print("%.4f" % RMSE(reference_image, normalized_output))


if __name__ == "__main__":
	main()