import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage import data #We will use sample images that come bundled with skimage data
import matplotlib.pyplot as plt

def plotGaussian():
	f = lambda x : np.exp(-1*(x**2))
	x = np.linspace(-5,5,num=100)
	plt.plot(x,map(f,x))
	plt.show()

## Using a mean kernel
def getMeanKernel(k=3):
    return np.ones(shape=(k,k)).astype(np.float32)/(k*k)

def getGaussianKernel(sx,sy):

	threshold = 1e-3

	f = lambda x,y : (1/(2*np.pi*sx*sy))*np.exp(-1*(((x/sx)**2)+((y/sy)**2)))
	win_size = 0
	y = 0

	print f(win_size,y)

	while f(win_size,y) > threshold:
		print win_size,f(win_size,y)
		win_size += 1

	kernel = np.zeros(shape=(2*win_size+1,2*win_size+1))

	centered_f = lambda x,y : (1/(2*np.pi*sx*sy))*np.exp(-1*((((x-win_size)/sx)**2)+(((y-win_size)/sy)**2))) 
	i = np.arange(kernel.shape[0])
	j = np.arange(kernel.shape[1])

	II,JJ = np.meshgrid(i,j,indexing='ij')

	kernel = centered_f(II.flatten(),JJ.flatten()).reshape(kernel.shape)

	kernel = kernel/np.sum(kernel)

	return kernel

def applyFilterImageNaive(image,kernel):
	# Padding the image by kernel size
	pad_1d = kernel.shape[0] / 2
	print 'pad_1d',pad_1d
	pad_shape = ((pad_1d,pad_1d),(pad_1d,pad_1d),(0,0))
	padded_image = np.pad(image,pad_shape,'reflect')

	I = np.arange(image.shape[0])
	J = np.arange(image.shape[1])
	II,JJ = np.meshgrid(I,J,indexing='ij')
	conv_coords = zip(II.flatten()+pad_1d,JJ.flatten()+pad_1d)

	conv_image = np.zeros_like(image)
	print 'conv_image.shape',conv_image.shape
	for i,j in conv_coords:
		conv_image[i-pad_1d,j-pad_1d,0] = np.sum(padded_image[i-pad_1d:i+pad_1d+1,j-pad_1d:j+pad_1d+1,0]*kernel)
		conv_image[i-pad_1d,j-pad_1d,1] = np.sum(padded_image[i-pad_1d:i+pad_1d+1,j-pad_1d:j+pad_1d+1,1]*kernel)
		conv_image[i-pad_1d,j-pad_1d,2] = np.sum(padded_image[i-pad_1d:i+pad_1d+1,j-pad_1d:j+pad_1d+1,2]*kernel)

	plt.subplot(121)
	plt.imshow(image)
	plt.subplot(122)
	plt.imshow(conv_image)
	plt.show()

def applyFilterImageNumpy(image,kernel):

	conv_image = np.zeros_like(image)
	print image.shape, conv_image.shape
	conv_image[:,:,0] = scipy.signal.convolve(image[:,:,0],kernel,mode='same')
	conv_image[:,:,1] = scipy.signal.convolve(image[:,:,1],kernel,mode='same')
	conv_image[:,:,2] = scipy.signal.convolve(image[:,:,2],kernel,mode='same')

	plt.subplot(121)
	plt.imshow(image)
	plt.subplot(122)
	plt.imshow(conv_image)
	plt.show()

def main():
	image = data.astronaut()
	kernel = getMeanKernel(5)
	# kernel = getGaussianKernel(1,1)
	# applyFilterImageNaive(image,kernel)
	applyFilterImageNumpy(image,kernel)

if __name__ == '__main__':
	main()