from sklearn.decomposition import PCA
import skimage.io as io
import numpy as np
from PIL import Image
from PIL import Image
import matplotlib.pyplot as plt
from random import randint
from matplotlib.pylab import subplots, cm
import matplotlib.image as mpimg

'''
: for data/image loading
: there are 190 images(neutral faces) in total, with the extension of a.jpg
'''
def load_data(path='/*.jpg'):
	assert isinstance(path,str)
	pictures=io.ImageCollection(path) # get all imgaes with the extension of .jpg
	data=[]
	for i in range(len(pictures)):
		data.append(np.ravel(pictures[i].reshape((1,pictures[i].shape[0]*pictures[i].shape[1])))) #flatten the matrix from 193*162 to 1*31266
	return np.matrix(data).T # transpose matrix to 31266*1

a=load_data(path='/*.jpg') # the training set of first 190 faces

'''
# PCA function 
def PCA(a,number_of_PCs=190):
	assert isinstance(number_of_PCs,int)
	avgImg=np.mean(a,1) # mean image
	diff_train=a-avgImg # each eigenface minus the mean image [31266,190]
	eig_Vals,eig_Vects=np.linalg.eig(np.mat(diff_train.T*diff_train)) # 
	eig_sortidx=np.argsort(-np.abs(eig_Vals)) # get index based on the order of decreasing eigenvalues(i.e: eig_sortdix[0] is the index of the largest eigenvalue)
	e=eig_sortidx[0:number_of_PCs] # filter the eigenvectors based on the first largest number_of_PCs eigenvalues(i.e.:50)
	new_Vects=eig_Vects[:,e] # decreasing the dimension from 190 to number_of_PCs(i.e.:50)
	cov_Vects=diff_train*new_Vects # eigenvector matrix [31266,190]*[190,number_of_PCs]=[31266,number_of_PCs]
	return avgImg,cov_Vects,diff_train

#avgImg,cov_Vects,diff_train=PCA(a,number_of_PCs=190)
#print('The %s matrix of PCs is: ' %str(cov_Vects.shape))
#print(cov_Vects)
'''

'''
: pick an image 
: neutral/smiling from first 190 faces
: neutral from last 10 faces

'''

def pick_image(m=188,neutral=True,first_190=True,number_of_PCs=80,path='/',path1='/',path2='/'):
	'''
	: pick your image from neutral training set/ neutral test set/ smiling test set
	and see the reconstruction performance by PCA with the neutral facial expression training set
	: m: index of face images in the file path
	: number_of_PCs: number of principal components used in PCA
	: path: store 190 neurtal faces as the traning set
	: path1: store 10 neutral faces as the test set
	: path2; store 200 smiling faces as the test set
	'''
	assert isinstance(number_of_PCs,int) and 1<=number_of_PCs<=190
	assert neutral+first_190>=1
	avgImg=np.mean(a,1)
	pca=PCA(number_of_PCs,whiten=True) # data processing with PCA
	train_data=pca.fit_transform(a) # traindata is the matrix of PCs(eigenfaces), each column is a PC(eigenface)
	if neutral==True:
		if first_190==True:
			img=mpimg.imread(path+str(m)+'a.jpg') # neutral
		else:
			img=mpimg.imread(path1+str(m)+'a.jpg')
	else:
		img=mpimg.imread(path2+str(m)+'b.jpg') # smiling

	im=np.matrix(np.ravel(img.reshape((1,img.shape[0]*img.shape[1])))) # flatten the matrix(1*31266)
	#print(im.shape)
	avg_im=im # minus the mean image/value [1,31266] avg_im=(im.T-avgImg).T
	#print(avg_im.shape)
	projection=avg_im*train_data # projection matrix: [1,31266]*[31266,number_of_PCs]=[1,number_of_PCs]
	#print(projection.shape)
	reconstruct=projection*train_data.T+avgImg.T # [1,50]*[50,31266]=[1,31266]
	reconstruct=reconstruct.reshape(193,162)

	return img,reconstruct


x=[1,range(5,55,5)]
fig,axs=subplots(3,4,sharex=True,sharey=True)
# number of principal components 
no=[190,100,80,50,40,20]
i=0
for ax in axs.flatten():
	'''
	: subplots of face reconstruction with different number of principal components used
	'''
	plt.title('Reconstructed Face(neutral')
	img,reconstruct=pick_image(m=195,neutral=True,first_190=False,number_of_PCs=no[i],path='/',path1='/',path2='/')
	ax.imshow(reconstruct,cmap=cm.gray)
	ax.set_title('PC Number:'+str(no[i]),weight='bold')
	i+=1

plt.show()
