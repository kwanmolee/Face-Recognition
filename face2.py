from sklearn.decomposition import PCA
import skimage.io as io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from random import randint
from matplotlib.pylab import subplots, cm
import matplotlib.image as mpimg

'''
: for data/image loading
: there are 190 images(neutral faces) in total, with the extension of a.jpg
'''
def load_data(path='/Users/momolee/Downloads/lee/*.jpg'):
	assert isinstance(path,str)
	pictures=io.ImageCollection(path) # get all imgaes with the extension of .jpg
	data=[]
	for i in range(len(pictures)):
		data.append(np.ravel(pictures[i].reshape((1,pictures[i].shape[0]*pictures[i].shape[1])))) #flatten the matrix from 193*162 to 1*31266
	return np.matrix(data).T # transpose matrix to 31266*1

a=load_data(path='/Users/momolee/Downloads/lee/*.jpg') # the training set of first 190 faces
'''
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
pca=PCA(190,whiten=True)
b=pca.fit_transform(a)[:,0]

def pick_image(neutral=True,first_190=True,number_of_PCs=80,path='/Users/momolee/Downloads/lee/',path1='/Users/momolee/Downloads/lee1/',path2='/Users/momolee/Downloads/lee2/'):
	assert isinstance(number_of_PCs,int) and 1<=number_of_PCs<=190
	assert neutral+first_190>=1
	avgImg=np.mean(a,1)
	pca=PCA(number_of_PCs,whiten=True) # data processing with PCA
	train_data=pca.fit_transform(a) # traindata is the matrix of PCs(eigenfaces), each column is a PC(eigenface)
	if first_190==True:
		m=randint(1,190) # randomly pick a face from first 190 
		if neutral==True:
			img=mpimg.imread(path+str(m)+'a.jpg') # neutral
		else:
			img=mpimg.imread(path2+str(m)+'b.jpg') # smiling
	else:
		m=randint(191,200) # randomly pick a face from last 10
		img=mpimg.imread(path1+str(m)+'a.jpg') # neutral

	im=np.matrix(np.ravel(img.reshape((1,img.shape[0]*img.shape[1])))) # flatten the matrix(1*31266)
	#print(im.shape)
	avg_im=im # minus the mean image/value [1,31266] avg_im=(im.T-avgImg).T
	#print(avg_im.shape)
	projection=avg_im*train_data # projection matrix: [1,31266]*[31266,number_of_PCs]=[1,number_of_PCs]
	#print(projection.shape)
	reconstruct=projection*train_data.T+avgImg.T # [1,50]*[50,31266]=[1,31266]
	reconstruct=reconstruct.reshape(193,162)

	return img,reconstruct

def nonhuman_image(path='/Users/momolee/Downloads/test_brunch.jpg'):
	assert isinstance(path,str)
	avgImg=np.mean(a,1)
	pca=PCA(190,whiten=True) # data processing with PCA
	train_data=pca.fit_transform(a) # traindata is the matrix of PCs(eigenfaces), each column is a PC(eigenface)
	img=Image.open(path).convert('L')
	im=np.array(img)
	im=np.matrix(np.ravel(im.reshape((1,im.shape[0]*im.shape[1])))) # flatten the matrix(1*31266)
	print(im.shape)
	avg_im=im # minus the mean image/value [1,31266] avg_im=(im.T-avgImg).T
	print(avg_im.shape)
	projection=avg_im*train_data # projection matrix: [1,31266]*[31266,number_of_PCs]=[1,number_of_PCs]
	print(projection.shape)
	reconstruct=projection*train_data.T+avgImg.T # [1,50]*[50,31266]=[1,31266]
	reconstruct=reconstruct.reshape(193,162)

	return img,reconstruct

def rotate_image(path='/Users/momolee/Downloads/78a.jpg',degree=20):
	assert isinstance(path,str)
	avgImg=np.mean(a,1)
	pca=PCA(190,whiten=True) # data processing with PCA
	train_data=pca.fit_transform(a) # traindata is the matrix of PCs(eigenfaces), each column is a PC(eigenface)
	img=Image.open(path)
	r=img.rotate(degree)
	im=np.array(r)
	im=np.matrix(np.ravel(im.reshape((1,im.shape[0]*im.shape[1])))) # flatten the matrix(1*31266)
	print(im.shape)
	avg_im=im # minus the mean image/value [1,31266] avg_im=(im.T-avgImg).T
	print(avg_im.shape)
	projection=avg_im*train_data # projection matrix: [1,31266]*[31266,number_of_PCs]=[1,number_of_PCs]
	print(projection.shape)
	reconstruct=projection*train_data.T+avgImg.T # [1,50]*[50,31266]=[1,31266]
	reconstruct=reconstruct.reshape(193,162)

	return r,reconstruct	
	
'''
fig,axs=subplots(1,2,sharex=True,sharey=True)
n=0
for ax in axs.flatten():
	t1=[img,reconstruct]
	t2=['Test Image(Simpson)','Reconstructed Face(fitting portrait)']
	ax.imshow(t1[n],cmap='gray')
	ax.set_title(t2[n],weight='bold')
	n+=1
'''

degree=[20,40,60,90,120,140,160,180]
degree1=[200,240,260,300,320,340]
degree2=[0]

for d in degree2:
	fig,axs=subplots(1,2,sharex=True,sharey=True)
	r,reconstruct=rotate_image(path='/Users/momolee/Downloads/lee/78a.jpg',degree=d)
	n=0
	for ax in axs.flatten():
		t1=[r,reconstruct]
		t2=['Original Image('+str(d)+'° rotated)','Reconstructed Face(all PCs)']
		ax.imshow(t1[n],cmap='gray')
		ax.set_title(t2[n],weight='bold')
		n+=1
plt.show()