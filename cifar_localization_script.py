from __future__ import print_function
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1 # for GPU
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.models import load_model
import sklearn.metrics as metrics 
import matplotlib
#matplotlib.use('Agg') # when in remote screen
import matplotlib.pyplot as plt
import numpy as np

from utilities.cifar_utils import generate_white_noise_perturbations, generate_pink_noise_perturbations
from utilities.cifar_utils import difference_of_averages_image,correlation_map_image,bounding_box_localization
from utilities.cifar_utils import average_power_spectrum,rgb2gray,rgb2gray_v2


############################################### SCRIPT FUNCTIONS FOR RESULTS ################################################################# 

def noise_comparison(i,original_img,label,certainty,cifar_objects,model,aps,correlation_type,correlation_subtype,num_perturbations):

	print('Noise Comparison')
	print('Image index: ',i)
	print('Object: ',cifar_objects[label])
	print('Perturbations: ',num_perturbations)
	print('Type: ',correlation_type)
	print('sub-type: ',correlation_subtype)

	fig=plt.figure(1,figsize=(8,15))
	#plt.title('Noise comparison for '+cifar_objects[label]+' ('+str(num_perturbations)+' perturbations)', fontsize=30)
	plt.axis('off')
	fig.subplots_adjust(hspace=0.1, wspace=0.05)

	ax=fig.add_subplot(5,3,2)
	ax.grid()
	plt.title('Original Image ('+str(cifar_objects[label])+')', fontsize=15)
	plt.imshow(original_img)
	plt.axis('off')

	############################# BOUNDING BOXES ######################################
	box_size = 2
	territory = bounding_box_localization(original_img,label,certainty,box_size,model)
	#np.save('boxes/img'+str(i)+'_box'+str(box_size)+'_region.npy',territory)
	#territory = np.load('boxes/img'+str(i)+'_box'+str(box_size)+'_region.npy')
	fig.add_subplot(5,3,4)
	plt.title(str(box_size)+'x'+str(box_size)+' with stride', fontsize=15)
	plt.imshow(rgb2gray_v2(territory), cmap='gray')
	plt.axis('off')


	box_size = 4
	territory = bounding_box_localization(original_img,label,certainty,box_size,model)
	#np.save('boxes/img'+str(i)+'_box'+str(box_size)+'_region.npy',territory)
	#territory = np.load('boxes/img'+str(i)+'_box'+str(box_size)+'_region.npy')
	fig.add_subplot(5,3,5)
	plt.title(str(box_size)+'x'+str(box_size)+' with stride', fontsize=15)
	plt.imshow(rgb2gray_v2(territory), cmap='gray')
	plt.axis('off')

	box_size = 8
	territory = bounding_box_localization(original_img,label,certainty,box_size,model)
	#np.save('boxes/img'+str(i)+'_box'+str(box_size)+'_region.npy',territory)
	#territory = np.load('boxes/img'+str(i)+'_box'+str(box_size)+'_region.npy')
	fig.add_subplot(5,3,6)
	plt.title(str(box_size)+'x'+str(box_size)+' with stride', fontsize=15)
	plt.imshow(rgb2gray_v2(territory), cmap='gray')
	plt.axis('off')


	if correlation_type == 'median':
		############################# CLASSIFICATION IMAGES ######################################
		difference_squared = True
		## White/Blurred Noise Perturbations
		noise_std = 0.1
		blur_std = 1.5
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,7)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(rgb2gray_v2(difference_noise),cmap='gray')
		plt.axis('off')

		noise_std = 0.5
		blur_std = 1.5
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,8)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(rgb2gray_v2(difference_noise),cmap='gray')
		plt.axis('off')

		noise_std = 1
		blur_std = 1.5
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,9)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(rgb2gray_v2(difference_noise),cmap='gray')
		plt.axis('off')

		noise_std = 0.1
		blur_std = 2
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,10)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(rgb2gray_v2(difference_noise),cmap='gray')
		plt.axis('off')

		noise_std = 0.5
		blur_std = 2
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,11)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(rgb2gray_v2(difference_noise),cmap='gray')
		plt.axis('off')
	
		noise_std = 1
		blur_std = 2
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,12)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(rgb2gray_v2(difference_noise),cmap='gray')
		plt.axis('off')


		##Pink Noise Perturbations
		amplitude=0.1
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,13)
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(rgb2gray_v2(difference_noise),cmap='gray')
		plt.axis('off')

		amplitude=0.25
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,14)
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(rgb2gray_v2(difference_noise),cmap='gray')
		plt.axis('off')

		amplitude=0.5
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		ax=fig.add_subplot(5,3,15)
		ax.grid()
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(rgb2gray_v2(difference_noise),cmap='gray')
		plt.axis('off')

		
	else:
		############################  CORRELATION ######################################
		## White/Blurred Noise Perturbations
		noise_std = 0.1
		blur_std = 1.5
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)

		fig.add_subplot(5,3,7)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 0.5
		blur_std = 1.5
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,8)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 1
		blur_std = 1.5
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,9)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 0.1
		blur_std = 2
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,10)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 0.5
		blur_std = 2
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,11)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 1
		blur_std = 2
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,12)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')




		##Pink Noise Perturbations
		amplitude=0.1
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,13)
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		amplitude=0.25
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,14)
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		amplitude=0.5
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		ax=fig.add_subplot(5,3,15)
		ax.grid()
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

	plt.savefig(os.path.expanduser('results/cifar10/img'+str(i)+'_'+str(correlation_type)+'_'+str(correlation_subtype)+'('+str(num_perturbations)+').png'),dpi=300)
	fig.clf()
	print('DONE')



def localization_demonstration(i,original_img,label,certainty,cifar_objects,model,aps,correlation_type,correlation_subtype,num_perturbations):


	print('Localization Demonstration')
	print('Object: ',cifar_objects[label])
	print('Perturbations: ',num_perturbations)
	print('Type: ',correlation_type)
	print('sub-type: ',correlation_subtype)


	fig=plt.figure(2)
	#plt.title('Noise comparison for '+cifar_objects[label]+' ('+str(num_perturbations)+' perturbations)', fontsize=30)
	plt.axis('off')
	fig.subplots_adjust(hspace=0.1, wspace=0.05)

	ax=fig.add_subplot(1,4,1)
	ax.grid()
	plt.title(str(cifar_objects[label]))
	plt.imshow(original_img)
	plt.axis('off')


	ratio = 0.9
	############################# BOUNDING BOX ######################################

	box_size = 4
	territory = bounding_box_localization(original_img,label,certainty,box_size,model)
	#np.save('boxes/img'+str(i)+'_box'+str(box_size)+'_region.npy',territory)
	#territory = np.load('boxes/img'+str(i)+'_box'+str(box_size)+'_region.npy')
	fig.add_subplot(1,4,2)
	plt.title(str(box_size)+'x'+str(box_size)+' with stride')
	plt.imshow(rgb2gray_v2(territory**2)*ratio +rgb2gray(original_img)*(1-ratio), cmap='coolwarm')
	plt.axis('off')

	if correlation_type =='median':
		difference_squared = True
		########## WHITE NOISE #############
		noise_std = 0.1
		blur_std = 2
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(1,4,3)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std))
		plt.imshow(rgb2gray_v2(difference_noise)*ratio+rgb2gray(original_img)*(1-ratio),cmap='coolwarm')
		plt.axis('off')

		########## PINK NOISE #############
		amplitude=0.1
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		ax=fig.add_subplot(1,4,4)
		ax.grid()
		plt.title('PN '+str(amplitude))
		im=plt.imshow(rgb2gray_v2(difference_noise)*ratio +rgb2gray(original_img)*(1-ratio),cmap='coolwarm')
		plt.axis('off')


	else:
		########## WHITE NOISE #############
		noise_std = 0.1
		blur_std = 2
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(1,4,3)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std))
		plt.imshow(correlation_image*ratio+ rgb2gray(original_img)*(1-ratio),cmap='coolwarm')
		plt.axis('off')

		########## PINK NOISE #############
		amplitude=0.1
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		ax=fig.add_subplot(1,4,4)
		ax.grid()
		plt.title('PN '+str(amplitude))
		im=plt.imshow(correlation_image*ratio + rgb2gray(original_img)*(1-ratio),cmap='coolwarm')
		plt.axis('off')


	fig.subplots_adjust(right=0.9)
	cbar_ax = fig.add_axes([0.91,0.34,0.015,0.31])
	fig.colorbar(im,cax=cbar_ax)
	
	plt.savefig(os.path.expanduser('results/cifar10/img'+str(i)+'_'+str(correlation_type)+'_'+str(correlation_subtype)+'('+str(num_perturbations)+')_single.png'),dpi=300)
	fig.clf()
	print('DONE')


def method_comparison(i,original_img,label,certainty,model,aps,num_perturbations):
	
	print('Method Comparison')
	print('Perturbations: ',num_perturbations)
	print('Original Certainty: ',certainty)

	fontsize = 17 

	fig=plt.figure(3,figsize=(19,6))
	#plt.title('Noise comparison for '+cifar_objects[label]+' ('+str(num_perturbations)+' perturbations)', fontsize=30)
	plt.axis('off')
	fig.subplots_adjust(hspace=0.05, wspace=0.05)

	##################### FOR WHITE NOISE ########################
	noise_std = 0.1
	blur_std = 2


	## DIFFERENCE OF AVERAGES
	ax=fig.add_subplot(2,6,1)
	perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared=False)
	plt.title('Median',fontsize=fontsize)
	plt.imshow(rgb2gray_v2(difference_noise),cmap='gray')
	plt.axis('off')

	##SQUARED DIFFERENCE OF AVERAGES
	ax=fig.add_subplot(2,6,2)
	perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared=True)
	plt.title('Median|Squared',fontsize=fontsize)
	plt.imshow(rgb2gray_v2(difference_noise),cmap='gray')
	plt.axis('off')



	## CORRELATION MAP
	correlation_type = 'pearson'
	correlation_subtype = 'absolute'
	ax=fig.add_subplot(2,6,3)
	perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	plt.title('Pearson|Abs',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')


	correlation_subtype = 'squared'
	ax=fig.add_subplot(2,6,4)
	perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	plt.title('Pearson|Squared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')


	correlation_type = 'spearman'
	correlation_subtype = 'absolute'
	ax=fig.add_subplot(2,6,5)
	perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	plt.title('Spearman|Abs',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')

	

	correlation_subtype = 'squared'
	ax=fig.add_subplot(2,6,6)
	perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	plt.title('Spearman|Squared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')


	##################### FOR PINK NOISE ########################
	
	## DIFFERENCE OF AVERAGES
	amplitude=0.1

	ax=fig.add_subplot(2,6,7)
	perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared=False)
	#plt.title('Median',fontsize=fontsize)
	plt.imshow(rgb2gray_v2(difference_noise),cmap='gray')
	plt.axis('off')

	##SQUARED DIFFERENCE OF AVERAGES
	ax=fig.add_subplot(2,6,8)
	perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared=True)
	#plt.title('Median|Squared',fontsize=fontsize)
	plt.imshow(rgb2gray_v2(difference_noise**2),cmap='gray')
	plt.axis('off')


	## CORRELATION MAP
	correlation_type = 'pearson'
	correlation_subtype = 'absolute'
	ax=fig.add_subplot(2,6,9)
	perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	#plt.title('Pearson|Abs',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')


	correlation_subtype = 'squared'
	ax=fig.add_subplot(2,6,10)
	perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	#plt.title('Pearson|Squared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')


	correlation_type = 'spearman'
	correlation_subtype = 'absolute'
	ax=fig.add_subplot(2,6,11)
	perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	#plt.title('Pearson|RootSquared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')


	correlation_subtype = 'squared'
	ax=fig.add_subplot(2,6,12)
	perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	#plt.title('Pearson|MaxAbs',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')

	plt.savefig(os.path.expanduser('results/cifar10/img'+str(i)+'_method_comparison('+str(num_perturbations)+').png'),dpi=300)
	fig.clf()
	print('DONE')


def fig_num_perturb(i,original_img,label,certainty,model,aps,noise_type):

	print('Number of Trials Effect')
	print('Image index: ',i)
	print('Noise Type: ',noise_type)
	print('Original Certainty: ',certainty)

	
	# fixed method
	correlation_type = 'pearson'
	correlation_subtype = 'squared'


	noise_title = 'White Noise - Trials' if noise_type == 'white' else 'Pink Noise - Trials'

	fig=plt.figure(4,figsize=(10,10))
	ttl = plt.title(noise_title, fontsize=30)
	ttl.set_position([.5,1.1])
	plt.axis('off')
	fig.subplots_adjust(hspace=0.1, wspace=0.05)

	fontsize = 20
	# White noise parameters
	noise_std = 0.1
	blur_std = 2
	# Pink noise parameters
	amplitude = 0.1


	
	num_perturbations = 200

	ax=fig.add_subplot(3,3,1)
	if noise_type == 'white':
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	else:	
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	plt.title(str(num_perturbations)+' Trials',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')
	
	ax=fig.add_subplot(3,3,4)
	if noise_type == 'white':
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	else:	
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	#plt.title('Pearson|Squared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')

	ax=fig.add_subplot(3,3,7)
	if noise_type == 'white':
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	else:	
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	#plt.title('Pearson|Squared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')

	
	num_perturbations = 2000

	ax=fig.add_subplot(3,3,2)
	if noise_type == 'white':
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	else:	
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	plt.title(str(num_perturbations)+' Trials',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')#
	plt.axis('off')

	ax=fig.add_subplot(3,3,5)
	if noise_type == 'white':
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	else:	
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	#plt.title('Pearson|Squared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')#plt.imshow(set_threshold(rgb2gray_v2(difference_noise)),cmap='gray')
	plt.axis('off')

	ax=fig.add_subplot(3,3,8)
	if noise_type == 'white':
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	else:	
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	#plt.title('Pearson|Squared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')#plt.imshow(set_threshold(rgb2gray_v2(difference_noise)),cmap='gray')
	plt.axis('off')

	
	num_perturbations = 20000

	ax=fig.add_subplot(3,3,3)
	if noise_type == 'white':
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	else:	
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	plt.title(str(num_perturbations)+' Trials',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')#
	plt.axis('off')

	ax=fig.add_subplot(3,3,6)
	if noise_type == 'white':
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	else:	
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	#plt.title('Pearson|Squared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')#plt.imshow(set_threshold(rgb2gray_v2(difference_noise)),cmap='gray')
	plt.axis('off')

	ax=fig.add_subplot(3,3,9)
	if noise_type == 'white':
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	else:	
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	#plt.title('Pearson|Squared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')#plt.imshow(set_threshold(rgb2gray_v2(difference_noise)),cmap='gray')
	plt.axis('off')

	plt.savefig(os.path.expanduser('results/cifar10/img'+str(i)+'_trial_comparison'+'_'+noise_type+'_noise.png'),dpi=300)
	fig.clf()
	print('DONE')

###################################################################################################################################################################################################

########################## GET IMAGES IN THE RIGHT FORM ###############

num_classes = 10
# Input image dimensions
img_rows, img_cols = 32, 32

# The data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)


############################# LOAD TRAINED MODEL ############################# 
model = load_model('models/densenet_40_12/densenet_40_12_model.h5')
#print(model.summary())

# Load probabilities and labels for test set coming from model
pred_prob = np.load('models/densenet_40_12/certainties.npy') #(10000,10)
pred_labels = np.load('models/densenet_40_12/predicted_labels.npy') #(10000,)

# Model's Accuracy
# score = metrics.accuracy_score(y_test, pred_labels) * 100
# print('Model Test set accuracy:', score)

# Pick sample of test images 
sample_size = 1000
pred_prob=pred_prob[0:sample_size]
pred_labels = pred_labels[0:sample_size]

#################################################################################

countlab = 10*[0]

## Average Power Spectrum of Test Images (for pink noise process)
aps = average_power_spectrum(x_test)


############################################# SCRIPT PARAMETERS ####################################################

cifar_objects = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

# Select Image from CIFAR-10 test set. For example i = 24,54,97,121,184,201,205,256,323,396,510,558,567,700

# Select number of perturbations
num_perturbations = 2000

i = 97
# Get initial predicted label and certainty for the selected Image
label = pred_labels[i]
certainty = pred_prob[i][label]
original_img = x_test[i]

# Method Comparison
method_comparison(i,original_img,label,certainty,model,aps,num_perturbations)

# Noise level comparison along with bounding box 
# Select correlation type for classification image
correlation_type = 'pearson'
correlation_subtype='squared'
noise_comparison(i,original_img,label,certainty,cifar_objects,model,aps,correlation_type,correlation_subtype,num_perturbations=num_perturbations)

# Localization Demonstration (one setting per type)
localization_demonstration(i,original_img,label,certainty,cifar_objects,model,aps,correlation_type,correlation_subtype,num_perturbations=num_perturbations)

# Compare the effect of Number of trials
noise_type='white' 
#noise_type='pink'
fig_num_perturb(i,original_img,label,certainty,model,aps,noise_type)




