from __future__ import print_function
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1 # for GPU
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import load_model
import numpy as np
import matplotlib
#matplotlib.use('Agg') # when in remote screen
import matplotlib.pyplot as plt

from utilities.mnist_utils import generate_white_noise_perturbations, generate_pink_noise_perturbations
from utilities.mnist_utils import difference_of_averages_image,correlation_map_image,bounding_box_localization
from utilities.mnist_utils import average_power_spectrum



############################################### SCRIPT FUNCTIONS FOR RESULTS ################################################################# 

def noise_comparison(i,original_img,label,certainty,mnist_objects,model,aps,correlation_type,correlation_subtype,num_perturbations):

	
	print('Noise Comparison')
	print('Image index: ',i)
	print('Object: ',mnist_objects[label])
	print('Perturbations: ',num_perturbations)
	print('Type: ',correlation_type)
	print('sub-type: ',correlation_subtype)
	print('Original Certainty: ',certainty)



	fig=plt.figure(1,figsize=(8,15))
	#plt.title('Noise comparison for '+mnist_objects[label]+' ('+str(num_perturbations)+' perturbations)', fontsize=30)
	plt.axis('off')
	fig.subplots_adjust(hspace=0.1, wspace=0.05)

	ax=fig.add_subplot(5,3,2)
	ax.grid()
	plt.title('Original Image ('+str(mnist_objects[label])+')', fontsize=15)
	plt.imshow(np.squeeze(original_img),cmap='gray')
	plt.axis('off')



	############################# BOUNDING BOXES ######################################
	box_size = 2
	territory = bounding_box_localization(original_img,label,certainty,box_size,model)
	fig.add_subplot(5,3,4)
	plt.title(str(box_size)+'x'+str(box_size)+' with stride', fontsize=15)
	plt.imshow(np.squeeze(territory), cmap='gray')
	plt.axis('off')


	box_size = 4
	territory = bounding_box_localization(original_img,label,certainty,box_size,model)
	fig.add_subplot(5,3,5)
	plt.title(str(box_size)+'x'+str(box_size)+' with stride', fontsize=15)
	plt.imshow(np.squeeze(territory), cmap='gray')
	plt.axis('off')

	box_size = 8
	territory = bounding_box_localization(original_img,label,certainty,box_size,model)
	fig.add_subplot(5,3,6)
	plt.title(str(box_size)+'x'+str(box_size)+' with stride', fontsize=15)
	plt.imshow(np.squeeze(territory), cmap='gray')
	plt.axis('off')


	if correlation_type == 'median':
		############################# CLASSIFICATION IMAGES ######################################

		difference_squared = True
		## White/Blurred Noise Perturbations
		noise_std = 0.5
		blur_std = 0
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,7)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')

		plt.axis('off')

		noise_std = 1
		blur_std = 0
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,8)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')

		noise_std = 2
		blur_std = 0
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,9)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')

		noise_std = 0.5
		blur_std = 1
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,10)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')

		noise_std = 1
		blur_std = 1
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,11)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')
	
		noise_std = 2
		blur_std = 1
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,12)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')


		##Pink Noise Perturbations
		amplitude=0.5
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,13)
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')

		amplitude=1
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,14)
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')

		amplitude=2
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		ax=fig.add_subplot(5,3,15)
		ax.grid()
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')

		
	else:
		############################  CORRELATION ######################################
		## White/Blurred Noise Perturbations
		noise_std = 0.5
		blur_std = 0
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)

		fig.add_subplot(5,3,7)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 1
		blur_std = 0
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,8)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 2
		blur_std = 0
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,9)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 0.5
		blur_std = 1
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,10)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 1
		blur_std = 1
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,11)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 2
		blur_std = 1
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,12)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')




		##Pink Noise Perturbations
		amplitude=0.5
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,13)
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		amplitude=1
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,14)
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		amplitude=2
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		ax=fig.add_subplot(5,3,15)
		ax.grid()
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

	plt.savefig(os.path.expanduser('results/mnist/img'+str(i)+'_'+str(correlation_type)+'_'+str(correlation_subtype)+'('+str(num_perturbations)+').png'),dpi=300)
	fig.clf()

	print('DONE')



def localization_demonstration(i,original_img,label,certainty,mnist_objects,model,aps,correlation_type,correlation_subtype,num_perturbations):


	print('Localization Demonstration')
	print('Image index: ',i)
	print('Object: ',mnist_objects[label])
	print('Perturbations: ',num_perturbations)
	print('Type: ',correlation_type)
	print('sub-type: ',correlation_subtype)
	print('Original Certainty: ',certainty)


	fig=plt.figure(2)
	#plt.title('Noise comparison for '+mnist_objects[label]+' ('+str(num_perturbations)+' perturbations)', fontsize=30)
	plt.axis('off')
	fig.subplots_adjust(hspace=0.1, wspace=0.05)

	ax=fig.add_subplot(1,4,1)
	ax.grid()
	plt.title(str(mnist_objects[label]))
	plt.imshow(np.squeeze(original_img),cmap='gray')
	plt.axis('off')


	ratio = 1
	############################# BOUNDING BOX ######################################

	box_size = 2
	territory = bounding_box_localization(original_img,label,certainty,box_size,model)
	fig.add_subplot(1,4,2)
	plt.title(str(box_size)+'x'+str(box_size)+' with stride')
	plt.imshow(np.squeeze(territory)*ratio +np.squeeze(original_img)*(1-ratio), cmap='coolwarm')
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
		plt.imshow(np.squeeze(difference_noise)*ratio+np.squeeze(original_img)*(1-ratio),cmap='coolwarm')
		plt.axis('off')

		########## PINK NOISE #############
		amplitude=0.1
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		ax=fig.add_subplot(1,4,4)
		ax.grid()
		plt.title('PN '+str(amplitude))
		im = plt.imshow(np.squeeze(difference_noise)*ratio +np.squeeze(original_img)*(1-ratio),cmap='coolwarm')
		plt.axis('off')


	else: #'pearson'
		########## WHITE NOISE #############
		noise_std = 1
		blur_std = 1
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(1,4,3)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std))
		plt.imshow(correlation_image*ratio+ np.squeeze(original_img)*(1-ratio),cmap='coolwarm')
		plt.axis('off')
		print(np.mean(correlation_image))
		########## PINK NOISE #############
		amplitude=1
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		ax=fig.add_subplot(1,4,4)
		ax.grid()
		plt.title('PN '+str(amplitude))
		im = plt.imshow(correlation_image*ratio + np.squeeze(original_img)*(1-ratio),cmap='coolwarm')
		plt.axis('off')
		print(np.mean(correlation_image))

	fig.subplots_adjust(right=0.9)
	cbar_ax = fig.add_axes([0.91,0.34,0.015,0.31])
	fig.colorbar(im,cax=cbar_ax)
	
	plt.savefig(os.path.expanduser('results/mnist/img'+str(i)+'_'+str(correlation_type)+'_'+str(correlation_subtype)+'('+str(num_perturbations)+')_single.png'),dpi=300)
	fig.clf()
	print('DONE')



def method_comparison(i,original_img,label,certainty,model,aps,num_perturbations):
	

	print('Method Comparison')
	print('Perturbations: ',num_perturbations)
	print('Original Certainty: ',certainty)

	fontsize = 17 
	#fig=plt.figure(3,figsize=(13,6))
	fig=plt.figure(3,figsize=(19,6))
	#plt.title('Noise comparison for '+mnist_objects[label]+' ('+str(num_perturbations)+' perturbations)', fontsize=30)
	plt.axis('off')
	fig.subplots_adjust(hspace=0.05, wspace=0.05)

	##################### FOR WHITE NOISE ########################
	noise_std = 1
	blur_std = 1
	perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared=False)
	ax=fig.add_subplot(2,6,1)
	ax.grid()
	plt.title('Median',fontsize=fontsize)
	plt.imshow(np.squeeze(difference_noise),cmap='gray')
	plt.axis('off')


	perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared=True)
	ax=fig.add_subplot(2,6,2)
	ax.grid()
	plt.title('Median|Squared',fontsize=fontsize)
	plt.imshow(np.squeeze(difference_noise),cmap='gray')
	plt.axis('off')



	## CORRELATIONS

	correlation_type = 'pearson'
	correlation_subtype = 'absolute'

	perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	ax=fig.add_subplot(2,6,3)
	ax.grid()
	plt.title('Pearson|Abs',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')


	correlation_type = 'pearson'
	correlation_subtype = 'squared'

	perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	ax=fig.add_subplot(2,6,4)
	ax.grid()
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


	correlation_type = 'spearman'
	correlation_subtype = 'squared'

	perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	ax=fig.add_subplot(2,6,6)
	ax.grid()
	plt.title('Spearman|Squared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')

	##################### FOR PINK NOISE ########################
	## MEDIAN
	amplitude=1
	perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared=False)
	ax=fig.add_subplot(2,6,7)
	ax.grid()
	#plt.title('Median',fontsize=fontsize)
	plt.imshow(np.squeeze(difference_noise),cmap='gray')
	plt.axis('off')

	perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared=True)
	ax=fig.add_subplot(2,6,8)
	ax.grid()
	#plt.title('Median|Squared',fontsize=fontsize)
	plt.imshow(np.squeeze(difference_noise),cmap='gray')
	plt.axis('off')


	## CORRELATIONS
	correlation_type = 'pearson'
	correlation_subtype = 'absolute'
	perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	ax=fig.add_subplot(2,6,9)
	ax.grid()
	#plt.title('Pearson|Abs',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')

	correlation_type = 'pearson'
	correlation_subtype = 'squared'
	perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	ax=fig.add_subplot(2,6,10)
	ax.grid()
	#plt.title('Pearson|Squared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')

	correlation_type = 'spearman'
	correlation_subtype = 'absolute'
	perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	ax=fig.add_subplot(2,6,11)
	ax.grid()
	#plt.title('Pearson|Abs',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')

	correlation_type = 'spearman'
	correlation_subtype = 'squared'
	perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	ax=fig.add_subplot(2,6,12)
	ax.grid()
	#plt.title('Pearson|Squared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')

	plt.savefig(os.path.expanduser('results/mnist/img'+str(i)+'_method_comparison('+str(num_perturbations)+').png'),dpi=300)
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

	# white noise parameters
	noise_std = 1
	blur_std = 1
	# pink noise parameters
	amplitude = 1


	
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
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')

	ax=fig.add_subplot(3,3,8)
	if noise_type == 'white':
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	else:	
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	#plt.title('Pearson|Squared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
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
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')

	ax=fig.add_subplot(3,3,9)
	if noise_type == 'white':
		perturbations, certainties = generate_white_noise_perturbations(original_img,label,num_perturbations,model,noise_std,blur_std)
	else:	
		perturbations, certainties = generate_pink_noise_perturbations(original_img,label,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	#plt.title('Pearson|Squared',fontsize=fontsize)
	plt.imshow(correlation_image,cmap='gray')
	plt.axis('off')

	plt.savefig(os.path.expanduser('results/mnist/img'+str(i)+'_trial_comparison'+'_'+noise_type+'_noise.png'),dpi=300)
	fig.clf()
	print('DONE')

###################################################################################################################################################################################################




#################################################### GET IMAGES IN THE RIGHT FORM ####################################################
num_classes = 10
# Input image dimensions
img_rows, img_cols = 28, 28

# The data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


############################# LOAD TRAINED MODEL ############################# 
model = load_model('models/mnist_model.h5')
#print(model.summary())


# Evaluate model on test set
score = model.evaluate(x_test, y_test, verbose=0)
print('Model Test set loss:', score[0])
print('Model Test set accuracy:', score[1])
print(score)

# pick sample of test images 
sample_size = 1000 

# get probabilities and labels for sample of images
pred_prob = model.predict(x_test[0:sample_size])
pred_labels = (np.argmax(pred_prob[0:sample_size], axis=1))




#################################################################################

countlab = 10*[0]

## Average Power Spectrum of Test Images (for pink noise process)
aps = average_power_spectrum(x_test)




############################################# SCRIPT PARAMETERS ####################################################

mnist_objects = ['0','1','2','3','4','5','6','7','8','9']


# Select Image from MNIST test set. For example i = 1,2,3,10,27,100,120,130,160,200,250,222

# Select number of perturbations
num_perturbations = 2000

i = 2
# Get initial predicted label and certainty for the selected Image
label = pred_labels[i]
certainty = pred_prob[i][label]
original_img = x_test[i]


# Method Comparison
method_comparison(i,original_img,label,certainty,model,aps,num_perturbations)

# Noise level comparison along with bounding box 
# Select correlation type for classification imagec
correlation_type = 'pearson'
correlation_subtype='squared'
noise_comparison(i,original_img,label,certainty,mnist_objects,model,aps,correlation_type,correlation_subtype,num_perturbations=num_perturbations)

# Localization Demonstration (one setting per type)
localization_demonstration(i,original_img,label,certainty,mnist_objects,model,aps,correlation_type,correlation_subtype,num_perturbations=num_perturbations)


# Compare the effect of Number of trials
noise_type='white' 
#noise_type='pink'
fig_num_perturb(i,original_img,label,certainty,model,aps,noise_type)