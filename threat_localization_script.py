from __future__ import print_function
import sys, os, os.path
#os.environ['CUDA_VISIBLE_DEVICES'] = '1 # for GPU
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.models import load_model 
import numpy as np
import matplotlib
#matplotlib.use('Agg') # when in remote screen
import matplotlib.pyplot as plt

from tfk.simple import load_wrapped, test_wrapped
from tfk.image import load_image

from utilities.threat_utils import generate_gun_white_noise_perturbations, generate_gun_pink_noise_perturbations
from utilities.threat_utils import difference_of_averages_image,correlation_map_image,bounding_box_threat_localization
from utilities.threat_utils import append_grid,rgb2gray

#####################################################


# pre-trained automatic threat detection (ATD) model
MODEL = 'trans_plane-vgg_i256_fHlogH_mplane-48.h5'
model = load_wrapped(MODEL)


#################################################################################
countlab = 10*[0]

# Load pre-computed average power spectrum from 1000 container images
aps = np.load('aps_container.npy')

def noise_comparison(image_name,original_img,certainty,model,aps,correlation_type,correlation_subtype,num_perturbations):

	print('Noise Comparison')
	print('Object: ',image_name)
	print('Perturbations: ',num_perturbations)
	print('Type: ',correlation_type)
	print('sub-type: ',correlation_subtype)

	fig=plt.figure(1,figsize=(8,16))
	#plt.title('Noise comparison for '+cifar_objects[label]+' ('+str(num_perturbations)+' perturbations)', fontsize=30)
	plt.axis('off')
	fig.subplots_adjust(hspace=0.1, wspace=0.05)

	ax=fig.add_subplot(5,3,2)
	plt.title(image_name, fontsize=15)
	plt.imshow(np.squeeze(original_img),cmap='gray')
	plt.axis('off')


	############################# BOUNDING BOXES ######################################
	box_size = 16
	#territory = bounding_box_threat_localization(original_img,certainty,box_size,model)
	#np.save('boxes/'+image_name+'_'+str(box_size)+'_region.npy',territory)
	territory = np.load('boxes/'+image_name+'_'+str(box_size)+'_region.npy')
	fig.add_subplot(5,3,4)
	plt.title(str(box_size)+'x'+str(box_size)+' with stride', fontsize=15)
	plt.imshow(np.squeeze(territory), cmap='gray')

	plt.axis('off')

	box_size = 24
	#territory = bounding_box_threat_localization(original_img,certainty,box_size,model)
	#np.save('boxes/'+image_name+'_'+str(box_size)+'_region.npy',territory)
	territory = np.load('boxes/'+image_name+'_'+str(box_size)+'_region.npy')
	fig.add_subplot(5,3,5)
	plt.title(str(box_size)+'x'+str(box_size)+' with stride', fontsize=15)
	plt.imshow(np.squeeze(territory), cmap='gray')

	plt.axis('off')

	box_size = 32
	#territory = bounding_box_threat_localization(original_img,certainty,box_size,model)
	#np.save('boxes/'+image_name+'_'+str(box_size)+'_region.npy',territory)
	territory = np.load('boxes/'+image_name+'_'+str(box_size)+'_region.npy')
	fig.add_subplot(5,3,6)
	plt.title(str(box_size)+'x'+str(box_size)+' with stride', fontsize=15)
	plt.imshow(np.squeeze(territory), cmap='gray')

	plt.axis('off')

	if correlation_type == 'median':
		############################# CLASSIFICATION IMAGES ######################################
		difference_squared = True
		## White/Blurred Noise Perturbations
		noise_std = 0.1
		blur_std = 5
		perturbations, certainties = generate_gun_white_noise_perturbations(original_img,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,7)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')

		plt.axis('off')

		noise_std = 0.5
		blur_std = 5
		perturbations, certainties = generate_gun_white_noise_perturbations(original_img,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,8)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')

		noise_std = 1
		blur_std = 5
		perturbations, certainties = generate_gun_white_noise_perturbations(original_img,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,9)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')

		noise_std = 0.1
		blur_std = 10
		perturbations, certainties = generate_gun_white_noise_perturbations(original_img,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,10)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')

		noise_std = 0.5
		blur_std = 10
		perturbations, certainties = generate_gun_white_noise_perturbations(original_img,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,11)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')
	
		noise_std = 1
		blur_std = 10
		perturbations, certainties = generate_gun_white_noise_perturbations(original_img,num_perturbations,model,noise_std,blur_std)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,12)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')

		##Pink Noise Perturbations
		amplitude=0.1
		perturbations, certainties = generate_gun_pink_noise_perturbations(original_img,num_perturbations,model,aps,amplitude)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,13)
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')

		amplitude=0.25
		perturbations, certainties = generate_gun_pink_noise_perturbations(original_img,num_perturbations,model,aps,amplitude)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		fig.add_subplot(5,3,14)
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')

		amplitude=0.5
		perturbations, certainties = generate_gun_pink_noise_perturbations(original_img,num_perturbations,model,aps,amplitude)
		difference_noise,positive_noise,negative_noise = difference_of_averages_image(perturbations,certainties,difference_squared)
		ax=fig.add_subplot(5,3,15)
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(np.squeeze(difference_noise),cmap='gray')
		plt.axis('off')

		
	else:
		############################  CORRELATION ######################################
		## White/Blurred Noise Perturbations
		noise_std = 0.1
		blur_std = 5
		perturbations, certainties = generate_gun_white_noise_perturbations(original_img,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,7)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 0.5
		blur_std = 5
		perturbations, certainties = generate_gun_white_noise_perturbations(original_img,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,8)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 1
		blur_std = 5
		perturbations, certainties = generate_gun_white_noise_perturbations(original_img,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,9)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 0.1
		blur_std = 10
		perturbations, certainties = generate_gun_white_noise_perturbations(original_img,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,10)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 0.5
		blur_std = 10
		perturbations, certainties = generate_gun_white_noise_perturbations(original_img,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,11)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		noise_std = 1
		blur_std = 10
		perturbations, certainties = generate_gun_white_noise_perturbations(original_img,num_perturbations,model,noise_std,blur_std)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,12)
		plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')




		##Pink Noise Perturbations
		amplitude=0.1
		perturbations, certainties = generate_gun_pink_noise_perturbations(original_img,num_perturbations,model,aps,amplitude)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,13)
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		amplitude=0.25
		perturbations, certainties = generate_gun_pink_noise_perturbations(original_img,num_perturbations,model,aps,amplitude)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		fig.add_subplot(5,3,14)
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

		amplitude = 0.5
		perturbations, certainties = generate_gun_pink_noise_perturbations(original_img,num_perturbations,model,aps,amplitude)
		correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
		ax=fig.add_subplot(5,3,15)
		plt.title('PN '+str(amplitude), fontsize=15)
		plt.imshow(correlation_image,cmap='gray')
		plt.axis('off')

	plt.savefig(os.path.expanduser('results/threat_localization/'+image_name+'_'+str(correlation_type)+'_('+str(num_perturbations)+').png'),dpi=300)
	fig.clf()
	print('DONE')


def localization_demonstration(image_name,original_img,certainty,model,aps,correlation_type,correlation_subtype,num_perturbations):

	print('Localization Demonstration')
	print('Object: ',image_name)
	print('Perturbations: ',num_perturbations)
	print('Type: ',correlation_type)
	print('sub-type: ',correlation_subtype)

	fig=plt.figure(2,figsize=(20,5))
	#plt.title('Noise comparison for '+cifar_objects[label]+' ('+str(num_perturbations)+' perturbations)', fontsize=30)
	plt.axis('off')
	fig.subplots_adjust(hspace=0.1, wspace=0.05)

	ax=fig.add_subplot(1,4,1)
	plt.title(image_name,fontsize=30)
	plt.imshow(append_grid(np.squeeze(original_img),np.max(original_img)/2),cmap='gray')
	plt.axis('off')

	box_size = 16
	#territory = bounding_box_threat_localization(original_img,certainty,box_size,model)
	#np.save('boxes/'+image_name+'_'+str(box_size)+'_region.npy',territory)
	territory = np.load('boxes/'+image_name+'_'+str(box_size)+'_region.npy')
	territory = np.divide(territory,np.max(territory))

	fig.add_subplot(1,4,2)
	plt.title(str(box_size)+'x'+str(box_size)+' + stride',fontsize=30)
	plt.imshow(append_grid(np.squeeze(territory),np.max(territory)/2), cmap='coolwarm')
	plt.axis('off')

	noise_std = 0.5
	blur_std = 10
	perturbations, certainties = generate_gun_white_noise_perturbations(original_img,num_perturbations,model,noise_std,blur_std)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	correlation_image=np.divide(correlation_image,np.max(correlation_image))

	fig.add_subplot(1,4,3)
	plt.title('WN '+str(noise_std)+'|Blur '+str(blur_std),fontsize=30)
	plt.imshow(append_grid(correlation_image,np.max(correlation_image)/2), cmap='coolwarm')
	plt.axis('off')

	amplitude=0.5
	perturbations, certainties = generate_gun_pink_noise_perturbations(original_img,num_perturbations,model,aps,amplitude)
	correlation_image = correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype)
	correlation_image=np.divide(correlation_image,np.max(correlation_image))


	fig.add_subplot(1,4,4)
	plt.title('PN '+str(amplitude),fontsize=30)
	im = plt.imshow(append_grid(correlation_image,np.max(correlation_image)/2), cmap='coolwarm')
	plt.axis('off')

	fig.subplots_adjust(right=0.9)
	cbar_ax = fig.add_axes([0.92,0.12,0.015,0.75])
	fig.colorbar(im,cax=cbar_ax)

	plt.savefig(os.path.expanduser('results/threat_localization/'+image_name+'_'+str(correlation_type)+'_('+str(num_perturbations)+')_single.png'),dpi=300)
	fig.clf()
	print('DONE')


# Define the number of perturbations
num_perturbations = 100
correlation_type = 'pearson'
correlation_subtype = 'squared'


IMAGE = 'threat_patches/threat_1.png'
image_name ='Gun 1'
image = load_image(IMAGE,mode='G' ,high=True)
########################## GET IMAGES IN THE RIGHT FORM ###############
images = [image]
certainty = test_wrapped(model, images)
print('initial certainty for gun 1: ',certainty)
original_img = image
localization_demonstration(image_name,original_img,certainty,model,aps,correlation_type,correlation_subtype,num_perturbations=num_perturbations)
noise_comparison(image_name,original_img,certainty,model,aps,correlation_type,correlation_subtype,num_perturbations=num_perturbations)


IMAGE = 'threat_patches/threat_2.png'
image_name ='Gun 2'
image = load_image(IMAGE,mode='G' ,high=True)
########################## GET IMAGES IN THE RIGHT FORM ###############
images = [image]
certainty = test_wrapped(model, images)
print('initial certainty for gun 2: ',certainty)
original_img = image
localization_demonstration(image_name,original_img,certainty,model,aps,correlation_type,correlation_subtype,num_perturbations=num_perturbations)
noise_comparison(image_name,original_img,certainty,model,aps,correlation_type,correlation_subtype,num_perturbations=num_perturbations)


IMAGE = 'threat_patches/threat_3.png'
image_name ='Gun 3'
image = load_image(IMAGE,mode='G' ,high=True)
########################## GET IMAGES IN THE RIGHT FORM ###############
images = [image]
certainty = test_wrapped(model, images)
print('initial certainty for gun 3: ',certainty)
original_img = image
localization_demonstration(image_name,original_img,certainty,model,aps,correlation_type,correlation_subtype,num_perturbations=num_perturbations)
noise_comparison(image_name,original_img,certainty,model,aps,correlation_type,correlation_subtype,num_perturbations=num_perturbations)

