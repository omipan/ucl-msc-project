#import matplotlib.pyplot as plt 
import numpy as np
from numpy import linalg as LA
from scipy import ndimage
from scipy.stats.stats import pearsonr,spearmanr
from sklearn import preprocessing
import matplotlib
#matplotlib.use('Agg') # when in remote screen
import matplotlib.pyplot as plt
from tfk.simple import load_wrapped, test_wrapped
from tfk.image import load_image

####################################### NOISE FUNCTIONS #######################################
def normalize_perturbation(noise):
    '''
    Description: Normalizing noise perturbation before its addition to image
    Returns:     Normalized (L2-norm) Noise  
    '''

    # Transform perturbation from width * height * 1 to width * height
    noise_2d = np.squeeze(noise)
    noise_normalized = preprocessing.normalize(noise_2d, norm='l2')
    # Transform perturbation from width * height back to to width * height * 1
    noise_back = np.expand_dims(noise_normalized, axis=2)
    

    return noise_back


#################### WHITE NOISE /BLURRED WHITE NOISE GENERATION ####################
def add_blurred_noise(img,noise_std,blur_std):
    '''
    Description: Image White noise generator of 0 mean, noise_std standard deviation
                 of noise and blur_std standard deviation of blur
    Returns:     Normalized (L2-norm) Noise and Distorted Image 
    '''

    noise_mean = 0
    noise = np.random.normal(noise_mean,noise_std,img.shape)
    # blur white noise
    blurred_noise = ndimage.gaussian_filter(noise, sigma=blur_std)

    # normalize noise perturbation array
    noise = normalize_perturbation(blurred_noise)*65535 #multiply with 2^16 -1 which corresponds to the range of gun images

    return noise,img + noise

def add_pink_noise(img,aps,amplitude):
        '''
        Description: Pink noise generator on 2D image give a base power spectrum and noise amplitude
                     Randomness is introduced to noise generation through random phases in range(-2pi,2pi)
        Returns:  Normalized (L2-norm) Noise multiplied by Amplitude and Distorted Image 
        '''

        # Generate random phases between [-2pi,2pi]
        theta = np.zeros(aps.shape)
        theta =-np.pi + 2*np.pi*np.random.random((aps.shape))

        # Inverse Fourier transform to obtain the spatial pattern
        noise = np.real(np.fft.ifft2(aps*np.exp(theta*1j),axes=(0,1)))
        noise = amplitude*normalize_perturbation(noise)*65535 #multiply with 2^16 -1 which corresponds to the range of gun images

        return noise,img+noise

def generate_gun_white_noise_perturbations(img,num_perturbations,model,noise_std,blur_std):
    '''
    Description: Generating a number of noise perturbations and label prediction for each of
                 the distorted images, after adding random noise perturbations to the original
                 image
    Returns:     Perturbations with the corresponding prediction certainties
    '''
    
    #perturbations = np.zeros((num_perturbations,28,28,1))
    perturbations = np.zeros((num_perturbations,img.shape[0],img.shape[1],img.shape[2]))
    certainties = []

    for k in range(num_perturbations):
        # adding normalized noise perturbation to images
        noise,distorted_image = add_blurred_noise(img,noise_std,blur_std)
        perturbations[k,] = noise
        # Predict with distorted image and get certainty


        certainty = test_wrapped(model, [distorted_image])
        certainties.append(certainty)

    return perturbations,certainties


############################## PINK NOISE GENERATION ##############################

def average_power_spectrum(images,single):
    '''
    Description: Calculating the average power spectrum given a set of images 
                 (used in pink noise generation)
    Returns:     2-D Average Power Spectrum with image dimensions   
    '''

    num_images = len(images)
    
    if single:
        shape = (images.shape[0],images.shape[1])
        
        img = subtract_image_mean(images)
        # n-D Discrete Fourier Transform of image 
        f = np.fft.fftn(img)
        power_spectrum = (np.absolute(f))
        return np.squeeze(power_spectrum)
    else:
        # Initιalize zero-2d (image_rows * image_columns) image
        shape = images[0].shape
        total_ps = np.zeros(shape)
   
        for k in range(num_images):


            img = subtract_image_mean(images[k])
            # n-D Discrete Fourier Transform of image 

            f = np.fft.fft2(img,axes=(0,1))
            
            # Shift the zero-frequency component to the center of the spectrum.
            #f_shift = np.fft.fftshift(f)
            
            # Power/Amplitude spectrum of Fourier S(f)
            # amplitude spectrum ==> np.absolute(f)
            # power spectrum ==> (np.absolute(f))**2
            power_spectrum = (np.absolute(f))
            total_ps += power_spectrum

        # Average power spectrum for all images
        average_ps = np.divide(total_ps,num_images)
        return average_ps


def generate_gun_pink_noise_perturbations(img,num_perturbations,model,aps,amplitude):
    '''
    Description: Generating a number of pink noise perturbations and label prediction for each of
                 the distorted images, after adding random phase perturbations of the average
                 power spectrum to the original image 
    Returns:     Perturbations with the corresponding prediction certainties
    '''
    
    # num_perturbationsx * image_rows * image_columns * 1 dimensions
    perturbations = np.zeros((num_perturbations,img.shape[0],img.shape[1],img.shape[2]))
    certainties = []

    for k in range(num_perturbations):

        noise,distorted_image = add_pink_noise(img,aps,amplitude)

        perturbations[k,] = noise

        certainty = test_wrapped(model, [distorted_image])
        certainties.append(certainty)

    return perturbations,certainties





################################# CLASSIFICATION IMAGE FUNCTIONS ###################################
def difference_of_averages_image(perturbations,certainties,difference_squared):
    '''
    Description: Traditional Classification Images experiment with the 2 averages being calculated on samples
                 that are split on the median of the distribution of certainties
    Returns: Noise+, Noise- and the classification image (their difference)
    '''

    num_perturbations = len(certainties)
    threshold = np.median(certainties)

    positive_noise = np.zeros(perturbations[0].shape)
    negative_noise = np.zeros(perturbations[0].shape)
    num_pos = 0
    num_neg = 0

    for k in range(num_perturbations):
        noise = perturbations[k,]
        
        if certainties[k]>=threshold:
            positive_noise += (noise)
            num_pos += 1
        else:
            negative_noise += (noise)
            num_neg += 1

    positive_noise = (np.divide(positive_noise,num_pos))
    negative_noise = (np.divide(negative_noise,num_neg))


    if difference_squared:
        difference_noise = (positive_noise - negative_noise)**2   #squared classification image
    else:
        difference_noise = positive_noise - negative_noise       #unsquared classification image
    

    return difference_noise,positive_noise,negative_noise

###########################f############################################################################
def correlation_map_image(perturbations,certainties,correlation_type,correlation_subtype):
    '''
    Description: Classification Images correlation map version using Pearson/Spearman's coefficient.
                 The correlation is found on pixel level and corresponds to the relationship between
                 perturbations and certainties.
    Returns: Noise+, Noise- and the classification image (their difference)
    '''

    num_perturbations = perturbations.shape[0]
    image_width = perturbations.shape[1]
    image_height = perturbations.shape[2]
    correlation_image = np.empty((image_width,image_height))

    # Correlate perturbation magnitude across each pixel with scores
    for row in range(image_width):
        for col in range(image_height):
            if correlation_type == 'pearson':
                corr,_ = pearsonr(perturbations[:,row,col,:],np.array(certainties))
            elif correlation_type == 'spearman':
                corr,_ = spearman(perturbations[:,row,col,:],np.array(certainties))

            if correlation_subtype == 'absolute':
                    correlation_score = abs(corr) #absolute
            elif correlation_subtype =='squared':
                    correlation_score = (corr)**2 #squared
            
            correlation_image[row,col] = correlation_score

    return correlation_image


#####################################################################################
def rgb2gray(img_rgb):
    ''' Function that converts RGB image to Grayscale'''

    ## Convert image to grayscale
    img_grayscale = np.dot(img_rgb[...,:3], [0.299, 0.587, 0.114])
    ##expand_dims to N,X,Y,1
    #img_grayscale = np.expand_dims(img_grayscale,axis=3)

    return img_grayscale

def rgb2gray_v2(img_rgb):
    ''' Function that converts RGB image to Grayscale'''

    ## Convert 3-channel image to grayscale via formula sqrt(R^2 + B^2 + G^2)
    img_combined_col = np.empty((img_rgb.shape[0],img_rgb.shape[1]))
    img_combined_col = np.sqrt(img_rgb[...,0]**2 + img_rgb[...,1]**2+ img_rgb[...,2]**2)
    return img_combined_col

def subtract_image_mean(img):
    '''
    Description: Subtract mean of each image from each of its pixels
                 Useful to deal with the DC component(constant part of FT) prior
                 to performing FT (Fourier Transformation) 
                 (used in pink noise generation)
    Returns:     Updated image
    '''

    ## Find average pixel intensity for each channel
    num_channels = img.shape[2]
    if num_channels == 1:     # in case we have 1 channel
        avg_val = np.mean(img[:,:,0])
        ## Subtract avg pixel value from each pixel of the image
        new_img = img - avg_val
        #return new_img
    elif num_channels == 3:     # in case we have 3 channels
        avg_R = np.mean(img[:,:,0])
        avg_G = np.mean(img[:,:,1])
        avg_B = np.mean(img[:,:,2])
        ## Subtract avg (R,G,B) triple from each pixel of the image (across channels)
        new_img = img - (avg_R,avg_G,avg_B)
        #return new_img
    
    return new_img

def append_grid(img,val):
    ''' Function that append grid lines to 1-channel image'''
    dx,dy = img.shape[0]//4,img.shape[1]//4
    grid_col = val
    img[:,::dy] = grid_col
    img[::dx,:] = grid_col
    return img

################################################# BOUNDING BOX ###############################################################

def bounding_box_threat_localization(original_img,certainty,square_size,model):
    '''
    Description: Perform bounding box localization given an image that belongs to a class and a neural network
                 that predicts predefined classes with a certainty.
    Returns:     A localization map of certainties
    '''

    image_width = original_img.shape[0]
    image_height = original_img.shape[1]
    territory = np.zeros(original_img.shape)

    # For STRIDE define stride<square_size, centre_row,centre_col = 0 and even size of square (odd sides) for symmetricity
    # For NO STRIDE define stride length = square_size, centre_row,centre_col = square_size/2
    stride = 4

    centre_row = 0
    centre_col = 0
    while (centre_row <= image_width):
        
        start_row = max(centre_row - square_size//2,0)
        end_row = min(centre_row + square_size//2,image_height)
        
        while (centre_col <= image_height):
            img = np.array(original_img)  
            
            ## Mask part of the image to 0 (black)
            start_col = max(centre_col - square_size//2,0)
            end_col = min(centre_col + square_size//2,image_width)

            img[start_row:end_row,start_col:end_col,0] = 0 # 0 for black
            
            # make prediction with the updated image
            certainty_box = test_wrapped(model, [img])
  
            ## Map image region to certainty amplitude
            territory[start_row:end_row,start_col:end_col,0] += certainty - certainty_box

            # Update box coordinates - move box centre to the right
            # Add stride 
            centre_col = centre_col + stride

        # Update box coordinates - move down
        centre_col = 0

        # Add stride
        centre_row = centre_row + stride

    return territory





