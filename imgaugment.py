from imgaug import augmenters as iaa
from imgaug import parameters as iap

import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt


class ImgAugment:

	def importImages(self, DATASET_SIZE):
		return np.array([mpimg.imread("data/frame{}.jpg".format(i)) for i in range(DATASET_SIZE)],   
    				dtype = np.uint8)


	def augment(self, images):
		seq = iaa.Sequential([
		    iaa.Fliplr(0.2), # horizontal flips
		    
		    # Small gaussian blur with random sigma between 0 and 0.5.
		    iaa.Sometimes(0.8,
		        iaa.GaussianBlur(sigma=(0, 0.5))
		        
		    ),
		    iaa.Sometimes(0.7,
		         iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
		    ),
		    
		    # Strengthen or weaken the contrast in each image.
		    iaa.ContrastNormalization((0.5, 1.5)),
		    # Add gaussian noise.
		    # For 50% of all images, we sample the noise once per pixel.
		    # For the other 50% of all images, we sample the noise per pixel AND
		    # channel. This can change the color (not only brightness) of the
		    # pixels.
		    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.1),
		    
		    # Apply affine transformations to each image.
		    # Scale/zoom them, translate/move them, rotate them and shear them.
		    iaa.Sometimes(0.5,
		        iaa.Affine(
		            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
		            rotate=(-10, 10),
		        )
		     )
		], random_order=True) # apply augmenters in random order

		return seq.augment_images(images)




