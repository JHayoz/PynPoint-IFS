import os
import urllib
import matplotlib.pyplot as plt
from pynpoint import Pypeline, Hdf5ReadingModule, PSFpreparationModule, PcaPsfSubtractionModule
dir = '/home/ipa/quanz/user_accounts/egarvin/IFS_pipeline/30_data/pynpoint_phot/'
# Get the dataset
urllib.request.urlretrieve('https://home.strw.leidenuniv.nl/~stolker/pynpoint/betapic_naco_mp.hdf5', str(dir)+'input_place/betapic_naco_mp.hdf5')

pipeline = Pypeline(working_place_in=str(dir)+'working_place/',
                    input_place_in=str(dir)+'input_place/',
                    output_place_in=str(dir)+'output_place/')


## PSF subtraction with PCA
# We start with the Hdf5ReadingModule which will import the preprocessed data from the HDF5 file that was downloaded into the current database. The instance of the Hdf5ReadingModule class is added to the Pypeline with the add_module method. The dataset that we need to import has the tag stack so we specify this name as input and output in the dictionary of tag_dictionary.

module = Hdf5ReadingModule(name_in='read',
                           input_filename='betapic_naco_mp.hdf5',
                           input_dir=None,
                           tag_dictionary={'stack': 'stack'})

pipeline.add_module(module)


#Next, we use the PSFpreparationModule to mask the central (saturated) area of the PSF and also pixels beyond 1.1 arcseconds.
module = PSFpreparationModule(name_in='prep',
                              image_in_tag='stack',
                              image_out_tag='prep',
                              mask_out_tag=None,
                              norm=False,
                              resize=None,
                              cent_size=0.15,
                              edge_size=1.1)

pipeline.add_module(module)


# The last pipeline module that we use is PcaPsfSubtractionModule. This module will run the PSF subtraction with PCA. Here we chose to subtract 20 principal components and store the median-collapsed residuals at the database tag residuals.

module = PcaPsfSubtractionModule(pca_numbers=[20, ],
                                 name_in='pca',
                                 images_in_tag='prep',
                                 reference_in_tag='prep',
                                 res_median_tag='residuals')

pipeline.add_module(module)

# We can now run the three pipeline modules that were added to the Pypeline with the run method.
pipeline.run()


# Accessing results in the database
# The Pypeline has several methods to access the datasets and attributes that are stored in the database. For example, we can use the get_shape method to check the shape of the residuals dataset that was stored by the PcaPsfSubtractionModule. The dataset contains 1 image since we ran the PSF subtraction only with 20 principal components.

pipeline.get_shape('residuals')

# read the median collapsed residuals of the psf subtraction
residuals = pipeline.get_data('residuals')

# We will also extract the pixel scale, which is stored as the PIXSCALE attribute of the dataset, by using the get_attribute method.
pixscale = pipeline.get_attribute('residuals', 'PIXSCALE')
print(f'Pixel scale = {pixscale*1e3} mas')

# Plotting the residuals
# Finally, let’s have a look at the residuals of the PSF subtraction. For simplicity, we define the image size in arcseconds.
size = pixscale * residuals.shape[-1]/2.

plt.imshow(residuals[0, ], origin='lower', extent=[size, -size, -size, size])
plt.xlabel('RA offset (arcsec)', fontsize=14)
plt.ylabel('Dec offset (arcsec)', fontsize=14)
cb = plt.colorbar()
cb.set_label('Flux (ADU)', size=14.)
plt.show()