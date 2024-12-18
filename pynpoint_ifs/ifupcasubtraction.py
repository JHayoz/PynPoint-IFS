"""
Pipeline modules for PCA PSF subtraction.
"""

import sys
import time
import math
import warnings
import copy

from typing import Union, Tuple
from typeguard import typechecked

import numpy as np
from sklearn.decomposition import PCA

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress

class IFUResidualsPCAModule(ProcessingModule):
    """
    Module to subtract the PSF using PCA.
    """
    __author__ = 'Gabriele Cugno'
    
    @typechecked
    def __init__(
        self,
        pc_number: int,
        name_in: str = "pca_sub",
        image_in_tag: str = "im_2D",
        image_out_tag: str = "im_2D_PCA",
        model_out_tag: str = "PCA_model"
    ) -> None:
        """
        Parameters
        ----------
        :param pc_number : int
            number of removed principal components
        :param name_in : str
            Unique name of the module instance.
        :param image_in_tag : str
            Tag of the database entry that is read as input.
        :param image_out_tag : str
            Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        :param model_out_ta : str
            Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        
        Returns
        -------
        NoneType
            None
        """
        
        super(IFUResidualsPCAModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_model_out_port = self.add_output_port(model_out_tag)
        
        self.m_pc_number = pc_number
    
    def run(self) -> None:
        """
        Run method of the module. Applies PCA to the images.
        
        Returns
        -------
        NoneType
            None
        """

        nim = self.m_image_in_port.get_shape()[0]
        im_shape = self.m_image_in_port.get_shape()
        
        sys.stdout.write('Running IFUResidualsPCAModule...')
        
        im_1D = []
        for k in range(nim):
            im_2D = self.m_image_in_port[k,:,:]-np.mean(self.m_image_in_port[k,:,:])
            im_1D.append(im_2D.reshape(-1))
        im_1D = np.array(im_1D-np.mean(im_1D, axis=0))

        pca_sklearn = PCA(n_components=self.m_pc_number, svd_solver="arpack")
        pca_sklearn.fit(im_1D)
        
        zeros = np.zeros((pca_sklearn.n_components - self.m_pc_number, im_1D.shape[0]))
        pca_rep = np.matmul(pca_sklearn.components_[:self.m_pc_number], im_1D.T)
        pca_rep = np.vstack((pca_rep, zeros)).T
        model = pca_sklearn.inverse_transform(pca_rep)
        im_1D_new = im_1D - model
        
        self.m_image_out_port.set_all(im_1D_new.reshape(im_shape))
        self.m_model_out_port.set_all(model.reshape(im_shape))

        
        sys.stdout.write('Running IFUResidualsPCAModule... [DONE]\n')
        sys.stdout.flush()
        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("PCA", "PC=")
        
        self.m_model_out_port.copy_attributes(self.m_image_in_port)
        self.m_model_out_port.add_history("PCA", "PC=")
        
        self.m_image_out_port.close_port()


