from base import ImageQuilting, BoxIndeces, QuiltingOutputs

import numpy as np
import cv2
import scipy.ndimage as ndimage
import PIL.Image as Image
import imageio
import os

# SCORE -1 if you imported additional modules


# SUBMISSION: You only need to submit this file 'implementation.py'

# Hint: You could refer to online sources for help as long as you cite it

# Hint: For clues on the implementation logic,
# view 'base.ImageQuiltingRandom'

# SCORE +5 for submitting this file (implementation.py)

# Feel free to add as much helper functions in the class
class ImageQuilting_AlgorithmAssignment(ImageQuilting):

    # SCORE +1 for implementing load image
    def load_image(self, path: str) -> np.ndarray:
        raise NotImplementedError()

    # SCORE +1 for implementing save image
    def save_image(self, path: str, image: np.ndarray):
        raise NotImplementedError()

    # SCORE +1 for finding the best matching patch using L2 similarity
    # SCORE +1 for using L2 similarity on the overlap areas only
    def find_matching_texture_patch_indeces(
        self,
        canvas_patch: np.ndarray,
        canvas_indeces: BoxIndeces,
        canvas: np.ndarray,
        block_size: int,
        block_overlap: int
    ) -> BoxIndeces:
        return BoxIndeces(0, 0, 0, 0)

    # SCORE +1 for finding a 'cut' that minimizes the L2 error
    # SCORE +1 for correctly converting a 'cut' to a 'mask' (Hint: mask is binary)
    # SCORE +1 for returning the correct mask for the initial case (row=0, column=0)
    # SCORE +1 for returning the correct mask for the first row (row=0, colums>1) and the first column (row>1, columns=0)
    # SCORE +1 for returning the correct mask for the other cases (row>1, columns>1)
    def compute_texture_mask(
        self,
        canvas_patch: np.ndarray,
        texture_patch: np.ndarray,
        canvas_indeces: BoxIndeces,
        texture_indeces: BoxIndeces,
        canvas: np.ndarray,
        block_size: int,
        block_overlap: int
    ) -> np.ndarray:
        raise NotImplementedError()

    # SCORE +1 if there are no errors when running the entire algorithm
