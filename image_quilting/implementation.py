""" 
Brian Gabini
S12

REFERENCES:
- https://github.com/axu2/image-quilting
"""

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
        return np.array(imageio.imread(path)) / 255

    # SCORE +1 for implementing save image
    def save_image(self, path: str, image: np.ndarray):
        image *= 255
        image = image.astype(np.uint8)
        imageio.imwrite(path, image)
        

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
        
        texture_height, texture_width, _ = self.texture_image.shape
        patch_height, patch_width, _ = canvas_patch.shape
        errors = np.zeros((texture_height - block_size, texture_width - block_size))
        
        # Initialize for the top left patch (first patch)
        if canvas_indeces.top == 0 and canvas_indeces.left == 0:
            return self.random_patch(patch_height, patch_width, texture_height, texture_width)
        
        else: 
            return self.random_best_patch(errors, canvas_patch, canvas_indeces, block_size, block_overlap, texture_height, texture_width) 
 
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
        # This implementation is random
        texture_height, texture_width, _ = texture_patch.shape
        mask = np.ones((texture_height, texture_width))

        # Initialization topmost-leftmost patch
        if canvas_indeces.top == 0 and canvas_indeces.left == 0:
            return mask
        
        
        mask = self.min_cut_patch(texture_patch, mask, block_overlap, canvas_patch, canvas_indeces.top, canvas_indeces.left)
        return mask

# HELPER METHODS     
    def random_patch(
        self, 
        patch_height: int,
        patch_width: int, 
        texture_height: int, 
        texture_width: int
    ) -> BoxIndeces:
        
        random_y = np.random.randint(0, texture_height - patch_height)
        random_x = np.random.randint(0, texture_width - patch_width)
        return BoxIndeces(
            top=random_y,
            bottom=random_y + patch_height,
            left=random_x,
            right=random_x + patch_width
        )
    
    def random_best_patch(
        self,
        errors,
        canvas_patch: np.ndarray,
        canvas_indeces: BoxIndeces,
        block_size: int,
        block_overlap: int,
        texture_height: int,
        texture_width: int,
    ):
        
        for i in range(texture_height - block_size):
            for j in range(texture_width - block_size):
                texture_patch = self.texture_image[i:i + block_size, j:j + block_size]
                error = self.compute_l2_overlap_diff(texture_patch, block_overlap, canvas_patch, canvas_indeces)
                errors[i, j] = error 
                
        i, j = np.unravel_index(np.argmin(errors), errors.shape)
        return BoxIndeces(
            top=i,
            bottom=i + block_size,
            left=j,
            right=j + block_size
        )

    def min_cut_path(self, errors):
            errors = np.pad(errors, [(0, 0), (1, 1)], 
                            mode='constant', 
                            constant_values=np.inf)

            cumError = errors[0].copy()
            paths = np.zeros_like(errors, dtype=int)    

            for i in range(1, len(errors)):
                M = cumError
                L = np.roll(M, 1)
                R = np.roll(M, -1)

                # optimize with np.choose?
                cumError = np.min((L, M, R), axis=0) + errors[i]
                paths[i] = np.argmin((L, M, R), axis=0)
            
            paths -= 1
            
            minCutPath = [np.argmin(cumError)]
            for i in reversed(range(1, len(errors))):
                minCutPath.append(minCutPath[-1] + paths[i][minCutPath[-1]])
            
            return map(lambda x: x - 1, reversed(minCutPath))
        
    def min_cut_patch(self, patch, mask, overlap, res, y, x):
    
        if x > 0:
            left = patch[:, :overlap] - res[:, :overlap]
            leftL2 = np.sum(left**2, axis=2)
            for i, j in enumerate(self.min_cut_path(leftL2)):
                mask[i, :j] = 0

        if y > 0:
            up = patch[:overlap, :] - res[:overlap, :]
            upL2 = np.sum(up**2, axis=2)
            for j, i in enumerate(self.min_cut_path(upL2.T)):
                mask[:i, j] = 0

        return mask
        
    def compute_l2_overlap_diff(self, texture_patch, overlap, canvas_patch, canvas_indeces):
        error = 0
        
        # Left
        if canvas_indeces.left > 0:
            left = texture_patch[:, :overlap] - canvas_patch[:, :overlap]
            error += np.sum(left**2)

        # Top
        if canvas_indeces.top > 0:
            up = texture_patch[:overlap, :] - canvas_patch[:overlap, :]
            error += np.sum(up**2)

        # Left and Top
        if canvas_indeces.left > 0 and canvas_indeces.top > 0:
            corner = texture_patch[:overlap, :overlap] - canvas_patch[:overlap, :overlap]
            error -= np.sum(corner**2)

        return error
    
    # SCORE +1 if there are no errors when running the entire algorithm
