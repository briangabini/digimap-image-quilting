from base import ImageQuilting, BoxIndeces, QuiltingOutputs

import numpy as np
import cv2
import scipy.ndimage as ndimage
import PIL.Image as Image
import imageio
import os
import heapq

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
        
    def random_patch(
        self, 
        block_size: int, 
        texture_height: int, 
        texture_width: int
    ) -> BoxIndeces:
        
        random_y = np.random.randint(0, texture_height - block_size)
        random_x = np.random.randint(0, texture_width - block_size)
        return BoxIndeces(
            top=random_y,
            bottom=random_y + block_size,
            left=random_x,
            right=random_x + block_size
        )
    
    def random_best_patch(
        self,
        canvas_patch: np.ndarray,
        canvas_indeces: BoxIndeces,
        block_size: int,
        block_overlap: int,
        texture_height: int,
        texture_width: int,
    ):
        errors = np.zeros((texture_height - block_size, texture_width - block_size))   
        
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
        # dijkstra's algorithm vertical
        pq = [(error, [i]) for i, error in enumerate(errors[0])]
        heapq.heapify(pq)

        h, w = errors.shape
        seen = set()

        while pq:
            error, path = heapq.heappop(pq)
            curDepth = len(path)
            curIndex = path[-1]

            if curDepth == h:
                return path

            for delta in -1, 0, 1:
                nextIndex = curIndex + delta

                if 0 <= nextIndex < w:
                    if (curDepth, nextIndex) not in seen:
                        cumError = error + errors[curDepth, nextIndex]
                        heapq.heappush(pq, (cumError, path + [nextIndex]))
                        seen.add((curDepth, nextIndex))
        
    def min_cut_patch(
        self,
        patch,
        block_size,
        block_overlap,
        canvas_patch,
        canvas_indeces
    ):
        patch = patch.copy()
        dy, dx, _ = patch.shape
        min_cut = np.zeros_like(patch, dtype=bool)
        
        if canvas_indeces.left > 0:
            left = patch[:, :block_overlap] - canvas_patch[canvas_indeces.top:canvas_indeces.top+dy, canvas_indeces.left:canvas_indeces.left+block_overlap]
            leftL2 = np.sum(left**2, axis=2)
            for i, j in enumerate(self.min_cut_path(leftL2)):
                min_cut[i, :j] = True
                
        if canvas_indeces.top > 0:
            up = patch[:block_overlap, :] - canvas_patch[canvas_indeces.top:canvas_indeces.top+block_overlap, canvas_indeces.left:canvas_indeces.left+dx]
            upL2 = np.sum(up**2, axis=2)
            for j, i in enumerate(self.min_cut_path(upL2.T)):
                min_cut[:i, j] = True
                
        np.copyto(patch, canvas_patch[canvas_indeces.top:canvas_indeces.top+dy, canvas_indeces.left:canvas_indeces.left+dx], where=min_cut)

        return BoxIndeces(
            top=canvas_indeces.top,
            bottom=canvas_indeces.top + block_size,
            left=canvas_indeces.left,
            right=canvas_indeces.left + block_size
        )
        
    """ 
        REFERENCES:
        Used this github repo as reference: https://github.com/axu2/image-quilting
    """
    def compute_l2_overlap_diff(self, texture_patch, block_overlap, canvas_patch, canvas_indeces):
        error = 0

        # Left
        if canvas_indeces.left > 0:
            left = texture_patch[:, :block_overlap] - canvas_patch[:, :block_overlap]
            error += np.sum(left**2)

        # Top
        if canvas_indeces.top > 0:
            up = texture_patch[:block_overlap, :] - canvas_patch[:block_overlap, :]
            error += np.sum(up**2)

        # Left and Top
        if canvas_indeces.left > 0 and canvas_indeces.top > 0:
            corner = texture_patch[:block_overlap, :block_overlap] - canvas_patch[:block_overlap, :block_overlap]
            error -= np.sum(corner**2)

        return error

    # SCORE +1 for finding the best matching patch using L2 similarity
    # SCORE +1 for using L2 similarity on the overlap areas only
    """ 
        REFERENCES:
        https://www.digitalocean.com/community/tutorials/norm-of-vector-python - L2 Norm according to this is the Euclidean distance
    """
    def find_matching_texture_patch_indeces(
        self,
        canvas_patch: np.ndarray,
        canvas_indeces: BoxIndeces,
        canvas: np.ndarray,
        block_size: int,
        block_overlap: int
    ) -> BoxIndeces:
        
        texture_height, texture_width, _ = self.texture_image.shape
        
        # Initialize for the top left patch (first patch)
        if canvas_indeces.top == 0 and canvas_indeces.left == 0:
            return self.random_patch(block_size, texture_height, texture_width)
        
        else: 
            patch_indeces = self.random_best_patch(canvas_patch, canvas_indeces, block_size, block_overlap, texture_height, texture_width)
            patch = self.extract_patch(canvas, patch_indeces)
            return self.min_cut_patch(patch, block_size, block_overlap, canvas, canvas_indeces)
            
            # return patch_indeces
            
            

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
        texture_height, texture_width, num_channels = texture_patch.shape
        mask = np.ones((texture_height, texture_width))

        # Initialization topmost-leftmost patch
        if canvas_indeces.top == 0 and canvas_indeces.left == 0:
            return mask
        
        # Left overlap / Topmost patches
        if canvas_indeces.top == 0:
            path_index = np.random.randint(0, block_overlap)
            for y in range(texture_height):
                mask[y, :path_index] = 0

                random_offset = np.random.choice([1, 0, -1])
                path_index += random_offset
                path_index = np.clip(path_index, 0, block_overlap - 1)

            return mask

        # Top overlap / Leftmost patches
        if canvas_indeces.left == 0:
            path_index = np.random.randint(0, block_overlap)
            for x in range(texture_width):
                mask[:path_index, x] = 0

                random_offset = np.random.choice([1, 0, -1])
                path_index += random_offset
                path_index = np.clip(path_index, 0, block_overlap - 1)
            return mask

        # Left and Top overlap / The other patches
        path_index = np.random.randint(0, block_overlap)
        for y in range(texture_height):
            mask[y, :path_index] = 0

            random_offset = np.random.choice([1, 0, -1])
            path_index += random_offset
            path_index = np.clip(path_index, 0, block_overlap - 1)

        path_index = np.random.randint(0, block_overlap)
        for x in range(texture_width):
            mask[:path_index, x] = 0

            random_offset = np.random.choice([1, 0, -1])
            path_index += random_offset
            path_index = np.clip(path_index, 0, block_overlap - 1)

        return mask
    
    # SCORE +1 if there are no errors when running the entire algorithm
