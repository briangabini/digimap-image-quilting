import typing
import abc

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np


class BoxIndeces(typing.NamedTuple):  # See 'ImageQuilting.extract_patch' for usage
    top: int
    bottom: int
    left: int
    right: int


class QuiltingOutputs(typing.NamedTuple):
    canvas_indeces: BoxIndeces
    canvas_patch: np.ndarray
    texture_indeces: BoxIndeces
    texture_patch: np.ndarray
    texture_mask: np.ndarray
    combined_patch: np.ndarray
    canvas: np.ndarray
    canvas_prev: np.ndarray


class ImageQuilting(abc.ABC):  # ABC means ABstract Class

    def __init__(self, texture_file: str):
        self.texture_image = self.load_image(texture_file)

    @abc.abstractmethod
    def load_image(self, path: str) -> np.ndarray: ...

    @abc.abstractmethod
    def save_image(self, path: str, image: np.ndarray): ...

    @abc.abstractmethod
    def find_matching_texture_patch_indeces(
        self,
        canvas_patch: np.ndarray,
        canvas_indeces: BoxIndeces,
        canvas: np.ndarray,
        block_size: int,
        block_overlap: int
    ) -> BoxIndeces: ...

    @abc.abstractmethod
    def compute_texture_mask(
        self,
        canvas_patch: np.ndarray,
        texture_patch: np.ndarray,
        canvas_indeces: BoxIndeces,
        texture_indeces: BoxIndeces,
        canvas: np.ndarray,
        block_size: int,
        block_overlap: int
    ) -> np.ndarray: ...

    def generate_blank_canvas(self, output_height: int, output_width: int) -> np.ndarray:
        num_channels = self.texture_image.shape[-1]
        return np.zeros((output_height, output_width, num_channels), dtype=self.texture_image.dtype)

    def extract_patch(self, image: np.ndarray, indeces: BoxIndeces) -> np.ndarray:
        # Examine this code to understand the content of class BoxIndeces
        return image[indeces.top:indeces.bottom, indeces.left:indeces.right, :].copy()

    def iterate_block_indeces(self, height: int, width: int, block_size: int, block_overlap: int) -> typing.Iterable[BoxIndeces]:
        def generator():
            for top in range(0, height - block_overlap, block_size - block_overlap):
                bottom = min(top + block_size, height)
                for left in range(0, width - block_overlap, block_size - block_overlap):
                    right = min(left + block_size, width)
                    yield BoxIndeces(top, bottom, left, right)
        return list(generator())

    def iterate_texture_generation(self, block_size: int = None, block_overlap: int = None, canvas_size: typing.Union[int, typing.Tuple[int, int]] = None):
        texture_height, texture_width, *_ = self.texture_image.shape

        if block_size is None:
            block_size = int(min(texture_height, texture_width) / 20)  # Default is 1/20 size of source_texture
        if block_overlap is None:
            block_overlap = int(block_size / 6)  # Default is 1/6 size of block_size

        if canvas_size is None:
            canvas_size = 2 * texture_height, 2 * texture_width  # Default is ×2 size of source_texture
        if isinstance(canvas_size, int):
            canvas_size = canvas_size, canvas_size  # If single value convert to tuple, e.g. 2 → (2, 2)
        canvas_height, canvas_width = canvas_size

        canvas_image = self.generate_blank_canvas(output_height=canvas_height, output_width=canvas_width)

        for canvas_indeces in self.iterate_block_indeces(height=canvas_height, width=canvas_width, block_size=block_size, block_overlap=block_overlap):
            canvas_patch = self.extract_patch(canvas_image, canvas_indeces)

            texture_indeces = self.find_matching_texture_patch_indeces(
                canvas_patch=canvas_patch,
                canvas_indeces=canvas_indeces,
                canvas=canvas_image,
                block_size=block_size,
                block_overlap=block_overlap
            )
            texture_patch = self.extract_patch(self.texture_image, texture_indeces)

            texture_mask = self.compute_texture_mask(
                canvas_patch=canvas_patch,
                texture_patch=texture_patch,
                canvas_indeces=canvas_indeces,
                texture_indeces=texture_indeces,
                canvas=canvas_image,
                block_size=block_size,
                block_overlap=block_overlap
            )
            combined_patch = texture_patch * texture_mask[:, :, np.newaxis] + canvas_patch * (1 - texture_mask[:, :, np.newaxis])

            canvas_image_prev = canvas_image.copy()
            top, bottom, left, right = canvas_indeces
            canvas_image[top:bottom, left:right, :] = combined_patch

            yield QuiltingOutputs(
                canvas_indeces=canvas_indeces,
                canvas_patch=canvas_patch,
                texture_indeces=texture_indeces,
                texture_patch=texture_patch,
                texture_mask=texture_mask,
                combined_patch=combined_patch,
                canvas=canvas_image.copy(),
                canvas_prev=canvas_image_prev.copy()
            )

    def generate_texture(self, canvas_file=None, block_size: int = None, block_overlap: int = None, canvas_size: typing.Union[int, typing.Tuple[int, int]] = None) -> np.ndarray:
        process = self.iterate_texture_generation(
            block_size=block_size,
            block_overlap=block_overlap,
            canvas_size=canvas_size
        )

        canvas = None
        for outputs in process:
            canvas = outputs.canvas

        if canvas_file is not None:
            self.save_image(canvas_file, canvas)

        return canvas


class ImageQuiltingRandom(ImageQuilting):

    def __init__(self, texture_file: str):
        super().__init__(texture_file)

    def load_image(self, path: str) -> np.ndarray:
        # A static image, generate on the spot
        dummy_red = np.linspace(0, 1, 200)[np.newaxis, :].repeat(200, axis=0)
        dummy_green = np.linspace(0, 1, 200)[:, np.newaxis].repeat(200, axis=1)
        dummy_blue = np.zeros((200, 200))
        dummy_image = np.stack([dummy_red, dummy_green, dummy_blue], axis=2)
        return dummy_image.astype(float)

    def save_image(self, path: str, image: np.ndarray):
        # You shouldn't save an image like this
        plt.imshow(image)
        plt.title(f"DUMMY IMAGE SAVE FILE")
        plt.savefig(path)

    def find_matching_texture_patch_indeces(
        self,
        canvas_patch: np.ndarray,
        canvas_indeces: BoxIndeces,
        canvas: np.ndarray,
        block_size: int,
        block_overlap: int
    ) -> BoxIndeces:
        # This implementation is random
        texture_height, texture_width, num_channels = self.texture_image.shape
        patch_height, patch_width, num_channels = canvas_patch.shape
        random_y = np.random.randint(0, texture_height - patch_height)
        random_x = np.random.randint(0, texture_width - patch_width)
        return BoxIndeces(
            top=random_y,
            bottom=random_y + patch_height,
            left=random_x,
            right=random_x + patch_width
        )

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


def visualize_process(quiltor: ImageQuilting, **kwds) -> np.ndarray:
    def draw_box(ax: plt.Axes, indeces: BoxIndeces):
        xy = (indeces.left, indeces.top)
        height = indeces.bottom - indeces.top
        width = indeces.right - indeces.left
        rect = patches.Rectangle(xy, height, width, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    print("Terminate/Exit the visualization early using KeyboardInterrupt on the terminal (KeyboardInterrupt for Windows/Linux: Ctrl+C, must be focused on the terminal)")
    print("Close the figure using keyboard shortcuts on the Figure window (Press Ctrl+W, must be focused on the App Window)")

    canvas = None
    for outputs in quiltor.iterate_texture_generation(**kwds):
        canvas = outputs.canvas

        fig, axs = plt.subplots(2, 3)

        ax: plt.Axes = axs[0, 0]
        ax.set_title("Canvas")
        ax.imshow(outputs.canvas_prev)
        draw_box(ax, outputs.canvas_indeces)

        ax: plt.Axes = axs[1, 0]
        ax.set_title("Source texture")
        ax.imshow(quiltor.texture_image)
        draw_box(ax, outputs.texture_indeces)

        ax: plt.Axes = axs[0, 1]
        ax.imshow(outputs.canvas_patch)
        ax.set_title("Query patch")

        ax: plt.Axes = axs[1, 1]
        ax.imshow(outputs.texture_patch)
        ax.set_title("Best match patch")

        ax: plt.Axes = axs[0, 2]
        ax.imshow(outputs.texture_mask, cmap="gray", vmin=0, vmax=1)
        ax.set_title("Texture Mask")

        ax: plt.Axes = axs[1, 2]
        ax.imshow(outputs.combined_patch)
        ax.set_title("Combined patch")

        fig.suptitle(f"Currently Generating: {outputs.canvas_indeces}")
        plt.show()

    return canvas
