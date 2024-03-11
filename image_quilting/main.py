from implementation import ImageQuilting_AlgorithmAssignment
import base

import argparse

# Run 'python main.py --demo'
# to view the overall pipeline
# In this version, patch selection and patch quilting is performed RANDOMLY

# Run 'python main.py --image Path/To/Source/Texture --output Path/To/Generated/Texture'
# to generate a new texture from a source texture

# Run 'python main.py --visualize --image Path/To/Source/Texture --output Path/To/Generated/Texture'
# to generate a new texture from a source texture, but the entire process will be visualized

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Runs the image quilting algorithm that uses randomly generates a texture from a static image")
    parser.add_argument("--visualize", action="store_true", help="Runs the image quilting algorithm that uses randomly generates a texture from a static image")
    parser.add_argument("--image", type=str, default="pattern.png", help="File path of the source texture")
    parser.add_argument("--output", type=str, default="_temp_generated_texture.png", help="File path where the generated texture will be saved")
    parser.add_argument("--block_size", type=int, default=40, help="See 'guide.jpg'")
    parser.add_argument("--block_overlap", type=int, default=10, help="See 'guide.jpg'")
    parser.add_argument("--output_size", type=int, default=160, help="Size of output image")

    args = parser.parse_args()

    if args.demo:
        quiltor = base.ImageQuiltingRandom(None)
        args.visualize = True
    else:
        quiltor = ImageQuilting_AlgorithmAssignment(args.image)

    if args.visualize:
        generated_texture = base.visualize_process(
            quiltor,
            block_size=args.block_size,
            block_overlap=args.block_overlap,
            canvas_size=args.output_size
        )
    else:
        generated_texture = quiltor.generate_texture(
            block_size=args.block_size,
            block_overlap=args.block_overlap,
            canvas_size=args.output_size
        )

    quiltor.save_image(args.output, generated_texture)
