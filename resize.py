import argparse
import os
from PIL import Image
import pdb

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, dirs, files in os.walk(image_dir):
        images = files
        num_images = len(images)    
        for i, image in enumerate(images):
            with open(os.path.join(subdir, image), 'r+b') as f:
                with Image.open(f) as img:
                    img = resize_image(img, size)
                    subdir_subname = subdir.split('/')[-1]
                    if not os.path.exists(output_dir + subdir_subname):
                        os.makedirs(output_dir + subdir_subname)
                    img.save(os.path.join(output_dir + subdir_subname, image), img.format)
            if i % 100 == 0:
                print ("[%d/%d] Resized the images and saved into '%s'."
                    %(i, num_images, output_dir+subdir_subname))
    
def main(args):
    splits = ['train', 'val']
    for split in splits:
        image_dir = args.image_dir
        output_dir = args.output_dir
        image_size = [args.image_size, args.image_size]
        resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_dir', type=str, default='./data/train2014/',
    parser.add_argument('--image_dir', type=str, default='./data/CUB_200_2011/images/',
                        help='directory for train images')
    # parser.add_argument('--output_dir', type=str, default='./data/resized2014/',
    parser.add_argument('--output_dir', type=str, default='./data/resized_CUB/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)
