import os
from pathlib import Path
import argparse
from glob import glob
from tqdm import tqdm
import imageio.v3 as iio

import numpy as np
import napari
from elf.io import open_file


def visualize_segmentation(args):
    img = None
    seg = None
    seg2 = None
    if args.image_path != "" and os.path.exists(args.image_path):
        with open_file(args.image_path, "r") as f:
            img = f["data"][:]
    else:
        raise Exception(f"Image path not found {args.image_path}")
    if args.segmentation_path != "" and os.path.exists(args.segmentation_path):
        with iio.imopen(args.segmentation_path, "r") as f:
            seg = f.read()
    elif args.second_segmentation_path != "" and os.path.exists(args.second_segmentation_path):
        with iio.imopen(args.second_segmentation_path, "r") as f:
            seg2 = f.read()
    # else:
    #     raise Exception(f"No segmentation path was found {args.segmentation_path} {args.second_segmentation_path}")

    v = napari.Viewer()
    if img is not None:
        v.add_image(img)
    if seg is not None:
        v.add_labels(seg)
    if seg2 is not None:
        v.add_labels(seg2)
    napari.run()


def main():
    parser = argparse.ArgumentParser(description="Segment mitochodria")
    parser.add_argument(
        "--image_path", "-i", default="",
        help="The path to the .mrc file containing the image/raw data."
    )
    parser.add_argument(
        "--segmentation_path", "-s", default="",
        help="The path to the .tif file containing the segmentation data. e.g. mitochondria"
    )
    parser.add_argument(
        "--second_segmentation_path", "-ss", default="",
        help="A second path to the .tif file containing the segmentation data. e.g. cristae"
    )

    args = parser.parse_args()

    visualize_segmentation(args)


if __name__ == "__main__":
    main()
