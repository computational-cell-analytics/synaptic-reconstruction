import os
from pathlib import Path
import argparse
from glob import glob
from tqdm import tqdm
from synaptic_reconstruction.inference.cristae import segment_cristae
import imageio.v3 as iio
import mrcfile
import numpy as np


def run_cristae_segmentation(args):
    if args.input_path != "" and os.path.exists(args.input_path):
        img_paths = sorted(glob(os.path.join(args.input_path, "**", "*.mrc"), recursive=True))
        if args.mitochondria_path != "" and os.path.exists(args.mitochondria_path):
            mito_paths = sorted(glob(os.path.join(args.mitochondria_path, "**", "*.tif"), recursive=True))
        else:
            raise Exception(f"Mitochondria path not found {args.mitochondria_path}")
    elif args.single_image_path != "" and os.path.exists(args.single_image_path):
        img_paths = [args.single_image_path]
        if args.single_mitochondria_path != "" and os.path.exists(args.single_mitochondria_path):
            mito_paths = [args.single_mitochondria_path]
        else:
            raise Exception(f"Mitochondria path not found {args.single_mitochondria_path}")
    else:
        raise Exception(f"Input path not found")
    # check if model path exists and remove best.pt if present
    if not os.path.exists(args.model_path):
        raise Exception(f"Model path not found {args.model_path}")
    if "best.pt" in args.model_path:
        model_path = args.model_path.replace("best.pt", "")
    else:
        model_path = args.model_path
    assert len(img_paths) == len(mito_paths), f"Number of images {len(img_paths)} does not match number of mito images {len(mito_paths)}"
    print(f"Processing {len(img_paths)} files")

    # get output path corresponding to input path, if not given
    if args.output_path == "":
        output_path = args.input_path
    elif not args.single_image_path == "":
        output_path = Path(args.single_image_path).parent
    else:
        output_path = args.output_path
        os.makedirs(output_path, exist_ok=True)

    for img_path, mito_path in tqdm(zip(img_paths, mito_paths)):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        assert filename in mito_path, f"{filename} not in {mito_path}"
        output_path = os.path.join(output_path, filename + "_prediction.tif")
        # load img volume
        with mrcfile.open(img_path, "r") as f:
            img = f.data
        # load mitochondria data and make it a mask
        with iio.imopen(mito_path, "r") as f:
            mito_img = f.read()
        mito_img = np.where(mito_img > 0, 1, 0).astype(np.float32)
        img = img.astype(np.float32)
        stacked_img = np.stack((img, mito_img), axis=0)
        # np.stack((img, mitochondria_mask), axis=0)
        seg = segment_cristae(stacked_img, model_path)
        # save tif with imageio
        iio.imwrite(output_path, seg, compression="zlib")
        print(f"Saved segmentation to {output_path}.")


def main():
    parser = argparse.ArgumentParser(description="Segment mitochodria")
    parser.add_argument(
        "--input_path", "-i", default="",
        help="The filepath to directory containing the mitochodria data."
    )
    parser.add_argument(
        "--mitochondria_path", "-im", default="",
        help="The filepath to directory containing the mitochondria data."
    )
    parser.add_argument(
        "--output_path", "-o", default="",
        help="The filepath to directory where the segmented images will be saved."
    )
    parser.add_argument(
        "--single_image_path", "-s", default="",
        help="The filepath to a single image to be segmented."
    )
    parser.add_argument(
        "--single_mitochondria_path", "-sm", default="",
        help="The filepath to a single mitochondria image to segment cristae."
    )
    parser.add_argument(
        "--model_path", "-m", default="",
        help="The filepath to the mitochondria model."
    )

    args = parser.parse_args()

    run_cristae_segmentation(args)


if __name__ == "__main__":
    main()
