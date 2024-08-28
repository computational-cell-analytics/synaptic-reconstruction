import argparse
from functools import partial

from synaptic_reconstruction.inference.mitochondria import segment_mitochondria
from synaptic_reconstruction.inference.util import inference_helper, parse_tiling


def run_mitochondria_segmentation(args):
    tiling = parse_tiling(args.tile_shape, args.halo)
    segmentation_function = partial(segment_mitochondria, model_path=args.model_path, verbose=False, tiling=tiling)
    inference_helper(args.input_path, args.output_path, segmentation_function)


def main():
    parser = argparse.ArgumentParser(description="Segment mitochodria")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="The filepath to mrc file or directory containing the mitochodria data."
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="The filepath to directory where the segmented images will be saved."
    )
    parser.add_argument(
        "--model_path", "-m", required=True,
        help="The filepath to the mitochondria model."
    )
    parser.add_argument(
        "--tile_shape", type=int, nargs=3,
        help="The tile shape for prediction. Lower the tile shape if GPU memory is insufficient."
    )
    parser.add_argument(
        "--halo", type=int, nargs=3,
        help="The halo for prediction. Increase the halo to minimize boundary artifacts."
    )

    args = parser.parse_args()
    run_mitochondria_segmentation(args)


if __name__ == "__main__":
    main()
