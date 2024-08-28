import multiprocessing
import os
import shutil
import tempfile

from concurrent import futures
from glob import glob
from subprocess import run
from typing import Optional, Tuple, Union

import imageio.v3 as imageio
import mrcfile
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from tqdm import tqdm


def convert_segmentation_to_spheres(
    segmentation: np.ndarray,
    verbose: bool = False,
    num_workers: Optional[int] = None,
    resolution: Optional[Tuple[float, float, float]] = None,
):
    """Extract spheres parameterized by center and radius from a segmentation.

    Args:
        segmentation: the segmentation.
        verbose: whether to print a progress bar.
        num_workers: number of workers to use for parallelization.
    Returns:
        np.array: the center coordinates
        np.array: the radii
    """
    num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers
    props = regionprops(segmentation)

    def coords_and_rads(prop):
        seg_id = prop.label

        bbox = prop.bbox
        bb = np.s_[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        mask = segmentation[bb] == seg_id
        seg_radii = distance_transform_edt(mask, sampling=resolution)

        max_coord = np.unravel_index(np.argmax(seg_radii), mask.shape)
        radius = seg_radii[max_coord]

        offset = np.array(bbox[:3])
        coord = np.array(max_coord) + offset
        return coord, radius

    with futures.ThreadPoolExecutor(num_workers) as tp:
        res = list(tqdm(
            tp.map(coords_and_rads, props), disable=not verbose, total=len(props),
            desc="Compute coordinates and radii"
        ))

    coords = [re[0] for re in res]
    rads = [re[1] for re in res]
    return np.array(coords), np.array(rads)


def write_points_to_imod(
    coordinates: np.ndarray,
    radii: np.ndarray,
    shape: Tuple[int, int, int],
    min_radius: Union[float, int],
    output_path: str,
    color: Optional[Tuple[int, int, int]] = None,
):
    """Write point annotations to a .mod file for IMOD.

    Args:
        coordinates: Array with the point coordinates.
        radii: Array with the point radii.
        shape: Shape of the volume to be exported.
        min_radius: Minimum radius for export.
        output_path: Where to save the .mod file.
        color: Optional color for writing out the points.
    """
    cmd = "point2model"
    cmd_path = shutil.which(cmd)
    assert cmd_path is not None, f"Could not find the {cmd} imod command."

    def _pad(inp, n=3):
        inp = int(round(inp))
        plen = 6 + (n - len(str(inp)))
        pw = plen * " "
        return f"{pw}{inp}.00"

    with tempfile.NamedTemporaryFile() as tmp_file:
        fname = tmp_file.name
        with open(fname, "w") as f:
            for coord, radius in zip(coordinates, radii):
                if radius < min_radius:
                    continue
                # IMOD needs peculiar whitespace padding
                x = _pad(coord[2])
                y = _pad(shape[1] - coord[1])
                z = _pad(coord[0])
                f.write(f"{x}{y}{z}{_pad(radius, 2)}\n")

        cmd = [cmd, "-si", "-scat", fname, output_path]
        if color is not None:
            assert len(color) == 3
            r, g, b = [str(co) for co in color]
            cmd += ["-co", f"{r} {g} {b}"]

        run(cmd)


def write_segmentation_to_imod_as_points(
    mrc_path: str,
    segmentation_path: str,
    output_path: str,
    min_radius: Union[int, float],
):
    """Write segmentation results to .mod file with imod point annotations.

    This approximates each segmented object as a sphere.

    Args:
        mrc_path: Filepath to the mrc volume that was segmented.
        segmentation_path: Filepath to the segmentation stored as .tif.
        min_radius: Minimum radius for export.
        output_path: Where to save the .mod file.
    """

    # Read the resolution information from the mrcfile.
    with mrcfile.open(mrc_path, "r") as f:
        resolution = f.voxel_size.tolist()

    # TODO double check this with Fidi / Ben
    # The resolution is stored in angstrom, we convert it to nanometer.
    resolution = [res / 10 for res in resolution]

    # Extract the center coordinates and radii from the segmentation.
    segmentation = imageio.imread(segmentation_path)
    coordinates, radii = convert_segmentation_to_spheres(segmentation, resolution=resolution)

    # Write the point annotations to imod.
    write_points_to_imod(coordinates, radii, segmentation.shape, min_radius, output_path)


# TODO we also need to support .rec files ...
def _get_file_paths(input_path, ext=".mrc"):
    if not os.path.exists(input_path):
        raise Exception(f"Input path not found {input_path}")

    if os.path.isfile(input_path):
        input_files = [input_path]
        input_root = None
    else:
        input_files = sorted(glob(os.path.join(input_path, "**", f"*{ext}"), recursive=True))
        input_root = None

    return input_files, input_root


def export_helper(
    input_path: str,
    segmentation_path: str,
    output_root: str,
    export_function: callable,
    force: bool = False,
):
    """
    Helper function to run imod export for files in a directory.

    Args:
        input_path: The path to the input data.
            Can either be a folder. In this case all mrc files below the folder will be exported.
            Or can be a single mrc file. In this case only this mrc file will be segmented.
        segmentation_path: Filepath to the segmentation results.
            The filestructure must exactly match `input_path`.
        output_root: The path to the output directory where the export results will be saved.
        export_function: The function performing the export to the desired imod format.
            This function must take the path to the input in a .mrc file,
            the path to the segmentation in a .tif file and the output path as only arguments.
            If you want to pass additional arguments to this function the use 'funtools.partial'
        force: Whether to rerun segmentation for output files that are already present.
    """
    input_files, input_root = _get_file_paths(input_path)
    segmentation_files, _ = _get_file_paths(segmentation_path, ext=".tif")
    assert len(input_files) == len(segmentation_files)

    for input_path, seg_path in tqdm(zip(input_files, segmentation_files), total=len(input_files)):
        # Determine the output file name.
        input_folder, input_name = os.path.split(input_path)
        fname = os.path.splitext(input_name)[0] + ".mod"
        if input_root is None:
            output_path = os.path.join(output_root, fname)
        else:  # If we have nested input folders then we preserve the folder structure in the output.
            rel_folder = os.path.relpath(input_folder, input_root)
            output_path = os.path.join(output_root, rel_folder, fname)

        # Check if the output path is already present.
        # If it is we skip the prediction, unless force was set to true.
        if os.path.exists(output_path) and not force:
            continue

        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        export_function(input_path, seg_path, output_path)