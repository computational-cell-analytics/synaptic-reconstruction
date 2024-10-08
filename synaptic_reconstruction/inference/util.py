import os
import time
import warnings
from glob import glob
from typing import Dict, Optional, Tuple

# Suppress annoying import warnings.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import bioimageio.core

import imageio.v3 as imageio
import elf.parallel as parallel
import numpy as np
import torch
import torch_em
import xarray

from elf.io import open_file
from skimage.transform import rescale, resize
from torch_em.util.prediction import predict_with_halo
from tqdm import tqdm


class _Scaler:
    def __init__(self, scale, verbose):
        self.scale = scale
        self.verbose = verbose
        self._original_shape = None

    def scale_input(self, input_volume, is_segmentation=False):
        if self.scale is None:
            return input_volume

        if self._original_shape is None:
            self._original_shape = input_volume.shape
        elif self._oringal_shape != input_volume.shape:
            raise RuntimeError(
                "Scaler was called with different input shapes. "
                "This is not supported, please create a new instance of the class for it."
            )

        if is_segmentation:
            input_volume = rescale(
                input_volume, self.scale, preserve_range=True, order=0, anti_aliasing=False,
            ).astype(input_volume.dtype)
        else:
            input_volume = rescale(input_volume, self.scale, preserve_range=True).astype(input_volume.dtype)

        if self.verbose:
            print("Rescaled volume from", self._original_shape, "to", input_volume.shape)
        return input_volume

    def rescale_output(self, output, is_segmentation):
        if self.scale is None:
            return output

        assert self._original_shape is not None
        out_shape = self._original_shape
        if output.ndim > len(out_shape):
            assert output.ndim == len(out_shape) + 1
            out_shape = (output.shape[0],) + out_shape

        if is_segmentation:
            output = resize(output, out_shape, preserve_range=True, order=0, anti_aliasing=False).astype(output.dtype)
        else:
            output = resize(output, out_shape, preserve_range=True).astype(output.dtype)

        return output


def get_prediction(
    input_volume: np.ndarray,  # [z, y, x]
    tiling: Optional[Dict[str, Dict[str, int]]],  # {"tile": {"z": int, ...}, "halo": {"z": int, ...}}
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    verbose: bool = True,
    with_channels: bool = False,
    mask: Optional[np.ndarray] = None,
):
    """
    Run prediction on a given volume.

    This function will automatically choose the correct prediction implementation,
    depending on the model type.

    Args:
        input_volume: The input volume to predict on.
        model_path: The path to the model checkpoint if 'model' is not provided.
        model: Pre-loaded model. Either model_path or model is required.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        with_channels: Whether to predict with channels.
        mask:

    Returns:
        The predicted volume.
    """
    # make sure either model path or model is passed
    if model is None and model_path is None:
        raise ValueError("Either 'model_path' or 'model' must be provided.")

    if model is not None:
        is_bioimageio = None
    else:
        is_bioimageio = model_path.endswith(".zip")

    if tiling is None:
        tiling = get_default_tiling()

    # We standardize the data for the whole volume beforehand.
    # If we have channels then the standardization is done independently per channel.
    if with_channels:
        # TODO Check that this is the correct axis.
        input_volume = torch_em.transform.raw.standardize(input_volume, axis=(1, 2, 3))
    else:
        input_volume = torch_em.transform.raw.standardize(input_volume)

    # Run prediction with the bioimage.io library.
    if is_bioimageio:
        # TODO determine if we use the old or new API and select the corresponding function
        if mask is not None:
            raise NotImplementedError
        pred = get_prediction_bioimageio_old(input_volume, model_path, tiling, verbose)

    # Run prediction with the torch-em library.
    else:
        if model is None:
            # torch_em expects the root folder of a checkpoint path instead of the checkpoint itself.
            if model_path.endswith("best.pt"):
                model_path = os.path.split(model_path)[0]
        print(f"tiling {tiling}")
        # Create updated_tiling with the same structure
        updated_tiling = {
            "tile": {},
            "halo": tiling["halo"]  # Keep the halo part unchanged
        }
        # Update tile dimensions
        for dim in tiling["tile"]:
            updated_tiling["tile"][dim] = tiling["tile"][dim] - 2 * tiling["halo"][dim]
        print(f"updated_tiling {updated_tiling}")
        pred = get_prediction_torch_em(
            input_volume, updated_tiling, model_path, model, verbose, with_channels, mask=mask
        )

    return pred


def get_prediction_bioimageio_old(
    input_volume: np.ndarray,  # [z, y, x]
    model_path: str,
    tiling: Dict[str, Dict[str, int]],  # {"tile": {"z": int, ...}, "halo": {"z": int, ...}}
    verbose: bool = True,
):
    """
    Run prediction using bioimage.io functionality on a given volume.

    Args:
        input_volume: The input volume to predict on.
        model_path: The path to the model checkpoint.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.

    Returns:
        The predicted volume.
    """
    # get foreground and boundary predictions from the model
    t0 = time.time()
    model = bioimageio.core.load_resource_description(model_path)
    with bioimageio.core.create_prediction_pipeline(model) as pp:
        input_ = xarray.DataArray(input_volume[None, None], dims=tuple("bczyx"))
        pred = bioimageio.core.predict_with_tiling(pp, input_, tiling=tiling, verbose=verbose)[0].squeeze()
    if verbose:
        print("Prediction time in", time.time() - t0, "s")
    return pred


def get_prediction_torch_em(
    input_volume: np.ndarray,  # [z, y, x]
    tiling: Dict[str, Dict[str, int]],  # {"tile": {"z": int, ...}, "halo": {"z": int, ...}}
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    verbose: bool = True,
    with_channels: bool = False,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Run prediction using torch-em on a given volume.

    Args:
        input_volume: The input volume to predict on.
        model_path: The path to the model checkpoint if 'model' is not provided.
        model: Pre-loaded model. Either model_path or model is required.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        with_channels: Whether to predict with channels.

    Returns:
        The predicted volume.
    """
    # get block_shape and halo
    block_shape = [tiling["tile"]["z"], tiling["tile"]["x"], tiling["tile"]["y"]]
    halo = [tiling["halo"]["z"], tiling["halo"]["x"], tiling["halo"]["y"]]

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Suppress warning when loading the model.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model is None:
            if os.path.isdir(model_path):  # Load the model from a torch_em checkpoint.
                model = torch_em.util.load_model(checkpoint=model_path, device=device)
            else:  # Load the model directly from a serialized pytorch model.
                model = torch.load(model_path)

    # Run prediction with the model.
    with torch.no_grad():

        # Deal with 2D segmentation case
        if len(input_volume.shape) == 2:
            block_shape = [block_shape[1], block_shape[2]]
            halo = [halo[1], halo[2]]

        if mask is not None:
            if verbose:
                print("Run prediction with mask.")
            mask = mask.astype("bool")

        pred = predict_with_halo(
            input_volume, model, gpu_ids=[device],
            block_shape=block_shape, halo=halo,
            preprocess=None, with_channels=with_channels, mask=mask,
        )
    if verbose:
        print("Prediction time in", time.time() - t0, "s")
    return pred


def _get_file_paths(input_path, ext=".mrc"):
    if not os.path.exists(input_path):
        raise Exception(f"Input path not found {input_path}")

    if os.path.isfile(input_path):
        input_files = [input_path]
        input_root = None
    else:
        input_files = sorted(glob(os.path.join(input_path, "**", f"*{ext}"), recursive=True))
        input_root = input_path

    return input_files, input_root


def _load_input(img_path, extra_files, i):
    # Load the input data data
    with open_file(img_path, "r") as f:

        # Try to automatically derive the key with the raw data.
        keys = list(f.keys())
        if len(keys) == 1:
            key = keys[0]
        elif "data" in keys:
            key = "data"
        elif "raw" in keys:
            key = "raw"

        input_volume = f[key][:]
    assert input_volume.ndim == 3

    # For now we assume this is always tif.
    if extra_files is not None:
        extra_input = imageio.imread(extra_files[i])
        assert extra_input.shape == input_volume.shape
        input_volume = np.stack([input_volume, extra_input], axis=0)

    return input_volume


def inference_helper(
    input_path: str,
    output_root: str,
    segmentation_function: callable,
    data_ext: str = ".mrc",
    extra_input_path: Optional[str] = None,
    extra_input_ext: str = ".tif",
    mask_input_path: Optional[str] = None,
    mask_input_ext: str = ".tif",
    force: bool = False,
    output_key: Optional[str] = None,
):
    """
    Helper function to run segmentation for mrc files.

    Args:
        input_path: The path to the input data.
            Can either be a folder. In this case all mrc files below the folder will be segmented.
            Or can be a single mrc file. In this case only this mrc file will be segmented.
        output_root: The path to the output directory where the segmentation results will be saved.
        segmentation_function: The function performing the segmentation.
            This function must take the input_volume as the only argument and must return only the segmentation.
            If you want to pass additional arguments to this function the use 'funtools.partial'
        data_ext: File extension for the image data. By default '.mrc' is used.
        extra_input_path: Filepath to extra inputs that need to be concatenated to the raw data loaded from mrc.
            This enables cristae segmentation with an extra mito channel.
        extra_input_ext: File extension for the extra inputs (by default .tif).
        mask_input_path: Filepath to mask(s) that will be used to restrict the segmentation.
        mask_input_ext: File extension for the mask inputs (by default .tif).
        force: Whether to rerun segmentation for output files that are already present.
        output_key: Output key for the prediction. If none will write an hdf5 file.
    """
    # Get the input files. If input_path is a folder then this will load all
    # the mrc files beneath it. Otherwise we assume this is an mrc file already
    # and just return the path to this mrc file.
    input_files, input_root = _get_file_paths(input_path, data_ext)

    # Load extra inputs if the extra_input_path was specified.
    if extra_input_path is None:
        extra_files = None
    else:
        extra_files, _ = _get_file_paths(extra_input_path, extra_input_ext)
        assert len(input_files) == len(extra_files)

    # Load the masks if they were specified.
    if mask_input_path is None:
        mask_files = None
    else:
        mask_files, _ = _get_file_paths(mask_input_path, mask_input_ext)
        assert len(input_files) == len(mask_files)

    for i, img_path in tqdm(enumerate(input_files), total=len(input_files)):
        # Determine the output file name.
        input_folder, input_name = os.path.split(img_path)

        if output_key is None:
            fname = os.path.splitext(input_name)[0] + "_prediction.tif"
        else:
            fname = os.path.splitext(input_name)[0] + "_prediction.h5"

        if input_root is None:
            output_path = os.path.join(output_root, fname)
        else:  # If we have nested input folders then we preserve the folder structure in the output.
            rel_folder = os.path.relpath(input_folder, input_root)
            output_path = os.path.join(output_root, rel_folder, fname)

        # Check if the output path is already present.
        # If it is we skip the prediction, unless force was set to true.
        if os.path.exists(output_path) and not force:
            continue

        # Load the input volume. If we have extra_files then this concatenates the
        # data across a new first axis (= channel axis).
        input_volume = _load_input(img_path, extra_files, i)
        # Load the mask (if given).
        mask = None if mask_files is None else imageio.imread(mask_files[i])

        # Run the segmentation.
        segmentation = segmentation_function(input_volume, mask=mask)

        # Write the result to tif or h5.
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)

        if output_key is None:
            imageio.imwrite(output_path, segmentation, compression="zlib")
        else:
            with open_file(output_path, "a") as f:
                f.create_dataset(output_key, data=segmentation, compression="gzip")

        print(f"Saved segmentation to {output_path}.")


def get_default_tiling():
    """Determine the tile shape and halo depending on the available VRAM.
    """
    if torch.cuda.is_available():
        print("Determining suitable tiling")

        # We always use the same default halo.
        halo = {"x": 64, "y": 64, "z": 16}  # before 64,64,8

        # Determine the GPU RAM and derive a suitable tiling.
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9

        if vram >= 80:
            tile = {"x": 640, "y": 640, "z": 80}
        elif vram >= 40:
            tile = {"x": 512, "y": 512, "z": 64}
        elif vram >= 20:
            tile = {"x": 352, "y": 352, "z": 48}
        else:
            # TODO determine tilings for smaller VRAM
            raise NotImplementedError

        print(f"Determined tile size: {tile}")
        tiling = {"tile": tile, "halo": halo}

    # I am not sure what is reasonable on a cpu. For now choosing very small tiling.
    # (This will not work well on a CPU in any case.)
    else:
        print("Determining default tiling")
        tiling = {
            "tile": {"x": 96, "y": 96, "z": 16},
            "halo": {"x": 16, "y": 16, "z": 4},
        }

    return tiling


def parse_tiling(tile_shape, halo):
    """
    Helper function to parse tiling parameter input from the command line.

    Args:
        tile_shape: The tile shape. If None the default tile shape is used.
        halo: The halo. If None the default halo is used.

    Returns:
        dict: the tiling specification
    """

    default_tiling = get_default_tiling()

    if tile_shape is None:
        tile_shape = default_tiling["tile"]
    else:
        assert len(tile_shape) == 3
        tile_shape = dict(zip("zyx", tile_shape))

    if halo is None:
        halo = default_tiling["halo"]
    else:
        assert len(halo) == 3
        halo = dict(zip("zyx", halo))

    tiling = {"tile": tile_shape, "halo": halo}
    return tiling


def apply_size_filter(
    segmentation: np.ndarray,
    min_size: int,
    verbose: bool = False,
    block_shape: Tuple[int, int, int] = (128, 256, 256),
) -> np.ndarray:
    """Apply size filter to the segmentation to remove small objects.

    Args:
        segmentation: The segmentation.
        min_size: The minimal object size in pixels.
        verbose: Whether to print runtimes.
        block_shape: Block shape for parallelizing the operations.

    Returns:
        The size filtered segmentation.
    """
    if min_size == 0:
        return segmentation
    t0 = time.time()
    ids, sizes = parallel.unique(segmentation, return_counts=True, block_shape=block_shape, verbose=verbose)
    filter_ids = ids[sizes < min_size]
    segmentation[np.isin(segmentation, filter_ids)] = 0
    if verbose:
        print("Size filter in", time.time() - t0, "s")
    return segmentation
