import time
from typing import Dict, List, Optional, Tuple, Union
import elf.parallel as parallel
import numpy as np

from skimage.transform import rescale, resize
from synaptic_reconstruction.inference.util import get_prediction_torch_em

DEFAULT_TILING = {
    "tile": {"x": 512, "y": 512, "z": 64},
    "halo": {"x": 64, "y": 64, "z": 8},
}


def _run_segmentation(
    foreground, verbose, min_size,
    # blocking shapes for parallel computation
    block_shape=(128, 256, 256),
):

    # get the segmentation via seeded watershed
    t0 = time.time()
    seg = parallel.label(foreground > 0.5, block_shape=block_shape, verbose=verbose)
    if verbose:
        print("Compute connected components in", time.time() - t0, "s")

    # size filter
    t0 = time.time()
    ids, sizes = parallel.unique(seg, return_counts=True, block_shape=block_shape, verbose=verbose)
    filter_ids = ids[sizes < min_size]
    seg[np.isin(seg, filter_ids)] = 0
    if verbose:
        print("Size filter in", time.time() - t0, "s")
    seg = np.where(seg > 0, 1, 0)
    return seg


def segment_cristae(
    input_volume: np.ndarray,
    model_path: str,
    tiling: Dict[str, Dict[str, int]] = DEFAULT_TILING,
    min_size: int = 500,
    verbose: bool = True,
    distance_based_segmentation: bool = False,
    return_predictions: bool = False,
    scale: Optional[List[float]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    
    """
    Segment cristae in an input volume.

    Args:
        input_volume: The input volume to segment. Expects 2 3D volumes: raw and mitochondria
        model_path: The path to the model checkpoint.
        tiling: The tiling configuration for the prediction.
        min_size: The minimum size of a cristae to be considered.
        verbose: Whether to print timing information.
        distance_based_segmentation: Whether to use distance-based segmentation.
        return_predictions: Whether to return the predictions (foreground, boundaries) alongside the segmentation.
        scale: The scale factor to use for rescaling the input volume before prediction.

    Returns:
        The segmentation mask as a numpy array, or a tuple containing the segmentation mask and the predictions if return_predictions is True.
    """
    if verbose:
        print("Segmenting cristae in volume of shape", input_volume.shape)

    if return_predictions:
        assert scale is None

    if scale is not None:
        original_shape = input_volume.shape
        input_volume = rescale(input_volume, scale, preserve_range=True).astype(input_volume.dtype)
        if verbose:
            print("Rescaled volume from", original_shape, "to", input_volume.shape)

    t0 = time.time()
    
    pred = get_prediction_torch_em(input_volume, model_path, tiling=tiling, with_channels=True)

    foreground, boundaries = pred[:2]
    if verbose:
        print("Run prediction in", time.time() - t0, "s")

    seg = _run_segmentation(
        foreground, verbose=verbose, min_size=min_size
    )

    if scale is not None:
        assert seg.ndim == input_volume.ndim
        seg = resize(seg, original_shape, preserve_range=True, order=0, anti_aliasing=False).astype(seg.dtype)
        assert seg.shape == original_shape

    if return_predictions:
        return seg, pred
    return seg
