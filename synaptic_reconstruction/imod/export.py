import shutil
import tempfile
from subprocess import run
from typing import Dict, Optional

import imageio.v3 as imageio
import numpy as np
from elf.io import open_file
from scipy.ndimage import maximum_filter1d
from skimage.morphology import ball

from tqdm import tqdm


def get_label_names(
    imod_path: str,
    return_types: bool = False,
    return_counts: bool = False,
    filter_empty_labels: bool = True,
) -> Dict[str, str]:
    """Get the ids and names of annotations in an imod file.

    Uses the imodinfo command to parse the contents of the imod file.

    Args:
        imod_path: The filepath to the imod annotation file.
        return_types: Whether to also return the annotation types.
        return_counts: Whether to also return the number of annotations per object.
        filter_empty_labels: Whether to filter out empty objects from the object list.

    Returns:
        Dictionary that maps the object ids to their names.
    """
    cmd = "imodinfo"
    cmd_path = shutil.which(cmd)
    assert cmd_path is not None, f"Could not find the {cmd} imod command."

    label_names, label_types, label_counts = {}, {}, {}

    with tempfile.NamedTemporaryFile() as f:
        tmp_path = f.name
        run([cmd, "-f", tmp_path, imod_path])

        object_id = None
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines:

                if line.startswith("OBJECT"):
                    object_id = int(line.rstrip("\n").strip().split()[-1])

                if line.startswith("NAME"):
                    name = line.rstrip("\n").strip().split()[-1]
                    assert object_id is not None
                    label_names[object_id] = name

                if "object uses" in line:
                    type_ = " ".join(line.rstrip("\n").strip().split()[2:]).rstrip(".")
                    label_types[object_id] = type_

                # Parse the number of annotions for this object.
                if "contours" in line:
                    try:
                        count = int(line.rstrip("\n").strip().split()[0])
                        label_counts[object_id] = count
                    # The condition also hits for 'closed countours' / 'open contours'.
                    # We just catch the exception thrown for these cases and continue.
                    except ValueError:
                        pass

                if "CONTOUR" in line and label_types[object_id] == "scattered points":
                    count = int(line.rstrip("\n").strip().split()[2])
                    label_counts[object_id] = count

    if filter_empty_labels:
        empty_labels = [k for k, v in label_counts.items() if v == 0]
        label_names = {k: v for k, v in label_names.items() if k not in empty_labels}
        label_types = {k: v for k, v in label_types.items() if k not in empty_labels}
        label_counts = {k: v for k, v in label_counts.items() if k not in empty_labels}

    if return_types and return_counts:
        return label_names, label_types, label_counts
    elif return_types:
        return label_names, label_types
    elif return_counts:
        return label_names, label_counts

    return label_names


def export_segmentation(
    imod_path: str,
    mrc_path: str,
    object_id: Optional[int] = None,
    output_path: Optional[int] = None,
    require_object: bool = True,
    depth_interpolation: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Export a segmentation mask from IMOD annotations.

    This function uses the imodmop command to extract masks.

    Args:
        imod_path: The filepath to the imod annotation file.
        mrc_path: The filepath to the mrc file with the corresponding tomogram data.
        object_id: The id of the object to be extracted.
            If none is given all objects in the mod file will be extracted.
        output_path: Optional path to a file where to save the extracted mask as tif.
            If not given the segmentation mask will be returned as a numpy array.
        require_object: Whether to throw an error if the object could not be extractd.
        depth_interpolation: Window for interpolating the extracted mask across the depth
            axis. This can be used to close masks which are only annotated in a subset of slices.

    Returns:
        The extracted mask. Will only be returned if `output_path` is not given.
    """
    cmd = "imodmop"
    cmd_path = shutil.which(cmd)
    assert cmd_path is not None, f"Could not find the {cmd} imod command."

    with tempfile.NamedTemporaryFile() as f:
        tmp_path = f.name

        if object_id is None:
            cmd = [cmd, "-ma", "1", imod_path, mrc_path, tmp_path]
        else:
            cmd = [cmd, "-ma", "1", "-o", str(object_id), imod_path, mrc_path, tmp_path]

        run(cmd)
        with open_file(tmp_path, ext=".mrc", mode="r") as f:
            data = f["data"][:]

    segmentation = data == 1
    if require_object and segmentation.sum() == 0:
        id_str = "" if object_id is None else f"for object {object_id}"
        raise RuntimeError(f"Segmentation extracted from {imod_path} {id_str} is empty.")

    if depth_interpolation is not None:
        segmentation = maximum_filter1d(segmentation, size=depth_interpolation, axis=0, mode="constant")

    if output_path is None:
        return segmentation

    imageio.imwrite(output_path, segmentation.astype("uint8"), compression="zlib")


def draw_spheres(coordinates, radii, shape, verbose=True):
    labels = np.zeros(shape, dtype="uint32")
    for label_id, (coord, radius) in tqdm(
        enumerate(zip(coordinates, radii), start=1), total=len(coordinates), disable=not verbose
    ):
        radius = int(radius)
        mask = ball(radius)
        full_mask = np.zeros(shape, dtype="bool")
        full_slice = tuple(
            slice(max(co - radius, 0), min(co + radius, sh)) for co, sh in zip(coord, shape)
        )
        radius_clipped_left = [co - max(co - radius, 0) for co in coord]
        radius_clipped_right = [min(co + radius, sh) - co for co, sh in zip(coord, shape)]
        mask_slice = tuple(
            slice(radius + 1 - rl, radius + 1 + rr) for rl, rr in zip(radius_clipped_left, radius_clipped_right)
        )
        full_mask[full_slice] = mask[mask_slice]
        labels[full_mask] = label_id
    return labels


def load_points_from_imodinfo(
    imod_path, full_shape, bb=None,
    exclude_labels=None, exclude_label_patterns=None,
    resolution=None,
):
    coordinates, sizes, labels = [], [], []
    label_names = {}

    if bb is not None:
        start = [b.start for b in bb]
        shape = [b.stop - sta for b, sta in zip(bb, start)]

    # first round: load the point sizes and labels
    with tempfile.NamedTemporaryFile() as f:
        tmp_path = f.name
        run(["imodinfo", "-p", "-f", tmp_path, imod_path])

        label_id = None
        label_name = None
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines:

                if line.startswith("#Object"):
                    line = line.rstrip("\n")
                    label_id = int(line.split(" ")[1])
                    label_name = line.split(",")[1]
                    label_names[label_id] = label_name
                    continue

                if label_id is None:
                    continue
                if exclude_labels is not None and label_id in exclude_labels:
                    continue
                if exclude_label_patterns is not None and any(
                    pattern.lower() in label_name.lower() for pattern in exclude_label_patterns
                ):
                    continue

                try:
                    size = float(line.rstrip("\n"))
                    sizes.append(size)
                    labels.append(label_id)
                except ValueError:
                    continue

    label_names = {lid: label_name for lid, label_name in label_names.items() if lid in labels}

    in_bounds = []
    with tempfile.NamedTemporaryFile() as f:
        tmp_path = f.name
        run(["imodinfo", "-vv", "-f", tmp_path, imod_path])

        label_id = None
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("OBJECT"):
                    label_id = int(line.rstrip("\n").split(" ")[-1])
                    continue
                if label_id not in label_names:
                    continue

                values = line.strip().rstrip("\n").split("\t")
                if len(values) != 3:
                    continue

                try:
                    x, y, z = float(values[0]), float(values[1]), float(values[2])
                    coords = [x, y, z]
                    coords = [int(np.round(float(coord))) for coord in coords]
                    # IMOD uses a very weird coordinate system:
                    coords = [coords[2], full_shape[1] - coords[1], coords[0]]

                    in_bound = True
                    if bb is not None:
                        coords = [co - sta for co, sta in zip(coords, start)]
                        if any(co < 0 or co > sh for co, sh in zip(coords, shape)):
                            in_bound = False

                    in_bounds.append(in_bound)
                    coordinates.append(coords)
                except ValueError:
                    continue

    # labelx = {seg_id: int(label_id) for seg_id, label_id in enumerate(labels, 1)}
    # print("Extracted the following labels:", label_names)
    # print("With counts:", {k: v for k, v in zip(*np.unique(list(labelx.values()), return_counts=True))})
    # breakpoint()
    assert len(coordinates) == len(sizes) == len(labels) == len(in_bounds), \
        f"{len(coordinates)}, {len(sizes)}, {len(labels)}, {len(in_bounds)}"

    coordinates, sizes, labels = np.array(coordinates), np.array(sizes), np.array(labels)
    in_bounds = np.array(in_bounds)

    # get rid of empty annotations
    in_bounds = np.logical_and(in_bounds, sizes > 0)

    coordinates, sizes, labels = coordinates[in_bounds], sizes[in_bounds], labels[in_bounds]

    if resolution is not None:
        sizes /= resolution

    if len(coordinates) == 0:
        raise RuntimeError(f"Empty annotations: {imod_path}")
    return coordinates, sizes, labels, label_names


def export_point_annotations(
    imod_path,
    shape,
    bb=None,
    exclude_labels=None,
    exclude_label_patterns=None,
    return_coords_and_radii=False,
    resolution=None,
):
    coordinates, radii, labels, label_names = load_points_from_imodinfo(
        imod_path, shape, bb=bb,
        exclude_labels=exclude_labels,
        exclude_label_patterns=exclude_label_patterns,
        resolution=resolution,
    )
    labels = {seg_id: int(label_id) for seg_id, label_id in enumerate(labels, 1)}
    segmentation = draw_spheres(coordinates, radii, shape)
    if return_coords_and_radii:
        return segmentation, labels, label_names, coordinates, radii
    return segmentation, labels, label_names
