import copy
import itertools
import json
import logging
import multiprocessing as mp
import os
from functools import partial
from glob import glob
from pathlib import Path
from typing import Literal, Tuple, Union

import numpy as np
from dask.array import to_zarr
from natsort import natsorted
from numpy.typing import NDArray
from tqdm import tqdm
from tqdm.contrib.itertools import product

from iohub.ngff import Plate, Position, TransformationMeta, open_ome_zarr

MAX_CHUNK_SIZE = 500e6  # in bytes

# Borrowed from mantis
def get_output_paths(
    input_paths: list[Path], output_zarr_path: Path
) -> list[Path]:
    """Generates a mirrored output path list given an input list of positions"""
    list_output_path = []
    for path in input_paths:
        # Select the Row/Column/FOV parts of input path
        path_strings = Path(path).parts[-3:]
        # Append the same Row/Column/FOV to the output zarr path
        list_output_path.append(Path(output_zarr_path, *path_strings))
    return list_output_path


# Borrowed from mantis
def create_empty_zarr(
    position_paths: list[Path],
    output_path: Path,
    output_zyx_shape: Tuple[int],
    chunk_zyx_shape: Tuple[int] = None,
    voxel_size: Tuple[int, float] = (1, 1, 1),
) -> None:
    """Create an empty zarr store mirroring another store"""
    DTYPE = np.float32
    bytes_per_pixel = np.dtype(DTYPE).itemsize

    # Load the first position to infer dataset information
    input_dataset = open_ome_zarr(str(position_paths[0]), mode="r")
    T, C, Z, Y, X = input_dataset.data.shape

    logging.info("Creating empty array...")

    # Handle transforms and metadata
    transform = TransformationMeta(
        type="scale",
        scale=2 * (1,) + voxel_size,
    )

    # Prepare output dataset
    channel_names = input_dataset.channel_names

    # Output shape based on the type of reconstruction
    output_shape = (T, len(channel_names)) + output_zyx_shape
    logging.info(f"Number of positions: {len(position_paths)}")
    logging.info(f"Output shape: {output_shape}")

    # Create output dataset
    output_dataset = open_ome_zarr(
        output_path, layout="hcs", mode="w", channel_names=channel_names
    )
    if chunk_zyx_shape is None:
        chunk_zyx_shape = list(output_zyx_shape)
        # chunk_zyx_shape[-3] > 1 ensures while loop will not stall if single
        # XY image is larger than MAX_CHUNK_SIZE
        while (
            chunk_zyx_shape[-3] > 1
            and np.prod(chunk_zyx_shape) * bytes_per_pixel > MAX_CHUNK_SIZE
        ):
            chunk_zyx_shape[-3] = np.ceil(chunk_zyx_shape[-3] / 2).astype(int)
        chunk_zyx_shape = tuple(chunk_zyx_shape)

    chunk_size = 2 * (1,) + chunk_zyx_shape
    logging.info(f"Chunk size: {chunk_size}")

    # This takes care of the logic for single position or multiple position by wildcards
    for path in position_paths:
        path_strings = Path(path).parts[-3:]
        pos = output_dataset.create_position(
            str(path_strings[0]), str(path_strings[1]), str(path_strings[2])
        )

        _ = pos.create_zeros(
            name="0",
            shape=output_shape,
            chunks=chunk_size,
            dtype=DTYPE,
            transform=[transform],
        )
    input_dataset.close()


def copy_n_paste(
    position: Position,
    output_path: Path,
    t_idx: int,
    c_idx: int,
) -> None:
    """Load a zyx array from a Position object, apply a transformation and save the result to file"""
    print(f"Processing c={c_idx}, t={t_idx}")

    data_array = open_ome_zarr(position)
    zyx_data = data_array[0][t_idx, c_idx]

    with open_ome_zarr(output_path, mode="r+") as output_dataset:
        output_dataset[0][t_idx, c_idx] = zyx_data

    data_array.close()
    logging.info(f"Finished Writing.. c={c_idx}, t={t_idx}")


def rechunking(
    input_zarr_path: Path,
    output_zarr_path: Path,
    chunk_size_zyx: Tuple,
    num_processes: int = 1,
):
    """
    Rechunk a ome-zarr dataset given the  3D rechunking size (Z,Y,X)
    """
    logging.info("Starting Rechunking")
    print(input_zarr_path, output_zarr_path, chunk_size_zyx)
    assert len(input_zarr_path) == 1

    input_zarr_path = input_zarr_path[0]
    output_zarr_path = Path(output_zarr_path)

    # Check we are given a plate
    with open_ome_zarr(input_zarr_path) as plate:
        assert isinstance(plate, Plate)
    # Check chunksize is 3D
    chunk_size_zyx = tuple(chunk_size_zyx)
    assert len(chunk_size_zyx) == 3

    # Convert to wildcard to process and mirror the input zarr
    input_zarr_path = input_zarr_path / "*" / "*" / "*"
    print(input_zarr_path)
    input_zarr_paths = natsorted(glob(str(input_zarr_path)))
    input_zarr_paths = [Path(path) for path in input_zarr_paths]
    output_zarr_paths = get_output_paths(input_zarr_paths, output_zarr_path)
    # Use FOV 0 for output_shape and
    with open_ome_zarr(input_zarr_paths[0]) as position:
        T, C, Z, Y, X = position[0].shape

    # Create empty zarr
    create_empty_zarr(
        position_paths=input_zarr_paths,
        output_path=output_zarr_path,
        output_zyx_shape=(Z, Y, X),
        chunk_zyx_shape=chunk_size_zyx,
    )

    for input_path, output_path in zip(input_zarr_paths, output_zarr_paths):
        with mp.Pool(num_processes) as p:
            p.starmap(
                partial(copy_n_paste, input_path, output_path),
                itertools.product(range(T), range(C)),
            )
