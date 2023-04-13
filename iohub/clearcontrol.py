import re
import json
from typing import Any, Tuple, TYPE_CHECKING, List, Sequence, Dict
from pathlib import Path

import blosc2
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


def blosc_buffer_to_array(
    buffer_path: "StrOrBytesPath",
    shape: Tuple[int, ...],
    dtype: np.dtype,
    nthreads: int = 4,
) -> np.ndarray:
    """Loads compressed "blosc" file and converts into numpy array.

    Parameters
    ----------
    buffer_path : StrOrBytesPath
        Compressed blosc buffer path.
    shape : Tuple[int, ...]
        Output array shape.
    dtype : np.dtype
        Output array data type.
    nthreads : int, optional
        Number of blosc decompression threads, by default 4

    Returns
    -------
    np.ndarray
        Output numpy array.
    """

    header_size = 32
    out_arr = np.empty(np.prod(shape), dtype=dtype)
    array_buffer = out_arr

    with open(buffer_path, "rb") as f:
        while True:
            # read header only
            blosc_header = bytes(f.read(header_size))
            if not blosc_header:
                break

            chunk_size, compress_chunk_size, _ = blosc2.get_cbuffer_sizes(blosc_header)

            # move to before the header and read chunk
            f.seek(f.tell() - header_size)
            chunk_buffer = f.read(compress_chunk_size)

            blosc2.decompress2(chunk_buffer, array_buffer, nthreads=nthreads)
            array_buffer = array_buffer[chunk_size // out_arr.itemsize:]

    return out_arr.reshape(shape)


class ClearControlFOV:
    """
    Reader class for Clear Control dataset (https://github.com/royerlab/opensimview).
    It provides a array-like API for the Clear Control dataset while loading the volumes lazily.

    It assumes the channels and volumes have the same shape, the minimum from each channel is used.
    """
    def __init__(self, data_path: "StrOrBytesPath"):
        super().__init__()
        self._data_path = Path(data_path)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Reads Clear Control index data of every data and returns the element-wise minimum shape."""

        # dummy maximum shape size
        shape = [65535,] * 4
        # guess of minimum line length, it might be wrong
        minimum_size = 64
        numbers = re.compile(r"\d+\.\d+|\d+")

        for index_filepath in self._data_path.glob("*.index.txt"):
            with open(index_filepath, "rb") as f:
                if index_filepath.stat().st_size > minimum_size:
                    f.seek(-minimum_size, 2)  # goes to a little bit before the last line
                last_line = f.readlines()[-1].decode("utf-8")

                values = list(numbers.findall(last_line))
                values = [int(values[0]), int(values[4]), int(values[3]), int(values[2])]

                shape = [min(s, v) for s, v in zip(shape, values)]
        
        shape.insert(1, len(self.channels))

        return tuple(shape)
    
    @property
    def channels(self) -> List[str]:
        """Return sorted channels name."""
        suffix = ".index.txt"
        return sorted([
            p.name.removesuffix(suffix)
            for p in self._data_path.glob(f"*{suffix}")
        ])
    
    def _read_volume(
        self,
        volume_shape: Tuple[int, int, int],
        channels: Sequence[str] | str,
        time_point: int,
        dtype: np.dtype = np.uint16,
    ) -> np.ndarray:
        """Reads a single or multiple channels of blosc compressed Clear Control volume.

        Parameters
        ----------
        volume_shape : Tuple[int, int, int]
            3-dimensional volume shape (z, y, x).
        channels : Sequence[str] | str]
            Channels names.
        time_point : int
            Volume time point.

        Returns
        -------
        np.ndarray
            Volume as an array can be single or multiple channels.

        Raises
        ------
        ValueError
            When expected volume path not found.
        """
        # single channel
        if isinstance(channels, str):
            volume_name = f"{str(time_point).zfill(6)}.blc"
            volume_path = self._data_path / "stacks" / channels / volume_name
            if not volume_path.exists():
                raise ValueError(f"{volume_path} not found.")
            return blosc_buffer_to_array(volume_path, volume_shape, dtype=dtype)
        
        return np.stack(
            [self._read_volume(volume_shape, ch, time_point) for ch in channels]
        )
        
    
    def __getitem__(
        self, key: (
            int |
            slice |
            List |
            Tuple[int, ...] |
            Tuple[slice, ...]
        ),
    ) -> np.ndarray:
        """Lazily load array as indexed.

        Parameters
        ----------
        key : int  |  slice  |  List  |  Tuple[int, ...]  |  Tuple[slice, ...]
            An indexing key as in numpy, but a bit more limited.

        Returns
        -------
        np.ndarray
            Output array.

        Raises
        ------
        NotImplementedError
            Not all numpy array of indexing are implemented.
        """

        # these are properties are loaded to avoid multiple reads per call
        shape = self.shape
        channels = self.channels
        time_pts = list(range(shape[0]))
        volume_shape = shape[-3:]

        err_msg = NotImplementedError(f"ClearControlFOV indexing not implemented for {key}."
                                       "Only Integer, List and slice indexing are available.")

        # querying a single time point
        if isinstance(key, int):
            self._read_volume(self._data_path, volume_shape, channels, key)

        # querying multiple time points
        elif isinstance(key, (List, slice)):
            return np.stack([
                self.__getitem__(t) for t in time_pts[key]
            ])

        # querying time points and channels at once
        elif isinstance(key, Tuple):

            if len(key) == 1:
                return self.__getitem__(key[0])

            if len(key) == 2:
                T, C = key
                arr_keys = ...

            else:
                T, C = key[:2]
                arr_keys = key[2:]
            
            # single time point
            if isinstance(T, int):
                out_arr = self._read_volume(volume_shape, channels[C], T) 
            
            # multiple time points
            elif isinstance(T, (List, slice)):
                out_arr = np.stack([
                    self._read_volume(volume_shape, channels[C], t)
                    for t in time_pts[T]
                ])
            
            else:
                raise err_msg
            
            return out_arr[arr_keys]
       
        else:
            raise err_msg

    def __setitem__(self, key: Any, value: Any) -> None:
        raise PermissionError("ClearControlFOV is read-only.")

    def metadata(self) -> Dict[str, Any]:
        """Summarizes Clear Control metadata into a dictionary."""
        cc_metadata = []
        for path in self._data_path.glob("*.metadata.txt"):
            with open(path, mode="r") as f:
                channel_metadata = pd.concat([
                    json.loads(s) for s in f.readlines()
                ])
            cc_metadata.append(channel_metadata)
        
        cc_metadata = pd.concat(cc_metadata)

        time_delta = cc_metadata.groupby("Channel")["TimeStampInNanoSeconds"].diff()
        acquisition_type = cc_metadata["AcquisitionType"].first()

        metadata = {
            "voxel_size_z": cc_metadata["VoxelDimZ"].mean(),     # micrometers
            "voxel_size_y": cc_metadata["VoxelDimY"].mean(),     # micrometers
            "voxel_size_x": cc_metadata["VoxelDimX"].mean(),     # micrometers
            "time_delta": time_delta.mean().mean() / 1_000_000,  # seconds
            "acquisition_type": acquisition_type,
        }

        return metadata
