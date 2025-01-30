from __future__ import annotations

import logging
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
from warnings import catch_warnings, filterwarnings

import dask.array as da
import numpy as np
import zarr
from natsort import natsorted
from numpy.typing import ArrayLike
from tifffile import TiffFile
from xarray import DataArray

from iohub.mm_fov import MicroManagerFOV, MicroManagerFOVMapping

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


__all__ = ["MMOmeTiffFOV", "MMStack"]
_logger = logging.getLogger(__name__)


def _normalize_mm_pos_key(key: str | int) -> int:
    try:
        return int(key)
    except TypeError:
        raise TypeError("Micro-Manager position keys must be integers.")


def find_first_ome_tiff_in_mmstack(data_path: Path) -> Path:
    if data_path.is_file():
        if "ome.tif" in data_path.name:
            return data_path
        else:
            raise ValueError("{data_path} is not a OME-TIFF file.")
    elif data_path.is_dir():
        files = data_path.glob("*.ome.tif")
        try:
            return next(files)
        except StopIteration:
            raise FileNotFoundError(
                f"Path {data_path} contains no OME-TIFF files."
            )
    raise FileNotFoundError(f"Path {data_path} does not exist.")


class MMOmeTiffFOV(MicroManagerFOV):
    def __init__(self, parent: MMStack, key: str) -> None:
        super().__init__(parent, key)
        self._xdata = parent.xdata[key]

    @property
    def axes_names(self) -> list[str]:
        return list(self._xdata.dims)

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return self._xdata.shape

    @property
    def dtype(self) -> np.dtype:
        return self._xdata.dtype

    @property
    def t_scale(self) -> float:
        return self.parent._t_scale

    def __getitem__(self, key: int | slice | tuple[int | slice]) -> ArrayLike:
        return self._xdata[key]

    @property
    def xdata(self) -> DataArray:
        return self._xdata

    def frame_metadata(self, t: int, c: int, z: int) -> dict | None:
        """Read image plane metadata from the OME-TIFF file."""
        return self.parent.read_image_metadata(self._position, t, c, z)


class MMStack(MicroManagerFOVMapping):
    """Micro-Manager multi-file OME-TIFF (MMStack) reader.

    Parameters
    ----------
    data_path : StrOrBytesPath
        Path to the directory containing OME-TIFF files
        or the path to the first OME-TIFF file in the series
    """

    def __init__(self, data_path: StrOrBytesPath):
        super().__init__()
        data_path = Path(data_path)
        first_file = find_first_ome_tiff_in_mmstack(data_path)
        self._root = first_file.parent
        self.dirname = self._root.name
        self._first_tif = TiffFile(first_file, is_mmstack=True)
        _logger.debug(f"Parsing {first_file} as MMStack.")
        with catch_warnings():
            # The IJMetadata tag (50839) is sometimes not written
            # See https://micro-manager.org/Micro-Manager_File_Formats
            filterwarnings("ignore", message=r".*50839.*", module="tifffile")
            self._parse_data()
        self._store = None

    def _parse_data(self):
        series = self._first_tif.series[0]
        raw_dims = dict(
            (axis, size)
            for axis, size in zip(series.get_axes(), series.get_shape())
        )
        axes = ("R", "T", "C", "Z", "Y", "X")
        dims = dict((ax, raw_dims.get(ax, 1)) for ax in axes)
        _logger.debug(f"Got dataset dimensions from tifffile: {dims}.")
        (
            self.positions,
            self.frames,
            self.channels,
            self.slices,
            self.height,
            self.width,
        ) = dims.values()
        self._set_mm_meta(self._first_tif.micromanager_metadata)
        self._store = series.aszarr()
        _logger.debug(f"Opened {self._store}.")
        data = da.from_zarr(zarr.open(self._store))
        self.dtype = data.dtype
        img = DataArray(data, dims=raw_dims, name=self.dirname)
        xarr = img.expand_dims(
            [ax for ax in axes if ax not in img.dims]
        ).transpose(*axes)
        if self.channels > len(self.channel_names):
            for c in range(self.channels):
                if c >= len(self.channel_names):
                    self.channel_names.append(f"Channel{c}")
            _logger.warning(
                "Number of channel names in the metadata is "
                "smaller than the number of channels. "
                f"Completing with fallback names: {self.channel_names}."
            )
        # not all positions in the position list may have been acquired
        xarr = xarr[: self.positions]
        xarr.coords["C"] = self.channel_names
        xset = xarr.to_dataset(dim="R")
        self._xdata = xset
        self._infer_image_meta()

    @property
    def xdata(self):
        return self._xdata

    def __len__(self) -> int:
        return len(self.xdata.keys())

    def __getitem__(self, key: str | int) -> MMOmeTiffFOV:
        key = _normalize_mm_pos_key(key)
        return MMOmeTiffFOV(self, key)

    def __setitem__(self, key, value) -> None:
        raise PermissionError("MMStack is read-only.")

    def __delitem__(self, key, value) -> None:
        raise PermissionError("MMStack is read-only.")

    def __contains__(self, key: str | int) -> bool:
        key = _normalize_mm_pos_key(key)
        return key in self.xdata

    def __iter__(self) -> Iterable[tuple[str, MMOmeTiffFOV]]:
        for key in self.xdata:
            yield str(key), self[key]

    def __enter__(self) -> MMStack:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Close file handles"""
        self._first_tif.close()

    def _set_mm_meta(self, mm_meta: dict) -> None:
        """Assign image metadata from summary metadata."""
        self._mm_meta = mm_meta
        self.channel_names = []
        mm_version = self._mm_meta["Summary"]["MicroManagerVersion"]
        if "beta" in mm_version:
            if self._mm_meta["Summary"]["Positions"] > 1:
                self._stage_positions = []

                for p in range(
                    len(self._mm_meta["Summary"]["StagePositions"])
                ):
                    pos = self._simplify_stage_position_beta(
                        self._mm_meta["Summary"]["StagePositions"][p]
                    )
                    self._stage_positions.append(pos)

            # MM beta versions sometimes don't have 'ChNames',
            # so I'm wrapping in a try-except and setting the
            # channel names to empty strings if it fails.
            try:
                for ch in self._mm_meta["Summary"]["ChNames"]:
                    self.channel_names.append(ch)
            except Exception:
                self.channel_names = self._mm_meta["Summary"]["Channels"] * [
                    ""
                ]  # empty strings

        elif mm_version == "1.4.22":
            for ch in self._mm_meta["Summary"].get("ChNames", []):
                self.channel_names.append(ch)

        # Parsing of data acquired with the OpenCell
        # acquisition script on the Dragonfly miroscope
        elif (
            mm_version == "2.0.1 20220920"
            and self._mm_meta["Summary"].get("Prefix", None) == "raw_data"
        ):
            files = natsorted(self.root.glob("*.ome.tif"))
            self.positions = len(files)  # not all positions are saved

            if self._mm_meta["Summary"]["Positions"] > 1:
                self._stage_positions = [None] * self.positions

                for p_idx, file_name in enumerate(files):
                    site_idx = int(str(file_name).split("_")[-1].split("-")[0])
                    pos = self._simplify_stage_position(
                        self._mm_meta["Summary"]["StagePositions"][site_idx]
                    )
                    self._stage_positions[p_idx] = pos

            for ch in self._mm_meta["Summary"]["ChNames"]:
                self.channel_names.append(ch)

        else:
            if self._mm_meta["Summary"].get("Positions", 1) > 1:
                self._stage_positions = []

                for p in range(self._mm_meta["Summary"]["Positions"]):
                    pos = self._simplify_stage_position(
                        self._mm_meta["Summary"]["StagePositions"][p]
                    )
                    self._stage_positions.append(pos)

            for ch in self._mm_meta["Summary"].get("ChNames", []):
                self.channel_names.append(ch)
        z_step_size = float(self._mm_meta["Summary"].get("z-step_um", 1.0))
        if z_step_size == 0:
            if self.slices == 1:
                z_step_size = 1.0
            else:
                _logger.warning(
                    f"Z-step size is {z_step_size} um in the metadata, "
                    "Using 1.0 um instead."
                )
        self._z_step_size = z_step_size
        self.height = self._mm_meta["Summary"]["Height"]
        self.width = self._mm_meta["Summary"]["Width"]
        self._t_scale = (
            float(self._mm_meta["Summary"].get("Interval_ms", 1e3)) / 1e3
        )

    def _simplify_stage_position(self, stage_pos: dict):
        """
        flattens the nested dictionary structure of stage_pos
        and removes superfluous keys

        Parameters
        ----------
        stage_pos:      (dict)
            dictionary containing a single position's device info

        Returns
        -------
        out:            (dict)
            flattened dictionary
        """

        out = copy(stage_pos)
        out.pop("DevicePositions")
        for dev_pos in stage_pos["DevicePositions"]:
            out.update({dev_pos["Device"]: dev_pos["Position_um"]})
        return out

    def _simplify_stage_position_beta(self, stage_pos: dict) -> dict:
        """
        flattens the nested dictionary structure of stage_pos
        and removes superfluous keys
        for MM2.0 Beta versions

        Parameters
        ----------
        stage_pos:      (dict)
            dictionary containing a single position's device info

        Returns
        -------
        new_dict:       (dict)
            flattened dictionary

        """

        new_dict = {}
        new_dict["Label"] = stage_pos["label"]
        new_dict["GridRow"] = stage_pos["gridRow"]
        new_dict["GridCol"] = stage_pos["gridCol"]

        for sub in stage_pos["subpositions"]:
            values = []
            for field in ["x", "y", "z"]:
                if sub[field] != 0:
                    values.append(sub[field])
            if len(values) == 1:
                new_dict[sub["stageName"]] = values[0]
            else:
                new_dict[sub["stageName"]] = values

        return new_dict

    def read_image_metadata(
        self, p: int, t: int, c: int, z: int
    ) -> dict | None:
        """Read image plane metadata from the OME-TIFF file."""
        multi_index = (p, t, c, z)
        tif_shape = (self.positions, self.frames, self.channels, self.slices)
        idx = np.ravel_multi_index(multi_index, tif_shape)
        return self._read_frame_metadata(idx)

    def _read_frame_metadata(self, idx: int) -> dict | None:
        # `TiffPageSeries` is not a collection of `TiffPage` objects
        # but a mixture of `TiffPage` and `TiffFrame` objects
        # https://github.com/cgohlke/tifffile/issues/179
        with catch_warnings():
            filterwarnings(
                "ignore", message=r".*from closed file.*", module="tifffile"
            )
            try:
                # virtual frames
                page = self._first_tif.pages[idx]
            except IndexError:
                page = self._first_tif.series[0].pages[idx]
            if page:
                try:
                    page = page.aspage()
                except ValueError:
                    _logger.warning("Cannot read tags from virtual frame.")
                    return None
            else:
                # invalid page
                _logger.warning(f"Page {idx} is not present in the dataset.")
                return None
            try:
                return page.tags["MicroManagerMetadata"].value
            except KeyError:
                _logger.warning("The Micro-Manager metadata tag is not found.")
                return None

    def _infer_image_meta(self) -> None:
        """
        Infer data type and pixel size from the first image plane metadata.
        """
        _logger.debug("Inferring image metadata.")
        metadata = self._read_frame_metadata(0)
        if metadata is not None:
            try:
                self._xy_pixel_size = float(metadata["PixelSizeUm"])
                if self._xy_pixel_size > 0:
                    return
            except Exception:
                _logger.warning(
                    "Micro-Manager image plane metadata cannot be loaded."
                )
        _logger.warning(
            "XY pixel size cannot be determined, defaulting to 1.0 um."
        )
        self._xy_pixel_size = 1.0

    @property
    def zyx_scale(self) -> tuple[float, float, float]:
        return (
            self._z_step_size,
            self._xy_pixel_size,
            self._xy_pixel_size,
        )
