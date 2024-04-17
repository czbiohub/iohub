import warnings
from typing import Literal, Union

import numpy as np
import zarr
from ndtiff import Dataset

from iohub.reader_base import ReaderBase


class NDTiffReader(ReaderBase):
    """Reader for ND-TIFF datasets acquired with Micro/Pycro-Manager,
    effectively a wrapper of the `ndtiff.Dataset` class.
    """

    def __init__(self, data_path: str):
        super().__init__()

        self.dataset = Dataset(data_path)
        self._axes = self.dataset.axes
        self._str_posistion_axis = self._check_str_axis("position")
        self._str_channel_axis = self._check_str_axis("channel")
        self.frames = (
            len(self._axes["time"]) if "time" in self._axes.keys() else 1
        )
        self.slices = len(self._axes["z"]) if "z" in self._axes.keys() else 1
        self.cameras = (
            self._axes["camera"] if "camera" in self._axes.keys() else []
        )
        self.channels = (
            len(self._axes["channel"]) if "channel" in self._axes.keys() else 1
        )

        self.height = self.dataset.image_height
        self.width = self.dataset.image_width
        self.dtype = self.dataset.dtype

        self.mm_meta = self._get_summary_metadata()
        self.channel_names = list(self.dataset.get_channel_names())
        self.stage_positions = self.mm_meta["Summary"]["StagePositions"]
        self.z_step_size = self.mm_meta["Summary"]["z-step_um"]
        self.xy_pixel_size = self.mm_meta["Summary"].get("PixelSize_um", 1.0)

        if len(self.cameras) > 1:
            channel_names = []
            self.channels = self.channels * len(self.cameras)
            for chan_name in self.channel_names:
                for cam in self.cameras:
                    channel_names.append(chan_name + "_" + cam)
            self.channel_names = channel_names

    def _get_summary_metadata(self):
        pm_metadata = self.dataset.summary_metadata
        pm_metadata["MicroManagerVersion"] = "pycromanager"
        pm_metadata["Positions"] = self.get_num_positions()

        img_metadata = self.get_image_metadata(0, 0, 0, 0, 0)

        pm_metadata["z-step_um"] = None
        if "ZPosition_um_Intended" in img_metadata.keys():
            pm_metadata["z-step_um"] = np.around(
                abs(
                    self.get_image_metadata(0, 0, 0, 0, 1)[
                        "ZPosition_um_Intended"
                    ]
                    - self.get_image_metadata(0, 0, 0, 0, 0)[
                        "ZPosition_um_Intended"
                    ]
                ),
                decimals=3,
            ).astype(float)

        pm_metadata["StagePositions"] = []
        if "position" in self._axes:
            for position in self._axes["position"]:
                position_metadata = {}
                img_metadata = self.get_image_metadata(position, 0, 0, 0, 0)

                if img_metadata is not None and all(
                    key in img_metadata.keys()
                    for key in [
                        "XPosition_um_Intended",
                        "YPosition_um_Intended",
                    ]
                ):
                    position_metadata[img_metadata["Core-XYStage"]] = (
                        img_metadata["XPosition_um_Intended"],
                        img_metadata["YPosition_um_Intended"],
                    )

                # Position label may also be obtained from "PositionName"
                # metadata key with AcqEngJ >= 0.29.0
                if isinstance(position, str):
                    position_metadata["Label"] = position

                pm_metadata["StagePositions"].append(position_metadata)

        return {"Summary": pm_metadata}

    def _check_str_axis(self, axis: Literal["position", "channel"]) -> bool:
        if axis in self._axes:
            coord_sample = next(iter(self._axes[axis]))
            return isinstance(coord_sample, str)
        else:
            return False

    @property
    def str_position_axis(self) -> bool:
        """Position axis is string-valued"""
        return self._str_posistion_axis

    @property
    def str_channel_axis(self) -> bool:
        """Channel axis is string-valued"""
        return self._str_channel_axis

    def _check_coordinates(
        self,
        p: Union[int, str],
        t: int,
        c: Union[int, str],
        camera: Union[int, str],
        z: int,
    ):
        """
        Check that the (p, t, c, z) coordinates are part of the ndtiff dataset.
        Replace coordinates with None or string values in specific cases - see
        below
        """
        coords = [p, t, c, camera, z]
        axes = ("position", "time", "channel", "camera", "z")

        for i, axis in enumerate(axes):
            coord = coords[i]

            # Check if the axis is part of the dataset axes
            if axis in self._axes.keys():
                # Check if coordinate is part of the dataset axis
                if coord in self._axes[axis]:
                    # all good
                    pass

                # The requested coordinate is not part of the axis
                else:
                    # If coord=0 is requested and the coordinate axis exists,
                    # but is string valued (e.g. {'Pos0', 'Pos1'}), a warning
                    # will be raised and the coordinate will be replaced by a
                    # random sample.

                    # Coordinates are in sets, here we get one sample from the
                    # set without removing it:
                    # https://stackoverflow.com/questions/59825
                    coord_sample = next(iter(self._axes[axis]))
                    if coord == 0 and isinstance(coord_sample, str):
                        coords[i] = coord_sample
                        warnings.warn(
                            f"Indices of {axis} are string-valued. "
                            f"Returning data at {axis} = {coord}"
                        )
                    else:
                        # If the coordinate is not part of the axis and
                        # nonzero, a ValueError will be raised
                        raise ValueError(
                            f"Image coordinate {axis} = {coord} is not "
                            "part of this dataset."
                        )

            # The axis is not part of the dataset axes
            else:
                # Nothing to do if coord == None
                if coord is not None:
                    # If coord = 0 is requested, the coordinate will be
                    # replaced with None
                    if coord == 0:
                        coords[i] = None
                    # If coord != 0 is requested and the axis is not part of
                    # the dataset, ValueError will be raised
                    else:
                        raise ValueError(
                            f"Axis {axis} is not part of this dataset"
                        )

        return (*coords,)

    def get_num_positions(self) -> int:
        return (
            len(self._axes["position"])
            if "position" in self._axes.keys()
            else 1
        )

    def get_image(
        self,
        p: Union[int, str],
        t: int,
        c: Union[int, str],
        camera: Union[int, str],
        z: int,
    ) -> np.ndarray:
        """return the image at the provided PTCZ coordinates

        Parameters
        ----------
        p : int or str
            position index
        t : int
            time index
        c : int or str
            channel index
        z : int
            slice/z index

        Returns
        -------
        np.ndarray
            numpy array of shape (Y, X) at given coordinate
        """

        image = None
        p, t, c, camera, z = self._check_coordinates(
            p,
            t,
            c,
            camera,
            z,
        )

        if self.dataset.has_image(
            position=p, time=t, channel=c, camera=camera, z=z
        ):
            image = self.dataset.read_image(
                position=p, time=t, channel=c, camera=camera, z=z
            )

        return image

    def get_zarr(self, position: Union[int, str]) -> zarr.array:
        """.. danger::
            The behavior of this function is different from other
            ReaderBase children as it return a Dask array
            rather than a zarr array.

        Return a lazy-loaded dask array with shape TCZYX at the given position.
        Data is not loaded into memory.


        Parameters
        ----------
        position:       (int) position index

        Returns
        -------
        position:       (zarr.array)

        """
        # TODO: try casting the dask array into a zarr array
        # using `dask.array.to_zarr()`.
        # Currently this call brings the data into memory
        if "position" in self._axes.keys():
            if position not in self._axes["position"]:
                raise ValueError(
                    f"Position index {position} is not part of this dataset. "
                    f'Valid positions are: {self._axes["position"]}'
                )
        else:
            if position not in (0, None):
                warnings.warn(
                    f"Position index {position} is not part of this dataset. "
                    "Returning data at the default position."
                )
                position = None

        da = self.dataset.as_array(position=position)

        shape = (
            self.frames,
            self.channels,
            self.slices,
            self.height,
            self.width,
        )
        # add singleton axes so output is 5D
        return da.reshape(shape)

    def get_array(self, position: Union[int, str]) -> np.ndarray:
        """
        return a numpy array with shape TCZYX at the given position

        Parameters
        ----------
        position:       (int) position index

        Returns
        -------
        position:       (np.ndarray)

        """

        return np.asarray(self.get_zarr(position))

    def get_image_metadata(
        self,
        p: Union[int, str],
        t: int,
        c: Union[int, str],
        z: int,
        camera: Union[int, str],
    ) -> dict:
        """Return image plane metadata at the requested PTCZ coordinates

        Parameters
        ----------
        p : int or str
            position index
        t : int
            time index
        c : int or str
            channel index
        z : int
            slice/z index

        Returns
        -------
        dict
            image plane metadata
        """
        metadata = None
        p, t, c, camera, z = self._check_coordinates(p, t, c, z, camera)

        if self.dataset.has_image(
            position=p, time=t, channel=c, z=z, camera=camera
        ):
            metadata = self.dataset.read_metadata(
                position=p, time=t, channel=c, z=z, camera=camera
            )

        return metadata
