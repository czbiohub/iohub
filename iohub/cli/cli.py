import click

from iohub._version import __version__
from iohub.cli.rechunk import rechunking
from iohub.convert import TIFFConverter
from iohub.reader import print_info

VERSION = __version__

_DATASET_PATH = click.Path(exists=True, file_okay=False, resolve_path=True)


@click.group()
@click.help_option("-h", "--help")
@click.version_option(version=VERSION)
def cli():
    """\u001b[34;1m iohub: N-dimensional bioimaging I/O \u001b[0m"""


@cli.command()
@click.help_option("-h", "--help")
@click.argument(
    "files",
    nargs=-1,
    required=True,
    type=_DATASET_PATH,
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show usage guide to open dataset in Python "
    "and full tree for HCS Plates in OME-Zarr",
)
def info(files, verbose):
    """View basic metadata of a list of FILES.

    Supported formats are Micro-Manager-acquired TIFF datasets
    (single-page TIFF, multi-page OME-TIFF, NDTIFF)
    and OME-Zarr (v0.1 linear HCS layout and all v0.4 layouts).
    """
    for file in files:
        click.echo(f"Reading file:\t {file}")
        print_info(file, verbose=verbose)


@cli.command()
@click.help_option("-h", "--help")
@click.option(
    "--input",
    "-i",
    required=True,
    type=_DATASET_PATH,
    help="Input Micro-Manager TIFF dataset directory",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(exists=False, resolve_path=True),
    help="Output zarr store (/**/converted.zarr)",
)
@click.option(
    "--format",
    "-f",
    required=False,
    type=str,
    help="Data type, 'ometiff', 'ndtiff', 'singlepagetiff'",
)
@click.option(
    "--scale-voxels",
    "-s",
    required=False,
    type=bool,
    default=True,
    help="Write voxel size (XY pixel size and Z-step, in micrometers) "
    "as scale coordinate transformation in NGFF. By default true.",
)
@click.option(
    "--grid-layout",
    "-g",
    required=False,
    is_flag=True,
    help="Arrange FOVs in a row/column grid layout for tiled acquisition",
)
@click.option(
    "--chunks",
    "-c",
    required=False,
    default="XY",
    help="Zarr chunk size given as 'XY', 'XYZ', or a tuple of chunk "
    "dimensions. If 'XYZ', chunk size will be limited to 500 MB.",
)
def convert(input, output, format, scale_voxels, grid_layout, chunks):
    """Converts Micro-Manager TIFF datasets to OME-Zarr"""
    converter = TIFFConverter(
        input_dir=input,
        output_dir=output,
        data_type=format,
        scale_voxels=scale_voxels,
        grid_layout=grid_layout,
        chunks=chunks,
    )
    converter.run()


@cli.command()
@click.argument(
    "input_zarr",
    nargs=-1,
    required=True,
    type=_DATASET_PATH,
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(exists=False, resolve_path=True),
    help="Output zarr store (/**/converted.zarr)",
)
@click.help_option("-h", "--help")
@click.option(
    "--chunks",
    "-c",
    required=False,
    type=(int, int, int),
    default=None,
    help="New chunksize given as (Z,Y,X) tuple argument. The ZYX chunk size will be limited to 500 MB.",
)
@click.option(
    "--num-processes",
    "-j",
    default=1,
    help="Number of simultaneous processes",
    required=False,
    type=int,
)
def rechunk(input_zarr, output, chunks, num_processes):
    """Rechunks OME-Zarr dataset to input chunk_size"""
    rechunking(input_zarr, output, chunks, num_processes)


# if __name__ == "__main__":
#     from iohub.cli.rechunk import rechunking

#     input = "/hpc/projects/comp.micro/mantis/2023_08_09_HEK_PCNA_H2B/xx-mbl_course_H2B/cropped_dataset_v3_small.zarr"
#     output = "./output_test.zarr"
#     chunks = (1, 1, 1)
#     num_processes = 4
#     rechunking(input, output, chunks, num_processes)
