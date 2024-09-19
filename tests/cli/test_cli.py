import csv
import re
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from iohub import open_ome_zarr
from iohub._version import __version__
from iohub.cli.cli import cli
from tests.conftest import (
    hcs_ref,
    mm2gamma_ome_tiffs,
    ndtiff_v2_datasets,
    ndtiff_v3_labeled_positions,
)

from ..ngff.test_ngff import _temp_copy


def pytest_generate_tests(metafunc):
    if "mm2gamma_ome_tiff" in metafunc.fixturenames:
        metafunc.parametrize("mm2gamma_ome_tiff", mm2gamma_ome_tiffs)
    if "verbose" in metafunc.fixturenames:
        metafunc.parametrize("verbose", ["-v", False])
    if "ndtiff_dataset" in metafunc.fixturenames:
        metafunc.parametrize(
            "ndtiff_dataset",
            ndtiff_v2_datasets + [ndtiff_v3_labeled_positions],
        )


def test_cli_entry():
    runner = CliRunner()
    result = runner.invoke(cli)
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_cli_help():
    runner = CliRunner()
    for option in ("-h", "--help"):
        result = runner.invoke(cli, option)
        assert result.exit_code == 0
        assert "Usage" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, "--version")
    assert result.exit_code == 0
    assert "version" in result.output
    assert str(__version__) in result.output


def test_cli_info_mock(mm2gamma_ome_tiff, verbose):
    runner = CliRunner()
    # resolve path with pathlib to be consistent with `click.Path`
    # this will not normalize partition symbol to lower case on Windows
    path = mm2gamma_ome_tiff.resolve()
    with patch("iohub.cli.cli.print_info") as mock:
        cmd = ["info", str(path)]
        if verbose:
            cmd.append(verbose)
        result = runner.invoke(cli, cmd)
        mock.assert_called_with(path, verbose=bool(verbose))
        assert result.exit_code == 0
        assert "Reading" in result.output


def test_cli_info_ndtiff(ndtiff_dataset, verbose):
    runner = CliRunner()
    cmd = ["info", str(ndtiff_dataset)]
    if verbose:
        cmd.append(verbose)
    result = runner.invoke(cli, cmd)
    assert result.exit_code == 0
    assert "Channel names" in result.output
    assert re.search(r"FOVs:\s+\d", result.output)
    assert "scale (um)" in result.output


def test_cli_info_ome_zarr(verbose):
    runner = CliRunner()
    cmd = ["info", str(hcs_ref)]
    if verbose:
        cmd.append(verbose)
    result = runner.invoke(cli, cmd)
    assert result.exit_code == 0
    assert re.search(r"Wells:\s+1", result.output)
    assert ("Chunk size" in result.output) == bool(verbose)
    # Test on single position
    result_pos = runner.invoke(cli, ["info", str(hcs_ref / "B" / "03" / "0")])
    assert "Channel names" in result_pos.output
    assert "scale (um)" in result_pos.output
    assert "Chunk size" in result_pos.output


@pytest.mark.parametrize("grid_layout", ["-g", None])
def test_cli_convert_ome_tiff(grid_layout, tmpdir):
    dataset = mm2gamma_ome_tiffs[0]
    runner = CliRunner()
    output_dir = tmpdir / "converted.zarr"
    cmd = ["convert", "-i", str(dataset), "-o", output_dir]
    if grid_layout:
        cmd.append(grid_layout)
    result = runner.invoke(cli, cmd)
    assert result.exit_code == 0, result.output
    assert "Converting" in result.output


def test_cli_rename_wells_help():
    runner = CliRunner()
    cmd = ["rename-wells"]
    for option in ("-h", "--help"):
        cmd.append(option)
        result = runner.invoke(cli, cmd)
        assert result.exit_code == 0
        assert ">> iohub rename-wells" in result.output


def test_cli_rename_wells(tmpdir):
    with _temp_copy(hcs_ref) as store_path:
        runner = CliRunner()
        test_csv = tmpdir / "well_names.csv"
        csv_data = [
            ["B/03", "D/4"],
        ]
        with open(test_csv, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)

        cmd = ["rename-wells", "-i", str(store_path), "-c", str(test_csv)]
        result = runner.invoke(cli, cmd)

        assert result.exit_code == 0
        assert "Renaming" in result.output

        with open_ome_zarr(store_path, mode="r") as plate:
            well_names = [well[0] for well in plate.wells()]
            assert "D/4" in well_names
            assert "B/03" not in well_names
            assert len(plate.metadata.wells) == 1
            assert len(plate.metadata.rows) == 1
            assert len(plate.metadata.columns) == 1
            assert plate.metadata.wells[0].path == "D/4"
            assert plate.metadata.rows[0].name == "D"
            assert plate.metadata.columns[0].name == "4"

        # Test round trip
        test_csv_2 = tmpdir / "well_names_2.csv"
        csv_data = [
            ["D/4", "B/03"],
        ]
        with open(test_csv_2, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)

        cmd = ["rename-wells", "-i", store_path, "-c", str(test_csv_2)]
        result = runner.invoke(cli, cmd)

        with open_ome_zarr(store_path, mode="r") as plate:
            well_names = [well[0] for well in plate.wells()]
            assert "D/4" not in well_names
            assert "B/03" in well_names
            assert len(plate.metadata.wells) == 1
            assert len(plate.metadata.rows) == 1
            assert len(plate.metadata.columns) == 1
            assert plate.metadata.wells[0].path == "B/03"
            assert plate.metadata.rows[0].name == "B"
            assert plate.metadata.columns[0].name == "03"
        


