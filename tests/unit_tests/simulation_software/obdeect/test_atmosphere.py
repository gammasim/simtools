"""Tests for the command line of the obdeect propagation."""

from pathlib import Path

from simtools.simulation_software.atmosphere import AtmosphereRequest
from simtools.simulation_software.obdeect.atmosphere import propagation_command
from simtools.simulation_software.programs import program_command


def _request(**kwargs):
    values = {
        "site": "North",
        "observation_level_m": 2158.0,
        "profile_file": Path("atmprof.dat"),
        "transmission_file": Path("transmission.dat"),
        "seed": 1,
    }
    values.update(kwargs)
    return AtmosphereRequest(**values)


def test_propagation_command_default_stdio():
    request = _request(telescope_pathlen_m={1: 28.0, 2: 0.0})

    assert propagation_command(request) == [
        *program_command("obdeect", "obdeect-propagation-run"),
        "--transmission",
        "transmission.dat",
        "--site-altitude",
        "2158.0",
        "--profile",
        "atmprof.dat",
        "--seed",
        "1",
        "--mode",
        "thinned",
        "--telescope-pathlen",
        "1=28.0",
        "--telescope-pathlen",
        "2=0.0",
        "-i",
        "-",
        "-o",
        "-",
    ]


def test_propagation_command_wavelength_range_and_weighted_mode():
    request = _request(mode="weighted", wavelength_range_nm=(250.0, 700.0))

    command = propagation_command(request, program="/opt/obdeect-propagation-run")

    assert command[0] == "/opt/obdeect-propagation-run"
    assert command[command.index("--mode") + 1] == "weighted"
    range_index = command.index("--wavelength-range")
    assert command[range_index + 1 : range_index + 3] == ["250.0", "700.0"]
    assert "--telescope-pathlen" not in command


def test_propagation_command_file_input_and_output():
    request = _request()

    command = propagation_command(request, input_file=Path("in.pst"), output_file=Path("out.pst"))

    assert command[-4:] == ["-i", "in.pst", "-o", "out.pst"]


def test_propagation_command_has_no_extinction_geometry_options():
    command = propagation_command(_request())

    assert "--no-extinction" not in command
    assert "--scattering" not in command
    assert "--envelope" not in command


def test_propagation_command_uses_cataloged_program():
    command = propagation_command(_request())

    assert command[:1] == program_command("obdeect", "obdeect-propagation-run")
    assert Path(command[0]).parent.name == "bin"


def test_propagation_command_program_tokens():
    command = propagation_command(_request(), program=["python", "-c", "code"])

    assert command[:4] == ["python", "-c", "code", "--transmission"]
