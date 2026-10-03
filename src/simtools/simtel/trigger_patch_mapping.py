"""Generate sim_telarray trigger-patch displays with ``pixled``."""

import os
import shlex
import shutil
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import astropy.units as u

from simtools import settings
from simtools.job_execution import job_manager
from simtools.runners.simtel_runner import SIM_TELARRAY_ENV


@dataclass(frozen=True)
class TriggerPatchMappingFiles:
    """Files produced while mapping trigger patches.

    Attributes
    ----------
    camera_file : pathlib.Path
        Generated camera definition passed to ``pixled``.
    iact_file : pathlib.Path
        LED event stream produced by ``pixled``.
    simtel_file : pathlib.Path
        sim_telarray event file.
    postscript_file : pathlib.Path
        Camera display produced by ``read_cta``.
    log_files : tuple[pathlib.Path, ...]
        Standard-output and standard-error logs for each program.
    """

    camera_file: Path
    iact_file: Path
    simtel_file: Path
    postscript_file: Path
    log_files: tuple[Path, ...]


def run_trigger_patch_mapping(
    telescope_model,
    site_model,
    output_directory,
    *,
    required_pixels=2,
    include_presum_slaves=False,
    photons_per_pixel=500,
    events_per_combination=1,
    run_number=1,
    output_name=None,
):
    """Generate a camera display for every trigger definition in a model.

    The model writer exports the camera file first. ``pixled`` then generates
    LED events from every trigger line in that exact file, sim_telarray
    processes the events, and ``read_cta`` writes the PostScript display.

    Parameters
    ----------
    telescope_model : simtools.model.telescope_model.TelescopeModel
        Telescope model whose camera trigger tables are mapped.
    site_model : simtools.model.site_model.SiteModel
        Site model supplying sim_telarray atmosphere settings.
    output_directory : str or pathlib.Path
        Directory for generated event files, display, and logs.
    required_pixels : int, optional
        Number of LEDs that ``pixled`` requires in each generated combination.
    include_presum_slaves : bool, optional
        Include pre-sum slave pixels when a master pixel fires.
    photons_per_pixel : int, optional
        Number of photons emitted by every selected LED.
    events_per_combination : int, optional
        Number of repeated events per selected LED combination.
    run_number : int, optional
        Run number recorded by ``pixled``.
    output_name : str, optional
        PostScript output filename. A deterministic name is used when omitted.

    Returns
    -------
    TriggerPatchMappingFiles
        Paths of all reviewable output files.

    Raises
    ------
    ValueError
        If a numeric option is not positive or the PostScript name is unsafe.
    FileNotFoundError
        If a required sim_telarray executable or generated camera file is absent.
    PermissionError
        If a required program cannot be executed.
    simtools.job_execution.job_manager.JobExecutionError
        If a program fails or does not produce a nonempty output file.
    """
    _validate_options(required_pixels, photons_per_pixel, events_per_combination, run_number)
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    prefix = f"trigger-patches-{telescope_model.name}-run{run_number}"
    postscript_file = _output_file(output_directory, output_name or f"{prefix}.ps")
    pixled, simtel, read_cta = _executables()
    camera_file = _export_camera_file(telescope_model, site_model)
    if camera_file.parent != output_directory:
        retained_camera = output_directory / f"{prefix}.camera.dat"
        shutil.copyfile(camera_file, retained_camera)
        camera_file = retained_camera
    atmospheric_transmission = _configuration_value(
        telescope_model.config_file_path, "atmospheric_transmission"
    )
    altitude = site_model.get_parameter_value_with_unit("corsika_observation_level").to_value(u.m)
    files = TriggerPatchMappingFiles(
        camera_file=camera_file,
        iact_file=output_directory / f"{prefix}.iact.gz",
        simtel_file=output_directory / f"{prefix}.simtel.gz",
        postscript_file=postscript_file,
        log_files=tuple(
            output_directory / f"{prefix}.{step}.{stream}.log"
            for step in ("pixled", "simtel", "read_cta")
            for stream in ("stdout", "stderr")
        ),
    )
    _run_pixled(
        pixled,
        files,
        required_pixels,
        include_presum_slaves,
        photons_per_pixel,
        events_per_combination,
        run_number,
        altitude,
    )
    _run_simtel(simtel, telescope_model, files, altitude, atmospheric_transmission)
    _run_read_cta(read_cta, files)
    return files


def _validate_options(*values):
    """Reject values outside pixled's positive integer range."""
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 2147483647
        for value in values
    ):
        raise ValueError("Trigger-patch mapping numeric options must be positive 32-bit integers")


def _output_file(directory, name):
    """Return a safe output file in ``directory``."""
    path = Path(name)
    if path.name != str(path) or path.suffix != ".ps":
        raise ValueError("output_name must be a PostScript filename without a directory")
    return directory / path


def _export_camera_file(telescope_model, site_model):
    """Export the telescope configuration and return its camera file."""
    telescope_model.write_sim_telarray_config_file(additional_models=site_model)
    config_file = Path(telescope_model.config_file_path).resolve()
    camera_file = config_file.parent / _configuration_value(config_file, "camera_config_file")
    if not camera_file.is_file():
        raise FileNotFoundError(f"Generated camera file not found: {camera_file}")
    return camera_file


def _configuration_value(config_file, parameter):
    """Read a filename from the exported configuration without guessing its name."""
    config_file = Path(config_file)
    for line in config_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() == parameter:
            tokens = shlex.split(value, comments=True)
            if len(tokens) == 1:
                return tokens[0]
            raise ValueError(f"Invalid {parameter} in generated config: {config_file}")
    raise ValueError(f"No {parameter} defined in generated config: {config_file}")


def _executables():
    """Return the required sim_telarray executable paths."""
    simtel_path = settings.config.sim_telarray_path
    paths = (
        simtel_path / "LightEmission" / "pixled",
        Path(settings.config.sim_telarray_exe),
        simtel_path / "bin" / "read_cta",
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required sim_telarray executable not found: {path}")
        if not os.access(path, os.X_OK):
            raise PermissionError(f"Required sim_telarray program is not executable: {path}")
    return paths


def _run_pixled(
    executable,
    files,
    required_pixels,
    include_presum_slaves,
    photons_per_pixel,
    events_per_combination,
    run_number,
    altitude,
):
    """Generate LED events from all trigger lines."""
    command = [
        str(executable),
        "--camera-file",
        str(files.camera_file),
        "--use-trg-all",
        "--require",
        str(required_pixels),
        "--photons",
        str(photons_per_pixel),
        "--events",
        str(events_per_combination),
        "--run",
        str(run_number),
        "--altitude",
        str(altitude),
        "-o",
        str(files.iact_file),
    ]
    if include_presum_slaves:
        command.append("--slaves")
    _submit(command, files.log_files[0:2], files.iact_file, files)


def _run_simtel(executable, telescope_model, files, altitude, atmospheric_transmission):
    """Process the LED event stream through sim_telarray."""
    command = [
        str(executable),
        "-c",
        str(Path(telescope_model.config_file_path).resolve()),
        f"-I{Path(telescope_model.config_file_directory).resolve()}",
        "-DNUM_TELESCOPES=1",
        "-C",
        "Bypass_Optics=2",
        "-C",
        "maximum_telescopes=1",
        "-C",
        "iobuf_maximum=1000000000",
        "-C",
        f"Altitude={altitude}",
        "-C",
        f"atmospheric_transmission={atmospheric_transmission}",
        "-C",
        "telescope_theta=0",
        "-C",
        "telescope_phi=0",
        "-r",
        str(files.simtel_file),
        str(files.iact_file),
    ]
    _submit(command, files.log_files[2:4], files.simtel_file, files)


def _run_read_cta(executable, files):
    """Write the sim_telarray camera display."""
    _submit(
        [str(executable), "-p", str(files.postscript_file), str(files.simtel_file)],
        files.log_files[4:6],
        files.postscript_file,
        files,
    )


def _submit(command, log_files, output_file, files):
    """Submit one command with explicit standard-output and standard-error logs."""
    try:
        output_file.unlink(missing_ok=True)
        job_manager.submit(
            command, out_file=log_files[0], err_file=log_files[1], env=SIM_TELARRAY_ENV
        )
        if not output_file.is_file() or output_file.stat().st_size == 0:
            raise job_manager.JobExecutionError(f"Missing or empty output: {output_file}")
    except (job_manager.JobExecutionError, OSError) as exc:
        raise job_manager.JobExecutionError(
            f"Trigger-patch step {Path(command[0]).name} failed: {exc}. "
            f"Camera: {files.camera_file}; IACT: {files.iact_file}; "
            f"simtel: {files.simtel_file}; PostScript: {files.postscript_file}; "
            f"logs: {log_files[0]}, {log_files[1]}"
        ) from exc
