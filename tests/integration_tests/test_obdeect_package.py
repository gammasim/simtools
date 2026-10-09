"""Integration checks for the published-style obdeect wheel."""

import csv
import json
import math
import subprocess
from pathlib import Path

import pytest


def test_published_obdeect_wheel_runs_reference_trace(tmp_path):
    obdeect = pytest.importorskip("obdeect")
    from obdeect.result_contract import read_arrivals

    result_file = tmp_path / "arrivals.csv"
    subprocess.run(
        [
            str(obdeect.executable_path()),
            "--telescope",
            "MST",
            "--photons",
            "8",
            "--output",
            str(result_file),
        ],
        check=True,
    )

    arrivals = read_arrivals(result_file)
    assert len(arrivals) == 8
    assert {arrival.source_kind for arrival in arrivals} == {"star"}


def test_obdeect_imaging_list_uses_common_analysis(mocker, tmp_path):
    pytest.importorskip("obdeect.imaging_list")
    from simtools.ray_tracing.psf_analysis import PSFImage
    from simtools.simtel.simulator_obdeect import SimulatorObdeect

    root = Path(tmp_path)
    model = root / "model.json"
    model.write_text(
        json.dumps(
            {
                "trace_model": {
                    "kind": "segmented",
                    "primary_facets": [{"centre_m": [0, 0, 0], "diameter_m": 2}],
                },
                "focal_length_m": 10,
                "camera": {"rotation_deg": 0},
                "optical_model_sha256": "a" * 64,
            }
        )
    )
    output = root / "obdeect.lis"
    mocker.patch(
        "simtools.simtel.simulator_obdeect.settings.config", mocker.Mock(obdeect_exe="obdeect")
    )
    simulator = SimulatorObdeect(
        telescope_model=None,
        config_data={"obdeect_optical_model_file": model, "number_of_photons": 5},
        output_file=output,
    )

    def run_native(command, **_kwargs):
        native = Path(command[command.index("--output") + 1])
        fields = (
            "contract_version,photon_id,source_kind,wavelength_nm,emission_time_ns,source_weight,"
            "throughput,status,point_count,path_length_m,x0_m,y0_m,z0_m,x1_m,y1_m,z1_m,"
            "x2_m,y2_m,z2_m,final_dx,final_dy,final_dz,incidence_primary_deg,incidence_focal_deg"
        ).split(",")
        with native.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, (x, throughput) in enumerate(
                zip([-0.03, -0.01, 0.01, 0.05, 0], [0, 0.1, 0.5, 1, 0], strict=True)
            ):
                writer.writerow(
                    dict(
                        zip(
                            fields,
                            [
                                "obdeect-arrival-v1",
                                index,
                                "star",
                                400,
                                0,
                                1,
                                throughput,
                                "detected" if index < 4 else "missed_primary",
                                3 if index < 4 else 1,
                                10,
                                0,
                                0,
                                2,
                                0,
                                0,
                                0,
                                x,
                                0,
                                10,
                                0,
                                0,
                                1,
                                20,
                                0,
                            ],
                            strict=True,
                        )
                    )
                )

    mocker.patch("simtools.simtel.simulator_obdeect.subprocess.run", side_effect=run_native)
    simulator.run()
    assert not output.with_suffix(".obdeect.csv").exists()
    reference = root / "simtel.lis"
    area = math.pi * 1.2**2
    reference.write_text(
        f"# Telescope 0 with 5 photons from 1 star(s) falling on an area of {area} m^2\n"
        "# Camera rotation angle = 0 deg\n"
        "0 -1 -3 0\n0 -1 -1 0\n0 -1 1 0\n0 -1 5 0\n"
    )
    actual_image = PSFImage(focal_length=1000)
    reference_image = PSFImage(focal_length=1000)
    for image, path in [(actual_image, output), (reference_image, reference)]:
        image.process_photon_list(path, use_rx=False)
    assert actual_image.centroid_x == pytest.approx(0.5)
    assert actual_image.get_effective_area() == pytest.approx(area * 4 / 5)
    assert actual_image.centroid_x_error == pytest.approx(reference_image.centroid_x_error)
    assert actual_image.get_psf() == pytest.approx(reference_image.get_psf())
    assert (
        actual_image.get_cumulative_data([0, 1, 2, 4, 5]).tolist()
        == reference_image.get_cumulative_data([0, 1, 2, 4, 5]).tolist()
    )
