from astropy.table import Table

from simtools.reporting.model_figures import (
    plot_nsb_altitude_correction,
)


def test_plot_nsb_altitude_correction_creates_heatmap(tmp_path):
    input_file = tmp_path / "altitude-correction.ecsv"
    Table(
        rows=[(300.0, 2.5, 0.2), (400.0, 2.5, 0.4), (300.0, 5.0, 0.6), (400.0, 5.0, 0.8)],
        names=("wavelength", "altitude", "extinction"),
    ).write(input_file, format="ascii.ecsv")

    plot_names = plot_nsb_altitude_correction(input_file, tmp_path)

    assert len(plot_names) == 1
    assert plot_names[0].startswith("altitude-correction_")
    assert (tmp_path / f"{plot_names[0]}.png").is_file()
