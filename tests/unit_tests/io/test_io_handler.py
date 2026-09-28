from simtools.io.io_handler import IOHandler


def test_get_output_file_creates_nested_parent(tmp_path):
    io_handler = IOHandler()
    io_handler.set_paths(tmp_path)

    output_file = io_handler.get_output_file("pm_photoelectron_spectrum/spectrum.ecsv")

    assert output_file == (tmp_path / "pm_photoelectron_spectrum/spectrum.ecsv").absolute()
    assert output_file.parent.is_dir()
