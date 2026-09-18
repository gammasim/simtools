"""Tests for the shared Matplotlib backend configuration."""

from simtools.visualization.matplotlib_backend import lazy_module, pyplot


def test_lazy_module_defers_matplotlib_import(mocker):
    matplotlib = mocker.Mock()
    module = mocker.Mock(answer=42)
    import_module = mocker.patch(
        "simtools.visualization.matplotlib_backend.import_module",
        side_effect=(matplotlib, module),
    )

    lazy_matplotlib_module = lazy_module("matplotlib.example")

    import_module.assert_not_called()
    assert lazy_matplotlib_module.answer == 42
    matplotlib.use.assert_called_once_with("Agg")
    assert import_module.call_args_list == [
        mocker.call("matplotlib"),
        mocker.call("matplotlib.example"),
    ]


def test_matplotlib_backend_is_non_interactive():
    """The shared pyplot module uses the non-interactive Agg backend."""
    assert pyplot.get_backend().lower() == "agg"
