"""Tests for the simulate_pedestals application."""

import simtools.applications.simulate_pedestals as app


def test_parser_excludes_redundant_telescope_argument():
    actions = {action.dest for action in app.APPLICATION.build_parser()._actions}

    assert "array_layout_name" in actions
    assert "telescope" not in actions
