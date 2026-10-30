"""Build and validate production definitions from resolved application options."""

from simtools.constants import SCHEMA_PATH
from simtools.data_model import schema
from simtools.io.ascii_handler import collect_data_from_file

_DEFINITION_SCHEMA = SCHEMA_PATH / "production_definition.schema.yml"
_SCHEMA_NAME = "production_definition"
_SCHEMA_VERSION = "1.0.0"
_SIMULATION_STAGES = {
    "corsika": ["corsika"],
    "sim_telarray": ["sim_telarray"],
    "corsika_sim_telarray": ["corsika", "sim_telarray"],
}


def production_definition_from_arguments(args_dict):
    """Create validated production data from resolved grid application arguments."""
    simulation_software = args_dict["simulation_software"]
    definition = {
        "schema_name": _SCHEMA_NAME,
        "schema_version": _SCHEMA_VERSION,
        "axes": args_dict.get("axis") or [],
        "limits_lookup": {
            "file": (
                str(args_dict["corsika_limits"])
                if args_dict.get("corsika_limits") is not None
                else None
            ),
            "interpolation": "point",
        },
        "run_number_allocation": {
            "offset": int(args_dict.get("run_number_offset", 0)),
            "runs": args_dict.get("number_of_runs"),
        },
        "job_sizes": {
            "showers_per_run": args_dict.get("showers_per_run"),
            "total_showers": args_dict.get("total_showers"),
            "max_showers_per_run": args_dict.get("max_showers_per_run"),
        },
        "simulation_stages": _SIMULATION_STAGES[simulation_software],
    }
    schema_definition = collect_data_from_file(_DEFINITION_SCHEMA)
    return schema.validate_dict_using_schema(
        definition,
        json_schema=schema_definition,
        offline=True,
    )
