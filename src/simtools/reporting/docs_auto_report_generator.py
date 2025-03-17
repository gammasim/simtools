#!/usr/bin/python3

r"""Class to automate generation of all reports according to command line inputs."""

import logging
from pathlib import Path

from simtools.db import db_handler
from simtools.reporting.docs_read_parameters import ReadParameters
from simtools.utils import names

logger = logging.getLogger()


# temporary setting during development
# pylint: disable=too-few-public-methods
class ReportGenerator:
    """Automate report generation."""

    def __init__(self, db_config, args, output_path):
        """Initialise class."""
        self._logger = logging.getLogger(__name__)
        self.db = db_handler.DatabaseHandler(mongo_db_config=db_config)
        self.db_config = db_config
        self.args = args
        self.output_path = output_path

    def _add_design_models_to_telescopes(self, model_version, telescopes):
        """Add design models to the list of telescopes for a given model version."""
        updated_telescopes = telescopes.copy()  # Copy original list to avoid modifying it directly
        for telescope in telescopes:
            design_model = self.db.get_design_model(model_version, telescope)  # Get design model
            if design_model and design_model not in updated_telescopes:
                updated_telescopes.append(design_model)  # Add design model if not already present
        return updated_telescopes

    def _filter_telescopes_by_site(self, telescopes, selected_sites):
        """Filter telescopes by selected sites."""
        filtered_telescopes = []
        for telescope in telescopes:
            sites = names.get_site_from_array_element_name(telescope)
            sites = sites if isinstance(sites, list) else [sites]
            if any(site in selected_sites for site in sites):
                filtered_telescopes.append(telescope)
        return filtered_telescopes

    def auto_generate_array_element_reports(self):
        """
        Generate all reports based on which --all_* flag is passed.

        Expands 'all' options to iterate over multiple values.
        """
        all_model_versions = self.db.get_model_versions()
        all_sites = {"North", "South"}

        model_versions = (
            all_model_versions
            if self.args.get("all_model_versions")
            else [self.args["model_version"]]
        )
        selected_sites = all_sites if self.args.get("all_sites") else {self.args["site"]}

        # Loop through each model version
        for model_version in model_versions:
            telescopes = self.db.get_array_elements(model_version)

            # Add design models to the list of telescopes
            all_telescopes = self._add_design_models_to_telescopes(model_version, telescopes)
            filtered_telescopes = self._filter_telescopes_by_site(all_telescopes, selected_sites)

            for telescope in filtered_telescopes:
                sites = names.get_site_from_array_element_name(telescope)
                sites = sites if isinstance(sites, list) else [sites]

                site = next((s for s in sites if s in selected_sites), None)

                if site:
                    self.args["telescope"], self.args["site"], self.args["model_version"] = (
                        telescope,
                        site,
                        model_version,
                    )

                    output_path = Path(self.output_path / f"{self.args['model_version']}")

                    ReadParameters(
                        self.db_config, self.args, output_path
                    ).produce_array_element_report()

                    logger.info(
                        f"Markdown report generated for {site} "
                        f"Telescope {telescope} (v{model_version}): {output_path}"
                    )
