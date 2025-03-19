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

    def _get_report_parameters(self):
        """Generate parameters for report generation."""
        all_model_versions = self.db.get_model_versions()
        all_sites = {"North", "South"}

        model_versions = (
            all_model_versions
            if self.args.get("all_model_versions")
            else [self.args["model_version"]]
        )
        selected_sites = all_sites if self.args.get("all_sites") else {self.args["site"]}

        def get_telescopes(model_version):
            if not self.args.get("all_telescopes"):
                return [self.args["telescope"]]
            telescopes = self.db.get_array_elements(model_version)
            all_telescopes = self._add_design_models_to_telescopes(model_version, telescopes)
            return self._filter_telescopes_by_site(all_telescopes, selected_sites)

        def get_valid_sites(telescope):
            sites = names.get_site_from_array_element_name(telescope)
            sites = sites if isinstance(sites, list) else [sites]
            return [site for site in sites if site in selected_sites]

        def generate_combinations():
            for version in model_versions:
                telescopes = get_telescopes(version)
                for telescope in telescopes:
                    for site in get_valid_sites(telescope):
                        yield version, telescope, site

        return generate_combinations()

    def _generate_single_array_element_report(self, model_version, telescope, site):
        """Generate a single report with given parameters."""
        self.args.update(
            {
                "telescope": telescope,
                "site": site,
                "model_version": model_version,
            }
        )

        output_path = Path(self.output_path) / str(model_version)
        ReadParameters(self.db_config, self.args, output_path).produce_array_element_report()

        logger.info(
            f"Markdown report generated for {site} "
            f"Telescope {telescope} (v{model_version}): {output_path}"
        )

    def auto_generate_array_element_reports(self):
        """Generate all reports based on which --all_* flag is passed."""
        for params in self._get_report_parameters():
            self._generate_single_array_element_report(*params)
