#!/usr/bin/python3

r"""Class to automate generation of all reports according to command line inputs."""

import logging
from collections.abc import Generator
from pathlib import Path

from simtools.db import db_handler
from simtools.reporting.docs_read_parameters import ReadParameters
from simtools.utils import names

logger = logging.getLogger()


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

    def _generate_array_element_report_combinations(self):
        """Generate combinations of model versions, telescopes, and sites for array element reports.

        This generator function yields tuples of (model_version, telescope, site) for array element
        report generation based on the input arguments and flags.

        The function filters telescopes by site and ensures that design models are included
        when necessary.

        Yields
        ------
            tuple[str, str, str]: A tuple containing (model_version, telescope, site) for each
                valid combination based on the input arguments.
        """
        all_sites = names.site_names()

        model_versions = (
            self.db.get_model_versions()
            if self.args.get("all_model_versions")
            else [self.args["model_version"]]
        )
        selected_sites = all_sites if self.args.get("all_sites") else {self.args["site"]}

        def get_telescopes(model_version):
            """Get list of telescopes depending on input arguments."""
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
        if self.args.get("observatory"):
            self.auto_generate_observatory_reports()
            return
        for params in self._generate_array_element_report_combinations():
            self._generate_single_array_element_report(*params)

    def _get_telescopes_from_layout(self, site: str) -> set[str]:
        """Get unique telescopes for a given site across all versions."""
        if not self.args.get("all_telescopes"):
            return {self.args["telescope"]}

        # Get layouts for all versions for this site
        layouts = self.db.get_model_parameters_for_all_model_versions(
            site=site, array_element_name=f"OBS-{site}", collection="sites"
        )

        # Collect all telescopes from hyper array layouts
        all_telescopes = set()
        for key, version_data in layouts.items():
            layout = version_data.get("array_layouts", {}).get("value", [])
            hyper_array = next(
                (layout["elements"] for layout in layout if layout["name"] == "hyper_array"), []
            )
            all_telescopes.update(hyper_array)
            all_telescopes.update(self._add_design_models_to_telescopes(key, list(all_telescopes)))

        return all_telescopes

    def _get_valid_sites_for_telescope(self, telescope: str) -> list[str]:
        """Get valid sites for a given telescope."""
        sites = names.get_site_from_array_element_name(telescope)
        return [sites] if not isinstance(sites, list) else sites

    def _generate_parameter_report_combinations(self) -> Generator[tuple[str, str], None, None]:
        """Generate combinations of telescopes and sites for report generation."""
        if not self.args.get("all_telescopes"):
            # For a specific telescope, get its valid sites regardless of --all_sites
            telescope = self.args["telescope"]
            for site in self._get_valid_sites_for_telescope(telescope):
                yield telescope, site
        else:
            # For --all_telescopes, use selected sites to get telescopes
            all_sites = names.site_names()
            selected_sites = all_sites if self.args.get("all_sites") else {self.args["site"]}

            for site in selected_sites:
                telescopes = self._get_telescopes_from_layout(site)
                yield from ((telescope, site) for telescope in telescopes)

    def auto_generate_parameter_reports(self):
        """Generate parameter reports for all telescope-site combinations."""
        for telescope, site in self._generate_parameter_report_combinations():
            self.args.update(
                {
                    "telescope": telescope,
                    "site": site,
                }
            )

            ReadParameters(
                self.db_config, self.args, self.output_path
            ).produce_model_parameter_reports()

            logger.info(
                f"Markdown report generated for {site} Telescope {telescope}: {self.output_path}"
            )

        # Also generate calibration device parameter comparison reports when using --all_telescopes
        if self.args.get("all_telescopes"):
            self._generate_calibration_device_parameter_reports()

    def _generate_observatory_report_combinations(self) -> Generator[tuple[str, str], None, None]:
        """Generate combinations of sites and model versions for observatory reports.

        Yields
        ------
            tuple[str, str]: A tuple containing (site, model_version) for each valid combination
                based on the input arguments.
        """
        all_sites = names.site_names()
        selected_sites = all_sites if self.args.get("all_sites") else {self.args["site"]}

        model_versions = (
            self.db.get_model_versions()
            if self.args.get("all_model_versions")
            else [self.args["model_version"]]
        )

        for site in selected_sites:
            for version in model_versions:
                yield site, version

    def _generate_single_observatory_report(self, site: str, model_version: str):
        """Generate a single observatory report with given parameters."""
        self.args.update(
            {
                "site": site,
                "model_version": model_version,
                "observatory": True,
            }
        )

        output_path = Path(self.output_path) / str(model_version)
        ReadParameters(self.db_config, self.args, output_path).produce_observatory_report()

        logger.info(f"Observatory report generated for {site} (v{model_version}): {output_path}")

    def auto_generate_observatory_reports(self):
        """Generate all observatory reports based on which --all_* flags are passed."""
        for params in self._generate_observatory_report_combinations():
            self._generate_single_observatory_report(*params)

    def auto_generate_simulation_configuration_reports(self):
        """Generate simulation configuration reports for one or all model versions.

        If --all_model_versions is set, produce reports for every model version in
        the DB; otherwise produce reports only for the configured model_version.
        """
        model_versions = (
            self.db.get_model_versions()
            if self.args.get("all_model_versions")
            else [self.args["model_version"]]
        )

        for version in model_versions:
            # update args and create a per-version output directory
            self.args.update({"model_version": version})
            output_path = Path(self.output_path) / str(version)

            ReadParameters(
                self.db_config, self.args, output_path
            ).produce_simulation_configuration_report()

            logger.info(f"Configuration reports for (v{version}) produced: {output_path}")

    def auto_generate_calibration_reports(self):
        """Generate calibration reports for one or all model versions.

        Mirrors the pattern used by other auto_generate_* methods: if
        --all_model_versions is set, produce reports for every model version in
        the DB; otherwise produce reports only for the configured model_version.
        """
        model_versions = (
            self.db.get_model_versions()
            if self.args.get("all_model_versions")
            else [self.args["model_version"]]
        )

        for version in model_versions:
            # update args and create a per-version output directory
            self.args.update({"model_version": version})
            output_path = Path(self.output_path) / str(version)

            try:
                ReadParameters(self.db_config, self.args, output_path).produce_calibration_reports()
                logger.info(f"Calibration reports for (v{version}) produced: {output_path}")
            except ValueError as err:
                # Some model versions do not have calibration_devices in the DB;
                msg = str(err)
                if "calibration_devices" in msg and "zero results" in msg:
                    logger.info(
                        f"Skipping model version {version}: no calibration devices defined ({msg})"
                    )
                    continue
                # re-raise unexpected ValueErrors
                raise

    def _generate_calibration_device_parameter_reports(self):
        """Generate parameter comparison reports for calibration devices for all model versions."""
        # Get all model versions since no specific version is provided when using --all_telescopes
        model_versions = self.db.get_model_versions()

        for version in model_versions:
            self._process_calibration_devices_for_version(version)

    def _process_calibration_devices_for_version(self, version):
        """Process calibration devices for a specific model version."""
        try:
            # Get all calibration devices for this version
            calibration_array_elements = self.db.get_array_elements(
                version, collection="calibration_devices"
            )
            array_elements = calibration_array_elements.copy()

            # Add design models
            for element in calibration_array_elements:
                design_model = self.db.get_design_model(version, element, "calibration_devices")
                if design_model and design_model not in array_elements:
                    array_elements.append(design_model)

            if array_elements:
                # Create a copy of args with the current version for this iteration
                version_args = self.args.copy()
                version_args["model_version"] = version

                # Generate parameter comparison reports for calibration devices
                ReadParameters(
                    self.db_config, version_args, self.output_path
                ).generate_model_parameter_reports_for_devices(array_elements)

                logger.info(
                    "Calibration device parameter reports generated for"
                    f" v{version}: {self.output_path}"
                )

        except ValueError as err:
            # Some model versions may not have calibration_devices
            msg = str(err)
            if "calibration_devices" in msg and "zero results" in msg:
                logger.info(
                    f"Skipping model version {version}: no calibration devices defined ({msg})"
                )
                return
            # re-raise unexpected ValueErrors
            raise
