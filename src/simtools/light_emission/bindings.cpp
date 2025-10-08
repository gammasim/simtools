#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <cstdio>
#ifndef _WIN32
#include <unistd.h>
#endif

// Include the real LightEmission interfaces
#include "IactLightEmission.hh"

namespace py = pybind11;

// Path-only ff_1m: accepts explicit atmosphere file path.
static int ff_1m(
    const std::string &output_path,
    double altitude_m,
    const std::string &atmosphere_file,
    double photons,
    int bunch_size,
    double x_cm,
    double y_cm,
    double distance_cm,
    double camera_radius_cm,
    int spectrum_nm,
    const std::string &lightpulse,
    const std::string &angular_distribution,
    int events
){
    try {
        // Clean up any previous output
        std::remove(output_path.c_str());

        if (atmosphere_file.empty()) {
            return -2; // invalid input
        }
        // Quick existence check
        std::ifstream test_in(atmosphere_file);
        if (!test_in.good()) {
            return -4; // atmosphere file missing
        }

        // Use Run constructor that accepts atmosphere file path directly
        Run run(1, altitude_m * 100.0, atmosphere_file.c_str(), camera_radius_cm);
        run.SetOutput(output_path);
        run.AddHistory("pybind11 ff_1m path");

        SpaceVect flasher_pos(x_cm, y_cm, distance_cm);
        std::string spectrum = std::to_string(spectrum_nm);
        LightSource flasher(flasher_pos, spectrum, lightpulse, angular_distribution);

        for (int iev = 0; iev < std::max(1, events); ++iev) {
            run.NextEvent();
            int rc = flasher.Emit(run, photons, static_cast<double>(bunch_size), 0.0);
            if (rc != 0) return -3;
        }
        return 0;
    } catch (...) {
        return -3; // generic failure
    }
}

PYBIND11_MODULE(_le, m) {
    m.doc() = "Bindings to sim_telarray LightEmission (ff-1m path-only bridge)";
    m.def("ff_1m", &ff_1m,
          py::arg("output_path"),
          py::arg("altitude_m"),
          py::arg("atmosphere_file"),
          py::arg("photons"),
          py::arg("bunch_size"),
          py::arg("x_cm"),
          py::arg("y_cm"),
          py::arg("distance_cm"),
          py::arg("camera_radius_cm"),
          py::arg("spectrum_nm"),
          py::arg("lightpulse"),
          py::arg("angular_distribution"),
          py::arg("events"),
          R"pbdoc(Returns 0 on success. Writes IACT file at output_path using supplied atmosphere file.)pbdoc");
}
