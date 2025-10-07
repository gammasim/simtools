#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Include the real LightEmission interfaces
#include "IactLightEmission.hh"

namespace py = pybind11;

// Minimal bridge that mirrors ff-1m invocation and writes an IACT file
static int ff_1m(
    const std::string &output_path,
    double altitude_m,
    int atmosphere_id,
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
) {
    try {
        // Clean up any previous output (ff-1m does unlink before writing)
        std::remove(output_path.c_str());

        // Run setup
        Run run(1, altitude_m * 100.0, atmosphere_id, camera_radius_cm);
        run.SetOutput(output_path);

        // Keep a little bit of history to ease debugging
        run.AddHistory("pybind11 ff_1m");

        // Build the flasher light source
        SpaceVect flasher_pos(x_cm, y_cm, distance_cm);
        std::string spectrum = std::to_string(spectrum_nm);
        LightSource flasher(flasher_pos, spectrum, lightpulse, angular_distribution);

        // Loop events and emit photons (bunch_size is advisory to LightSource)
        for (int iev = 0; iev < std::max(1, events); ++iev) {
            run.NextEvent();
            int rc = flasher.Emit(run, photons, static_cast<double>(bunch_size), 0.0);
            if (rc != 0) {
                return -3; // emission failed
            }
        }
        // Run destructor will flush final buffers
        return 0;
    } catch (const std::exception &e) {
        // Map any exception to negative error code
        return -3;
    }
}

PYBIND11_MODULE(_le, m) {
    m.doc() = "Bindings to sim_telarray LightEmission (ff-1m minimal bridge)";
    m.def("ff_1m", &ff_1m,
          py::arg("output_path"),
          py::arg("altitude_m"),
          py::arg("atmosphere_id"),
          py::arg("photons"),
          py::arg("bunch_size"),
          py::arg("x_cm"),
          py::arg("y_cm"),
          py::arg("distance_cm"),
          py::arg("camera_radius_cm"),
          py::arg("spectrum_nm"),
          py::arg("lightpulse"),
          py::arg("angular_distribution"),
          py::arg("events") ,
          R"pbdoc(Returns 0 on success. Writes IACT file at output_path.)pbdoc");
}
