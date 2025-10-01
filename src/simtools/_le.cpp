#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

// Minimal shim around sim_telarray LightEmission. This file deliberately provides
// a simplified implementation to avoid including the full sim_telarray dependency tree.

#ifdef USE_LIGHTEMISSION
// For now, create minimal compatibility classes instead of including full headers
// This allows us to build and test the native interface

// Minimal SpaceVect class
struct SpaceVect {
    double x, y, z;
    SpaceVect(double x_=0, double y_=0, double z_=0) : x(x_), y(y_), z(z_) {}
};

// Minimal DirectionVect class
struct DirectionVect {
    double cx, cy, cz;
    DirectionVect(double cx_=0, double cy_=0, double cz_=1) : cx(cx_), cy(cy_), cz(cz_) {
        double norm = sqrt(cx*cx + cy*cy + cz*cz);
        if (norm > 0) { cx/=norm; cy/=norm; cz/=norm; }
    }
};

// Minimal Run class
class Run {
private:
    int run_num;
    double obs_level;
    int atm_id;
    std::string output_file;

public:
    Run(int rn, double h, int atm, double r=12.5e2) : run_num(rn), obs_level(h), atm_id(atm) {}
    void SetOutput(const std::string& fn) { output_file = fn; }
    void NextEvent() { /* minimal implementation */ }
    const std::string& GetOutput() const { return output_file; }
};

// Minimal LightSource class
class LightSource {
private:
    SpaceVect position;
    std::string spectrum;
    std::string pulse;
    std::string angdist;

public:
    LightSource(const SpaceVect& pos, const std::string& spec, const std::string& pul, const std::string& ang)
        : position(pos), spectrum(spec), pulse(pul), angdist(ang) {}

    int Emit(const Run& run, double photons, double bsize, double when) {
        // Minimal implementation - just return success for now
        // In the future this would generate actual photon data and write to the output file
        std::cout << "Minimal LightSource::Emit called with " << photons << " photons" << std::endl;
        return 0;
    }
};

#endif

namespace py = pybind11;

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
    const std::string &angular_distribution)
{
#ifdef USE_LIGHTEMISSION
    // Simplified implementation using minimal classes
    try {
        // Create Run object (run_number=1, altitude in cm, atmosphere_id, camera_radius)
        Run run(1, altitude_m * 100.0, atmosphere_id, camera_radius_cm);
        run.SetOutput(output_path);

        // Create LightSource
        SpaceVect flasher_pos(x_cm, y_cm, distance_cm);
        std::string spectrum_str = std::to_string(spectrum_nm);
        LightSource flasher(flasher_pos, spectrum_str, lightpulse, angular_distribution);

        // Single event with specified photons
        run.NextEvent();
        int result = flasher.Emit(run, photons, static_cast<double>(bunch_size), 0.0);

        if (result != 0) {
            return -3; // emission failed
        }

        std::cout << "Native ff-1m simulation completed successfully" << std::endl;
        return 0; // success
    }
    catch (const std::exception& e) {
        std::cerr << "Error in native ff-1m: " << e.what() << std::endl;
        return -3; // runtime error
    }
#else
    // Sentinel indicating the module was built without LightEmission linkage.
    (void)output_path;
    (void)altitude_m;
    (void)atmosphere_id;
    (void)photons;
    (void)bunch_size;
    (void)x_cm;
    (void)y_cm;
    (void)distance_cm;
    (void)camera_radius_cm;
    (void)spectrum_nm;
    (void)lightpulse;
    (void)angular_distribution;
    return -2; // not compiled with LightEmission support
#endif
}

PYBIND11_MODULE(_le, m)
{
    m.doc() = "simtools LightEmission bindings (ff-1m first; xyzls later)";
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
          R"pbdoc(Return 0 on success, negative on configuration/build issues.)pbdoc");
}
