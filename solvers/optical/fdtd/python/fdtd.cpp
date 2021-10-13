/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../fdtd.hpp"
using namespace plask::solvers::optical_fdtd;


static HarminvValue HarminvResults__getitem__(const HarminvResults& self, int k) {
    int n = self.n_res;
    if (k < 0) k = n - k;
    if (k < 0 || k >= n) throw plask::python::IndexError("Value outside of the possible index.");
    return HarminvValue(k, self.data);
}

static int HarminvResults__len__(const HarminvResults& self) {
    return self.n_res;
}



BOOST_PYTHON_MODULE(fdtd) {
    CLASS(FDTDSolver, "FDTD2D", "Finite-differences time-domain solver")
    METHOD(step, step, "Performs a single time-step.");
    METHOD(compute, compute, "Performs a computation for a given simulation time.");
    METHOD(output_hdf5, outputHDF5, "Outputs field data into a HDF5 file.");
    METHOD(add_flux, addFluxDFT, "Add DFT flux over a given region,");
    METHOD(do_harminv, doHarminv, "",
            (py::arg("component"), py::arg("point"), py::arg("wavelength"), py::arg("dl"), py::arg("time") = 200.,
            py::arg("nbands") = 100));
    METHOD(add_field, addFieldDFT, "");
    solver.def_readonly("courant_factor", &__Class__::courantFactor, "Courant factor");
    solver.def_readonly("time", &__Class__::elapsedTime, "Elapsed time in given in fs");
    solver.def_readonly("mesh", &__Class__::fieldMesh, "MEEP mesh equivalent");
    PROVIDER(outLightH, "Provider for the H component");
    PROVIDER(outLightE, "Provider for the E component");

    py::class_<HarminvValue, shared_ptr<HarminvValue>>("_HarminvValue", "Dokumentacja", py::no_init)
        .add_property("omega", &HarminvValue::omega, "Complex angular frequency")
        .add_property("omega_error", &HarminvValue::omega_err, "Error of the complex angular frequency")
        .add_property("amp", &HarminvValue::amp, "Complex amplitude")
        .add_property("amp_mag", &HarminvValue::amp_mag, "Magnitude of the complex amplitude")
        .add_property("wavelength", &HarminvValue::wavelength, "Wavelength of the found mode")
        .add_property("Q", &HarminvValue::q_factor, "Quality factor of the resonant mode");

    py::class_<HarminvResults, shared_ptr<HarminvResults>>("_HarminvResults", "Dokumentacja", py::no_init)
        .def("__len__", &HarminvResults__len__)
        .def("__getitem__", &HarminvResults__getitem__);

    py::class_<FieldsDFT, shared_ptr<FieldsDFT>>("_FieldsDFT", "Dokumentacja", py::no_init)
        .add_property("wavelength", &FieldsDFT::freq, "Wavelength")
        .def("field_array", &FieldsDFT::field, "Array");

    py::class_<FluxDFT, shared_ptr<FluxDFT>>("_FluxDFT", "Dokumnetacja", py::no_init)
        .def("__len__", &FluxDFT::n_wave)
        .add_property("flux", &FluxDFT::get_flux, "Stored flux values.")
        .add_property("wavelengths", &FluxDFT::get_waves, "Stored wavelengths.")
        .def("load_negative_flux", &FluxDFT::load_minus_flux_data, "Loads the negative DFT fields.");
}
