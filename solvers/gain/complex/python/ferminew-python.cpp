/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
#include <boost/python/raw_function.hpp>

#include <boost/math/special_functions/erf.hpp>
#include <boost/regex.hpp> // tylko do wczytywania z pliku
#include <boost/lexical_cast.hpp>

#include <util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../ferminew.h"
using namespace plask::solvers::ferminew;

template <typename GeometryT>
static GainSpectrum<GeometryT> FerminewGetGainSpectrum2(FerminewGainSolver<GeometryT>* solver, double c0, double c1) {
    return solver->getGainSpectrum(Vec<2>(c0,c1));
}

template <typename GeometryT>
static py::object FerminewGainSpectrum__call__(GainSpectrum<GeometryT>& self, py::object wavelengths) {
   return UFUNC<double>([&](double x){return self.getGain(x);}, wavelengths);
}

/*template <typename GeometryT>
static py::object FerminewGain_setLevels(py::tuple args, py::dict kwargs)
{
    if (py::len(args) != 1) {
        throw TypeError("set_levels() takes exactly 1 non-keyword argument1 (%1% given)", py::len(args));
    }

    double* el = nullptr;
    double* hh = nullptr;
    double* lh = nullptr;
    try {
        py::stl_input_iterator<std::string> begin(kwargs), end;
        for (auto key = begin; key != end; ++key) {
            if (*key == "el") {
                size_t n = py::len(kwargs["el"]);
                el = new double[n+1]; el[n] = 1.;
                for (size_t i = 0; i != n; ++i) el[i] = - py::extract<double>(kwargs["el"][i]);
            } else if (*key == "hh") {
                size_t n = py::len(kwargs["hh"]);
                hh = new double[n+1]; hh[n] = 1.;
                for (size_t i = 0; i != n; ++i) hh[i] = - py::extract<double>(kwargs["hh"][i]);
            } else if (*key == "lh") {
                size_t n = py::len(kwargs["lh"]);
                lh = new double[n+1]; lh[n] = 1.;
                for (size_t i = 0; i != n; ++i) lh[i] = - py::extract<double>(kwargs["lh"][i]);
            } else if (*key != "Fc" && *key != "Fv")
                throw TypeError("set_levels() got an unexpected keyword argument '%s'", *key);
        }
        if (!el || !hh || !lh) {
            throw ValueError("All 'el', 'hh', and 'lh' levels must be set");
        }
    } catch(...) {
        delete[] el; delete[] hh; delete[] lh;
        throw;
    }

    return py::object();
}*/

/**
 * Initialization of Ferminew gain solver class to Python
 */
BOOST_PYTHON_MODULE(complex)
{
    plask_import_array();

    {CLASS(FerminewGainSolver<Geometry2DCartesian>, "Ferminew2D", "Gain solver based on Fermi Golden Rule for Cartesian 2D geometry.")
        solver.def_readwrite("strained", &__Class__::if_strain, "Consider strain in QW? (True or False)");
        RECEIVER(inTemperature, "");
        RECEIVER(inCarriersConcentration, "");
        PROVIDER(outGain, "");
        PROVIDER(outGainOverCarriersConcentration, "");
        RW_PROPERTY(roughness, getRoughness, setRoughness, "Roughness [-]");
        RW_PROPERTY(matrix_elem, getMatrixElem, setMatrixElem, "optical matrix element [m0*eV]");
        solver.def("spectrum", &__Class__::getGainSpectrum, "Get gain spectrum at given point", py::arg("point"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("spectrum", FerminewGetGainSpectrum2<Geometry2DCartesian>, "Get gain spectrum at given point", (py::arg("c0"), "c1"),
                   py::with_custodian_and_ward_postcall<0,1>());

        py::scope scope = solver;
        py::class_<GainSpectrum<Geometry2DCartesian>, plask::shared_ptr<GainSpectrum<Geometry2DCartesian>>>("Spectrum",
            "Gain spectrum class. You can call it like a function to get gains for different vavelengths.",
            py::no_init)
            .def("__call__", &FerminewGainSpectrum__call__<Geometry2DCartesian>)
        ;
    }
    {CLASS(FerminewGainSolver<Geometry2DCylindrical>, "FerminewCyl", "Gain solver based on Fermi Golden Rule for Cylindrical 2D geometry.")
        solver.def_readwrite("strained", &__Class__::if_strain, "Consider strain in QW? (True or False)");
        RECEIVER(inTemperature, "");
        RECEIVER(inCarriersConcentration, "");
        PROVIDER(outGain, "");
        PROVIDER(outGainOverCarriersConcentration, "");
        RW_PROPERTY(roughness, getRoughness, setRoughness, "Roughness [-]");
        RW_PROPERTY(matrix_elem, getMatrixElem, setMatrixElem, "optical matrix element [m0*eV]");
        solver.def("spectrum", &__Class__::getGainSpectrum, "Get gain spectrum at given point", py::arg("point"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("spectrum", FerminewGetGainSpectrum2<Geometry2DCylindrical>, "Get gain spectrum at given point", (py::arg("c0"), "c1"),
                   py::with_custodian_and_ward_postcall<0,1>());

        py::scope scope = solver;
        py::class_<GainSpectrum<Geometry2DCylindrical>, plask::shared_ptr<GainSpectrum<Geometry2DCylindrical>>>("Spectrum",
            "Gain spectrum class. You can call it like a function to get gains for different vavelengths.",
            py::no_init)
            .def("__call__", &FerminewGainSpectrum__call__<Geometry2DCylindrical>)
        ; // LUKASZ
    }

}

