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
using namespace plask::solvers::FermiNew;

template <typename GeometryT>
static GainSpectrum<GeometryT> FermiNewGetGainSpectrum2(FermiNewGainSolver<GeometryT>* solver, double c0, double c1) {
    return solver->getGainSpectrum(Vec<2>(c0,c1));
}

template <typename GeometryT>
static py::object FermiNewGainSpectrum__call__(GainSpectrum<GeometryT>& self, py::object wavelengths) {
   return UFUNC<double>([&](double x){return self.getGain(x);}, wavelengths);
}

template <typename GeometryT>
static LuminescenceSpectrum<GeometryT> FermiNewGetLuminescenceSpectrum2(FermiNewGainSolver<GeometryT>* solver, double c0, double c1) {
    return solver->getLuminescenceSpectrum(Vec<2>(c0,c1));
}

template <typename GeometryT>
static py::object FermiNewLuminescenceSpectrum__call__(LuminescenceSpectrum<GeometryT>& self, py::object wavelengths) {
   return UFUNC<double>([&](double x){return self.getLuminescence(x);}, wavelengths);
}

/*template <typename GeometryT>
static py::object FermiNewGain_setLevels(py::tuple args, py::dict kwargs)
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
 * Initialization of FermiNew gain solver class to Python
 */
BOOST_PYTHON_MODULE(complex)
{
    plask_import_array();

    {CLASS(FermiNewGainSolver<Geometry2DCartesian>, "FermiNew2D", "Gain solver based on Fermi Golden Rule for Cartesian 2D geometry.")
        solver.add_property("strained", &__Class__::getStrains, &__Class__::setStrains,
                            "Consider strain in QW and barriers? (True or False).");
        solver.add_property("fixed_qw_widths", &__Class__::getFixedQwWidths, &__Class__::setFixedQwWidths,
                            "Fix widths of the QWs? (True or False).");
        solver.add_property("fast_levels", &__Class__::getBuildStructOnce, &__Class__::setBuildStructOnce,
                            "Compute levels only once and simply shift for different temperatures?\n"
                            "Setting this to True stongly increases computation speed, but makes the results\n"
                            "less accurate for high gains. (True or False).");
        RECEIVER(inTemperature, "");
        RECEIVER(inCarriersConcentration, "");
        PROVIDER(outGain, "");
        PROVIDER(outLuminescence, "");
        PROVIDER(outGainOverCarriersConcentration, "");
        RW_PROPERTY(roughness, getRoughness, setRoughness, "Roughness [-]");
        RW_PROPERTY(lifetime, getLifeTime, setLifeTime, "Lifetime [ps]");
        RW_PROPERTY(matrix_elem, getMatrixElem, setMatrixElem, "Optical matrix element [m0*eV]");
        RW_PROPERTY(matrix_elem_scaling, getMatrixElemScFact, setMatrixElemScFact, "Scale factor for optical matrix element [-]");
        RW_PROPERTY(cond_shift, getCondQWShift, setCondQWShift, "Additional conduction band shift for QW [eV]");
        RW_PROPERTY(vale_shift, getValeQWShift, setValeQWShift, "Additional valence band shift for QW [eV]");
        RW_PROPERTY(Tref, getTref, setTref,
                    "Reference temperature. If *fast_levels* is True, this is the temperature used\n"
                    "for initial computation of the energy levels [K].");
        solver.def("spectrum", &__Class__::getGainSpectrum, "Get gain spectrum at given point", py::arg("point"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("spectrum", FermiNewGetGainSpectrum2<Geometry2DCartesian>, "Get gain spectrum at given point", (py::arg("c0"), "c1"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("luminescence_spectrum", &__Class__::getLuminescenceSpectrum, "Get luminescence spectrum at given point", py::arg("point"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("luminescence_spectrum", FermiNewGetLuminescenceSpectrum2<Geometry2DCartesian>, "Get luminescence spectrum at given point", (py::arg("c0"), "c1"),
                   py::with_custodian_and_ward_postcall<0,1>());

        py::scope scope = solver;
        py::class_<GainSpectrum<Geometry2DCartesian>, plask::shared_ptr<GainSpectrum<Geometry2DCartesian>>>("Spectrum",
            "Gain spectrum class. You can call it like a function to get gains for different vavelengths.",
            py::no_init)
            .def("__call__", &FermiNewGainSpectrum__call__<Geometry2DCartesian>)
        ;
        py::class_<LuminescenceSpectrum<Geometry2DCartesian>, plask::shared_ptr<LuminescenceSpectrum<Geometry2DCartesian>>>("LuminescenceSpectrum",
            "Luminescence spectrum class. You can call it like a function to get luminescences for different vavelengths.",
            py::no_init)
            .def("__call__", &FermiNewLuminescenceSpectrum__call__<Geometry2DCartesian>)
        ;
    }
    {CLASS(FermiNewGainSolver<Geometry2DCylindrical>, "FermiNewCyl", "Gain solver based on Fermi Golden Rule for Cylindrical 2D geometry.")
        solver.add_property("strained", &__Class__::getStrains, &__Class__::setStrains,
                            "Consider strain in QW and barriers? (True or False).");
        solver.add_property("fixed_qw_widths", &__Class__::getFixedQwWidths, &__Class__::setFixedQwWidths,
                            "Fix widths of the QWs? (True or False).");
        solver.add_property("fast_levels", &__Class__::getBuildStructOnce, &__Class__::setBuildStructOnce,
                            "Compute levels only once and simply shift for different temperatures?\n"
                            "Setting this to True stongly increases computation speed, but makes the results\n"
                            "less accurate for high gains. (True or False).");
        RECEIVER(inTemperature, "");
        RECEIVER(inCarriersConcentration, "");
        PROVIDER(outGain, "");
        PROVIDER(outLuminescence, "");
        PROVIDER(outGainOverCarriersConcentration, "");
        RW_PROPERTY(roughness, getRoughness, setRoughness, "Roughness [-]");
        RW_PROPERTY(lifetime, getLifeTime, setLifeTime, "Lifetime [ps]");
        RW_PROPERTY(matrix_elem, getMatrixElem, setMatrixElem, "optical matrix element [m0*eV]");
        RW_PROPERTY(matrix_elem_scaling, getMatrixElemScFact, setMatrixElemScFact, "Scale factor for optical matrix element [-]");
        RW_PROPERTY(cond_shift, getCondQWShift, setCondQWShift, "Additional conduction band shift for QW [eV]");
        RW_PROPERTY(vale_shift, getValeQWShift, setValeQWShift, "Additional valence band shift for QW [eV]");
        RW_PROPERTY(Tref, getTref, setTref,
                    "Reference temperature. If *fast_levels* is True, this is the temperature used\n"
                    "for initial computation of the energy levels [K].");
        solver.def("spectrum", &__Class__::getGainSpectrum, "Get gain spectrum at given point", py::arg("point"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("spectrum", FermiNewGetGainSpectrum2<Geometry2DCylindrical>, "Get gain spectrum at given point", (py::arg("c0"), "c1"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("luminescencespectrum", &__Class__::getLuminescenceSpectrum, "Get luminescence spectrum at given point", py::arg("point"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("luminescencespectrum", FermiNewGetLuminescenceSpectrum2<Geometry2DCylindrical>, "Get luminescence spectrum at given point", (py::arg("c0"), "c1"),
                   py::with_custodian_and_ward_postcall<0,1>());

        py::scope scope = solver;
        py::class_<GainSpectrum<Geometry2DCylindrical>, plask::shared_ptr<GainSpectrum<Geometry2DCylindrical>>>("Spectrum",
            "Gain spectrum class. You can call it like a function to get gains for different vavelengths.",
            py::no_init)
            .def("__call__", &FermiNewGainSpectrum__call__<Geometry2DCylindrical>)
        ;
        py::class_<LuminescenceSpectrum<Geometry2DCylindrical>, plask::shared_ptr<LuminescenceSpectrum<Geometry2DCylindrical>>>("LuminescenceSpectrum",
            "Luminescence spectrum class. You can call it like a function to get luminescences for different vavelengths.",
            py::no_init)
            .def("__call__", &FermiNewLuminescenceSpectrum__call__<Geometry2DCylindrical>)
        ;
    }

}

