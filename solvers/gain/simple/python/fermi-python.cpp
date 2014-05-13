/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
#include <boost/python/raw_function.hpp>
#include <util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../fermi.h"
using namespace plask::solvers::fermi;

template <typename GeometryT>
static GainSpectrum<GeometryT> FermiGetGainSpectrum2(FermiGainSolver<GeometryT>* solver, double c0, double c1) {
    return solver->getGainSpectrum(Vec<2>(c0,c1));
}

template <typename GeometryT>
static py::object FermiGainSpectrum__call__(GainSpectrum<GeometryT>& self, py::object wavelengths) {
   return PARALLEL_UFUNC<double>([&](double x){return self.getGain(x);}, wavelengths);
}

template <typename GeometryT>
static py::object FermiGain_determineLevels(FermiGainSolver<GeometryT>& self, double T, double n)
{
    py::list result;
    for (const auto& data: self.determineLevels(T, n)) {
        py::dict info;
        info["el"] = std::get<0>(data);
        info["hh"] = std::get<1>(data);
        info["lh"] = std::get<2>(data);
        info["Fc"] = std::get<3>(data);
        info["Fv"] = std::get<4>(data);
        result.append(info);
    }
    return result;
}

template <typename GeometryT>
static py::object FermiGain_setLevels(py::tuple args, py::dict kwargs)
{
    if (py::len(args) != 1) {
        throw TypeError("set_levels() takes exactly 1 non-keyword argument1 (%1% given)", py::len(args));
    }

    FermiGainSolver<GeometryT>& self = py::extract<FermiGainSolver<GeometryT>&>(args[0]);

    if (len(kwargs) == 0) {
        if (self.extern_levels) {
            delete[] self.extern_levels->el;
            delete[] self.extern_levels->hh;
            delete[] self.extern_levels->lh;
        }
        self.extern_levels.reset();
        return py::object();
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

    if (self.extern_levels) {
        delete[] self.extern_levels->el;
        delete[] self.extern_levels->hh;
        delete[] self.extern_levels->lh;
    }
    self.extern_levels.reset(QW::ExternalLevels(el, hh, lh));

    return py::object();
}

/**
 * Initialization of Fermi gain solver class to Python
 */
BOOST_PYTHON_MODULE(simple)
{
    plask_import_array();

    {CLASS(FermiGainSolver<Geometry2DCartesian>, "Fermi2D", "Gain solver based on Fermi Golden Rule for Cartesian 2D geometry.")
        solver.def("determine_levels", &FermiGain_determineLevels<Geometry2DCartesian>,
                   "Determine quasi-Fermi levels and carriers levels inside QW", (py::arg("T"), "n"));
        solver.def("set_levels", py::raw_function(&FermiGain_setLevels<Geometry2DCartesian>),
                   "set_levels(**kwargs)\n\n"
                   "Determine quasi-Fermi levels and carriers levels inside QW.\n"
                  );
        solver.def_readwrite("strained", &__Class__::if_strain, "Consider strain in QW? (True or False)");
        RECEIVER(inTemperature, "");
        RECEIVER(inCarriersConcentration, "");
        PROVIDER(outGain, "");
        PROVIDER(outGainOverCarriersConcentration, "");
        RW_PROPERTY(lifetime, getLifeTime, setLifeTime, "Stimulated emission lifetime [ps]");
        RW_PROPERTY(matrix_elem, getMatrixElem, setMatrixElem, "optical matrix element [m0*eV]");
        // solver.def_readwrite("cond_depth", &__Class__::cond_waveguide_depth, "Waveguide conduction band depth [eV]");
        // solver.def_readwrite("vale_depth", &__Class__::vale_waveguide_depth, "Waveguide valence band depth [eV]");
        solver.def("spectrum", &__Class__::getGainSpectrum, "Get gain spectrum at given point", py::arg("point"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("spectrum", FermiGetGainSpectrum2<Geometry2DCartesian>, "Get gain spectrum at given point", (py::arg("c0"), "c1"),
                   py::with_custodian_and_ward_postcall<0,1>());

        py::scope scope = solver;
        py::class_<GainSpectrum<Geometry2DCartesian>,shared_ptr<GainSpectrum<Geometry2DCartesian>>>("Spectrum",
            "Gain spectrum class. You can call it like a function to get gains for different vavelengths.",
            py::no_init)
            .def("__call__", &FermiGainSpectrum__call__<Geometry2DCartesian>)
        ;
    }
    {CLASS(FermiGainSolver<Geometry2DCylindrical>, "FermiCyl", "Gain solver based on Fermi Golden Rule for Cylindrical 2D geometry.")
        solver.def("determine_levels", &FermiGain_determineLevels<Geometry2DCylindrical>,
                   "Determine quasi-Fermi levels and carriers levels inside QW", (py::arg("T"), "n"));
        solver.def("set_levels", py::raw_function(&FermiGain_setLevels<Geometry2DCylindrical>),
                   "set_levels(**kwargs)\n\n"
                   "Determine quasi-Fermi levels and carriers levels inside QW.\n"
                  );
        solver.def_readwrite("strained", &__Class__::if_strain, "Consider strain in QW? (True or False)");
        RECEIVER(inTemperature, "");
        RECEIVER(inCarriersConcentration, "");
        PROVIDER(outGain, "");
        PROVIDER(outGainOverCarriersConcentration, "");
        RW_PROPERTY(lifetime, getLifeTime, setLifeTime, "Stimulated emission lifetime [ps]");
        RW_PROPERTY(matrix_elem, getMatrixElem, setMatrixElem, "optical matrix element [m0*eV]");
        // solver.def_readwrite("cond_depth", &__Class__::cond_waveguide_depth, "Waveguide conduction band depth [eV]");
        // solver.def_readwrite("vale_depth", &__Class__::vale_waveguide_depth, "Waveguide valence band depth [eV]");
        solver.def("spectrum", &__Class__::getGainSpectrum, "Get gain spectrum at given point", py::arg("point"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("spectrum", FermiGetGainSpectrum2<Geometry2DCylindrical>, "Get gain spectrum at given point", (py::arg("c0"), "c1"),
                   py::with_custodian_and_ward_postcall<0,1>());

        py::scope scope = solver;
        py::class_<GainSpectrum<Geometry2DCylindrical>,shared_ptr<GainSpectrum<Geometry2DCylindrical>>>("Spectrum",
            "Gain spectrum class. You can call it like a function to get gains for different vavelengths.",
            py::no_init)
            .def("__call__", &FermiGainSpectrum__call__<Geometry2DCylindrical>)
        ;
    }

}

