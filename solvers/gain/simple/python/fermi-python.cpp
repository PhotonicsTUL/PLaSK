/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
#include <util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../fermi.h"
using namespace plask::solvers::fermi;

template <typename GeometryT>
GainSpectrum<GeometryT> FermiGetGainSpectrum2(FermiGainSolver<GeometryT>* solver, double c0, double c1) {
    return solver->getGainSpectrum(Vec<2>(c0,c1));
}

template <typename GeometryT>
py::object FermiGainSpectrum__call__(GainSpectrum<GeometryT>& self, py::object wavelengths) {
   return UFUNC<double>([&](double x){return self.getGain(x);}, wavelengths);
}

/**
 * Initialization of Fermi gain solver class to Python
 */
BOOST_PYTHON_MODULE(simple)
{
    plask_import_array();

    {CLASS(FermiGainSolver<Geometry2DCartesian>, "Fermi2D", "Gain solver based on Fermi Golden Rule for Cartesian 2D geometry.")
        METHOD(determine_levels, determineLevels, "Determine quasi-Fermi levels and carriers levels inside QW", "T", "n");
        solver.add_receiver("inTemperature",
                            reinterpret_cast<ReceiverFor<Temperature,Geometry2DCartesian>__Class__::*>(&__Class__::inTemperature),
                            "Temperature distribution");
        solver.add_receiver("inCarriersConcentration",
                            reinterpret_cast<ReceiverFor<CarriersConcentration,Geometry2DCartesian>__Class__::*>(&__Class__::inCarriersConcentration),
                            "Carrier pairs concentration");
        PROVIDER(outGain, "Optical gain in the active region");
        PROVIDER(outGainOverCarriersConcentration, "Optical gain over carriers concentration derivative in the active region");
        RW_PROPERTY(lifetime, getLifeTime, setLifeTime, "Stimulated emission lifetime [ps]");
        RW_PROPERTY(matrix_elem, getMatrixElem, setMatrixElem, "optical matrix element [m0*eV]");
        solver.def_readwrite("cond_depth", &__Class__::cond_waveguide_depth, "Waveguide conduction band depth [eV]");
        solver.def_readwrite("vale_depth", &__Class__::vale_waveguide_depth, "Waveguide valence band depth [eV]");
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
        METHOD(determine_levels, determineLevels, "Determine quasi-Fermi levels and carriers levels inside QW", "T", "n");
        RECEIVER(inTemperature, "Temperature distribution");
        RECEIVER(inCarriersConcentration, "Carrier pairs concentration");
        PROVIDER(outGain, "Optical gain in the active region");
        PROVIDER(outGainOverCarriersConcentration, "Optical gain over carriers concentration derivative in the active region");
        RW_PROPERTY(lifetime, getLifeTime, setLifeTime, "Stimulated emission lifetime [ps]");
        RW_PROPERTY(matrix_elem, getMatrixElem, setMatrixElem, "optical matrix element [m0*eV]");
        solver.def_readwrite("cond_depth", &__Class__::cond_waveguide_depth, "Waveguide conduction band depth [eV]");
        solver.def_readwrite("vale_depth", &__Class__::vale_waveguide_depth, "Waveguide valence band depth [eV]");
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

