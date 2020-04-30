/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
#include <boost/python/raw_function.hpp>
#include <plask/python_util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../fermi.h"
#include "../ferminew.h"
using namespace plask::solvers;


template <typename GeometryT>
static fermi::GainSpectrum<GeometryT> FermiGetGainSpectrum2(fermi::FermiGainSolver<GeometryT>* solver, double c0, double c1) {
    return solver->getGainSpectrum(Vec<2>(c0,c1));
}

template <typename GeometryT>
static py::object FermiGainSpectrum__call__(fermi::GainSpectrum<GeometryT>& self, py::object wavelengths) {
   return PARALLEL_UFUNC<double>([&](double x){return self.getGain(x);}, wavelengths);
}

template <typename GeometryT>
static py::object FermiGain_determineLevels(fermi::FermiGainSolver<GeometryT>& self, double T, double n)
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
        throw TypeError("set_levels() takes exactly 1 non-keyword argument1 ({0} given)", py::len(args));
    }

    fermi::FermiGainSolver<GeometryT>& self = py::extract<fermi::FermiGainSolver<GeometryT>&>(args[0]);

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
                throw TypeError("set_levels() got an unexpected keyword argument '{}'", *key);
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



template <typename GeometryT>
static FermiNew::GainSpectrum<GeometryT> FermiNewGetGainSpectrum2(FermiNew::FermiNewGainSolver<GeometryT>* solver, double c0, double c1) {
    return solver->getGainSpectrum(Vec<2>(c0,c1));
}

template <typename GeometryT>
static py::object FermiNewGainSpectrum__call__(FermiNew::GainSpectrum<GeometryT>& self, py::object wavelengths) {
   return UFUNC<double>([&](double x){return self.getGain(x);}, wavelengths);
}

template <typename GeometryT>
static FermiNew::LuminescenceSpectrum<GeometryT> FermiNewGetLuminescenceSpectrum2(FermiNew::FermiNewGainSolver<GeometryT>* solver, double c0, double c1) {
    return solver->getLuminescenceSpectrum(Vec<2>(c0,c1));
}

template <typename GeometryT>
static py::object FermiNewLuminescenceSpectrum__call__(FermiNew::LuminescenceSpectrum<GeometryT>& self, py::object wavelengths) {
   return UFUNC<double>([&](double x){return self.getLuminescence(x);}, wavelengths);
}


/*template <typename GeometryT>
static py::object FermiNewGain_setLevels(py::tuple args, py::dict kwargs)
{
    if (py::len(args) != 1) {
        throw TypeError("set_levels() takes exactly 1 non-keyword argument1 ({0} given)", py::len(args));
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
                throw TypeError("set_levels() got an unexpected keyword argument '{}'", *key);
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

template <typename GeometryT>
static py::object FermiNew_getLevels(FermiNew::FermiNewGainSolver<GeometryT>& self, py::object To, py::object No)
{
    self.initCalculation();
    py::list result;
    double T;
    if (To.is_none())
        T = self.Tref;
    else
        T = py::extract<double>(To);
    for (size_t reg = 0; reg < self.region_levels.size(); ++reg) {
        py::dict info;
        py::list el, hh, lh;
        self.findEnergyLevels(self.region_levels[reg], self.regions[reg], T);
        if (self.region_levels[reg].mpStrEc) for (auto stan: self.region_levels[reg].mpStrEc->rozwiazania) el.append(stan.poziom);
        if (self.region_levels[reg].mpStrEvhh) for (auto stan: self.region_levels[reg].mpStrEvhh->rozwiazania) hh.append(-stan.poziom);
        if (self.region_levels[reg].mpStrEvlh) for (auto stan: self.region_levels[reg].mpStrEvlh->rozwiazania) lh.append(-stan.poziom);
        info["el"] = el;
        info["hh"] = hh;
        info["lh"] = lh;
//         if (!No.is_none()) {
//             double n = py::extract<double>(No);
//             const auto& region = self.regios[reg];
//             QW::Gain gainModule;
//             gainModule.setGain(levels.aktyw, n*(region.qwtotallen*1e-8), T, 1, 
//                                region.getLayerMaterial(0)->CB(T,0.)-region.getLayerMaterial(0)->VB(T,0.));
//             double tFe = gainModule.policz_qFlc();
//             double tFp = gainModule.policz_qFlv();
//         }
        result.append(info);
    }
    return result;
}




/**
 * Initialization of Fermi gain solver class to Python
 */
BOOST_PYTHON_MODULE(wasiak)
{
    plask_import_array();

    {CLASS(fermi::FermiGainSolver<Geometry2DCartesian>, "Wasiak2D", "Gain solver based on Fermi Golden Rule for Cartesian 2D geometry.")
        solver.def("determine_levels", &FermiGain_determineLevels<Geometry2DCartesian>,
                   "Determine quasi-Fermi levels and carriers levels inside QW", (py::arg("T"), "n"));
        solver.def("set_levels", py::raw_function(&FermiGain_setLevels<Geometry2DCartesian>),
                   "Set quasi-Fermi levels and carriers levels inside QW.\n"
                  );
        solver.def_readwrite("strained", &__Class__::if_strain, "Consider strain in QW? (True or False)");
        RECEIVER(inTemperature, "");
        RECEIVER(inCarriersConcentration, "");
        PROVIDER(outGain, "");
        RW_PROPERTY(lifetime, getLifeTime, setLifeTime, "Stimulated emission lifetime [ps]");
        RW_PROPERTY(matrix_elem, getMatrixElem, setMatrixElem, "Optical matrix element [m0*eV]");
        RW_PROPERTY(matrix_elem_scaling, getMatrixElemScFact, setMatrixElemScFact, "Scale factor for optical matrix element [-]");
        RW_PROPERTY(cond_shift, getCondQWShift, setCondQWShift, "Additional conduction band shift for QW [eV].");
        RW_PROPERTY(vale_shift, getValeQWShift, setValeQWShift, "Additional valence band shift for QW [eV].");
        // solver.def_readwrite("cond_depth", &__Class__::cond_waveguide_depth, "Waveguide conduction band depth [eV]");
        // solver.def_readwrite("vale_depth", &__Class__::vale_waveguide_depth, "Waveguide valence band depth [eV]");
        solver.def("spectrum", &__Class__::getGainSpectrum, "Get gain spectrum at given point", py::arg("point"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("spectrum", FermiGetGainSpectrum2<Geometry2DCartesian>, "Get gain spectrum at given point", (py::arg("c0"), "c1"),
                   py::with_custodian_and_ward_postcall<0,1>());

        py::scope scope = solver;
        py::class_<fermi::GainSpectrum<Geometry2DCartesian>,plask::shared_ptr<fermi::GainSpectrum<Geometry2DCartesian>>>("Spectrum",
            "Gain spectrum class. You can call it like a function to get gains for different vavelengths.",
            py::no_init)
            .def("__call__", &FermiGainSpectrum__call__<Geometry2DCartesian>)
        ;
    }
    {CLASS(fermi::FermiGainSolver<Geometry2DCylindrical>, "WasiakCyl", "Gain solver based on Fermi Golden Rule for Cylindrical 2D geometry.")
        solver.def("determine_levels", &FermiGain_determineLevels<Geometry2DCylindrical>,
                   "Determine quasi-Fermi levels and carriers levels inside QW", (py::arg("T"), "n"));
        solver.def("set_levels", py::raw_function(&FermiGain_setLevels<Geometry2DCylindrical>),
                   "Determine quasi-Fermi levels and carriers levels inside QW.\n"
                  );
        solver.def_readwrite("strained", &__Class__::if_strain, "Consider strain in QW? (True or False)");
        RECEIVER(inTemperature, "");
        RECEIVER(inCarriersConcentration, "");
        PROVIDER(outGain, "");
        RW_PROPERTY(lifetime, getLifeTime, setLifeTime, "Stimulated emission lifetime [ps]");
        RW_PROPERTY(matrix_elem, getMatrixElem, setMatrixElem, "Optical matrix element [m0*eV]");
        RW_PROPERTY(matrix_elem_scaling, getMatrixElemScFact, setMatrixElemScFact, "Scale factor for optical matrix element [-]");
        RW_PROPERTY(cond_shift, getCondQWShift, setCondQWShift, "Additional conduction band shift for QW [eV]");
        RW_PROPERTY(vale_shift, getValeQWShift, setValeQWShift, "Additional valence band shift for QW [eV]");
        // solver.def_readwrite("cond_depth", &__Class__::cond_waveguide_depth, "Waveguide conduction band depth [eV]");
        // solver.def_readwrite("vale_depth", &__Class__::vale_waveguide_depth, "Waveguide valence band depth [eV]");
        solver.def("spectrum", &__Class__::getGainSpectrum, "Get gain spectrum at given point", py::arg("point"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("spectrum", FermiGetGainSpectrum2<Geometry2DCylindrical>, (py::arg("c0"), "c1"),
                   py::with_custodian_and_ward_postcall<0,1>());

        py::scope scope = solver;
        py::class_<fermi::GainSpectrum<Geometry2DCylindrical>,plask::shared_ptr<fermi::GainSpectrum<Geometry2DCylindrical>>>("Spectrum",
            "Gain spectrum class. You can call it like a function to get gains for different vavelengths.",
            py::no_init)
            .def("__call__", &FermiGainSpectrum__call__<Geometry2DCylindrical>)
        ;
    }

    {CLASS(FermiNew::FermiNewGainSolver<Geometry2DCartesian>, "WasiakNew2D", "Gain solver based on Fermi Golden Rule for Cartesian 2D geometry.")
        solver.add_property("strained", &__Class__::getStrains, &__Class__::setStrains,
                            "Consider strain in QW and barriers? (True or False).");
        solver.add_property("adjust_layers", &__Class__::getAdjustWidths, &__Class__::setAdjustWidths,
                            "Adjust thicknesses of quantum wells?\n\n"
                            "Setting this to True, allows to adjust the widths of the gain region layers\n"
                            "by few angstroms to improve numerical stability.");
        solver.add_property("fast_levels", &__Class__::getBuildStructOnce, &__Class__::setBuildStructOnce,
                            "Compute levels only once and simply shift for different temperatures?\n\n"
                            "Setting this to True stongly increases computation speed, but makes the results\n"
                            "less accurate for high gains.");
        RECEIVER(inTemperature, "");
        RECEIVER(inCarriersConcentration, "");
        PROVIDER(outGain, "");
        PROVIDER(outLuminescence, "");
        RW_PROPERTY(roughness, getRoughness, setRoughness, "Roughness of the layers [-].");
        RW_PROPERTY(lifetime, getLifeTime, setLifeTime, "Carriers lifetime [ps].");
        RW_PROPERTY(matrix_elem, getMatrixElem, setMatrixElem, "Optical matrix element [m0*eV]");
        RW_PROPERTY(matrix_elem_scaling, getMatrixElemScFact, setMatrixElemScFact, "Scale factor for optical matrix element [-]");
        RW_PROPERTY(cond_shift, getCondQWShift, setCondQWShift, "Additional conduction band shift for QW [eV]");
        RW_PROPERTY(vale_shift, getValeQWShift, setValeQWShift, "Additional valence band shift for QW [eV]");
        RW_PROPERTY(Tref, getTref, setTref,
                    "Reference temperature. If *fast_levels* is True, this is the temperature used\n"
                    "for initial computation of the energy levels [K].");
        solver.def("get_levels", &FermiNew_getLevels<Geometry2DCartesian>, py::arg("T")=py::object(), py::arg("n")=py::object());
        solver.def("spectrum", &__Class__::getGainSpectrum, "Get gain spectrum at given point", py::arg("point"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("spectrum", FermiNewGetGainSpectrum2<Geometry2DCartesian>, "Get gain spectrum at given point", (py::arg("c0"), "c1"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("luminescence_spectrum", &__Class__::getLuminescenceSpectrum, "Get luminescence spectrum at given point", py::arg("point"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("luminescence_spectrum", FermiNewGetLuminescenceSpectrum2<Geometry2DCartesian>, "Get luminescence spectrum at given point", (py::arg("c0"), "c1"),
                   py::with_custodian_and_ward_postcall<0,1>());

        py::scope scope = solver;
        py::class_<FermiNew::GainSpectrum<Geometry2DCartesian>, plask::shared_ptr<FermiNew::GainSpectrum<Geometry2DCartesian>>>("Spectrum",
            "Gain spectrum class. You can call it like a function to get gains for different vavelengths.",
            py::no_init)
            .def("__call__", &FermiNewGainSpectrum__call__<Geometry2DCartesian>)
        ;
        py::class_<FermiNew::LuminescenceSpectrum<Geometry2DCartesian>, plask::shared_ptr<FermiNew::LuminescenceSpectrum<Geometry2DCartesian>>>("LuminescenceSpectrum",
            "Luminescence spectrum class. You can call it like a function to get luminescences for different vavelengths.",
            py::no_init)
            .def("__call__", &FermiNewLuminescenceSpectrum__call__<Geometry2DCartesian>)
        ;
    }
    {CLASS(FermiNew::FermiNewGainSolver<Geometry2DCylindrical>, "WasiakNewCyl", "Gain solver based on Fermi Golden Rule for Cylindrical 2D geometry.")
        solver.add_property("strained", &__Class__::getStrains, &__Class__::setStrains,
                            "Consider strain in QW and barriers? (True or False).");
        solver.add_property("adjust_layers", &__Class__::getAdjustWidths, &__Class__::setAdjustWidths,
                            "Adjust thicknesses of quantum wells?\n\n"
                            "Setting this to True, allows to adjust the widths of the gain region layers\n"
                            "by few angstroms to improve numerical stability.");
        solver.add_property("fast_levels", &__Class__::getBuildStructOnce, &__Class__::setBuildStructOnce,
                            "Compute levels only once and simply shift for different temperatures?\n\n"
                            "Setting this to True stongly increases computation speed, but makes the results\n"
                            "less accurate for high gains.");
        RECEIVER(inTemperature, "");
        RECEIVER(inCarriersConcentration, "");
        PROVIDER(outGain, "");
        PROVIDER(outLuminescence, "");
        RW_PROPERTY(roughness, getRoughness, setRoughness, "Roughness of the layers [-].");
        RW_PROPERTY(lifetime, getLifeTime, setLifeTime, "Carriers lifetime [ps].");
        RW_PROPERTY(matrix_elem, getMatrixElem, setMatrixElem, "optical matrix element [m0*eV]");
        RW_PROPERTY(matrix_elem_scaling, getMatrixElemScFact, setMatrixElemScFact, "Scale factor for optical matrix element [-]");
        RW_PROPERTY(cond_shift, getCondQWShift, setCondQWShift, "Additional conduction band shift for QW [eV]");
        RW_PROPERTY(vale_shift, getValeQWShift, setValeQWShift, "Additional valence band shift for QW [eV]");
        RW_PROPERTY(Tref, getTref, setTref,
                    "Reference temperature. If *fast_levels* is True, this is the temperature used\n"
                    "for initial computation of the energy levels [K].");
        solver.def("get_levels", &FermiNew_getLevels<Geometry2DCylindrical>, py::arg("T")=py::object(), py::arg("n")=py::object());
        solver.def("spectrum", &__Class__::getGainSpectrum, "Get gain spectrum at given point", py::arg("point"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("spectrum", FermiNewGetGainSpectrum2<Geometry2DCylindrical>, "Get gain spectrum at given point", (py::arg("c0"), "c1"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("luminescencespectrum", &__Class__::getLuminescenceSpectrum, "Get luminescence spectrum at given point", py::arg("point"),
                   py::with_custodian_and_ward_postcall<0,1>());
        solver.def("luminescencespectrum", FermiNewGetLuminescenceSpectrum2<Geometry2DCylindrical>, "Get luminescence spectrum at given point", (py::arg("c0"), "c1"),
                   py::with_custodian_and_ward_postcall<0,1>());

        py::scope scope = solver;
        py::class_<FermiNew::GainSpectrum<Geometry2DCylindrical>, plask::shared_ptr<FermiNew::GainSpectrum<Geometry2DCylindrical>>>("Spectrum",
            "Gain spectrum class. You can call it like a function to get gains for different vavelengths.",
            py::no_init)
            .def("__call__", &FermiNewGainSpectrum__call__<Geometry2DCylindrical>)
        ;
        py::class_<FermiNew::LuminescenceSpectrum<Geometry2DCylindrical>, plask::shared_ptr<FermiNew::LuminescenceSpectrum<Geometry2DCylindrical>>>("LuminescenceSpectrum",
            "Luminescence spectrum class. You can call it like a function to get luminescences for different vavelengths.",
            py::no_init)
            .def("__call__", &FermiNewLuminescenceSpectrum__call__<Geometry2DCylindrical>)
        ;
    }

}

