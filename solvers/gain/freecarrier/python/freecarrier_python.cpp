/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
#include <util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../freecarrier.h"
using namespace plask::gain::freecarrier;

#ifndef NDEBUG
template <typename GeometryT>
static py::object FreeCarrier_detEl(FreeCarrierGainSolver<GeometryT>* self, py::object E, size_t reg=0, size_t well=0) {
    self->initCalculation();
    typename FreeCarrierGainSolver<GeometryT>::ActiveRegionParams params(self, self->regions[reg], self->getT0());
    return PARALLEL_UFUNC<double>([self,&params,well](double x){return self->detEl(x, params, well);}, E);
}

template <typename GeometryT>
static py::object FreeCarrier_detHh(FreeCarrierGainSolver<GeometryT>* self, py::object E, size_t reg=0, size_t well=0) {
    self->initCalculation();
    typename FreeCarrierGainSolver<GeometryT>::ActiveRegionParams params(self, self->regions[reg], self->getT0());
    return PARALLEL_UFUNC<double>([self,&params,well](double x){return self->detHh(x, params, well);}, E);
}

template <typename GeometryT>
static py::object FreeCarrier_detLh(FreeCarrierGainSolver<GeometryT>* self, py::object E, size_t reg=0, size_t well=0) {
    self->initCalculation();
    typename FreeCarrierGainSolver<GeometryT>::ActiveRegionParams params(self, self->regions[reg], self->getT0());
    return PARALLEL_UFUNC<double>([self,&params,well](double x){return self->detLh(x, params, well);}, E);
}

template <typename GeometryT>
static py::object FreeCarrierGainSolver_getN(FreeCarrierGainSolver<GeometryT>* self, py::object F, py::object pT, size_t reg=0) {
    double T = (pT == py::object())? self->getT0() : py::extract<double>(pT);
    self->initCalculation();
    typename FreeCarrierGainSolver<GeometryT>::ActiveRegionParams params(self, self->params0[reg], T);
    return PARALLEL_UFUNC<double>([self,T,reg,&params](double x){return self->getN(x, T, params);}, F);
}

template <typename GeometryT>
static py::object FreeCarrierGainSolver_getP(FreeCarrierGainSolver<GeometryT>* self, py::object F, py::object pT, size_t reg=0) {
    double T = (pT == py::object())? self->getT0() : py::extract<double>(pT);
    self->initCalculation();
    typename FreeCarrierGainSolver<GeometryT>::ActiveRegionParams params(self, self->params0[reg], T);
    return PARALLEL_UFUNC<double>([self,T,reg,&params](double x){return self->getP(x, T, params);}, F);
}
#endif

template <typename GeometryT>
static py::object FreeCarrier_getLevels(FreeCarrierGainSolver<GeometryT>& self, py::object To)
{
    static const char* names[3] = { "el", "hh", "lh" };

    //TODO consider temperature
    self.initCalculation();
    py::list result;
    for (size_t reg = 0; reg < self.regions.size(); ++reg) {
        py::dict info;
        for (size_t i = 0; i < 3; ++i) {
            py::list lst;
            for (const auto& l: self.params0[reg].levels[i]) lst.append(l.E);
            info[names[i]] = lst; 
        }
        result.append(info);
    }
    return result;
}

template <typename GeometryT>
static py::object FreeCarrier_getFermiLevels(FreeCarrierGainSolver<GeometryT>* self, double N, py::object To, int reg)
{
    double T = (To == py::object())? self->getT0() : py::extract<double>(To);
    if (reg < 0) reg = self->regions.size() + reg;
    if (reg < 0 || reg >= self->regions.size()) throw IndexError("%s: Bad active region index", self->getId());
    self->initCalculation();
    double Fc{NAN}, Fv{NAN};
    typename FreeCarrierGainSolver<GeometryT>::ActiveRegionParams params(self, self->params0[reg], T);
    self->findFermiLevels(Fc, Fv, N, T, params);
    return py::make_tuple(Fc, Fv);
}

template <typename GeometryT>
static shared_ptr<GainSpectrum<GeometryT>> FreeCarrierGetGainSpectrum2(FreeCarrierGainSolver<GeometryT>* solver, double c0, double c1) {
    return solver->getGainSpectrum(Vec<2>(c0,c1));
}

template <typename GeometryT>
static py::object FreeCarrierGainSpectrum__call__(GainSpectrum<GeometryT>& self, py::object wavelengths) {
   return PARALLEL_UFUNC<double>([&](double x){return self.getGain(x);}, wavelengths);
}


BOOST_PYTHON_MODULE(freecarrier)
{
    plask_import_array();

    {CLASS(FreeCarrierGainSolver<Geometry2DCylindrical>, "FreeCarrierCyl", "Quantum-well gain using free-carrier approximation for cylindrical geometry.")
#ifndef NDEBUG
        solver.def("det_El", &FreeCarrier_detEl<Geometry2DCylindrical>, (arg("E"), arg("reg")=0, arg("well")=0));
        solver.def("det_Hh", &FreeCarrier_detHh<Geometry2DCylindrical>, (arg("E"), arg("reg")=0, arg("well")=0));
        solver.def("det_Lh", &FreeCarrier_detLh<Geometry2DCylindrical>, (arg("E"), arg("reg")=0, arg("well")=0));
        solver.def("getN", &FreeCarrierGainSolver_getN<Geometry2DCylindrical>, (arg("F"), arg("T")=py::object(), arg("reg")=0));
        solver.def("getP", &FreeCarrierGainSolver_getP<Geometry2DCylindrical>, (arg("F"), arg("T")=py::object(), arg("reg")=0));
#endif
//         RW_FIELD(quick_levels, 
//                  "Compute levels only once and simply shift for different temperatures?\n\n"
//                  "Setting this to True strongly increases computation speed, but canis  make the results\n"
//                  "less accurate for high temperatures.");
        solver.def("get_energy_levels", &FreeCarrier_getLevels<Geometry2DCylindrical>, arg("T")=py::object(),
            "Get energy levels in quantum wells.\n\n"
            "Compute energy levels in quantum wells for electrons, heavy holes and\n"
            "light holes.\n\n"
            "Args:\n"
            "    T (float or ``None``): Temperature to get the levels. If this argument is\n"
            "                           ``None``, the estimates for temperature :py:attr:`T0`\n"
            "                           are returned.\n\n"
            "Returns:\n"
            "    list: List with dictionaries with keys `el`, `hh`, and `lh` with levels for\n"
            "          electrons, heavy holes and light holes. Each list element corresponds\n"
            "          to one active region.\n"
        );
        solver.def("get_fermi_levels", &FreeCarrier_getFermiLevels<Geometry2DCylindrical>, (arg("n"), arg("T")=py::object(), arg("reg")=0),
            "Get quasi-Fermi levels.\n\n"
            "Compute quasi-Fermi levels in specified active region.\n"
            "Args:\n"
            "    n (float): Carriers concentration to determine the levels for\n"
            "               [1/cm\\ :sup:`3`\\ ].\n"
            "    T (float or ``None``): Temperature to get the levels. If this argument is\n"
            "                           ``None``, the estimates for temperature :py:attr:`T0`\n"
            "                           are returned.\n\n"
            "    reg (int): Active region number.\n"
            "Returns:\n"
            "    tuple: Two-element tuple with quasi-Fermi levels for electrons and holes.\n"
        );
        RW_PROPERTY(T0, getT0, setT0, "Reference temperature.\n\nIn this temperature levels estimates are computed.");
        RECEIVER(inTemperature, "");
        RECEIVER(inCarriersConcentration, "");
        PROVIDER(outGain, "");
        solver.def("spectrum", &__Class__::getGainSpectrum, py::arg("point"), py::with_custodian_and_ward_postcall<0,1>(),
                   "Get gain spectrum at given point.\n\n"
                   "Args:\n"
                   "    point (vec): Point to get gain at.\n"
                   "    c0, c1 (float): Coordinates of the point to get gain at.\n\n"
                   "Returns:\n"
                   "    :class:`XFermiCyl.Spectrum`: Spectrum object.\n");
        solver.def("spectrum", FreeCarrierGetGainSpectrum2<Geometry2DCylindrical>, (py::arg("c0"), "c1"), py::with_custodian_and_ward_postcall<0,1>());

        py::scope scope = solver;
        py::class_<GainSpectrum<Geometry2DCylindrical>,plask::shared_ptr<GainSpectrum<Geometry2DCylindrical>>, boost::noncopyable>("Spectrum",
            "Gain spectrum object. You can call it like a function to get gains for different vavelengths.",
            py::no_init)
            .def("__call__", &FreeCarrierGainSpectrum__call__<Geometry2DCylindrical>, py::arg("lam"),
                 "Get gain at specified wavelength.\n\n"
                 "Args:\n"
                 "    lam (float): Wavelength to get the gain at.\n"
            )
        ;
    }
}

