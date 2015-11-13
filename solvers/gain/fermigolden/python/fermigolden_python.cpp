/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
#include <util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../fermigolden.h"
using namespace plask::gain::fermigolden;

#ifndef NDEBUG
template <typename GeometryT>
static py::object FermiGolden_detEl(FermiGoldenGainSolver<GeometryT>* self, py::object E, size_t reg=0, size_t well=0) {
    self->initCalculation();
    typename FermiGoldenGainSolver<GeometryT>::ActiveRegionParams params(self, self->regions[reg], self->getT0());
    return PARALLEL_UFUNC<double>([self,&params,well](double x){return self->detEl(x, params, well);}, E);
}

template <typename GeometryT>
static py::object FermiGolden_detHh(FermiGoldenGainSolver<GeometryT>* self, py::object E, size_t reg=0, size_t well=0) {
    self->initCalculation();
    typename FermiGoldenGainSolver<GeometryT>::ActiveRegionParams params(self, self->regions[reg], self->getT0());
    return PARALLEL_UFUNC<double>([self,&params,well](double x){return self->detHh(x, params, well);}, E);
}

template <typename GeometryT>
static py::object FermiGolden_detLh(FermiGoldenGainSolver<GeometryT>* self, py::object E, size_t reg=0, size_t well=0) {
    self->initCalculation();
    typename FermiGoldenGainSolver<GeometryT>::ActiveRegionParams params(self, self->regions[reg], self->getT0());
    return PARALLEL_UFUNC<double>([self,&params,well](double x){return self->detLh(x, params, well);}, E);
}

template <typename GeometryT>
static py::object FermiGoldenGainSolver_getN(FermiGoldenGainSolver<GeometryT>* self, py::object F, py::object pT, size_t reg=0) {
    double T = (pT == py::object())? self->getT0() : py::extract<double>(pT);
    self->initCalculation();
    typename FermiGoldenGainSolver<GeometryT>::ActiveRegionParams params(self, self->regions[reg], T);
    return PARALLEL_UFUNC<double>([self,T,reg,&params](double x){return self->getN(x, T, reg, params);}, F);
}

template <typename GeometryT>
static py::object FermiGoldenGainSolver_getP(FermiGoldenGainSolver<GeometryT>* self, py::object F, py::object pT, size_t reg=0) {
    double T = (pT == py::object())? self->getT0() : py::extract<double>(pT);
    self->initCalculation();
    typename FermiGoldenGainSolver<GeometryT>::ActiveRegionParams params(self, self->regions[reg], T);
    return PARALLEL_UFUNC<double>([self,T,reg,&params](double x){return self->getP(x, T, reg, params);}, F);
}
#endif

template <typename GeometryT>
static py::object FermiGolden_getLevels(FermiGoldenGainSolver<GeometryT>& self, py::object To)
{
    //TODO consider temperature
    self.initCalculation();
    py::list result;
    for (size_t reg = 0; reg < self.levels_el.size(); ++reg) {
        py::dict info;
        info["el"] = self.levels_el[reg];
        info["hh"] = self.levels_hh[reg];
        info["lh"] = self.levels_lh[reg];
        result.append(info);
    }
    return result;
}

template <typename GeometryT>
static py::object FermiGolden_getFermiLevels(FermiGoldenGainSolver<GeometryT>& self, double N, py::object To, int reg)
{
    double T = (To == py::object())? self.getT0() : py::extract<double>(To);
    if (reg < 0) reg = self.regions.size() + reg;
    if (reg < 0 || reg >= self.regions.size()) throw IndexError("%s: Bad active region index", self.getId());
    self.initCalculation();
    double Fc{NAN}, Fv{NAN};
    self.findFermiLevels(Fc, Fv, N, T, reg);
    return py::make_tuple(Fc, Fv);
}


BOOST_PYTHON_MODULE(fermigolden)
{
    plask_import_array();

    {CLASS(FermiGoldenGainSolver<Geometry2DCylindrical>, "XFermiCyl", "Gain solver based on Fermi Golden Rule for cylindrical geometry.")
#ifndef NDEBUG
        solver.def("det_El", &FermiGolden_detEl<Geometry2DCylindrical>, (arg("E"), arg("reg")=0, arg("well")=0));
        solver.def("det_Hh", &FermiGolden_detHh<Geometry2DCylindrical>, (arg("E"), arg("reg")=0, arg("well")=0));
        solver.def("det_Lh", &FermiGolden_detLh<Geometry2DCylindrical>, (arg("E"), arg("reg")=0, arg("well")=0));
        solver.def("getN", &FermiGoldenGainSolver_getN<Geometry2DCylindrical>, (arg("F"), arg("T")=py::object(), arg("reg")=0));
        solver.def("getP", &FermiGoldenGainSolver_getP<Geometry2DCylindrical>, (arg("F"), arg("T")=py::object(), arg("reg")=0));
#endif
//         RW_FIELD(quick_levels, 
//                  "Compute levels only once and simply shift for different temperatures?\n\n"
//                  "Setting this to True strongly increases computation speed, but canis  make the results\n"
//                  "less accurate for high temperatures.");
        solver.def("get_energy_levels", &FermiGolden_getLevels<Geometry2DCylindrical>, arg("T")=py::object(),
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
        solver.def("get_fermi_levels", &FermiGolden_getFermiLevels<Geometry2DCylindrical>, (arg("n"), arg("T")=py::object(), arg("reg")=0),
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
//         METHOD(python_method_name, method_name, "Short documentation", "name_or_argument_1", arg("name_of_argument_2")=default_value_of_arg_2, ...);
//         RO_FIELD(field_name, "Short documentation"); // read-only field
//         RW_FIELD(field_name, "Short documentation"); // read-write field
//         RO_PROPERTY(python_property_name, get_method_name, "Short documentation"); // read-only property
//         RW_PROPERTY(python_property_name, get_method_name, set_method_name, "Short documentation"); // read-write property
//         RECEIVER(inReceiver, ""); // receiver in the solver (string is an optional additional documentation)
//         PROVIDER(outProvider, ""); // provider in the solver (string is an optional additional documentation)
//         BOUNDARY_CONDITIONS(boundary_conditions_name, "Short documentation"); // boundary conditions
        RECEIVER(inTemperature, "");
        RECEIVER(inCarriersConcentration, "");
    }

}

