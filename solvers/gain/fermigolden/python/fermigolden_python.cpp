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
    return PARALLEL_UFUNC<double>([self,reg,well](double x){return self->detEl(x, reg, well);}, E);
}

template <typename GeometryT>
static py::object FermiGolden_detHh(FermiGoldenGainSolver<GeometryT>* self, py::object E, size_t reg=0, size_t well=0) {
    self->initCalculation();
    return PARALLEL_UFUNC<double>([self,reg,well](double x){return self->detHh(x, reg, well);}, E);
}

template <typename GeometryT>
static py::object FermiGolden_detLh(FermiGoldenGainSolver<GeometryT>* self, py::object E, size_t reg=0, size_t well=0) {
    self->initCalculation();
    return PARALLEL_UFUNC<double>([self,reg,well](double x){return self->detLh(x, reg, well);}, E);
}
#endif

BOOST_PYTHON_MODULE(fermigolden)
{
    plask_import_array();

    {CLASS(FermiGoldenGainSolver<Geometry2DCylindrical>, "XFermiCyl", "Gain solver based on Fermi Golden Rule for cylindrical geometry.")
#ifndef NDEBUG
        solver.def("det_El", &FermiGolden_detEl<Geometry2DCylindrical>, (arg("E"), arg("reg")=0, arg("well")=0));
        solver.def("det_Hh", &FermiGolden_detHh<Geometry2DCylindrical>, (arg("E"), arg("reg")=0, arg("well")=0));
        solver.def("det_Lh", &FermiGolden_detLh<Geometry2DCylindrical>, (arg("E"), arg("reg")=0, arg("well")=0));
#endif
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

