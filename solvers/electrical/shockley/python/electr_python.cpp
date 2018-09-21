#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../electr2d.h"
#include "../electr3d.h"
using namespace plask::electrical::shockley;

static py::object outPotential(const py::object& self) {
    throw TypeError(u8"{}: 'outPotential' is reserved for drift-diffusion model; use 'outVoltage' instead",
                    std::string(py::extract<std::string>(self.attr("id"))));
    return py::object();
}

// template <typename Cls>
// static PythonDataVector<const double, 2> getCondJunc(const Cls* self) {
//     if (self->getMesh() && self->getGeometry()) {
//         auto midmesh = self->getMesh()->getElementMesh();
//         shared_ptr<OrderedAxis> line1 = plask::make_shared<OrderedAxis>();
//         for (size_t n = 0; n < self->getActNo(); ++n)
//             line1->addPoint(self->getMesh()->axis1->at((self->getActLo(n)+self->getActHi(n))/2));
//         auto mesh = plask::make_shared<RectangularMesh<2>>(midmesh->axis0->clone(), line1);
//         return PythonDataVector<const double,2>(self->getCondJunc(), mesh);
//     } else {
//         auto mesh = plask::make_shared<RectangularMesh<2>>(plask::make_shared<OrderedAxis>(std::initializer_list<double>{NAN}),
//                                                     plask::make_shared<OrderedAxis>(std::initializer_list<double>{NAN}));
//         return PythonDataVector<const double,2>(self->getCondJunc(), mesh);
//     }
// }
//
// template <typename Cls>
// static void setCondJunc(Cls* self, py::object value) {
//     try {
//         double val = py::extract<double>(value);
//         self->setCondJunc(val);
//         return;
//     } catch (py::error_already_set) {
//         PyErr_Clear();
//     }
//     if (!self->getMesh()) throw NoMeshException(self->getId());
//     size_t len = self->getMesh()->axis0->size()-1;
//     try {
//         const PythonDataVector<const double,2>& val = py::extract<PythonDataVector<const double,2>&>(value);
//         {
//             auto mesh = dynamic_pointer_cast<RectangularMesh<2>>(val.mesh);
//             if (mesh && mesh->axis1->size() == self->getActNo() && val.size() == len) {
//                 self->setCondJunc(val);
//                 return;
//             }
//         }
//     } catch (py::error_already_set) {
//     //    PyErr_Clear();
//     //}
//     //try {
//     //    if (py::len(value) != len) throw py::error_already_set();
//     //    DataVector<double> data(len);
//     //    for (size_t i = 0; i != len; ++i) data[i] = py::extract<double>(value[i]);
//     //    self->setCondJunc(DataVector<const double>(std::move(data)));
//     //} catch (py::error_already_set) {
//         throw ValueError("pnjcond can be set either to float or data read from it", len);
//     }
// }

template <typename Class> static double Shockley_getBeta(const Class& self) { return self.getBeta(0); }
template <typename Class> static void Shockley_setBeta(Class& self, double value) { self.setBeta(0, value); }

template <typename Class> static double Shockley_getVt(const Class& self) { return self.getVt(0); }
template <typename Class> static void Shockley_setVt(Class& self, double value) { self.setVt(0, value); }

template <typename Class> static double Shockley_getJs(const Class& self) { return self.getJs(0); }
template <typename Class> static void Shockley_setJs(Class& self, double value) { self.setJs(0, value); }

template <typename Class> py::object Shockley__getattr__(const Class& self, const std::string& attr)
{
    try {
        if (attr.substr(0,4) == "beta") return py::object(self.getBeta(boost::lexical_cast<size_t>(attr.substr(4))));
        if (attr.substr(0,2) == "Vt") return py::object(self.getVt(boost::lexical_cast<size_t>(attr.substr(2))));
        if (attr.substr(0,2) == "js") return py::object(self.getJs(boost::lexical_cast<size_t>(attr.substr(2))));
    } catch (boost::bad_lexical_cast&) {
        throw AttributeError(u8"{0} object has no attribute '{1}'", self.getClassName(), attr);
    }
    return py::object();
}

template <typename Class> void Shockley__setattr__(const py::object& oself, const std::string& attr, const py::object& value)
{
    Class& self = py::extract<Class&>(oself);

    try {
        if (attr.substr(0,4) == "beta") { self.setBeta(boost::lexical_cast<size_t>(attr.substr(4)), py::extract<double>(value)); return; }
        if (attr.substr(0,2) == "Vt") { self.setVt(boost::lexical_cast<size_t>(attr.substr(2)), py::extract<double>(value)); return; }
        if (attr.substr(0,2) == "js") { self.setJs(boost::lexical_cast<size_t>(attr.substr(2)), py::extract<double>(value)); return; }
    } catch (boost::bad_lexical_cast&) {}

    oself.attr("__class__").attr("__base__").attr("__setattr__")(oself, attr, value);
}


template <typename __Class__>
inline static void register_electrical_solver(const char* name, const char* geoname)
{
    ExportSolver<__Class__> solver(name, format(

        u8"{0}(name=\"\")\n\n"

        u8"Finite element thermal solver for {1} geometry."

        , name, geoname).c_str(), py::init<std::string>(py::arg("name")=""));
    METHOD(compute, compute, u8"Run electrical calculations", py::arg("loops")=0);
    METHOD(get_total_current, getTotalCurrent, u8"Get total current flowing through active region [mA]", py::arg("nact")=0);
    RO_PROPERTY(err, getErr, u8"Maximum estimated error");
    RECEIVER(inWavelength, u8"It is required only if :attr:`heat` is equal to *wavelength*.");
    RECEIVER(inTemperature, u8"");
    PROVIDER(outVoltage, u8"");
    PROVIDER(outCurrentDensity, u8"");
    PROVIDER(outHeat, u8"");
    PROVIDER(outConductivity, u8"");
    BOUNDARY_CONDITIONS(voltage_boundary, u8"Boundary conditions of the first kind (constant potential)");
    RW_FIELD(maxerr, u8"Limit for the potential updates");
    RW_FIELD(algorithm, u8"Chosen matrix factorization algorithm");
    solver.def_readwrite("heat", &__Class__::heatmet, "Chosen method used for computing heats");
    RW_PROPERTY(include_empty, usingFullMesh, useFullMesh, "Should empty regions (e.g. air) be included into computation domain?");
    solver.add_property("beta", &Shockley_getBeta<__Class__>, &Shockley_setBeta<__Class__>,
                        u8"Junction coefficient [1/V].\n\n"
                        u8"In case there is more than one junction you may set $\\beta$ parameter for any\n"
                        u8"of them by using ``beta#`` property, where # is the junction number (specified\n"
                        u8"by a role ``junction#`` or ``active#``).\n\n"
                        u8"``beta`` is an alias for ``beta0``.\n"
                       );
    solver.add_property("Vt", &Shockley_getVt<__Class__>, &Shockley_setVt<__Class__>,
                        u8"Junction thermal voltage [V].\n\n"
                        u8"In case there is more than one junction you may set $V_t$ parameter for any\n"
                        u8"of them by using ``Vt#`` property, where # is the junction number (specified\n"
                        u8"by a role ``junction#`` or ``active#``).\n\n"
                        u8"``Vt`` is an alias for ``Vt0``.\n"
                       );
    solver.add_property("js", &Shockley_getJs<__Class__>, &Shockley_setJs<__Class__>,
                        u8"Reverse bias current density [A/m\\ :sup:`2`\\ ].\n\n"
                        u8"In case there is more than one junction you may set $j_s$ parameter for any\n"
                        u8"of them by using ``js#`` property, where # is the junction number (specified\n"
                        u8"by a role ``junction#`` or ``active#``).\n\n"
                        u8"``js`` is an alias for ``js0``.\n"
                       );
    solver.def("__getattr__",  &Shockley__getattr__<__Class__>);
    solver.def("__setattr__",  &Shockley__setattr__<__Class__>);
    RW_PROPERTY(pcond, getCondPcontact, setCondPcontact, u8"Conductivity of the p-contact");
    RW_PROPERTY(ncond, getCondNcontact, setCondNcontact, u8"Conductivity of the n-contact");
    solver.add_property("pnjcond", &__Class__::getCondJunc, (void(__Class__::*)(double))&__Class__::setCondJunc,
                        u8"Default effective conductivity of the p-n junction.\n\n"
                        u8"Effective junction conductivity will be computed starting from this value.\n"
                        u8"Note that the actual junction conductivity after convergence can be obtained\n"
                        u8"with :attr:`outConductivity`.");
    solver.add_property("outPotential", outPotential, u8"Not available in this solver. Use :attr:`outVoltage` instead.");
    RW_FIELD(itererr, u8"Allowed residual iteration for iterative method");
    RW_FIELD(iterlim, u8"Maximum number of iterations for iterative method");
    RW_FIELD(logfreq, u8"Frequency of iteration progress reporting");
    METHOD(get_electrostatic_energy, getTotalEnergy,
           u8"Get the energy stored in the electrostatic field in the analyzed structure.\n\n"
           u8"Return:\n"
           u8"    Total electrostatic energy [J].\n"
    );
    METHOD(get_capacitance, getCapacitance,
           u8"Get the structure capacitance.\n\n"
           u8"Return:\n"
           u8"    Total capacitance [pF].\n\n"
           u8"Note:\n"
           u8"    This method can only be used it there are exactly two boundary conditions\n"
           u8"    specifying the voltage. Otherwise use :meth:`get_electrostatic_energy` to\n"
           u8"    obtain the stored energy :math:`W` and compute the capacitance as:\n"
           u8"    :math:`C = 2 \\, W / U^2`, where :math:`U` is the applied voltage.\n"
    );
    METHOD(get_total_heat, getTotalHeat,
           u8"Get the total heat produced by the current flowing in the structure.\n\n"
           u8"Return:\n"
           u8"    Total produced heat [mW].\n"
    );
}

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(shockley)
{
    py_enum<Algorithm>()
        .value("CHOLESKY", ALGORITHM_CHOLESKY)
        .value("GAUSS", ALGORITHM_GAUSS)
        .value("ITERATIVE", ALGORITHM_ITERATIVE)
    ;

    py_enum<HeatMethod>()
        .value("JOULES", HEAT_JOULES)
        .value("WAVELENGTH", HEAT_BANDGAP)
    ;

    register_electrical_solver<FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>>("Shockley2D", "2D Cartesian");

    register_electrical_solver<FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>>("ShockleyCyl", "2D cylindrical");

    register_electrical_solver<FiniteElementMethodElectrical3DSolver>("Shockley3D", "3D Cartesian");

}

