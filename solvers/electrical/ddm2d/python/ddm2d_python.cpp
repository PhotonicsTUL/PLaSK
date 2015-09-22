#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../ddm2d.h"
using namespace plask::solvers::drift_diffusion;

// template <typename Cls>
// static DataVectorWrap<const double, 2> getCondJunc(const Cls* self) {
//     if (self->getMesh() && self->getGeometry()) {
//         auto midmesh = self->getMesh()->getMidpointsMesh();
//         shared_ptr<OrderedAxis> line1 = make_shared<OrderedAxis>();
//         for (size_t n = 0; n < self->getActNo(); ++n)
//             line1->addPoint(self->getMesh()->axis1->at((self->getActLo(n)+self->getActHi(n))/2));
//         auto mesh = make_shared<RectangularMesh<2>>(midmesh->axis0->clone(), line1);
//         return DataVectorWrap<const double,2>(self->getCondJunc(), mesh);
//     } else {
//         auto mesh = make_shared<RectangularMesh<2>>(make_shared<OrderedAxis>(std::initializer_list<double>{NAN}),
//                                                     make_shared<OrderedAxis>(std::initializer_list<double>{NAN}));
//         return DataVectorWrap<const double,2>(self->getCondJunc(), mesh);
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
//         const DataVectorWrap<const double,2>& val = py::extract<DataVectorWrap<const double,2>&>(value);
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

/*template <typename Class> static double Shockley_getBeta(const Class& self) { return self.getBeta(0); }
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
    } catch (boost::bad_lexical_cast) {
        throw AttributeError("%1% object has no attribute '%2%'", self.getClassName(), attr);
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
    } catch (boost::bad_lexical_cast) {}

    oself.attr("__class__").attr("__base__").attr("__setattr__")(oself, attr, value);
}*/


template <typename GeometryT>
inline static void register_drift_diffusion_solver(const char* name, const char* geoname)
{
    typedef DriftDiffusionModel2DSolver<GeometryT>  __Class__;
    ExportSolver<DriftDiffusionModel2DSolver<GeometryT>> solver(name, format(

        "%1%(name=\"\")\n\n"

        "Finite element drift-diffusion electrical solver for 2D %2% geometry."

        , name, geoname).c_str(), py::init<std::string>(py::arg("name")=""));
    //METHOD(compute, compute, "Run drift_diffusion calculations", py::arg("loops")=0);
    METHOD(compute, compute, "Run drift_diffusion calculations"/*, py::arg("loops")=0*/);
    METHOD(compute_initial_potential, computePsiI, "Run drift_diffusion calculations"/*, py::arg("loops")=0*/);
    /*METHOD(get_total_current, getTotalCurrent, "Get total current flowing through active region [mA]", py::arg("nact")=0);
    RO_PROPERTY(err, getErr, "Maximum estimated error");
    RECEIVER(inWavelength, "It is required only if :attr:`heat` is equal to *wavelength*.");
    RECEIVER(inTemperature, "");*/
    PROVIDER(outPotential, "");
    PROVIDER(outQuasiFermiElectronLevel, "");
    PROVIDER(outQuasiFermiHoleLevel, "");
    /*PROVIDER(outPotential, "");
    PROVIDER(outCurrentDensity, "");
    PROVIDER(outHeat, "");
    PROVIDER(outConductivity, "");
    BOUNDARY_CONDITIONS(voltage_boundary, "Boundary conditions of the first kind (constant potential)");
    RW_FIELD(maxerr, "Limit for the potential updates");
    RW_FIELD(algorithm, "Chosen matrix factorization algorithm");
    solver.def_readwrite("heat", &__Class__::heatmet, "Chosen method used for computing heats");
    solver.add_property("beta", &Shockley_getBeta<__Class__>, &Shockley_setBeta<__Class__>,
                        "Junction coefficient [1/V].\n\n"
                        "In case there is more than one junction you may set $\\beta$ parameter for any\n"
                        "of them by using ``beta#`` property, where # is the junction number (specified\n"
                        "by a role ``junction#`` or ``active#``).\n\n"
                        "``beta`` is an alias for ``beta0``.\n"
                       );
    solver.add_property("Vt", &Shockley_getVt<__Class__>, &Shockley_setVt<__Class__>,
                        "Junction thermal voltage [V].\n\n"
                        "In case there is more than one junction you may set $V_t$ parameter for any\n"
                        "of them by using ``Vt#`` property, where # is the junction number (specified\n"
                        "by a role ``junction#`` or ``active#``).\n\n"
                        "``Vt`` is an alias for ``Vt0``.\n"
                       );
    solver.add_property("js", &Shockley_getJs<__Class__>, &Shockley_setJs<__Class__>,
                        "Reverse bias current density [A/m\\ :sup:2:].\n\n"
                        "In case there is more than one junction you may set $j_s$ parameter for any\n"
                        "of them by using ``js#`` property, where # is the junction number (specified\n"
                        "by a role ``junction#`` or ``active#``).\n\n"
                        "``js`` is an alias for ``js0``.\n"
                       );
    solver.def("__getattr__",  &Shockley__getattr__<__Class__>);
    solver.def("__setattr__",  &Shockley__setattr__<__Class__>);
    RW_PROPERTY(pcond, getCondPcontact, setCondPcontact, "Conductivity of the p-contact");
    RW_PROPERTY(ncond, getCondNcontact, setCondNcontact, "Conductivity of the n-contact");
    solver.add_property("pnjcond", &__Class__::getDefaultCondJunc, (void(__Class__::*)(double))&__Class__::setCondJunc, "Effective conductivity of the p-n junction");*/
    /*solver.setattr("outVoltage", solver.attr("outPotential"));
    RW_FIELD(itererr, "Allowed residual iteration for iterative method");
    RW_FIELD(iterlim, "Maximum number of iterations for iterative method");
    RW_FIELD(logfreq, "Frequency of iteration progress reporting");
    METHOD(get_electrostatic_energy, getTotalEnergy,
           "Get the energy stored in the electrostatic field in the analyzed structure.\n\n"
           "Return:\n"
           "    Total electrostatic energy [J].\n"
    );
    METHOD(get_capacitance, getCapacitance,
           "Get the structure capacitance.\n\n"
           "Return:\n"
           "    Total capacitance [pF].\n\n"
           "Note:\n"
           "    This method can only be used it there are exactly two boundary conditions\n"
           "    specifying the voltage. Otherwise use :meth:`get_electrostatic_energy` to\n"
           "    obtain the stored energy :math:`W` and compute the capacitance as:\n"
           "    :math:`C = 2 \\, W / U^2`, where :math:`U` is the applied voltage.\n"
    );
    METHOD(get_total_heat, getTotalHeat,
           "Get the total heat produced by the current flowing in the structure.\n\n"
           "Return:\n"
           "    Total produced heat [mW].\n"
    );*/
}

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(ddm2d)
{
    /*py_enum<Algorithm>()
        .value("CHOLESKY", ALGORITHM_CHOLESKY)
        .value("GAUSS", ALGORITHM_GAUSS)
        .value("ITERATIVE", ALGORITHM_ITERATIVE)
    ;

    py_enum<HeatMethod>()
        .value("JOULES", HEAT_JOULES)
        .value("WAVELENGTH", HEAT_BANDGAP)
    ;*/

    register_drift_diffusion_solver<Geometry2DCartesian>("DriftDiffusion2D", "Cartesian");

    //register_namespace drift_diffusion_solver<Geometry2DCylindrical>("ShockleyCyl", "cylindrical");
}

