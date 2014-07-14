#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../femV.h"
using namespace plask::solvers::electrical;

template <typename Cls>
static DataVectorWrap<const double, 2> getCondJunc(const Cls* self) {
    if (self->getMesh() && self->getGeometry()) {
        auto midmesh = self->getMesh()->getMidpointsMesh();
        shared_ptr<OrderedAxis> line1 = make_shared<OrderedAxis>();
        for (size_t n = 0; n < self->getActNo(); ++n)
            line1->addPoint(self->getMesh()->axis1->at((self->getActLo(n)+self->getActHi(n))/2));
        auto mesh = make_shared<RectangularMesh<2>>(midmesh->axis0->clone(), line1);
        return DataVectorWrap<const double,2>(self->getCondJunc(), mesh);
    } else {
        auto mesh = make_shared<RectangularMesh<2>>(make_shared<OrderedAxis>(std::initializer_list<double>{NAN}),
                                                    make_shared<OrderedAxis>(std::initializer_list<double>{NAN}));
        return DataVectorWrap<const double,2>(self->getCondJunc(), mesh);
    }
}

template <typename Cls>
static void setCondJunc(Cls* self, py::object value) {
    try {
        double val = py::extract<double>(value);
        self->setCondJunc(val);
        return;
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    if (!self->getMesh()) throw NoMeshException(self->getId());
    size_t len = self->getMesh()->axis0->size()-1;
    try {
        const DataVectorWrap<const double,2>& val = py::extract<DataVectorWrap<const double,2>&>(value);
        {
            auto mesh = dynamic_pointer_cast<RectangularMesh<2>>(val.mesh);
            if (mesh && mesh->axis1->size() == self->getActNo() && val.size() == len) {
                self->setCondJunc(val);
                return;
            }
        }
    } catch (py::error_already_set) {
    //    PyErr_Clear();
    //}
    //try {
    //    if (py::len(value) != len) throw py::error_already_set();
    //    DataVector<double> data(len);
    //    for (size_t i = 0; i != len; ++i) data[i] = py::extract<double>(value[i]);
    //    self->setCondJunc(DataVector<const double>(std::move(data)));
    //} catch (py::error_already_set) {
        throw ValueError("pnjcond can be set either to float or data read from it", len);
    }
}


//TODO remove after 1.06.2014
py::object outHeatDensity_get(const py::object& self) {
    writelog(LOG_WARNING, "'outHeatDensity' is obsolete. Use 'outHeat' instead!");
    return self.attr("outHeat");
}


template <typename GeometryT>
inline static void register_electrical_solver(const char* name, const char* geoname)
{
    typedef FiniteElementMethodElectrical2DSolver<GeometryT>  __Class__;
    ExportSolver<FiniteElementMethodElectrical2DSolver<GeometryT>> solver(name, format(

        "%1%(name=\"\")\n\n"

        "Finite element thermal solver for 2D %2% geometry."

        , name, geoname).c_str(), py::init<std::string>(py::arg("name")=""));
    METHOD(compute, compute, "Run electrical calculations", py::arg("loops")=0);
    METHOD(get_total_current, getTotalCurrent, "Get total current flowing through active region [mA]", py::arg("nact")=0);
    RO_PROPERTY(err, getErr, "Maximum estimated error");
    RECEIVER(inWavelength, "It is required only if :attr:`heat` is equal to *wavelength*.");
    RECEIVER(inTemperature, "");
    PROVIDER(outPotential, "");
    PROVIDER(outCurrentDensity, "");
    PROVIDER(outHeat, "");
    PROVIDER(outConductivity, "");
    BOUNDARY_CONDITIONS(voltage_boundary, "Boundary conditions of the first kind (constant potential)");
    RW_FIELD(maxerr, "Limit for the potential updates");
    RW_FIELD(algorithm, "Chosen matrix factorization algorithm");
    solver.def_readwrite("heat", &__Class__::heatmet, "Chosen method used for computing heats");
    RW_PROPERTY(beta, getBeta, setBeta, "Junction coefficient [1/V]");
    RW_PROPERTY(Vt, getVt, setVt, "Junction thermal voltage [V]");
    RW_PROPERTY(js, getJs, setJs, "Reverse bias current density [A/m²]");
    RW_PROPERTY(pcond, getCondPcontact, setCondPcontact, "Conductivity of the p-contact");
    RW_PROPERTY(ncond, getCondNcontact, setCondNcontact, "Conductivity of the n-contact");
    solver.add_property("pnjcond", &getCondJunc<__Class__>, &setCondJunc<__Class__>, "Effective conductivity of the p-n junction");
    solver.setattr("outVoltage", solver.attr("outPotential"));
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
}

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(fem)
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

    register_electrical_solver<Geometry2DCartesian>("Shockley2D", "Cartesian");

    register_electrical_solver<Geometry2DCylindrical>("ShockleyCyl", "cylindrical");
}

