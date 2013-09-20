#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../electr3d.h"
using namespace plask::solvers::electrical3d;

static DataVectorWrap<const double,3> getCondJunc(const FiniteElementMethodElectrical3DSolver* self) {
    if (self->getMesh() && self->getGeometry()) {
        auto midmesh = self->getMesh()->getMidpointsMesh();
        RectilinearAxis line2;
        for (size_t n = 0; n < self->getActNo(); ++n)
            line2.addPoint(self->getMesh()->axis1[(self->getActLo(n)+self->getActHi(n))/2]);
        auto mesh = make_shared<RectilinearMesh3D>(midmesh->axis0, midmesh->axis1, line2);
        return DataVectorWrap<const double,3>(self->getCondJunc(), mesh);
    } else {
        auto mesh = make_shared<RectilinearMesh3D>(RectilinearAxis({NAN}), RectilinearAxis({NAN}), RectilinearAxis({NAN}));
        return DataVectorWrap<const double,3>(self->getCondJunc(), mesh);
    }
}

static void setCondJunc(FiniteElementMethodElectrical3DSolver* self, py::object value) {
    try {
        double val = py::extract<double>(value);
        self->setCondJunc(val);
        return;
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    if (!self->getMesh()) throw NoMeshException(self->getId());
    size_t len = (self->getMesh()->axis0.size()-1) * (self->getMesh()->axis1.size()-1);
    try {
        const DataVectorWrap<const double,3>& val = py::extract<DataVectorWrap<const double,3>&>(value);
        {
            auto mesh = dynamic_pointer_cast<RectilinearMesh3D>(val.mesh);
            if (mesh && mesh->axis2.size() == self->getActNo() && val.size() == len) {
                self->setCondJunc(val);
                return;
            }
        }{
            auto mesh = dynamic_pointer_cast<RegularMesh3D>(val.mesh);
            if (mesh && mesh->axis2.size() == self->getActNo() && val.size() == len) {
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

BOOST_PYTHON_MODULE(fem3d)
{
    py_enum<Algorithm>("Algorithm", "Algorithms used for matrix factorization")
        .value("CHOLESKY", ALGORITHM_CHOLESKY)
        .value("ITERATIVE", ALGORITHM_ITERATIVE)
    ;

    py_enum<HeatMethod>("HeatType", "Methods used for computing heats")
        .value("JOULES", HEAT_JOULES)
        .value("WAVELENGTH", HEAT_BANDGAP)
    ;

    {CLASS(FiniteElementMethodElectrical3DSolver, "Shockley3D", "Finite element thermal solver for 3D Geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("loops")=0);
        METHOD(get_total_current, getTotalCurrent, "Get total current flowing through active region [mA]", py::arg("nact")=0);
        RO_PROPERTY(err, getErr, "Maximum estimated error");
        RECEIVER(inWavelength, "Wavelength specifying the bad-gap");
        RECEIVER(inTemperature, "Receiver of temperature");
        PROVIDER(outPotential, "Provider of potential");
        PROVIDER(outCurrentDensity, "Provider of current density");
        PROVIDER(outHeat, "Provider of heat density");
        solver.setattr("outHeatDensity", solver.attr("outHeat"));
        BOUNDARY_CONDITIONS(voltage_boundary, "Boundary conditions of the first kind (constant potential)");
        RW_FIELD(maxerr, "Limit for the potential updates");
        RW_PROPERTY(algorithm, getAlgorithm, setAlgorithm, "Chosen matrix factorization algorithm");
        solver.def_readwrite("heat", &__Class__::heatmet, "Chosen method used for computing heats");
        RW_PROPERTY(beta, getBeta, setBeta, "Junction coefficient [1/V]");
        RW_PROPERTY(Vt, getVt, setVt, "Junction thermal voltage [V]");
        RW_PROPERTY(js, getJs, setJs, "Reverse bias current density [A/m²]");
        RW_PROPERTY(pcond, getPcond, setPcond, "Conductivity of the p-contact");
        RW_PROPERTY(ncond, getNcond, setNcond, "Conductivity of the n-contact");
        solver.add_property("pnjcond", &getCondJunc, &setCondJunc, "Effective conductivity of the p-n junction");
        solver.setattr("outVoltage", solver.attr("outPotential"));
        RW_FIELD(itererr, "Allowed residual iteration for iterative method");
        RW_FIELD(iterlim, "Maximum number of iterations for iterative method");
        RW_FIELD(logfreq, "Frequency of iteration progress reporting");
        py::scope().attr("Beta3D") = solver;
    }

}
