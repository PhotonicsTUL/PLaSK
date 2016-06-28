#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../electr3d.h"
using namespace plask::solvers::electrical3d;

static py::object outPotential(const py::object& self) {
    throw TypeError("{}: 'outPotential' is reserved for drift-diffusion model; use 'outVoltage' instead",
                    std::string(py::extract<std::string>(self.attr("id"))));
    return py::object();
}

static DataVectorWrap<const double,3> getCondJunc(const FiniteElementMethodElectrical3DSolver* self) {
    if (self->getMesh() && self->getGeometry()) {
        auto midmesh = self->getMesh()->getMidpointsMesh();
        shared_ptr<OrderedAxis> line2 = plask::make_shared<OrderedAxis>();
        for (size_t n = 0; n < self->getActNo(); ++n)
            line2->addPoint(self->getMesh()->axis1->at((self->getActLo(n)+self->getActHi(n))/2));
        auto mesh = plask::make_shared<RectangularMesh<3>>(midmesh->axis0->clone(), midmesh->axis1->clone(), line2);
        return DataVectorWrap<const double,3>(self->getCondJunc(), mesh);
    } else {
        auto mesh = plask::make_shared<RectangularMesh<3>>(plask::make_shared<OrderedAxis>(std::initializer_list<double>{NAN}),
                                                    plask::make_shared<OrderedAxis>(std::initializer_list<double>{NAN}),
                                                    plask::make_shared<OrderedAxis>(std::initializer_list<double>{NAN}));
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
    size_t len = (self->getMesh()->axis0->size()-1) * (self->getMesh()->axis1->size()-1);
    try {
        const DataVectorWrap<const double,3>& val = py::extract<DataVectorWrap<const double,3>&>(value);
        {
            auto mesh = dynamic_pointer_cast<RectangularMesh<3>>(val.mesh);
            if (mesh && mesh->axis2->size() == self->getActNo() && val.size() == len) {
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


BOOST_PYTHON_MODULE(fem3d)
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

    {CLASS(FiniteElementMethodElectrical3DSolver, "Shockley3D", u8"Finite element thermal solver for 3D Geometry.")
        METHOD(compute, compute, u8"Run thermal calculations", py::arg("loops")=0);
        METHOD(get_total_current, getTotalCurrent, u8"Get total current flowing through active region [mA]", py::arg("nact")=0);
        RO_PROPERTY(err, getErr, u8"Maximum estimated error");
        RECEIVER(inWavelength, u8"It is required only if :attr:`heat` is eual to *wavelength*.");
        RECEIVER(inTemperature, u8"");
        PROVIDER(outVoltage, u8"");
        PROVIDER(outCurrentDensity, u8"");
        PROVIDER(outHeat, u8"");
        PROVIDER(outConductivity, u8"");
        BOUNDARY_CONDITIONS(voltage_boundary, u8"Boundary conditions of the first kind (constant potential)");
        RW_FIELD(maxerr, u8"Limit for the potential updates");
        RW_PROPERTY(algorithm, getAlgorithm, setAlgorithm, u8"Chosen matrix factorization algorithm");
        solver.def_readwrite("heat", &__Class__::heatmet, u8"Chosen method used for computing heats");
        RW_PROPERTY(beta, getBeta, setBeta, u8"Junction coefficient [1/V]");
        RW_PROPERTY(Vt, getVt, setVt, u8"Junction thermal voltage [V]");
        RW_PROPERTY(js, getJs, setJs, u8"Reverse bias current density [A/m²]");
        RW_PROPERTY(pcond, getPcond, setPcond, u8"Conductivity of the p-contact");
        RW_PROPERTY(ncond, getNcond, setNcond, u8"Conductivity of the n-contact");
        solver.add_property("pnjcond", &getCondJunc, &setCondJunc, u8"Effective conductivity of the p-n junction");
        solver.add_property("outPotential", outPotential, u8"Removed: use :attr:`outVoltage` instead.");
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

}
