#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../femV.h"
using namespace plask::solvers::electrical;

static shared_ptr<SolverOver<Geometry2DCartesian>> DriftDiffusion2D(const std::string& name) {
    throw NotImplemented("DriftDiffusion2D: I want it to be implemented too!");
}

static shared_ptr<SolverOver<Geometry2DCylindrical>> DriftDiffusionCyl(const std::string& name) {
    throw NotImplemented("DriftDiffusionCyl: I want it to be implemented too!");
}

template <typename Cls>
static DataVectorWrap<const double,2> getCondJunc(const Cls* self) {
    auto midmesh = self->getMesh()->getMidpointsMesh();
    RectilinearMesh1D line1;
    for (size_t n = 0; n < self->getActNo(); ++n)
        line1.addPoint(self->getMesh()->axis1[(self->getActLo(n)+self->getActHi(n))/2]);
    auto mesh = make_shared<RectilinearMesh2D>(midmesh->axis0, line1);
    return DataVectorWrap<const double,2>(self->getCondJunc(), mesh);
}

template <typename Cls>
static void setCondJunc(Cls* self, py::object value) {
    size_t len = self->getMesh()->axis0.size()-1;
    try {
        double val = py::extract<double>(value);
        self->setCondJunc(val);
        return;
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    try {
        const DataVectorWrap<const double,2>& val = py::extract<DataVectorWrap<const double,2>&>(value);
        {
            auto mesh = dynamic_pointer_cast<RectilinearMesh2D>(val.mesh);
            if (mesh && mesh->axis1.size() == self->getActNo() && val.size() == len) {
                self->setCondJunc(val);
                return;
            }
        }{
            auto mesh = dynamic_pointer_cast<RegularMesh2D>(val.mesh);
            if (mesh && mesh->axis1.size() == self->getActNo() && val.size() == len) {
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

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(fem)
{
    py_enum<Algorithm>("Algorithm", "Algorithms used for matrix factorization")
        .value("CHOLESKY", ALGORITHM_CHOLESKY)
        .value("GAUSS", ALGORITHM_GAUSS)
        .value("ITERATIVE", ALGORITHM_ITERATIVE)
    ;

    py_enum<CorrectionType>("CorrectionType", "Types of the returned correction")
        .value("ABSOLUTE", CORRECTION_ABSOLUTE)
        .value("RELATIVE", CORRECTION_RELATIVE)
    ;

    py_enum<HeatMethod>("HeatType", "Methods used for computing heats")
        .value("JOULES", HEAT_JOULES)
        .value("WAVELENGTH", HEAT_BANDGAP)
    ;

    {CLASS(FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>, "Beta2D", "Finite element thermal solver for 2D Cartesian Geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("loops")=0);
        METHOD(get_total_current, getTotalCurrent, "Get total current flowing through active region [mA]", py::arg("nact")=0);
        RO_PROPERTY(abscorr, getMaxAbsVCorr, "Maximum absolute correction for potential");
        RO_PROPERTY(relcorr, getMaxRelVCorr, "Maximum relative correction for potential");
        RECEIVER(inWavelength, "Wavelength specifying the bad-gap");
        RECEIVER(inTemperature, "Temperatures");
        PROVIDER(outPotential, "Potentials");
        PROVIDER(outCurrentDensity, "CurrentDensities");
        PROVIDER(outHeatDensity, "HeatDensities");
        solver.add_boundary_conditions("voltage_boundary", &__Class__::mVConst, "Boundary conditions of the first kind (constant potential)");
        RW_PROPERTY(corrlim, getVCorrLim, setVCorrLim, "Limit for the potential updates");
        solver.def_readwrite("corrtype", &__Class__::mCorrType, "Type of returned correction");
        solver.def_readwrite("algorithm", &__Class__::mAlgorithm, "Chosen matrix factorization algorithm");
        solver.def_readwrite("heat", &__Class__::mHeatMethod, "Chosen method used for computing heats");
        RW_PROPERTY(beta, getBeta, setBeta, "Junction coefficient");
        RW_PROPERTY(js, getJs, setJs, "Reverse bias current density [A/m²]");
        RW_PROPERTY(pcond, getCondPcontact, setCondPcontact, "Conductivity of the p-contact");
        RW_PROPERTY(ncond, getCondNcontact, setCondNcontact, "Conductivity of the n-contact");
        solver.add_property("pnjcond", &getCondJunc<__Class__>, &setCondJunc<__Class__>, "Effective conductivity of the p-n junction");
        solver.setattr("outVoltage", solver.attr("outPotential"));
        solver.def_readwrite("itererr", &__Class__::mIterErr, "Allowed residual iteration for iterative method");
        solver.def_readwrite("iterlim", &__Class__::mIterLim ,"Maximum number of iterations for iterative method");
        solver.def_readwrite("logfreq", &__Class__::mLogFreq ,"Frequency of iteration progress reporting");
    }

    {CLASS(FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>, "BetaCyl", "Finite element thermal solver for 2D Cylindrical Geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("loops")=0);
        METHOD(get_total_current, getTotalCurrent, "Get total current flowing through active region [mA]", py::arg("nact")=0);
        RO_PROPERTY(abscorr, getMaxAbsVCorr, "Maximum absolute correction for potential");
        RO_PROPERTY(relcorr, getMaxRelVCorr, "Maximum relative correction for potential");
        RECEIVER(inWavelength, "Wavelength specifying the bad-gap");
        RECEIVER(inTemperature, "Temperatures");
        PROVIDER(outPotential, "Potentials");
        PROVIDER(outCurrentDensity, "CurrentDensities");
        PROVIDER(outHeatDensity, "HeatDensities");
        solver.add_boundary_conditions("voltage_boundary", &__Class__::mVConst, "Boundary conditions of the first kind (constant potential)");
        RW_PROPERTY(corrlim, getVCorrLim, setVCorrLim, "Limit for the potential updates");
        solver.def_readwrite("corrtype", &__Class__::mCorrType, "Type of returned correction");
        solver.def_readwrite("algorithm", &__Class__::mAlgorithm, "Chosen matrix factorization algorithm");
        solver.def_readwrite("heat", &__Class__::mHeatMethod, "Chosen method used for computing heats");
        RW_PROPERTY(beta, getBeta, setBeta, "Junction coefficient");
        RW_PROPERTY(js, getJs, setJs, "Reverse bias current density [A/m²]");
        RW_PROPERTY(pcond, getCondPcontact, setCondPcontact, "Conductivity of the p-contact");
        RW_PROPERTY(ncond, getCondNcontact, setCondNcontact, "Conductivity of the n-contact");
        solver.add_property("pnjcond", &getCondJunc<__Class__>, &setCondJunc<__Class__>, "Effective conductivity of the p-n junction");
        solver.setattr("outVoltage", solver.attr("outPotential"));
        solver.def_readwrite("itererr", &__Class__::mIterErr, "Allowed residual iteration for iterative method");
        solver.def_readwrite("iterlim", &__Class__::mIterLim ,"Maximum number of iterations for iterative method");
        solver.def_readwrite("logfreq", &__Class__::mLogFreq ,"Frequency of iteration progress reporting");
    }

    py::def("DriftDiffusion2D", DriftDiffusion2D, py::arg("name")="");
    py::def("DriftDiffusionCyl", DriftDiffusionCyl, py::arg("name")="");
}

