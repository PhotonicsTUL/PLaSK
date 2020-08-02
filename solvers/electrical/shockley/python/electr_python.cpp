#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../beta.h"
#include "../electr2d.h"
#include "../electr3d.h"
using namespace plask::electrical::shockley;

static py::object outPotential(const py::object& self) {
    throw TypeError(u8"{}: 'outPotential' is reserved for drift-diffusion model; use 'outVoltage' instead",
                    std::string(py::extract<std::string>(self.attr("id"))));
    return py::object();
}

template <typename GeometryT>
struct Shockley: BetaSolver<GeometryT> {

    std::vector<py::object> beta_function, js_function;

    Shockley(const std::string& id=""): BetaSolver<GeometryT>(id) {}

    py::object getBeta0() const { return getBeta(0); }
    void setBeta0(const py::object& value) { setBeta(0, value); }

    py::object getJs0() const { return getJs(0); }
    void setJs0(const py::object& value) { setJs(0, value); }

    py::object __getattr__(const std::string& attr) const {
        try {
            if (attr.substr(0, 4) == "beta") return py::object(getBeta(boost::lexical_cast<size_t>(attr.substr(4))));
            if (attr.substr(0, 2) == "js") return py::object(getJs(boost::lexical_cast<size_t>(attr.substr(2))));
        } catch (boost::bad_lexical_cast&) {
            throw AttributeError(u8"{0} object has no attribute '{1}'", this->getClassName(), attr);
        }
        return py::object();
    }

    py::object getBeta(size_t n) const {
        if (n < beta_function.size() && !beta_function[n].is_none()) return beta_function[n];
        return py::object(BetaSolver<GeometryT>::getBeta(n));
    }

    void setBeta(size_t n, const py::object& value) {
        py::extract<double> val(value);
        if (val.check()) {
            BetaSolver<GeometryT>::setBeta(n, val());
        } else if (PyCallable_Check(value.ptr())) {
            if (this->beta_function.size() <= n) this->beta_function.resize(n + 1);
            beta_function[n] = value;
            this->invalidate();
        } else {
            throw TypeError("{}: beta{} must be a float or a callable", this->getId(), n);
        }
    }

    py::object getJs(size_t n) const {
        if (n < js_function.size() && !js_function[n].is_none()) return js_function[n];
        return py::object(BetaSolver<GeometryT>::getJs(n));
    }

    void setJs(size_t n, const py::object& value) {
        py::extract<double> val(value);
        if (val.check()) {
            BetaSolver<GeometryT>::setJs(n, val());
        } else if (PyCallable_Check(value.ptr())) {
            if (this->js_function.size() <= n) this->js_function.resize(n + 1);
            js_function[n] = value;
            this->invalidate();
        } else {
            throw TypeError("{}: js{} must be a float or a callable", this->getId(), n);
        }
    }

    double activeVoltage(size_t n, double jy, double T) override {
        double beta = (n < beta_function.size() && !beta_function[n].is_none()) ? py::extract<double>(beta_function[n](T))
                                                                                : BetaSolver<GeometryT>::getBeta(n);
        double js = (n < js_function.size() && !js_function[n].is_none()) ? py::extract<double>(js_function[n](T))
                                                                          : BetaSolver<GeometryT>::getJs(n);
        return log(1e7 * jy / js + 1.) / beta;
    }
};

template <typename Class> void Shockley__setattr__(const py::object& oself, const std::string& attr, const py::object& value) {
    Class& self = py::extract<Class&>(oself);

    try {
        if (attr.substr(0, 4) == "beta") {
            self.setBeta(boost::lexical_cast<size_t>(attr.substr(4)), value);
            return;
        }
        if (attr.substr(0, 2) == "js") {
            self.setJs(boost::lexical_cast<size_t>(attr.substr(2)), value);
            return;
        }
    } catch (boost::bad_lexical_cast&) {
    }

    oself.attr("__class__").attr("__base__").attr("__setattr__")(oself, attr, value);
}

template <typename __Class__> inline static void register_electrical_solver(const char* name, const char* geoname) {
    ExportSolver<__Class__> solver(name,
                                   format(
                                       u8"{0}(name=\"\")\n\n"
                                       u8"Finite element thermal solver for {1} geometry.",
                                       name, geoname)
                                       .c_str(),
                                   py::init<std::string>(py::arg("name") = ""));
    METHOD(compute, compute, u8"Run electrical calculations", py::arg("loops") = 0);
    METHOD(get_total_current, getTotalCurrent, u8"Get total current flowing through active region [mA]", py::arg("nact") = 0);
    RO_PROPERTY(err, getErr, u8"Maximum estimated error");
    RECEIVER(inTemperature, u8"");
    PROVIDER(outVoltage, u8"");
    PROVIDER(outCurrentDensity, u8"");
    PROVIDER(outHeat, u8"");
    PROVIDER(outConductivity, u8"");
    BOUNDARY_CONDITIONS(voltage_boundary, u8"Boundary conditions of the first kind (constant potential)");
    RW_FIELD(maxerr, u8"Limit for the potential updates");
    RW_FIELD(algorithm, u8"Chosen matrix factorization algorithm");
    RW_PROPERTY(include_empty, usingFullMesh, useFullMesh, "Should empty regions (e.g. air) be included into computation domain?");
    solver.add_property("beta", &__Class__::getBeta0, &__Class__::setBeta0,
                        u8"Junction coefficient [1/V].\n\n"
                        u8"In case there is more than one junction you may set $\\beta$ parameter for any\n"
                        u8"of them by using ``beta#`` property, where # is the junction number (specified\n"
                        u8"by a role ``junction#`` or ``active#``).\n\n"
                        u8"``beta`` is an alias for ``beta0``.\n");
    solver.add_property("js", &__Class__::getJs0, &__Class__::setJs0,
                        u8"Reverse bias current density [A/m\\ :sup:`2`\\ ].\n\n"
                        u8"In case there is more than one junction you may set $j_s$ parameter for any\n"
                        u8"of them by using ``js#`` property, where # is the junction number (specified\n"
                        u8"by a role ``junction#`` or ``active#``).\n\n"
                        u8"``js`` is an alias for ``js0``.\n");
    solver.def("__getattr__", &__Class__::__getattr__);
    solver.def("__setattr__", &Shockley__setattr__<__Class__>);
    RW_PROPERTY(pcond, getCondPcontact, setCondPcontact, u8"Conductivity of the p-contact");
    RW_PROPERTY(ncond, getCondNcontact, setCondNcontact, u8"Conductivity of the n-contact");
    solver.add_property("pnjcond", &__Class__::getCondJunc, (void (__Class__::*)(double)) & __Class__::setCondJunc,
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
           u8"    Total electrostatic energy [J].\n");
    METHOD(get_capacitance, getCapacitance,
           u8"Get the structure capacitance.\n\n"
           u8"Return:\n"
           u8"    Total capacitance [pF].\n\n"
           u8"Note:\n"
           u8"    This method can only be used it there are exactly two boundary conditions\n"
           u8"    specifying the voltage. Otherwise use :meth:`get_electrostatic_energy` to\n"
           u8"    obtain the stored energy :math:`W` and compute the capacitance as:\n"
           u8"    :math:`C = 2 \\, W / U^2`, where :math:`U` is the applied voltage.\n");
    METHOD(get_total_heat, getTotalHeat,
           u8"Get the total heat produced by the current flowing in the structure.\n\n"
           u8"Return:\n"
           u8"    Total produced heat [mW].\n");
}

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(shockley) {
    py_enum<Algorithm>()
        .value("CHOLESKY", ALGORITHM_CHOLESKY)
        .value("GAUSS", ALGORITHM_GAUSS)
        .value("ITERATIVE", ALGORITHM_ITERATIVE);

    register_electrical_solver<Shockley<Geometry2DCartesian>>("Shockley2D", "2D Cartesian");

    register_electrical_solver<Shockley<Geometry2DCylindrical>>("ShockleyCyl", "2D cylindrical");

    register_electrical_solver<Shockley<Geometry3D>>("Shockley3D", "3D Cartesian");
}
