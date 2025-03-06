/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include <cmath>
#include <plask/python.hpp>
#include <plask/common/fem/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../beta.hpp"
#include "../electr2d.hpp"
#include "../electr3d.hpp"
using namespace plask::electrical::shockley;

static py::object outPotential(const py::object& self) {
    throw TypeError(u8"{}: 'outPotential' is reserved for drift-diffusion model; use 'outVoltage' instead",
                    std::string(py::extract<std::string>(self.attr("id"))));
    return py::object();
}

template <typename GeometryT> struct Shockley : BetaSolver<GeometryT> {
    std::vector<py::object> beta_function, js_function;

    Shockley(const std::string& id = "") : BetaSolver<GeometryT>(id) {}

    py::object getBeta0() const { return getBeta(0); }
    void setBeta0(const py::object& value) { setBeta(0, value); }

    py::object getJs0() const { return getJs(0); }
    void setJs0(const py::object& value) { setJs(0, value); }

    py::object __getattr__(const std::string& attr) const {
        try {
            if (attr.substr(0, 4) == "beta") return py::object(getBeta(boost::lexical_cast<size_t>(attr.substr(4))));
            if (attr.substr(0, 2) == "js") return py::object(getJs(boost::lexical_cast<size_t>(attr.substr(2))));
        } catch (boost::bad_lexical_cast&) {
        }
        throw AttributeError(u8"'{0}' object has no attribute '{1}'", this->getClassName(), attr);
    }

    static void __setattr__(const py::object& oself, const std::string& attr, const py::object& value) {
        Shockley<GeometryT>& self = py::extract<Shockley<GeometryT>&>(oself);
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

    Tensor2<double> activeCond(size_t n, double U, double jy, double T) override {
        double beta = (n < beta_function.size() && !beta_function[n].is_none()) ? py::extract<double>(beta_function[n](T))
                                                                                : BetaSolver<GeometryT>::getBeta(n);
        double js = (n < js_function.size() && !js_function[n].is_none()) ? py::extract<double>(js_function[n](T))
                                                                          : BetaSolver<GeometryT>::getJs(n);
        jy = abs(jy);
        return Tensor2<double>(0., 10. * jy * beta * this->active[n].height / log(1e7 * jy / js + 1.));
    }
};

template <typename GeometryT>
struct PythonCondSolver : public std::conditional<std::is_same<GeometryT, Geometry3D>::value,
                                                  ElectricalFem3DSolver,
                                                  ElectricalFem2DSolver<GeometryT>>::type {
    typedef typename std::conditional<std::is_same<GeometryT, Geometry3D>::value,
                                      ElectricalFem3DSolver,
                                      ElectricalFem2DSolver<GeometryT>>::type BaseClass;

    std::vector<py::object> cond_function;

    PythonCondSolver(const std::string& id = "") : BaseClass(id) {}

    py::object getCond0() const { return getCond(0); }
    void setCond0(const py::object& value) { setCond(0, value); }

    py::object getCond(size_t n) const {
        if (n < cond_function.size()) return cond_function[n];
        return py::object();
    }

    void setCond(size_t n, const py::object& value) {
        if (PyCallable_Check(value.ptr())) {
            if (this->cond_function.size() <= n) this->cond_function.resize(n + 1);
            cond_function[n] = value;
            this->invalidate();
        } else {
            throw TypeError("{}: cond{} must be a a callable", this->getId(), n);
        }
    }

    py::object __getattr__(const std::string& attr) const {
        try {
            if (attr.substr(0, 4) == "cond") return py::object(getCond(boost::lexical_cast<size_t>(attr.substr(4))));
        } catch (boost::bad_lexical_cast&) {
        }
        throw AttributeError(u8"'{0}' object has no attribute '{1}'", this->getClassName(), attr);
    }

    static void __setattr__(const py::object& oself, const std::string& attr, const py::object& value) {
        PythonCondSolver<GeometryT>& self = py::extract<PythonCondSolver<GeometryT>&>(oself);

        try {
            if (attr.substr(0, 4) == "cond") {
                self.setCond(boost::lexical_cast<size_t>(attr.substr(4)), value);
                return;
            }
        } catch (boost::bad_lexical_cast&) {
        }

        oself.attr("__class__").attr("__base__").attr("__setattr__")(oself, attr, value);
    }

    /** Compute voltage drop of the active region
     *  \param n active region number
     *  \param U junction voltage (V)
     *  \param jy vertical current (kA/cm²)
     *  \param T temperature (K)
     */
    Tensor2<double> activeCond(size_t n, double U, double jy, double T) override {
        if (n >= this->active.size() || n >= cond_function.size() || cond_function[n].is_none())
            throw IndexError("no conductivity for active region {}", n);
        py::object cond = cond_function[n](U, jy, T);
        py::extract<double> double_cond(cond);
        if (double_cond.check()) return Tensor2<double>(0., double_cond());
        return py::extract<Tensor2<double>>(cond);
    }

    std::string getClassName() const override;
};

template <typename Class> void setCondJunc(Class& self, const py::object& value) {
    py::extract<double> double_cond(value);
    if (double_cond.check())
        self.setCondJunc(double_cond());
    else
        self.setCondJunc(py::extract<Tensor2<double>>(value)());
}

template <> std::string PythonCondSolver<Geometry2DCartesian>::getClassName() const { return "electrical.ActiveCond2D"; }
template <> std::string PythonCondSolver<Geometry2DCylindrical>::getClassName() const { return "electrical.ActiveCondCyl"; }
template <> std::string PythonCondSolver<Geometry3D>::getClassName() const { return "electrical.ActiveCond3D"; }

template <typename __Class__>
inline static ExportSolver<__Class__> register_electrical_solver(const char* name, const char* geoname) {
    ExportSolver<__Class__> solver(name,
                                   format(u8"{0}(name=\"\")\n\n"
                                          u8"Finite element thermal solver for {1} geometry.",
                                          name, geoname)
                                       .c_str(),
                                   py::init<std::string>(py::arg("name") = ""));
    METHOD(compute, compute, u8"Run electrical calculations", py::arg("loops") = 0);
    METHOD(get_total_current, getTotalCurrent, u8"Get total current flowing through active region (mA)", py::arg("nact") = 0);
    RO_PROPERTY(err, getErr, u8"Maximum estimated error");
    RECEIVER(inTemperature, u8"");
    PROVIDER(outVoltage, u8"");
    PROVIDER(outCurrentDensity, u8"");
    PROVIDER(outHeat, u8"");
    PROVIDER(outConductivity, u8"");
    BOUNDARY_CONDITIONS(voltage_boundary, u8"Boundary conditions of the first kind (constant potential)");
    RW_FIELD(maxerr, u8"Limit for the potential updates");
    RW_FIELD(convergence, u8"Convergence method.\n\nIf stable, n is slowed down to ensure stability.");
    RW_PROPERTY(pcond, getCondPcontact, setCondPcontact, u8"Conductivity of the p-contact");
    RW_PROPERTY(ncond, getCondNcontact, setCondNcontact, u8"Conductivity of the n-contact");
    solver.add_property("start_cond", &__Class__::getCondJunc, &setCondJunc<__Class__>,
                        u8"Default effective conductivity of the active region.\n\n"
                        u8"Effective junction conductivity will be computed starting from this value.\n"
                        u8"Note that the actual junction conductivity after convergence can be obtained\n"
                        u8"with :attr:`outConductivity`.");
    solver.attr("pnjcond") = solver.attr("start_cond");
    solver.add_property("outPotential", outPotential, u8"Not available in this solver. Use :attr:`outVoltage` instead.");
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
           u8"    obtain the stored energy $W$ and compute the capacitance as:\n"
           u8"    $C = 2 \\, W / U^2$, where $U$ is the applied voltage.\n");
    METHOD(get_total_heat, getTotalHeat,
           u8"Get the total heat produced by the current flowing in the structure.\n\n"
           u8"Return:\n"
           u8"    Total produced heat (mW).\n");
    registerFemSolverWithMaskedMesh(solver);

    return solver;
}

template <typename GeoT> inline static void register_shockley_solver(const char* name, const char* geoname) {
    typedef Shockley<GeoT> __Class__;
    ExportSolver<__Class__> solver = register_electrical_solver<__Class__>(name, geoname);
    solver.add_property("beta", &__Class__::getBeta0, &__Class__::setBeta0,
                        u8"Junction coefficient (1/V).\n\n"
                        u8"In case, there is more than one junction you may set $\\\\beta$ parameter for any\n"
                        u8"of them by using ``beta#`` property, where # is the junction number (specified\n"
                        u8"by a role ``junction#`` or ``active#``).\n\n"
                        u8"``beta`` is an alias for ``beta0``.\n");
    solver.add_property("js", &__Class__::getJs0, &__Class__::setJs0,
                        u8"Reverse bias current density (A/m\\ :sup:`2`\\ ).\n\n"
                        u8"In case, there is more than one junction you may set $j_s$ parameter for any\n"
                        u8"of them by using ``js#`` property, where # is the junction number (specified\n"
                        u8"by a role ``junction#`` or ``active#``).\n\n"
                        u8"``js`` is an alias for ``js0``.\n");
    solver.def("__getattr__", &__Class__::__getattr__);
    solver.def("__setattr__", &__Class__::__setattr__);
}

template <typename GeoT> inline static void register_cond_solver(const char* name, const char* geoname) {
    typedef PythonCondSolver<GeoT> __Class__;
    ExportSolver<__Class__> solver = register_electrical_solver<__Class__>(name, geoname);
    solver.add_property("cond", &__Class__::getCond0, &__Class__::setCond0,
                        u8"Junction conductivity function [S/m].\n\n"
                        u8"This function should take junction voltage (V), current density (kA/cm²)\n"
                        u8"and temperature (K) as arguments and return a conductivity [S/m]. In case,\n"
                        u8"there is more than one junction you may set such function for any of them\n"
                        u8"by using ``cond#`` property, where # is the junction number (specified by\n"
                        u8"a role ``junction#`` or ``active#``).\n\n"
                        u8"Example:\n\n"
                        u8"    >>> def cond(U, j, T):\n"
                        u8"    ...     return 1e-3 * j**2 + 1e-4 * T**2\n"
                        u8"    >>> solver.cond = cond\n\n"
                        u8"``cond`` is an alias for ``cond0``.\n");
    solver.def("__getattr__", &__Class__::__getattr__);
    solver.def("__setattr__", &__Class__::__setattr__);
}

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(shockley) {
    register_shockley_solver<Geometry2DCartesian>("Shockley2D", "2D Cartesian");
    register_shockley_solver<Geometry2DCylindrical>("ShockleyCyl", "2D cylindrical");
    register_shockley_solver<Geometry3D>("Shockley3D", "3D Cartesian");

    register_cond_solver<Geometry2DCartesian>("ActiveCond2D", "2D Cartesian");
    register_cond_solver<Geometry2DCylindrical>("ActiveCondCyl", "2D cylindrical");
    register_cond_solver<Geometry3D>("ActiveCond3D", "3D Cartesian");
}
