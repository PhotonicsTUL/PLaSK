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
#define PY_ARRAY_UNIQUE_SYMBOL PLASK_OPTICAL_SLAB_ARRAY_API
#define NO_IMPORT_ARRAY

#include <plask/python_numpy.hpp>

#include "linescyl-python.hpp"
#include "slab-python.hpp"

namespace plask { namespace optical { namespace slab { namespace python {

template <>
py::object Eigenmodes<LinesSolverCyl>::array(const dcomplex* data, size_t N) const {
    const int dim = 2, strid = 2;
    npy_intp dims[] = { npy_intp(N / strid), npy_intp(strid) };
    npy_intp strides[] = { strid * sizeof(dcomplex), sizeof(dcomplex) };
    PyObject* arr = PyArray_New(&PyArray_Type, dim, dims, NPY_CDOUBLE, strides, (void*)data, 0, 0, NULL);
    if (arr == nullptr) throw plask::CriticalException("Cannot create array");
    return py::object(py::handle<>(arr));
}

std::string LinesSolverCyl_Mode_str(const LinesSolverCyl::Mode& self) {
    return format(u8"<m: {:d}, lam: {}nm, power: {:.2g}mW>", self.m, str(2e3*PI/self.k0, u8"({:.3f}{:+.3g}j)"), self.power);
}
std::string LinesSolverCyl_Mode_repr(const LinesSolverCyl::Mode& self) {
    return format(u8"LinesCyl.Mode(m={:d}, lam={}, power={:g})", self.m, str(2e3*PI/self.k0), self.power);
}

py::object LinesSolverCyl_getDeterminant(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError(u8"get_determinant() takes exactly one non-keyword argument ({0} given)", py::len(args));
    LinesSolverCyl* self = py::extract<LinesSolverCyl*>(args[0]);

    enum What {
        WHAT_NOTHING = 0,
        WHAT_WAVELENGTH,
        WHAT_K0,
    };
    What what = WHAT_NOTHING;
    py::object array;
    int m = self->getM();

    plask::optional<dcomplex> k0;
    py::stl_input_iterator<std::string> begin(kwargs), end;
    for (auto i = begin; i != end; ++i) {
        if (*i == "lam" || *i == "wavelength") {
            if (what == WHAT_K0 || k0)
                throw BadInput(self->getId(), u8"'lam' and 'k0' are mutually exclusive");
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError(u8"Only one key may be an array");
                what = WHAT_WAVELENGTH; array = kwargs[*i];
            } else
                k0.reset(2e3*PI / py::extract<dcomplex>(kwargs[*i])());
        } else if (*i == "kNumpyDataDeleter0") {
            if (what == WHAT_WAVELENGTH || k0)
                throw BadInput(self->getId(), u8"'lam' and 'k0' are mutually exclusive");
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError(u8"Only one key may be an array");
                what = WHAT_K0; array = kwargs[*i];
            } else
                k0.reset(py::extract<dcomplex>(kwargs[*i]));
        } else if (*i == "m") {
            m = py::extract<int>(kwargs[*i]);
        } else
            throw TypeError(u8"get_determinant() got unexpected keyword argument '{0}'", *i);
    }

    self->Solver::initCalculation();

    if (k0) self->expansion.setK0(*k0);
    self->expansion.setM(m);

    switch (what) {
        case WHAT_NOTHING:
            if (!k0) self->expansion.setK0(self->getK0());
            return py::object(self->getDeterminant());
        case WHAT_WAVELENGTH:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->expansion.setK0(2e3*PI / x); return self->getDeterminant(); },
                array,
                "LinesSolverCyl.getDeterminant",
                "lam"
            );
        case WHAT_K0:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->expansion.setK0(x); return self->getDeterminant(); },
                array,
                "LinesSolverCyl.getDeterminant",
                "k0"
            );
    }
    return py::object();
}

static size_t LinesSolverCyl_findMode(LinesSolverCyl& self, dcomplex start, const py::object& pym) {
    int m;
    if (pym == py::object()) {
        m = self.getM();
    } else {
        m = py::extract<int>(pym);
    }
    return self.findMode(start, m);
}

static size_t LinesSolverCyl_setMode(LinesSolverCyl* self, dcomplex lam, const py::object& pym) {
    self->Solver::initCalculation();

    self->expansion.setK0(2e3*PI / lam);

    if (pym != py::object()) {
        self->expansion.setM(py::extract<int>(pym));
    }

    return self->setMode();
}

static py::object LinesSolverCyl_getFieldVectorE(LinesSolverCyl& self, int num, double z) {
    if (num < 0) num += int(self.modes.size());
    if (std::size_t(num) >= self.modes.size()) throw IndexError(u8"Bad mode number {:d}", num);
    return arrayFromVec2D<NPY_CDOUBLE>(self.getFieldVectorE(num, z), false, 2);
}

static py::object LinesSolverCyl_getFieldVectorH(LinesSolverCyl& self, int num, double z) {
    if (num < 0) num += int(self.modes.size());
    if (std::size_t(num) >= self.modes.size()) throw IndexError(u8"Bad mode number {:d}", num);
    return arrayFromVec2D<NPY_CDOUBLE>(self.getFieldVectorH(num, z), false, 2);
}


void export_LinesSolverCyl()
{
    CLASS(LinesSolverCyl, "LinesCyl",
        u8"Optical Solver using finite differences in cylindrical coordinates.\n\n"
        u8"It calculates optical modes and optical field distribution using method of lines\n"
        u8"and reflection transfer in two-dimensional cylindrical space.")
        export_base(solver);
    PROVIDER(outLoss, "");
    solver.add_property("lam", &__Class__::getLam, &Solver_setLam<__Class__>,
                u8"Wavelength of the light [nm].\n");
    solver.add_property("wavelength", &__Class__::getLam, &Solver_setLam<__Class__>,
                u8"Alias for :attr:`lam`");
    solver.add_property("k0", &__Class__::getK0, &Solver_setK0<__Class__>,
                u8"Normalized frequency of the light [1/µm].\n");
    solver.add_property("m", &__Class__::getM, &__Class__::setM, "Angular dependence parameter.");
    solver.def("find_mode", &LinesSolverCyl_findMode,
           u8"Compute the mode near the specified effective index.\n\n"
           u8"Only one of the following arguments can be given through a keyword.\n"
           u8"It is the starting point for search of the specified parameter.\n\n"
           u8"Args:\n"
           u8"    lam (complex): Starting wavelength.\n"
           u8"    m (int): HE/EH Mode angular number. If ``None``, use :attr:`m` attribute.\n",
           (arg("lam"), arg("m")=py::object())
          );
    solver.def("set_mode", &LinesSolverCyl_setMode,
                u8"Set the mode for specified parameters.\n\n"
                u8"This method should be used if you have found a mode manually and want to insert\n"
                u8"it into the solver in order to determine the fields. Calling this will raise an\n"
                u8"exception if the determinant for the specified parameters is too large.\n\n"
                u8"Arguments can be given through keywords only.\n\n"
                u8"Args:\n"
                u8"    lam (complex): Wavelength.\n"
                u8"    m (int): HE/EH Mode angular number.\n"
              );
    RW_FIELD(emission, "Direction of the useful light emission.\n\n"
                       u8"Necessary for the over-threshold model to correctly compute the output power.\n"
                       u8"Currently the fields are normalized only if this parameter is set to\n"
                       u8"``top`` or ``bottom``. Otherwise, it is ``undefined`` (default) and the fields\n"
                       u8"are not normalized.");
    solver.def("get_determinant", py::raw_function(LinesSolverCyl_getDeterminant),
               u8"Compute discontinuity matrix determinant.\n\n"
               u8"Arguments can be given through keywords only.\n\n"
               u8"Args:\n"
               u8"    lam (complex): Wavelength.\n"
               u8"    k0 (complex): Normalized frequency.\n"
               u8"    m (int): HE/EH Mode angular number.\n"
              );
    solver.def("compute_reflectivity", &Solver_computeReflectivity_index<LinesSolverCyl>,
               (py::arg("lam"), "side", "index"));
    solver.def("compute_reflectivity", &Solver_computeReflectivity_array<LinesSolverCyl>,
               (py::arg("lam"), "side", "coffs"),
               u8"Compute reflection coefficient on planar incidence [%].\n\n"
               u8"Args:\n"
               u8"    lam (float or array of floats): Incident light wavelength.\n"
               u8"    side (`top` or `bottom`): Side of the structure where the incident light is\n"
               u8"        present.\n"
               u8"    idx: Eigenmode number.\n"
               u8"    coeffs: expansion coefficients of the incident vector.\n");
    solver.def("compute_transmittivity", &Solver_computeTransmittivity_index<LinesSolverCyl>,
               (py::arg("lam"), "side", "index"));
    solver.def("compute_transmittivity", &Solver_computeTransmittivity_array<LinesSolverCyl>,
               (py::arg("lam"), "side", "coffs"),
               u8"Compute transmission coefficient on planar incidence [%].\n\n"
               u8"Args:\n"
               u8"    lam (float or array of floats): Incident light wavelength.\n"
               u8"    side (`top` or `bottom`): Side of the structure where the incident light is\n"
               u8"        present.\n"
               u8"    idx: Eigenmode number.\n"
               u8"    coeffs: expansion coefficients of the incident vector.\n");
    solver.def("scattering", Scattering<LinesSolverCyl>::from_index, py::with_custodian_and_ward_postcall<0,1>(), (py::arg("side"), "idx"));
    solver.def("scattering", Scattering<LinesSolverCyl>::from_array, py::with_custodian_and_ward_postcall<0,1>(), (py::arg("side"), "coeffs"),
               u8"Access to the reflected field.\n\n"
               u8"Args:\n"
               u8"    side (`top` or `bottom`): Side of the structure where the incident light is\n"
               u8"        present.\n"
               u8"    polarization: Specification of the incident light polarization.\n"
               u8"        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
               u8"        of the non-vanishing electric field component.\n"
               u8"    idx: Eigenmode number.\n"
               u8"    coeffs: expansion coefficients of the incident vector.\n\n"
               u8":rtype: Fourier2D.Scattering\n"
              );
    solver.def("get_raw_E", LinesSolverCyl_getFieldVectorE, (py::arg("num"), "level"),
               u8"Get Lines expansion coefficients for the electric field.\n\n"
               u8"This is a low-level function returning :math:`E_s` and :math:`E_p` Lines\n"
               u8"expansion coefficients. Please refer to the detailed solver description for their\n"
               u8"interpretation.\n\n"
               u8"Args:\n"
               u8"    num (int): Computed mode number.\n"
               u8"    level (float): Vertical lever at which the coefficients are computed.\n\n"
               u8":rtype: numpy.ndarray\n"
              );
    solver.def("get_raw_H", LinesSolverCyl_getFieldVectorH, (py::arg("num"), "level"),
               u8"Get Lines expansion coefficients for the magnetic field.\n\n"
               u8"This is a low-level function returning :math:`H_s` and :math:`H_p` Lines\n"
               u8"expansion coefficients. Please refer to the detailed solver description for their\n"
               u8"interpretation.\n\n"
               u8"Args:\n"
               u8"    num (int): Computed mode number.\n"
               u8"    level (float): Vertical lever at which the coefficients are computed.\n\n"
               u8":rtype: numpy.ndarray\n"
              );
    solver.add_property("pml", py::make_function(&Solver_getPML<LinesSolverCyl>, py::with_custodian_and_ward_postcall<0,1>()),
                        &Solver_setPML<LinesSolverCyl>,
                        "Side Perfectly Matched Layers boundary conditions.\n\n"
                        PML_ATTRS_DOC
                       );
    RO_FIELD(modes, "Computed modes.");

    solver.def("layer_eigenmodes", &Eigenmodes<LinesSolverCyl>::fromZ, py::arg("level"),
        u8"Get eignemodes for a layer at specified level.\n\n"
        u8"This is a low-level function to access diagonalized eigenmodes for a specific\n"
        u8"layer. Please refer to the detailed solver description for the interpretation\n"
        u8"of the returned values.\n\n"
        u8"Args:\n"
        u8"    level (float): Vertical level at which the coefficients are computed.\n\n"
        u8":rtype: :class:`~optical.slab.LinesCyl.Eigenmodes`\n",
        py::with_custodian_and_ward_postcall<0, 1>()
    );

    py::scope scope = solver;
    (void) scope;   // don't warn about unused variable scope

    register_vector_of<LinesSolverCyl::Mode>("Modes");
    py::class_<LinesSolverCyl::Mode>("Mode", u8"Detailed information about the mode.", py::no_init)
        .add_property("lam", &getModeWavelength<LinesSolverCyl::Mode>, u8"Mode wavelength [nm].")
        .add_property("loss", &getModeLoss<LinesSolverCyl::Mode>, u8"Mode loss [1/cm].")
        .add_property("wavelength", &getModeWavelength<LinesSolverCyl::Mode>, u8"Mode wavelength [nm].")
        .def_readonly("k0", &LinesSolverCyl::Mode::k0, u8"Mode normalized frequency [1/µm].")
        .def_readonly("m", &LinesSolverCyl::Mode::m, u8"Angular mode order.")
        .def_readwrite("power", &LinesSolverCyl::Mode::power, u8"Total power emitted into the mode.")
        .def("__str__", &LinesSolverCyl_Mode_str)
        .def("__repr__", &LinesSolverCyl_Mode_repr)
    ;

    Eigenmodes<LinesSolverCyl>::registerClass("LinesCyl", "Cyl");
}

}}}} // namespace plask::optical::slab::python
