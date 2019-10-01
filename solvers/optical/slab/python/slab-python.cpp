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

#include "slab-python.hpp"
using namespace plask;
using namespace plask::python;

#include "fourier2d-python.hpp"
#include "fourier3d-python.hpp"
#include "besselcyl-python.hpp"
#include "linescyl-python.hpp"
using namespace plask::optical::slab;
using namespace plask::optical::slab::python;

#ifndef NDEBUG
template <typename T>
struct Matrix_Python {
    Matrix<T> data;
    Matrix_Python(const Matrix<T>& data): data(data) {}
    Matrix_Python(const Matrix_Python<T>& src): data(src.data) {}

    static PyObject* convert(const Matrix<T>& self) {
        npy_intp dims[2] = { static_cast<npy_intp>(self.rows()), static_cast<npy_intp>(self.cols()) };
        npy_intp strides[2] = { sizeof(dcomplex), npy_intp(self.rows() * sizeof(dcomplex)) };

        PyObject* arr = PyArray_New(&PyArray_Type, 2, dims, plask::python::detail::typenum<T>(), strides,
                                    (void*)self.data(), 0, 0, NULL);
        if (arr == nullptr) throw plask::CriticalException(u8"Cannot create array from matrix");
        // Make sure the data vector stays alive as long as the array
        py::object oself {Matrix_Python<T>(self)};
        py::incref(oself.ptr());
        PyArray_SetBaseObject((PyArrayObject*)arr, oself.ptr());
        return arr;
    }
};

template <typename T>
struct Diagonal_Python {
    MatrixDiagonal<T> data;
    Diagonal_Python(const MatrixDiagonal<T>& data): data(data) {}
    Diagonal_Python(const Diagonal_Python<T>& src): data(src.data) {}

    static PyObject* convert(const MatrixDiagonal<T>& self) {
        npy_intp dims[1] = { static_cast<npy_intp>(self.size()) };
        npy_intp strides[1] = { sizeof(dcomplex) };

        PyObject* arr = PyArray_New(&PyArray_Type, 1, dims, plask::python::detail::typenum<T>(), strides,
                                    (void*)self.data(), 0, 0, NULL);
        if (arr == nullptr) throw plask::CriticalException(u8"Cannot create array from matrix");
        // Make sure the data vector stays alive as long as the array
        py::object oself {Diagonal_Python<T>(self)};
        py::incref(oself.ptr());
        PyArray_SetBaseObject((PyArrayObject*)arr, oself.ptr());
        return arr;
    }
};
#endif


void* CoeffsArray::convertible(PyObject* obj) {
    if (PyArray_Check(obj))
        return obj;
    return nullptr;

}

void CoeffsArray::construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
    PyArrayObject* arr = (PyArrayObject*)obj;
    if (PyArray_NDIM(arr) != 1)
        throw TypeError("only rank 1 arrays allowed");
    size_t size = PyArray_DIMS(arr)[0];

    if (PyArray_TYPE(arr) != NPY_CDOUBLE || PyArray_STRIDES(arr)[0] != sizeof(dcomplex)) {
    writelog(LOG_DEBUG, u8"Copying numpy array to make it contiguous");
        npy_intp sizes[] = { (npy_int)size };
        npy_intp strides[] = { sizeof(dcomplex) };
        PyObject* newarr = PyArray_New(&PyArray_Type, 1, sizes,
                                        PyArray_TYPE(arr), strides,
                                        nullptr, 0, 0, nullptr);
        PyArray_CopyInto((PyArrayObject*)newarr, arr);
        arr = (PyArrayObject*)newarr;
    }

    void* storage = ((boost::python::converter::rvalue_from_python_storage<CoeffsArray>*)data)->storage.bytes;
    new(storage) CoeffsArray(arr);
    data->convertible = storage;
}


BOOST_PYTHON_MODULE(slab)
{
    plask_import_array();

#ifndef NDEBUG
    py::class_<Matrix_Python<dcomplex>>("_cmatrix", py::no_init);
    py::delattr(py::scope(), "_cmatrix");
    py::to_python_converter<cmatrix, Matrix_Python<dcomplex>>();
    py::class_<Diagonal_Python<dcomplex>>("_cdiagonal", py::no_init);
    py::delattr(py::scope(), "_cdiagonal");
    py::to_python_converter<cdiagonal, Diagonal_Python<dcomplex>>();
    py::class_<Matrix_Python<double>>("_dmatrix", py::no_init);
    py::delattr(py::scope(), "_dmatrix");
    py::to_python_converter<dmatrix, Matrix_Python<double>>();
    // py::class_<Diagonal_Python<double>>("_ddiagonal", py::no_init);
    // py::delattr(py::scope(), "_ddiagonal");
    // py::to_python_converter<MatrixDiagonal<double>, Diagonal_Python<double>>();
#endif

    py::to_python_converter<Expansion::Component, PythonComponentConventer>();
    py::converter::registry::push_back(&PythonComponentConventer::convertible, &PythonComponentConventer::construct,
                                       py::type_id<Expansion::Component>());

    // py::converter::registry::push_back(&PythonFourierSolver3DWhatConverter::convertible, &PythonFourierSolver3DWhatConverter::construct,
    //                                    py::type_id<FourierSolver3D::What>());

    py::class_<PmlWrapper, shared_ptr<PmlWrapper>>("PML", "Perfectly matched layer details.", py::no_init)
        .def("__init__", py::make_constructor(&PmlWrapper::__init__, py::default_call_policies(),
                                              (py::arg("factor"), "size", "dist", py::arg("shape")=2)))
        .add_property("factor", &PmlWrapper::get_factor, &PmlWrapper::set_factor, "PML scaling factor.")
        .add_property("size", &PmlWrapper::get_size, &PmlWrapper::set_size, u8"PML size.")
        .add_property("dist", &PmlWrapper::get_dist, &PmlWrapper::set_dist, u8"PML distance from the structure.")
        .add_property("shape", &PmlWrapper::get_order, &PmlWrapper::set_order, u8"PML shape order (0 → flat, 1 → linearly increasing, 2 → quadratic, etc.).")
        .def("__str__", &PmlWrapper::__str__)
        .def("__repr__", &PmlWrapper::__repr__)
    ;

    py_enum<Transfer::Method>()
        .value("AUTO", Transfer::METHOD_AUTO)
        .value("REFLECTION", Transfer::METHOD_REFLECTION_ADMITTANCE)
        .value("REFLECTION_ADMITTANCE", Transfer::METHOD_REFLECTION_ADMITTANCE)
        .value("REFLECTION_IMPEDANCE", Transfer::METHOD_REFLECTION_IMPEDANCE)
        .value("REFLECTION", Transfer::METHOD_REFLECTION_ADMITTANCE)
        .value("ADMITTANCE", Transfer::METHOD_ADMITTANCE)
        .value("IMPEDANCE", Transfer::METHOD_IMPEDANCE)
    ;

    py_enum<Transfer::Determinant>()
        .value("EIGEN", Transfer::DETERMINANT_EIGENVALUE)
        .value("EIGENVALUE", Transfer::DETERMINANT_EIGENVALUE)
        .value("FULL", Transfer::DETERMINANT_FULL)
    ;

    py_enum<RootDigger::Method>()
        .value("MULLER", RootDigger::ROOT_MULLER)
        .value("BROYDEN", RootDigger::ROOT_BROYDEN)
        .value("BRENT", RootDigger::ROOT_BRENT)
    ;

    py_enum<typename Transfer::IncidentDirection>()
        .value("TOP", Transfer::INCIDENCE_TOP)
        .value("BOTTOM", Transfer::INCIDENCE_BOTTOM)
    ;

    py_enum<SlabBase::Emission>()
        .value("UNDEFINED", SlabBase::EMISSION_UNSPECIFIED)
        .value("TOP", SlabBase::EMISSION_TOP)
        .value("BOTTOM", SlabBase::EMISSION_BOTTOM)
        .value("FRONT", SlabBase::EMISSION_FRONT)
        .value("BACK", SlabBase::EMISSION_BACK)
    ;

    py::class_<RootDigger::Params, boost::noncopyable>("RootParams", u8"Configuration of the root finding algorithm.", py::no_init)
        .def_readwrite("method", &RootDigger::Params::method, u8"Root finding method ('muller', 'broyden',  or 'brent')")
        .def_readwrite("tolx", &RootDigger::Params::tolx, u8"Absolute tolerance on the argument.")
        .def_readwrite("tolf_min", &RootDigger::Params::tolf_min, u8"Sufficient tolerance on the function value.")
        .def_readwrite("tolf_max", &RootDigger::Params::tolf_max, u8"Required tolerance on the function value.")
        .def_readwrite("maxiter", &RootDigger::Params::maxiter, u8"Maximum number of iterations.")
        .def_readwrite("maxstep", &RootDigger::Params::maxstep, u8"Maximum step in one iteration (Broyden method only).")
        .def_readwrite("alpha", &RootDigger::Params::maxstep, u8"Parameter ensuring sufficient decrease of determinant in each step\n(Broyden method only).")
        .def_readwrite("lambd", &RootDigger::Params::maxstep, u8"Minimum decrease ratio of one step (Broyden method only).")
        .def_readwrite("initial_range", &RootDigger::Params::initial_dist, u8"Initial range size (Muller and Brent methods only).")
    ;

    export_FourierSolver2D();
    export_FourierSolver3D();
    export_BesselSolverCyl();
    export_LinesSolverCyl();

    py::converter::registry::push_back(&CoeffsArray::convertible, &CoeffsArray::construct, boost::python::type_id<CoeffsArray>());
}
