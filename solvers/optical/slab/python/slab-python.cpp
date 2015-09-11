#include "slab-python.h"
using namespace plask;
using namespace plask::python;

#include "fourier2d-python.h"
#include "fourier3d-python.h"
#include "besselcyl-python.h"
using namespace plask::solvers::slab;
using namespace plask::solvers::slab::python;

#ifndef NDEBUG
struct CMatrix_Python {
    cmatrix data;
    CMatrix_Python(const cmatrix& data): data(data) {}
    CMatrix_Python(const CMatrix_Python& src): data(src.data) {}
    
    static PyObject* convert(const cmatrix& self) {
        npy_intp dims[2] = { self.rows(), self.cols() };
        npy_intp strides[2] = {sizeof(dcomplex), self.rows() * sizeof(dcomplex)};
        
        PyObject* arr = PyArray_New(&PyArray_Type, 2, dims, NPY_CDOUBLE, strides,
                                    (void*)self.data(), 0, 0, NULL);
        if (arr == nullptr) throw plask::CriticalException("Cannot create array from matrix");
        // Make sure the data vector stays alive as long as the array
        py::object oself {CMatrix_Python(self)};
        py::incref(oself.ptr());        
        PyArray_SetBaseObject((PyArrayObject*)arr, oself.ptr());
        return arr;
    }
};
#endif

BOOST_PYTHON_MODULE(slab)
{
    plask_import_array();

#ifndef NDEBUG
    py::class_<CMatrix_Python>("_cmatrix", py::no_init);
    py::delattr(py::scope(), "_cmatrix");
    py::to_python_converter<cmatrix, CMatrix_Python>();
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
        .add_property("size", &PmlWrapper::get_size, &PmlWrapper::set_size, "PML size.")
        .add_property("dist", &PmlWrapper::get_dist, &PmlWrapper::set_dist, "PML distance from the structure.")
        .add_property("shape", &PmlWrapper::get_order, &PmlWrapper::set_order, "PML shape order (0 → flat, 1 → linearly increasing, 2 → quadratic, etc.).")
        .def("__str__", &PmlWrapper::__str__)
        .def("__repr__", &PmlWrapper::__repr__)
    ;

    py_enum<Transfer::Method>()
        .value("AUTO", Transfer::METHOD_AUTO)
        .value("REFLECTION", Transfer::METHOD_REFLECTION)
        .value("ADMITTANCE", Transfer::METHOD_ADMITTANCE)
    ;

    py_enum<RootDigger::Method>()
        .value("MULLER", RootDigger::ROOT_MULLER)
        .value("BROYDEN", RootDigger::ROOT_BROYDEN)
        .value("BRENT", RootDigger::ROOT_BRENT)
    ;

    py_enum<typename ReflectionTransfer::IncidentDirection>()
        .value("TOP", ReflectionTransfer::INCIDENCE_TOP)
        .value("BOTTOM", ReflectionTransfer::INCIDENCE_BOTTOM)
    ;

    py::class_<RootDigger::Params, boost::noncopyable>("RootParams", "Configuration of the root finding algorithm.", py::no_init)
        .def_readwrite("method", &RootDigger::Params::method, "Root finding method ('muller', 'broyden',  or 'brent')")
        .def_readwrite("tolx", &RootDigger::Params::tolx, "Absolute tolerance on the argument.")
        .def_readwrite("tolf_min", &RootDigger::Params::tolf_min, "Sufficient tolerance on the function value.")
        .def_readwrite("tolf_max", &RootDigger::Params::tolf_max, "Required tolerance on the function value.")
        .def_readwrite("maxiter", &RootDigger::Params::maxiter, "Maximum number of iterations.")
        .def_readwrite("maxstep", &RootDigger::Params::maxstep, "Maximum step in one iteration (Broyden method only).")
        .def_readwrite("alpha", &RootDigger::Params::maxstep, "Parameter ensuring sufficient decrease of determinant in each step\n(Broyden method only).")
        .def_readwrite("lambd", &RootDigger::Params::maxstep, "Minimum decrease ratio of one step (Broyden method only).")
        .def_readwrite("initial_range", &RootDigger::Params::initial_dist, "Initial range size (Muller and Brent methods only).")
    ;
    
    export_FourierSolver2D();
    export_FourierSolver3D();
    export_BesselSolverCyl();
}

