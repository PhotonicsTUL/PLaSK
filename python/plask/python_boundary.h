#ifndef PLASK__PYTHON_BOUNDARY_H
#define PLASK__PYTHON_BOUNDARY_H

#include <cmath>

// Important includes
#include "python_globals.h"
#include <plask/mesh/boundary.h>

namespace plask { namespace python {

namespace py = boost::python;

// Generic declaration of boundary class for a specifie dmesh type
template <typename MeshT>
struct ExportBoundary {

    struct PythonPredicate {

        PyObject* pyfun;

        PythonPredicate(PyObject* fun) : pyfun(fun) { Py_INCREF(pyfun); }

        ~PythonPredicate() { Py_XDECREF(pyfun); }

        bool operator()(const MeshT& mesh, std::size_t indx) const {
            py::tuple args = py::make_tuple(mesh, indx);
            PyObject* pyresult = PyObject_CallObject(pyfun, args.ptr());
            bool result;
            try {
                result = py::extract<bool>(pyresult);
                Py_XDECREF(pyresult);
            } catch (py::error_already_set) {
                Py_XDECREF(pyresult);
                throw TypeError("Boundary predicate did not return Boolean value");
            }
            return result;
        }

        static void* convertible(PyObject* obj) {
            if (PyCallable_Check(obj)) return obj;
            return nullptr;
        }
        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<typename MeshT::Boundary>*)data)->storage.bytes;
            PythonPredicate predicate(obj);
            new (storage) typename MeshT::Boundary { makePredicateBoundary<MeshT>(predicate) };
            data->convertible = storage;
        }

    };

    static typename MeshT::Boundary::WithMesh Boundary__call__(const typename MeshT::Boundary& self, const MeshT& mesh) {
        return self(mesh);
    }

    ExportBoundary(const std::string& name) {

        py::class_<typename MeshT::Boundary::WithMesh, shared_ptr<typename MeshT::Boundary::WithMesh>>((name+"BoundaryInstance").c_str(),
            ("Boundary specification for particular "+name+" mesh object").c_str(), py::no_init)
            .def("__contains__", &MeshT::Boundary::WithMesh::includes)
            .def("__iter__", py::range(&MeshT::Boundary::WithMesh::begin, &MeshT::Boundary::WithMesh::end))
        ;

        py::class_<typename MeshT::Boundary, shared_ptr<typename MeshT::Boundary>>((name+"Boundary").c_str(),
            ("Generic boundary specification for "+name+"mesh").c_str(), py::no_init)
            .def("__call__", &Boundary__call__, py::arg("mesh"), "Get boundary instance for particular mesh",
                 py::with_custodian_and_ward_postcall<0,1,
                 py::with_custodian_and_ward_postcall<0,2>>())
        ;

        boost::python::converter::registry::push_back(&PythonPredicate::convertible, &PythonPredicate::construct, boost::python::type_id<typename MeshT::Boundary>());
    }

};

}} // namespace plask::python

#endif // PLASK__PYTHON_BOUNDARY_H
