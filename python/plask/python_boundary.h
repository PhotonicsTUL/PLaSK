#ifndef PLASK__PYTHON_BOUNDARY_H
#define PLASK__PYTHON_BOUNDARY_H

#include <cmath>

// Important includes
#include "python_globals.h"
#include <plask/mesh/boundary.h>

namespace plask { namespace python {

namespace py = boost::python;

// Generic declaration of boundary class for a specific mesh type
template <typename MeshType>
struct ExportBoundary {

    struct PythonPredicate {

        py::object pyfun;

        PythonPredicate(PyObject* obj) : pyfun(py::object(py::handle<>(py::incref(obj)))) { }

        bool operator()(const MeshType& mesh, std::size_t indx) const {
            py::object pyresult = pyfun(mesh, indx);
            bool result;
            try {
                result = py::extract<bool>(pyresult);
            } catch (py::error_already_set) {
                throw TypeError("Boundary predicate did not return Boolean value");
            }
            return result;
        }

        static void* convertible(PyObject* obj) {
            if (PyCallable_Check(obj)) return obj;
            return nullptr;
        }
        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<typename MeshType::Boundary>*)data)->storage.bytes;
            PythonPredicate predicate(obj);
            new (storage) typename MeshType::Boundary { makePredicateBoundary<MeshType>(predicate) };
            data->convertible = storage;
        }

    };

    static typename MeshType::Boundary::WithMesh Boundary__call__(const typename MeshType::Boundary& self, const MeshType& mesh) {
        return self(mesh);
    }

    ExportBoundary(const std::string& name) {

        py::class_<typename MeshType::Boundary::WithMesh, shared_ptr<typename MeshType::Boundary::WithMesh>>((name+"BoundaryInstance").c_str(),
            ("Boundary specification for particular "+name+" mesh object").c_str(), py::no_init)
            .def("__contains__", &MeshType::Boundary::WithMesh::includes)
            .def("__iter__", py::range(&MeshType::Boundary::WithMesh::begin, &MeshType::Boundary::WithMesh::end))
        ;

        py::class_<typename MeshType::Boundary, shared_ptr<typename MeshType::Boundary>>((name+"Boundary").c_str(),
            ("Generic boundary specification for "+name+"mesh").c_str(), py::no_init)
            .def("__call__", &Boundary__call__, py::arg("mesh"), "Get boundary instance for particular mesh",
                 py::with_custodian_and_ward_postcall<0,1,
                 py::with_custodian_and_ward_postcall<0,2>>())
        ;

        boost::python::converter::registry::push_back(&PythonPredicate::convertible, &PythonPredicate::construct, boost::python::type_id<typename MeshType::Boundary>());
    }

};

}} // namespace plask::python

#endif // PLASK__PYTHON_BOUNDARY_H
