#ifndef PLASK__PYTHON_MESH_H
#define PLASK__PYTHON_MESH_H

#include <cmath>

// Important includes
#include "python_globals.h"
#include <plask/mesh/mesh.h>
#include <plask/mesh/boundary.h>

namespace plask { namespace python {

namespace py = boost::python;

/// Generic declaration of mesh generator
template <typename MeshType>
py::class_<MeshGeneratorOf<MeshType>, shared_ptr<MeshGeneratorOf<MeshType>>, py::bases<MeshGenerator>, boost::noncopyable>
ExportMeshGenerator(const std::string name) {
    py::scope scope = py::object(py::scope().attr(name.c_str()));
    py::class_<MeshGeneratorOf<MeshType>, shared_ptr<MeshGeneratorOf<MeshType>>, py::bases<MeshGenerator>, boost::noncopyable>
    pyclass("Generator", ("Base class for all "+name+" mesh generators.").c_str(), py::no_init);
    pyclass.def("__call__", &MeshGeneratorOf<MeshType>::operator(), "Generate mesh for given geometry or load it from the cache", py::arg("geometry"));
    pyclass.def("generate", &MeshGeneratorOf<MeshType>::generate, "Generate mesh for given geometry omitting the cache", py::arg("geometry"));
    pyclass.def("clearCache", &MeshGeneratorOf<MeshType>::clearCache, "Clear cache of generated meshes");
    return pyclass;
}

/// Generic declaration of boundary class for a specific mesh type
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

    ExportBoundary(py::object mesh_class) {

        py::scope scope = mesh_class;

        std::string name = py::extract<std::string>(mesh_class.attr("__name__"));

        py::class_<typename MeshType::Boundary::WithMesh, shared_ptr<typename MeshType::Boundary::WithMesh>>("BoundaryInstance",
            ("Boundary specification for particular "+name+" mesh object").c_str(), py::no_init)
            .def("__contains__", &MeshType::Boundary::WithMesh::includes)
            .def("__iter__", py::range(&MeshType::Boundary::WithMesh::begin, &MeshType::Boundary::WithMesh::end))
        ;
        py::delattr(scope, "BoundaryInstance");

        py::class_<typename MeshType::Boundary, shared_ptr<typename MeshType::Boundary>>("Boundary",
            ("Generic boundary specification for "+name+" mesh").c_str(), py::no_init)
            .def("__call__", &Boundary__call__, py::arg("mesh"), "Get boundary instance for particular mesh",
                 py::with_custodian_and_ward_postcall<0,1,
                 py::with_custodian_and_ward_postcall<0,2>>())
        ;

        boost::python::converter::registry::push_back(&PythonPredicate::convertible, &PythonPredicate::construct, boost::python::type_id<typename MeshType::Boundary>());
    }
};

}} // namespace plask::python

#endif // PLASK__PYTHON_MESH_H
