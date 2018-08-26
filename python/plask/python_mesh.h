#ifndef PLASK__PYTHON_MESH_H
#define PLASK__PYTHON_MESH_H

#include <cmath>

// Important contains
#include "python_globals.h"
#include "python_boundaries.h"
#include <plask/mesh/mesh.h>
#include <plask/mesh/boundary.h>

namespace plask { namespace python {

namespace py = boost::python;

/// Custom Python mesh
template <int dim>
struct MeshWrap: public MeshD<dim>, Overriden<MeshD<dim>> {

    MeshWrap(PyObject* self): Overriden<MeshD<dim>> (self) {}

    typename MeshD<dim>::LocalCoords at(std::size_t index) const override;

    size_t size() const override;

    static shared_ptr<MeshD<dim>> __init__(py::tuple args, py::dict kwargs);
};

/// Generic declaration of boundary class for a specific mesh type
template <typename Boundary>
struct ExportBoundary {

    typedef typename Boundary::MeshType MeshType;

    struct PythonPredicate {

        py::object pyfun;

        PythonPredicate(PyObject* obj) : pyfun(py::object(py::handle<>(py::incref(obj)))) {}

        bool operator()(const MeshType& mesh, std::size_t indx) const {
            py::object pyresult = pyfun(boost::ref(mesh), indx);
            bool result;
            try {
                result = py::extract<bool>(pyresult);
            } catch (py::error_already_set&) {
                throw TypeError(u8"Boundary predicate did not return Boolean value");
            }
            return result;
        }

        static void* convertible(PyObject* obj) {
            if (PyCallable_Check(obj)) return obj;
            return nullptr;
        }
        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<Boundary>*)data)->storage.bytes;
            PythonPredicate predicate(obj);
            new (storage) Boundary { makePredicateBoundary<Boundary>(predicate) };
            data->convertible = storage;
        }

    };

    static BoundaryNodeSet Boundary__call__(
        const Boundary& self, const MeshType& mesh, shared_ptr<const GeometryD<MeshType::DIM>> geometry
    ) {
        return self(mesh, geometry);
    }

    ExportBoundary(py::object mesh_class) {

        py::scope scope = mesh_class;

        std::string name = py::extract<std::string>(mesh_class.attr("__name__"));

        if (py::converter::registry::lookup(py::type_id<BoundaryNodeSet>()).m_class_object == nullptr) {
            py::class_<BoundaryNodeSet, shared_ptr<BoundaryNodeSet>>("BoundaryInstance",
                ("Boundary specification for particular "+name+" mesh object").c_str(), py::no_init)
                .def("__contains__", &BoundaryNodeSet::contains)
                .def("__iter__", py::range(&BoundaryNodeSet::begin, &BoundaryNodeSet::end))
                .def("__len__", &BoundaryNodeSet::size)
            ;
            py::delattr(scope, "BoundaryInstance");
        }

        py::class_<Boundary, shared_ptr<Boundary>>("Boundary",
            ("Generic boundary specification for "+name+" mesh").c_str(), py::no_init)
            .def("__call__", &Boundary__call__, (py::arg("mesh"), "geometry"), u8"Get boundary instance for particular mesh",
                 py::with_custodian_and_ward_postcall<0,1,
                 py::with_custodian_and_ward_postcall<0,2>>())
        ;

        detail::RegisterBoundaryConditions<Boundary, py::object>(false);

        boost::python::converter::registry::push_back(&PythonPredicate::convertible, &PythonPredicate::construct, boost::python::type_id<Boundary>());
    }
};

}} // namespace plask::python

#endif // PLASK__PYTHON_MESH_H
