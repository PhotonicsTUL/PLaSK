#include "geometry.h"
#include "../../util/py_set.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <plask/geometry/path.h>
#include <plask/geometry/element.h>


namespace plask { namespace python {

template <typename... Args>
static shared_ptr<Path> Path__init__(Args... args) {
    return make_shared<Path>(std::forward<Args>(args)...);
}

namespace detail {

    struct Element_List_from_Python {

        Element_List_from_Python();

        static void* convertible(PyObject* obj_ptr) {
            if (!PySequence_Check(obj_ptr)) return nullptr;
            int n = PySequence_Size(obj_ptr);
            try {
                for(int i = 0; i < n; i++) py::extract<shared_ptr<GeometryElement>>(PySequence_GetItem(obj_ptr, i));
            } catch (py::error_already_set) {
                PyErr_Clear();
                return nullptr;
            }
            return obj_ptr;
        }

        static void construct(PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data) {
            int n = PySequence_Size(obj_ptr);

            // Grab pointer to memory into which to construct the new QString
            void* storage = ((boost::python::converter::rvalue_from_python_storage<std::vector<shared_ptr<const GeometryElement>>>*)data)->storage.bytes;

            std::vector<shared_ptr<const GeometryElement>>* vec = new(storage) std::vector<shared_ptr<const GeometryElement>>;
            vec->reserve(n);

            for(int i = 0; i < n; i++) {
                shared_ptr<GeometryElement> p = py::extract<shared_ptr<GeometryElement>>(PySequence_GetItem(obj_ptr, i));
                vec->push_back(const_pointer_cast<const GeometryElement>(p));
            }

            // Stash the memory chunk pointer for later use by boost.python
            data->convertible = vec;
        }
    };
}

static shared_ptr<PathHints> Hint__add__(const PathHints::Hint& first, const PathHints::Hint& second) {
    auto hints = make_shared<PathHints>();
    *hints += first;
    *hints += second;
    return hints;
}

static std::string PathHints__repr__(const PathHints& self) {
    if (self.hintFor.size() == 0) return "plask.geometry.PathHints()";
    return format("plask.geometry.PathHints(<%1% hints>)", self.hintFor.size());
}

namespace detail {
    struct PathHints_from_None {
        PathHints_from_None() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<PathHints>());
        }

        // Determine if obj_ptr can be converted into an Aligner
        static void* convertible(PyObject* obj_ptr) {
            if (obj_ptr != Py_None) return 0;
            return obj_ptr;
        }

        static void construct(PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data) {
            // Grab pointer to memory into which to construct the new Aligner
            void* storage = ((boost::python::converter::rvalue_from_python_storage<PathHints>*)data)->storage.bytes;
            new(storage) PathHints;
            // Stash the memory chunk pointer for later use by boost.python
            data->convertible = storage;
        }

    };
}

static shared_ptr<GeometryElement> Path__getitem__(const Path& self, long int index) {
    long int i = (index >= 0) ? index : self.elements.size() + index;
    if (i < 0 || std::size_t(i) >= self.elements.size()) throw IndexError("Index %1% out of range for Path (%2% < index < %3%)", index, -self.elements.size(), self.elements.size());
    return const_pointer_cast<GeometryElement>(self.elements[i]);
}

static size_t Path__len__(const Path& self) { return self.elements.size(); }

static bool Subtree__nonzero__(const GeometryElement::Subtree& self) { return !self.empty(); }
static Path Subtree_getLastBranch(const GeometryElement::Subtree& self) { return Path(self.getLastPath()); }

void register_geometry_path()
{
    py::class_<PathHints::Hint>("PathHint",
                                "PathHint stores weak references to container and its child with translation.\n\n"
                                "It may only be used as an intermediate object to either add it to Path, PathHints, or\n"
                                "to retrieve the container, child, or translation elements.",
                                py::no_init)
        .def("__add__", &Hint__add__)
    ;

    set_to_python_list_conventer<shared_ptr<GeometryElement>>();
    // export_frozenset<shared_ptr<GeometryElement>>("GeometryElement_set");

    py::class_<PathHints, shared_ptr<PathHints>>("PathHints",
                          "PathHints is used to resolve ambiguities if any element is present in the geometry\n"
                          "tree more than once. It contains a set of PathHint objects holding weak references\n"
                          "to containers and their childred.")
        .def("__repr__", &PathHints__repr__)
        .def("add", (void (PathHints::*)(const PathHints::Hint&))&PathHints::addHint, "Append hint to the path.", (py::arg("hint")))
        .def(py::self += py::other<PathHints::Hint>())
        .def("getChildren", (std::set<shared_ptr<GeometryElement>> (PathHints::*)(const GeometryElement& container) const)&PathHints::getChildren,
             "Get all children of a container present in the Hints", (py::arg("container")))
        .def("cleanDeleted",  &PathHints::cleanDeleted, "Remove all hints which refer to deleted objects")
    ;

    py::implicitly_convertible<PathHints::Hint,PathHints>();
    detail::PathHints_from_None();

    py::class_<Path, shared_ptr<Path>>("Path",
                     "Path is used to specify unique instance of every element in the geometry,\n"
                     "even if this element is inserted to the geometry tree in more than one place.\n\n"
                     "It contains a sequence of objects in the geometry tree.", py::no_init)
        .def("__init__", py::make_constructor(Path__init__<const PathHints::Hint&>))
        .def("__init__", py::make_constructor(Path__init__<const GeometryElement&>))
        .def("__init__", py::make_constructor(Path__init__<const GeometryElement::Subtree&>))
        .def("__init__", py::make_constructor(Path__init__<const std::vector<shared_ptr<const GeometryElement>>&>))
        .def("__getitem__", &Path__getitem__)
        .def("__len__", &Path__len__)
        .def("append", (Path& (Path::*)(const Path&, const PathHints*))&Path::append,
             "Append another path to this path. Optional hints are used to resolve ambiguities.",
             (py::arg("path"), py::arg("hints")=py::object()), py::return_self<>())
        .def("append", (Path& (Path::*)(const PathHints::Hint&, const PathHints*))&Path::append,
             "Append a hint to this path. Optional hints are used to resolve ambiguities.",
             (py::arg("hint"), py::arg("hints")=py::object()), py::return_self<>())
        .def("append", (Path& (Path::*)(const GeometryElement&, const PathHints*))&Path::append,
             "Append an element to this path. Optional hints are used to resolve ambiguities.",
             (py::arg("element"), py::arg("hints")=py::object()), py::return_self<>())
        .def("append", (Path& (Path::*)(const GeometryElement::Subtree&, const PathHints*))&Path::append,
             "Append a subtree to this path. Optional hints are used to resolve ambiguities.",
             (py::arg("subtree"), py::arg("hints")=py::object()), py::return_self<>())
        .def("append", (Path& (Path::*)(const std::vector<shared_ptr<const GeometryElement>>&, const PathHints*))&Path::append,
             "Append an element list to this path. Optional hints are used to resolve ambiguities.",
             (py::arg("elements"), py::arg("hints")=py::object()), py::return_self<>())
        .def(py::self += py::other<Path>())
        .def(py::self += py::other<PathHints::Hint>())
        .def(py::self += py::other<GeometryElement>())
        .def(py::self += py::other<GeometryElement::Subtree>())
        .def(py::self += py::other<std::vector<shared_ptr<const GeometryElement>>>())
    ;
    py::implicitly_convertible<Path,PathHints>();

    boost::python::converter::registry::push_back(&detail::Element_List_from_Python::convertible, &detail::Element_List_from_Python::construct,
                                                  boost::python::type_id<std::vector<shared_ptr<const GeometryElement>>>());

    py::class_<GeometryElement::Subtree, shared_ptr<GeometryElement::Subtree>>("Subtree", "Class holding some part of geometry tree", py::no_init)
        .def("__nonzero__", &Subtree__nonzero__)
        .add_property("brached", &GeometryElement::Subtree::hasBranches, "Indicates whether the subtree has more than one branch")
        .add_property("last_path", &Subtree_getLastBranch, "Last (topmost) branch of the subtree")
    ;
    py::implicitly_convertible<GeometryElement::Subtree,PathHints>();
    py::implicitly_convertible<GeometryElement::Subtree,Path>();
}



}} // namespace plask::python
