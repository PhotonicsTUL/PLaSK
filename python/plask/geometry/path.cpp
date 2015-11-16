#include "geometry.h"
#include "../../util/py_set.h"

#include <plask/geometry/path.h>
#include <plask/geometry/object.h>


namespace plask { namespace python {

template <typename... Args>
static shared_ptr<Path> Path__init__(Args... args) {
    return plask::make_shared<Path>(std::forward<Args>(args)...);
}

static bool Hint__eq__(const PathHints::Hint& first, const PathHints::Hint& second) {
    return first.first == second.second && first.second == second.second;
}

static shared_ptr<PathHints> Hint__add__(const PathHints::Hint& first, const PathHints::Hint& second) {
    auto hints = plask::make_shared<PathHints>();
    *hints += first;
    *hints += second;
    return hints;
}

static std::string PathHints__repr__(const PathHints& self) {
    if (self.hintFor.size() == 0) return "plask.geometry.PathHints()";
    return format("plask.geometry.PathHints(<%1% hint%2%>)", self.hintFor.size(), (self.hintFor.size()==1)?"":"s" );
}

namespace detail {
    struct PathHints_from_None {
        PathHints_from_None() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<PathHints>());
        }

        // Determine if obj can be converted into an Aligner
        static void* convertible(PyObject* obj) {
            if (obj != Py_None) return 0;
            return obj;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
            // Grab pointer to memory into which to construct the new Aligner
            void* storage = ((boost::python::converter::rvalue_from_python_storage<PathHints>*)data)->storage.bytes;
            new(storage) PathHints;
            // Stash the memory chunk pointer for later use by boost.python
            data->convertible = storage;
        }

    };
}

static shared_ptr<GeometryObject> Path__getitem__(const Path& self, long int index) {
    long int i = (index >= 0) ? index : self.objects.size() + index;
    if (i < 0 || std::size_t(i) >= self.objects.size()) throw IndexError("Index %1% out of range for Path (%2% < index < %3%)", index, -self.objects.size(), self.objects.size());
    return const_pointer_cast<GeometryObject>(self.objects[i]);
}

static size_t Path__len__(const Path& self) { return self.objects.size(); }

static bool Subtree__nonzero__(const GeometryObject::Subtree& self) { return !self.empty(); }

void register_geometry_path()
{
    py::class_<PathHints::Hint>("PathHint",
                                "PathHint stores weak references to container and its child with translation.\n\n"
                                "It may only be used as an intermediate object to either add it to Path,\n"
                                "PathHints, or to retrieve the container, child, or translation objects.",
                                py::no_init)
        .def("__add__", &Hint__add__)
        .def("__eq__", Hint__eq__)
    ;

    set_to_python_list_conventer<shared_ptr<GeometryObject>>();

    py::class_<PathHints, shared_ptr<PathHints>>("PathHints",
                          "Hint used for resolving ambiguities in a geometry tree.\n\n"
                          "PathHints is used to resolve ambiguities if any object is present in the"
                          "geometry\n tree more than once. It contains a set of PathHint objects holding\n"
                          "weak references to containers and their items.")
        .def("__repr__", &PathHints__repr__)
        .def("add", (void (PathHints::*)(const PathHints::Hint&))&PathHints::addHint, py::arg("hint"),
             "Append hint to the path.\n\n"
             "Args:\n"
             "    hint (PathHint): Hint to add.")
        .def(py::self += py::other<PathHints::Hint>())
        .def("get_items", (std::set<shared_ptr<GeometryObject>> (PathHints::*)(const GeometryObject& container) const)&PathHints::getChildren, py::arg("container"),
             "Get all items in a container present in the Hints.\n\n"
             "Args:\n"
             "    container (GeometryObject): Container to get items from.")
        .def("cleanup",  &PathHints::cleanDeleted, "Remove all hints which refer to deleted objects.")
        .def(py::self == py::other<PathHints>())
        .def("__hash__", __hash__<PathHints>)
        // .def(py::self < py::other<PathHints>())
    ;

    py::implicitly_convertible<PathHints::Hint,PathHints>();
    detail::PathHints_from_None();

    py::class_<Path, shared_ptr<Path>>("Path",
                     "Sequence of objects in the geometry tree, used for resolving ambiguities.\n\n"
                     "Path is used to specify unique instance of every object in the geometry,\n"
                     "even if this object is inserted to the geometry tree in more than one place.\n\n",
                     py::no_init)
        .def("__init__", py::make_constructor(Path__init__<const PathHints::Hint&>))
        .def("__init__", py::make_constructor(Path__init__<const GeometryObject&>))
        .def("__init__", py::make_constructor(Path__init__<const GeometryObject::Subtree&>))
        .def("__getitem__", &Path__getitem__)
        .def("__len__", &Path__len__)
        .def("append", (Path& (Path::*)(shared_ptr<const GeometryObject>, const PathHints*))&Path::append,
             (py::arg("object"), py::arg("hints")=py::object()), py::return_self<>())
        .def("append", (Path& (Path::*)(const PathHints::Hint&, const PathHints*))&Path::append,
             (py::arg("hint"), py::arg("hints")=py::object()), py::return_self<>())
        .def("append", (Path& (Path::*)(const GeometryObject::Subtree&, const PathHints*))&Path::append,
             (py::arg("subtree"), py::arg("hints")=py::object()), py::return_self<>())
        .def("append", (Path& (Path::*)(const Path&, const PathHints*))&Path::append,
             (py::arg("path"), py::arg("hints")=py::object()), py::return_self<>(),
             "Append an object/hint/subtree/path list to this path.\n\n"
             "Args:\n"
             "object (GeometryObject): Geometry object to append to the path. It must be\n"
             "                         an item of a container already present in it.\n"
             "hint (PathHint): Hint returned by a addition of an object to the container\n"
             "                 already present in the path.\n"
             "subtree (Subtree): Subtree to add to the path. It must somehow be connected\n"
             "                   with it."
             "path (Path): Another path to join with the exising one. There must be some\n"
             "             connection between the two paths."
             "hints (PathHint, optional): Optional hints used for resolving ambiguities.\n"
            )
        .def(py::self += py::other<Path>())
        .def(py::self += py::other<PathHints::Hint>())
        .def(py::self += py::other<shared_ptr<const GeometryObject>>())
        .def(py::self += py::other<GeometryObject::Subtree>())
        .def("__eq__", __is__<Path>)
        .def("__hash__", __hash__<Path>)
    ;
    py::implicitly_convertible<Path,PathHints>();

    py::class_<GeometryObject::Subtree, shared_ptr<GeometryObject::Subtree>>("Subtree", "A selected part of a geometry tree.", py::no_init)
        .def("__nonzero__", &Subtree__nonzero__)
        .add_property("brached", &GeometryObject::Subtree::hasBranches, "Bool indicating whether the subtree has more than one branch.")
        .add_property("last_path", &GeometryObject::Subtree::getLastPath, "Last (topmost) branch of the subtree.")
        .def("__eq__", __is__<GeometryObject::Subtree>)
        .def("__hash__", __hash__<GeometryObject::Subtree>)
    ;
    py::implicitly_convertible<GeometryObject::Subtree,PathHints>();
    py::implicitly_convertible<GeometryObject::Subtree,Path>();
}



}} // namespace plask::python
