#include "geometry.h"
#include "../../util/py_set.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/raw_function.hpp>

#include <plask/geometry/container.h>
#include <plask/geometry/translation_container.h>

namespace plask { namespace python {

template <int dim>
static bool Container__contains__(const GeometryObjectContainer<dim>& self, shared_ptr<typename GeometryObjectContainer<dim>::ChildType> child) {
    for (auto trans: self.getChildrenVector()) {
        if (trans->getChild() == child) return true;
    }
    return false;
}

shared_ptr<GeometryObject> GeometryObject__getitem__(py::object oself, int i);

template <int dim>
static std::set<shared_ptr<GeometryObject>> Container__getitem__hints(const GeometryObjectContainer<dim>& self, const PathHints& hints) {
    std::set<shared_ptr<GeometryObject>> result = hints.getChildren(self);
    return result;
}

template <int dim>
static void Container__delitem__(GeometryObjectContainer<dim>& self, py::object item) {
    try {
        int i = py::extract<int>(item);
        if (i < 0) i = self.getRealChildrenCount() + i;
        self.removeAt(i);
        return;
    } catch (py::error_already_set) { PyErr_Clear(); }
    try {
        PathHints::Hint* hint = py::extract<PathHints::Hint*>(item);
        self.remove(PathHints(*hint));
        return;
    } catch (py::error_already_set) { PyErr_Clear(); }
    try {
        PathHints* hints = py::extract<PathHints*>(item);
        self.remove(*hints);
        return;
    } catch (py::error_already_set) { PyErr_Clear(); }
    try {
        shared_ptr<typename GeometryObjectContainer<dim>::TranslationT> child = py::extract<shared_ptr<typename GeometryObjectContainer<dim>::TranslationT>>(item);
        self.removeT(child);
        return;
    } catch (py::error_already_set) { PyErr_Clear(); }
    try {
        shared_ptr<typename GeometryObjectContainer<dim>::ChildType> child = py::extract<shared_ptr<typename GeometryObjectContainer<dim>::ChildType>>(item);
        self.remove(child);
        return;
    } catch (py::error_already_set) { PyErr_Clear(); }
    throw TypeError("unrecognized object %s delete from container", std::string(py::extract<std::string>(py::str(item))));
}


DECLARE_GEOMETRY_ELEMENT_23D(GeometryObjectContainer, "Container", "Base class for all "," containers.") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryObjectContainer, GeometryObjectD<dim>)
        .def("__contains__", &Container__contains__<dim>)
        .def("__getitem__", &GeometryObject__getitem__)
        .def("__getitem__", &Container__getitem__hints<dim>)
        .def("__len__", &GeometryObjectD<dim>::getChildrenCount)
        .def("__delitem__", &Container__delitem__<dim>)
    ;
}

static PathHints::Hint TranslationContainer2_add
    (TranslationContainer<2>& self, shared_ptr<typename TranslationContainer<2>::ChildType> el, double c0, double c1) {
    return self.add(el, Vec<2>(c0, c1));
}

static PathHints::Hint TranslationContainer3_add
    (TranslationContainer<3>& self, shared_ptr<typename TranslationContainer<3>::ChildType> el, double c0, double c1, double c2) {
    return self.add(el, Vec<3>(c0, c1, c2));
}


template <typename ContainerT>
PathHints::Hint TranslationContainer_add(py::tuple args, py::dict kwargs) {
    parseKwargs("append", args, kwargs, "self", "item");
    ContainerT* self = py::extract<ContainerT*>(args[0]);
    shared_ptr<typename ContainerT::ChildType> child = py::extract<shared_ptr<typename ContainerT::ChildType>>(args[1]);
    if (py::len(kwargs) == 0)
        return self->add(child);
    else
        return self->add(child, py::extract<typename ContainerT::ChildAligner>(kwargs));
}


void register_geometry_container_stack();

void register_geometry_container_lattice();

void register_geometry_container()
{
    init_GeometryObjectContainer<2>();
    init_GeometryObjectContainer<3>();

    py::class_<TranslationContainer<2>, shared_ptr<TranslationContainer<2>>, py::bases<GeometryObjectContainer<2>>, boost::noncopyable>
    ("Align2D",
     "Align2D()\n\n"
     "Container with its items located according to specified alignment.\n\n"
     "Container in which every item is located according to its alignment\n"
     "specification. Items in this container can overlap with each other so, although\n"
     "their order matches, you can cretate any arrangement you wish.\n\n"
     "Example:\n"
     "    To create a trapezoid of the height 2µm and lower and upper bases of 4µm and\n"
     "    3µm, respectively, you can issue the following commands:\n\n"
     "    >>> trapezoid = plask.geometry.Align2D()\n"
     "    >>> trapezoid.append(plask.geometry.Rectangle(4, 2, 'GaAs'), 0, 0)\n"
     "    >>> trapezoid.append(plask.geometry.Triangle((0, 2), (-1, 2), 'air'),\n"
     "                         bottom=0, right=4)\n\n"
     "    The triangle now overlaps a part of the rectangle.\n"
    )
        .def("append", py::raw_function(&TranslationContainer_add<TranslationContainer<2>>))
        .def("append", (PathHints::Hint(TranslationContainer<2>::*)(shared_ptr<TranslationContainer<2>::ChildType>,const Vec<2>&))&TranslationContainer<2>::add,
             (py::arg("item"), py::arg("translation")=Vec<2>(0.,0.)))
        .def("append", &TranslationContainer2_add, (py::arg("item"), "c0", "c1"),
            "append(item, **alignments)\n\n"
            "Add new object to the container with provided alignment.\n\n"
            "Args:\n"
            "    translation (:class:`plask.vec`): Two-dimensional vector specifying\n"
            "                                      the position of the item origin.\n"
            "    c0 (float): Horizontal component of the vector specifying the position\n"
            "                of the item origin.\n"
            "    c1 (float): Vertical component of the vector specifying the position\n"
            "                of the item origin.\n"
            "    alignments (dict): Alignment specifications. The keys in this dictionary\n"
            "                       can be ``left``, ``right``, ``top``, ``bottom``,\n"
            "                       ``#center``, and `#`` where `#` is an axis name.\n"
            "                       The corresponding values are positions of a given\n"
            "                       edge/center/origin of the item. Exactly one alignment\n"
            "                       for each axis must be given.\n"
        )

        // alias to append
        .def("add", py::raw_function(&TranslationContainer_add<TranslationContainer<2>>), "add(item, **alignments)" )
        .def("add", (PathHints::Hint(TranslationContainer<2>::*)(shared_ptr<TranslationContainer<2>::ChildType>,const Vec<2>&))&TranslationContainer<2>::add,
            (py::arg("item"), py::arg("translation")=Vec<2>(0.,0.)))
        .def("add", &TranslationContainer2_add, (py::arg("item"), "c0", "c1"),
            "Alias for :meth:`~plask.geometry.Align2D.append`.\n"
        )

        .def("move_item", py::raw_function(&Container_move<TranslationContainer<2>>),
            "move_item(path, **alignments)\n\n"
            "Move item existing in the container, setting its position according to the new\n"
            "aligners.\n\n"
            "Args:\n"
            "    path (Path): Path returned by :meth:`~plask.geometry.Align2D.append`\n"
            "                 specifying the object to move.\n"
            "    alignments (dict): Alignment specifications. The keys in this dictionary\n"
            "                       are can be ``left``, ``right``, ``top``, ``bottom``,\n"
            "                       ``#center``, and `#`` where `#` is an axis name.\n"
            "                       The corresponding values are positions of a given\n"
            "                       edge/center/origin of the item. Exactly one alignment\n"
            "                       for each axis must be given.\n"
            )
       ;

    py::class_<TranslationContainer<3>, shared_ptr<TranslationContainer<3>>, py::bases<GeometryObjectContainer<3>>, boost::noncopyable>
    ("Align3D",
     "Align3D()\n\n"
     "Container with its items located according to specified alignment.\n\n"
     "Container in which every item is located according to its alignment\n"
     "specification. Items in this container can overlap with each other so, although\n"
     "their order matches, you can cretate any arrangement you wish.\n\n"
     "Example:\n"
     "    To create a hollow cylinder, you can issue the following commands:\n\n"
     "    >>> hollow = plask.geometry.Align3D()\n"
     "    >>> hollow.append(plask.geometry.Cylinder(10, 2, 'GaAs'), 0, 0, 0)\n"
     "    >>> hollow.append(plask.geometry.Cylinder(8, 2, 'air'), 0, 0, 0)\n\n"
     "    The small cylinder (hole) now overlaps a part of the large one.\n"
    )
        .def("append", py::raw_function(&TranslationContainer_add<TranslationContainer<3>>))
        .def("append", (PathHints::Hint(TranslationContainer<3>::*)(shared_ptr<TranslationContainer<3>::ChildType>,const Vec<3>&))&TranslationContainer<3>::add,
             (py::arg("item"), py::arg("translation")=Vec<3>(0.,0.,0.)))
        .def("append", &TranslationContainer3_add, (py::arg("item"), "c0", "c1", "c2"),
            "append(item, **alignments)\n\n"
            "Add new object to the container with provided alignment.\n\n"
            "Args:\n"
            "    translation (:class:`plask.vec`): Three-dimensional vector specifying\n"
            "                                      the position of the item origin.\n"
            "    c0 (float): Longitudinal component of the vector specifying the position\n"
            "                of the item origin.\n"
            "    c1 (float): Transverse component of the vector specifying the position\n"
            "                of the item origin.\n"
            "    c2 (float): Vertical component of the vector specifying the position\n"
            "                of the item origin.\n"
            "    alignments (dict): Alignment specifications. The keys in this dictionary\n"
            "                       can be ``left``, ``right``, ``top``, ``bottom``,\n"
            "                       ``front``, ``back``, ``#center``, and `#`` where `#`\n"
            "                       is an axis name. The corresponding values are positions\n"
            "                       of a given edge/center/origin of the item. Exactly one\n"
            "                       alignment for each axis must be given.\n"
        )

        // alias to append
        .def("add", py::raw_function(&TranslationContainer_add<TranslationContainer<3>>), "add(item, **alignments)")
        .def("add", (PathHints::Hint(TranslationContainer<3>::*)(shared_ptr<TranslationContainer<3>::ChildType>,const Vec<3>&))&TranslationContainer<3>::add,
             (py::arg("item"), py::arg("translation")=Vec<3>(0.,0.,0.)))
        .def("add", &TranslationContainer3_add, (py::arg("item"), "c0", "c1", "c2"),
            "Alias for :meth:`~plask.geometry.Align3D.append`.\n"
        )

        .def("move_item", py::raw_function(&Container_move<TranslationContainer<3>>), "Move item in container")
    ;

    register_geometry_container_stack();
    
    register_geometry_container_lattice();
}



}} // namespace plask::python
