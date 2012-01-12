#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <plask/geometry/container.h>

#include "geometry.h"

namespace plask { namespace python {

DECLARE_GEOMETRY_ELEMENT_23D(TranslationContainer, "TranslationContainer",
                             "Geometry elements container in which every child has an associated translation vector ("," version)")
{
    GEOMETRY_ELEMENT_23D_DEFAULT(TranslationContainer, GeometryElementContainer<dim>)
        .def("add", &TranslationContainer<dim>::add, "Add new element to the container")
    ;
}

DECLARE_GEOMETRY_ELEMENT_23D(MultiStackContainer, "MultiStackContainer",
                             "Stack container which repeats its contents\n\n"
                             "MultiStackContainer","(repeatCount = 1, baseLevel = 0) -> Create new multi-stack with repeatCount repetitions")
{
    GEOMETRY_ELEMENT_23D_DEFAULT(MultiStackContainer, typename MultiStackContainer<dim>::UpperClass)
        .def_readwrite("repeats", &MultiStackContainer<dim>::repeat_count, "Number of repeats of the stack content")
    ;
}

void register_geometry_container()
{
    // Path hints (jest for future use)

    py::class_<PathHints::Hint>("PathHint",
                                "Objects of this class are returned by methods which add new elements to containers and can be added to path Hints",
                                py::no_init);

    py::class_<PathHints>("PathHints", "Hints are used to to find unique path for every element in the geometry tree, "
                                       "even if this element is inserted to geometry graph in more than one place.")
        .def("add", (void (PathHints::*)(const PathHints::Hint&)) &PathHints::addHint, "Add hint to hints map.")
        .def(py::self += py::other<PathHints::Hint>())
    ;

    // Translation container
    init_TranslationContainer<2>();
    init_TranslationContainer<3>();

    // Stack container

    py::class_<StackContainer2d>("StackContainer2D",
        "Container that organizes its childern in vertical stack (2D version)\n\n"
        "StackContainer2D(baseLevel = 0) -> Create the stack with the bottom side of the first element at the baseLevel (in container local coordinates)",
        py::init<double>())
        .def("add", &StackContainer2d::add, "Add new element to the container")
    ;

    py::class_<StackContainer3d>("StackContainer3D",
        "Container that organizes its childern in vertical stack (3D version)\n\n"
        "StackContainer3D(baseLevel = 0) -> Create the stack with the bottom side of the first element at the baseLevel (in container local coordinates)",
        py::init<double>())
        .def("add", &StackContainer3d::add, "Add new element to the container")
    ;

}



}} // namespace plask::python
