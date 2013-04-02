#ifndef PLASK__PYTHON_GEOMETRY_H
#define PLASK__PYTHON_GEOMETRY_H

#include "../python_globals.h"
#include <plask/geometry/path.h>


#define DECLARE_GEOMETRY_ELEMENT_23D(cls, pyname, pydoc1, pydoc2) \
    template <int dim> inline static const char* cls##_pyname () { return pyname; } \
    template <> inline const char* cls##_pyname<2> () { return pyname "2D"; } \
    template <> inline const char* cls##_pyname<3> () { return pyname "3D"; } \
    template <int dim> inline static const char* cls##_pydoc () { return pydoc1 pydoc2; } \
    template <> inline const char* cls##_pydoc<2> () { return pydoc1 "2D" pydoc2; } \
    template <> inline const char* cls##_pydoc<3> () { return pydoc1 "3D" pydoc2; } \
    template <int dim> inline static void init_##cls()

#define ABSTRACT_GEOMETRY_ELEMENT_23D(cls, base) \
    py::class_<cls<dim>, shared_ptr<cls<dim>>, py::bases<base>, boost::noncopyable> \
    cls##_registrant (cls##_pyname<dim>(), cls##_pydoc<dim>(), py::no_init); cls##_registrant


#define GEOMETRY_ELEMENT_23D_DEFAULT(cls, base) \
    py::class_<cls<dim>, shared_ptr<cls<dim>>, py::bases<base>, boost::noncopyable> cls##_registrant (cls##_pyname<dim>(), cls##_pydoc<dim>()); \
     cls##_registrant

#define GEOMETRY_ELEMENT_23D(cls, base, init) \
    py::class_<cls<dim>, shared_ptr<cls<dim>>, py::bases<base>, boost::noncopyable> cls##_registrant (cls##_pyname<dim>(), cls##_pydoc<dim>(), init); \
    cls##_registrant

namespace plask { namespace python {

template <typename ContainerT>
py::object Container_move(py::tuple args, py::dict kwargs) {
    parseKwargs("move", args, kwargs, "self", "item");
    ContainerT* self = py::extract<ContainerT*>(args[0]);
    typename ContainerT::ChildAligner aligner = py::extract<typename ContainerT::ChildAligner>(kwargs);
    try {
        size_t index = py::extract<size_t>(args[1]);
        self->move(index, aligner);
    } catch (py::error_already_set) {
        PyErr_Clear();
        PathHints path = py::extract<PathHints>(args[1]);
        auto children = path.getTranslationChildren<ContainerT::DIM>(*self);
        if (children.size() != 1)
            throw ValueError("Non-unique item specified");
        self->move(*children.begin(), aligner);
    }
    return py::object();
}

}} // namespace plask::python
#endif // PLASK__PYTHON_GEOMETRY_H
