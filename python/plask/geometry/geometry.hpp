#ifndef PLASK__PYTHON_GEOMETRY_H
#define PLASK__PYTHON_GEOMETRY_H

#include "../python_globals.hpp"
#include "plask/geometry/path.hpp"
#include "plask/geometry/align.hpp"


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

#define GEOMETRY_ELEMENT_23D_DOC(cls, method, doc2, doc3) \
    template <int dim> inline static const char* cls##_##method##_doc(); \
    template <> inline const char* cls##_##method##_doc<2>() { return doc2; }; \
    template <> inline const char* cls##_##method##_doc<3>() { return doc3; };

#define USE_23D_DOC(cls, method) cls##_##method##_doc<dim>()

#define GEOMETRY_ELEMENT_23D_DEFAULT(cls, base) \
    py::class_<cls<dim>, shared_ptr<cls<dim>>, py::bases<base>, boost::noncopyable> cls##_registrant (cls##_pyname<dim>(), cls##_pydoc<dim>()); \
     cls##_registrant

#define GEOMETRY_ELEMENT_23D(cls, base, init) \
    py::class_<cls<dim>, shared_ptr<cls<dim>>, py::bases<base>, boost::noncopyable> cls##_registrant (cls##_pyname<dim>(), cls##_pydoc<dim>(), init); \
    cls##_registrant

namespace plask { namespace python {

template<typename AlignerT>
inline AlignerT update_aligner_from_dict(const AlignerT& original_aligner, const py::dict& dict) {
    std::map<std::string, double> map = dict_to_map<std::string, double>(dict);
    AlignerT aligner = align::fromDictionary([&](const std::string& name) -> plask::optional<double> {
                                                plask::optional<double> result;
                                                auto found = map.find(name);
                                                if (found != map.end()) {
                                                    result.reset(found->second);
                                                    map.erase(found);
                                                }
                                                return result;
                                            },
                                            current_axes,
                                            original_aligner
                                            );
    if (!map.empty()) throw TypeError(u8"Got unexpected alignment keyword '{0}'", map.begin()->first);
    return aligner;
}


template <typename ContainerT>
py::object Container_move(py::tuple args, py::dict kwargs) {
    parseKwargs("move_item", args, kwargs, "self", "path");
    ContainerT* self = py::extract<ContainerT*>(args[0]);
    py::extract<int> ei(args[1]);
    int i;
    if (ei.check()) {
        i = ei();
        if (i < 0) i += int(self->getChildrenCount());
        if (i < 0 || std::size_t(i) >= self->getChildrenCount()) {
            throw IndexError("{0} index {1} out of range (0 <= index < {2})",
                std::string(py::extract<std::string>(args[0].attr("__class__").attr("__name__"))), i, self->getChildrenCount());
        }
    } else {
        PathHints path = py::extract<PathHints>(args[1]);
        auto specified = path.getTranslationChildren<ContainerT::DIM>(*self);
        if (specified.size() != 1)
            throw ValueError("Non-unique item specified");
        auto children = self->getChildrenVector();
        auto it = std::find(children.begin(), children.end(), *specified.begin());
        if (it != children.end()) i = it - children.begin();
    }
    auto current_aligner = self->getAligners()[i];
    typename ContainerT::ChildAligner aligner = update_aligner_from_dict(current_aligner, kwargs);
    self->move(i, aligner);
    return py::object();
}


template <int dim> struct GeometryObjectD_vector_args;

template<> struct GeometryObjectD_vector_args<2> {
    static inline const py::detail::keywords<2> args() {
        return py::arg("c0"), py::arg("c1");
    }
    template <size_t nargs>
    static inline py::detail::keywords<nargs+2> args(const py::detail::keywords<nargs>& other) {
        return other, py::arg("c0"), py::arg("c1");
    }
};

template<> struct GeometryObjectD_vector_args<3> {
    static inline const py::detail::keywords<3> args() {
        return py::arg("c0"), py::arg("c1"), py::arg("c2");
    }
    template <size_t nargs>
    static inline const py::detail::keywords<nargs+3> args(const py::detail::keywords<nargs>& other) {
        return other, py::arg("c0"), py::arg("c1"), py::arg("c2");
    }
};


struct PythonGeometryObjectChanger: public GeometryObject::Changer {

    py::object callable;

    PythonGeometryObjectChanger(const py::object& callable): callable(callable) {}

    bool apply(shared_ptr<GeometryObject>& to_change, Vec<3, double>* translation = 0) const override;
};


}} // namespace plask::python
#endif // PLASK__PYTHON_GEOMETRY_H
