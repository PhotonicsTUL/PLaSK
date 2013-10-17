#include <plask/geometry/align_container.h>

#include "geometry.h"
#include "../../util/raw_constructor.h"

namespace plask { namespace python {

extern AxisNames current_axes;
    
template <int dim, typename Primitive<dim>::Direction direction>
static shared_ptr<AlignContainer<dim,direction>> AlignContainer__init__(py::tuple args, py::dict kwargs)
{
    if (py::len(args) != 1)
        throw TypeError("__init__() takes exactly 1 non-keyword argument1 (%1% given)", py::len(args));
    typedef AlignContainer<dim,direction> AlignContainerT;
    return make_shared<AlignContainerT>(py::extract<typename AlignContainerT::Aligner>(kwargs));
}

template <int dim, typename Primitive<dim>::Direction direction>
PathHints::Hint AlignContainer_add(py::tuple args, py::dict kwargs) {
    parseKwargs("append", args, kwargs, "self", "item");
    typedef AlignContainer<dim,direction> AlignContainerT;
    AlignContainerT* self = py::extract<AlignContainerT*>(args[0]);
    shared_ptr<typename AlignContainerT::ChildType> child = py::extract<shared_ptr<typename AlignContainerT::ChildType>>(args[1]);
    if (py::len(kwargs) == 0)
        return self->add(child);
    else
        return self->add(child, py::extract<typename AlignContainerT::ChildAligner>(kwargs));
}

#define LONG Primitive<3>::DIRECTION_LONG
#define TRAN Primitive<3>::DIRECTION_TRAN
#define VERT Primitive<3>::DIRECTION_VERT

py::object makeAlignContainer2D(py::tuple args, py::dict kwargs)
{

    auto dict_fun = [&](const std::string& name) {
        return kwargs.has_key(name)? boost::optional<double>(py::extract<double>(kwargs[name])) : boost::optional<double>();
    };

    auto aligner_tran = align::fromDictionary<TRAN>(dict_fun, current_axes);
    auto aligner_vert = align::fromDictionary<VERT>(dict_fun, current_axes);

    if (!aligner_tran.isNull()) {
        if (!aligner_vert.isNull())
            throw TypeError("Inconsistent aligner specification");
        else
            return py::object(make_shared<AlignContainer<2,Primitive<2>::DIRECTION_TRAN>>(aligner_tran));
    }
    if (!aligner_vert.isNull())
        return py::object(make_shared<AlignContainer<2,Primitive<2>::DIRECTION_VERT>>(aligner_vert));

    throw TypeError("AlignContainer2D() got an unexpected keyword argument '%1%'", std::string(py::extract<std::string>(kwargs.keys()[0])));
}

py::object makeAlignContainer3D(py::tuple args, py::dict kwargs)
{

    auto dict_fun = [&](const std::string& name) {
        return kwargs.has_key(name)? boost::optional<double>(py::extract<double>(kwargs[name])) : boost::optional<double>();
    };

    auto aligner_long = align::fromDictionary<LONG>(dict_fun, current_axes);
    auto aligner_tran = align::fromDictionary<TRAN>(dict_fun, current_axes);
    auto aligner_vert = align::fromDictionary<VERT>(dict_fun, current_axes);

    if (!aligner_long.isNull()) {
        if (!aligner_tran.isNull() || !aligner_vert.isNull())
            throw TypeError("Inconsistent aligner specification");
        else
            return py::object(make_shared<AlignContainer<3,LONG>>(aligner_long));
    }
    if (!aligner_tran.isNull()) {
        if (!aligner_vert.isNull())
            throw TypeError("Inconsistent aligner specification");
        else
            return py::object(make_shared<AlignContainer<3,TRAN>>(aligner_tran));
    }
    if (!aligner_vert.isNull())
        return py::object(make_shared<AlignContainer<3,VERT>>(aligner_vert));

    throw TypeError("AlignContainer3D() got an unexpected keyword argument '%1%'", std::string(py::extract<std::string>(kwargs.keys()[0])));
}

template <int dim, typename Primitive<dim>::Direction direction>
static void register_geometry_aligncontainer(const std::string& suffix)
{
    typedef AlignContainer<dim,direction> AlignContainerT;

    py::class_<AlignContainerT, shared_ptr<AlignContainerT>, py::bases<GeometryObjectContainer<dim>>, boost::noncopyable>
        (("AlignContainer"+suffix).c_str(),
        format("Container that aligns its content along axis%1%\n\n"
        "AlignContainer%2%(**kwargs)\n"
        "    Create the container, with its alignment specified in kwargs.\n\n"
        "See geometry.AlignContainer3D().\n", current_axes[3-dim+int(direction)], suffix).c_str(),
        py::no_init)
        .def("__init__", raw_constructor(AlignContainer__init__<dim,direction>, 1))
        .def("add", raw_function(AlignContainer_add<dim,direction>), "Add object to the container")
        .add_property("aligner", py::make_function(&AlignContainerT::getAligner, py::return_value_policy<py::return_by_value>()),
                                 &AlignContainerT::setAligner, "Container alignment")
        .def("move_item", py::raw_function(&Container_move<AlignContainerT>), "Move item in container")
    ;
}

void register_geometry_aligncontainers()
{
    register_geometry_aligncontainer<2, Primitive<2>::DIRECTION_TRAN>("Tran2D");
    register_geometry_aligncontainer<2, Primitive<2>::DIRECTION_VERT>("Vert2D");

    register_geometry_aligncontainer<3, Primitive<3>::DIRECTION_LONG>("Long3D");
    register_geometry_aligncontainer<3, Primitive<3>::DIRECTION_TRAN>("Tran3D");
    register_geometry_aligncontainer<3, Primitive<3>::DIRECTION_VERT>("Vert3D");

    py::def("AlignContainer2D", py::raw_function(&makeAlignContainer2D));
    py::scope().attr("AlignContainer2D").attr("__doc__") = "Create container that aligns its content along one axis (2D version)";

    py::def("AlignContainer3D", py::raw_function(&makeAlignContainer3D));
    py::scope().attr("AlignContainer3D").attr("__doc__") = "Create container that aligns its content along one axis (3D version)";
}


}} // namespace  plask::python
