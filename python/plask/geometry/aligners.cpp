#include "geometry.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <boost/algorithm/string.hpp>

#include <plask/geometry/container.h>


namespace plask { namespace python {

typedef align::Aligner2D<align::DIRECTION_TRAN> A2;
typedef align::Aligner2D<align::DIRECTION_LON> A2l;
typedef align::Aligner3D<align::DIRECTION_LON, align::DIRECTION_TRAN> A3;


struct Aligners_from_Python
{
    Aligners_from_Python() {
        boost::python::converter::registry::push_back(&convertible, &construct2, boost::python::type_id<A2>());
        boost::python::converter::registry::push_back(&convertible, &construct3, boost::python::type_id<A3>());
    }

    // Determine if obj_ptr can be converted into an Aligner
    static void* convertible(PyObject* obj_ptr) {
        if (!PyString_Check(obj_ptr)) return 0;
        return obj_ptr;
    }

    static void construct2(PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data) {
        std::string str = PyString_AsString(obj_ptr);
        boost::algorithm::to_lower(str);

        // Grab pointer to memory into which to construct the new Aligner
        void* storage = ((boost::python::converter::rvalue_from_python_storage<A2>*)data)->storage.bytes;

        if (str == "left" || str == "l") new(storage) align::Left();
        else if (str == "right" || str == "r") new(storage) align::Right();
        else if (str == "center" || str == "c") new(storage) align::Center();
        else {
            throw ValueError("wrong alignment specification");
        }

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }

    static void construct3(PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data) {
        std::string str = PyString_AsString(obj_ptr);
        boost::algorithm::to_lower(str);

        // Grab pointer to memory into which to construct the new Aligner
        void* storage = ((boost::python::converter::rvalue_from_python_storage<A3>*)data)->storage.bytes;

             if (str == "front left" || str == "fl" || str == "left front" || str == "lf") new(storage) align::FrontLeft();
        else if (str == "center left" || str == "cl" || str == "left center" || str == "lc") new(storage) align::CenterLeft();
        else if (str == "back left" || str == "bl" || str == "left back" || str == "lb") new(storage) align::BackLeft();
        else if (str == "front center" || str == "fc" || str == "center front" || str == "lf") new(storage) align::FrontCenter();
        else if (str == "center center" || str == "cc" || str == "center" || str == "c") new(storage) align::CenterCenter();
        else if (str == "back center" || str == "bl" || str == "center back" || str == "lb") new(storage) align::BackCenter();
        else if (str == "front right" || str == "fr" || str == "right front" || str == "rf") new(storage) align::FrontRight();
        else if (str == "center right" || str == "cr" || str == "right center" || str == "rc") new(storage) align::CenterRight();
        else if (str == "back right" || str == "br" || str == "right back" || str == "rb") new(storage) align::BackRight();
        else {
            throw ValueError("wrong alignment specification");
        }

        // Stash the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }
};

void register_geometry_aligners()
{
    py::object align_module { py::handle<>(py::borrowed(PyImport_AddModule("plask.geometry.align"))) };
    py::scope().attr("align") = align_module;
    py::scope scope = align_module;

    scope.attr("__doc__") =
        "This solver lists available aligners for geometry containers."; //TODO maybe more extensive description

    py::class_<A2, shared_ptr<A2>, boost::noncopyable>("Aligner2D", "Base for all 2D aligners", py::no_init)
            .def(py::self & py::other<A2l>())
    ;

    py::class_<A2l, shared_ptr<A2l>, boost::noncopyable>("AlignerLon", "Base for all longitudal aligners", py::no_init)
        .def(py::self & py::other<A2>())
    ;

    py::class_<A3, shared_ptr<A3>, boost::noncopyable>("Aligner3D", "Base for all 3D aligners", py::no_init);

    py::class_<align::ComposeAligner3D<plask::align::DIRECTION_LON, plask::align::DIRECTION_TRAN>,
               shared_ptr<align::ComposeAligner3D<plask::align::DIRECTION_LON, plask::align::DIRECTION_TRAN>>,
               py::bases<A3>>("ComposedAligner3D", "Three-dimensional aligner composed of two 2D aligners", py::no_init);
    py::class_<align::ComposeAligner3D<plask::align::DIRECTION_TRAN, plask::align::DIRECTION_LON>,
               shared_ptr<align::ComposeAligner3D<plask::align::DIRECTION_TRAN, plask::align::DIRECTION_LON>>,
               py::bases<A3>>("ComposedAligner3D", "Three-dimensional  aligner composed of two 2D aligners", py::no_init);


    py::class_<align::Tran, shared_ptr<align::Tran>, py::bases<A2>>("Tran",
                                                     "Two-dimensional aligner with arbitrary child shift\n\n"
                                                     "Tran(tran=0.0)\n    create aligner with child shifted to shift\n",
                                                     py::init<double>((py::arg("tran")=0.))
                                                    );

    py::class_<align::Lon, shared_ptr<align::Lon>, py::bases<A2l>>("Lon",
                                                     "Aligner with arbitrary longitudal child shift for construction of 3D aligners\n\n"
                                                     "Lon(lon=0.0)\n    create aligner with child shifted to shift\n",
                                                     py::init<double>((py::arg("lon")=0.))
                                                    );

    py::class_<align::LonTran, shared_ptr<align::LonTran>, py::bases<A3>>("LonTran",
                                                     "Three-dimensional aligner with arbitrary child shift\n\n"
                                                     "LonTran(lon_shift=0.0, tran_shift=0.0)\n    create aligner with child shifted to [lon, tran]\n",
                                                     py::init<double, double>((py::arg("lon")=0., py::arg("tran")=0.))
                                                    );

    py::class_<align::Left, shared_ptr<align::Left>, py::bases<A2>>("Left", "Two-dimensional aligner: left");
    py::class_<align::Right, shared_ptr<align::Right>, py::bases<A2>>("Right", "Two-dimensional aligner: right");
    py::class_<align::Center, shared_ptr<align::Center>, py::bases<A2>> center("Center", "Two-dimensional aligner: center");
    scope.attr("TranCenter") = center;

    py::class_<align::Front, shared_ptr<align::Front>, py::bases<A2l>>("Front", "Longitudal aligner: front");
    py::class_<align::Back, shared_ptr<align::Back>, py::bases<A2l>>("Back", "Longitudal aligner: back");
    py::class_<align::LonCenter, shared_ptr<align::LonCenter>, py::bases<A2l>>("LonCenter", "Longitudal aligner: center");

    py::class_<align::FrontLeft, shared_ptr<align::FrontLeft>, py::bases<A3>>("FrontLeft", "Three-dimesional aligner: front left");
    py::class_<align::FrontRight, shared_ptr<align::FrontRight>, py::bases<A3>>("FrontRight", "Three-dimesional aligner: front right");
    py::class_<align::FrontCenter, shared_ptr<align::FrontCenter>, py::bases<A3>>("FrontCenter", "Three-dimesional aligner: front center");
    py::class_<align::BackLeft, shared_ptr<align::BackLeft>, py::bases<A3>>("BackLeft", "Three-dimesional aligner: back left");
    py::class_<align::BackRight, shared_ptr<align::BackRight>, py::bases<A3>>("BackRight", "Three-dimesional aligner: back right");
    py::class_<align::BackCenter, shared_ptr<align::BackCenter>, py::bases<A3>>("BackCenter", "Three-dimesional aligner: back center");
    py::class_<align::CenterLeft, shared_ptr<align::CenterLeft>, py::bases<A3>>("CenterLeft", "Three-dimesional aligner: center left");
    py::class_<align::CenterRight, shared_ptr<align::CenterRight>, py::bases<A3>>("CenterRight", "Three-dimesional aligner: center right");
    py::class_<align::CenterCenter, shared_ptr<align::CenterCenter>, py::bases<A3>>("CenterCenter", "Three-dimesional aligner: center");

    // Register string conventers
    Aligners_from_Python();

}





}} // namespace plask::python
