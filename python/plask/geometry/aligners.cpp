#include "geometry.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <boost/algorithm/string.hpp>

#include <plask/geometry/container.h>


namespace plask { namespace python {

typedef align::AxisAligner<Primitive<3>::DIRECTION_TRAN> A2;
typedef align::AxisAligner<Primitive<3>::DIRECTION_LONG> A2l;
typedef align::Aligner3D<Primitive<3>::DIRECTION_LONG, Primitive<3>::DIRECTION_TRAN> A3;

namespace detail {

//     template <typename AlignerT>
//     struct Aligner_to_Python {
//         static PyObject* convert(const AlignerT& aligner) {
//             py::dict dict;
//             for (auto i: aligner.asDict(config.axes)) {
//                 dict[i.first] = i.second;
//             }
//             return py::incref(dict.ptr());
//         }
//     };
//
//     static void* aligner_convertible(PyObject* obj) {
//         if (!PyDict_Check(obj)) return 0;
//         return obj;
//     }
//
//     static void construct2(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
//         // Grab pointer to memory into which to construct the new Aligner
//         void* storage = ((boost::python::converter::rvalue_from_python_storage<AlignerT>*)data)->storage.bytes;
//
//         py::dict dict = py::handle<>(obj);
//         py::stl_input_iterator<std::string> begin(dict), end;
//         std::map<std::string, double> map;
//         for (auto key = begin; key != end; ++key)
//             map[key] = py::extract<double>(dict[key]);
//
//         if (str == "left" || str == "l") new(storage) align::Left(0.0);
//
//         // Stash the memory chunk pointer for later use by boost.python
//         data->convertible = storage;
//     }
}

void register_geometry_aligners()
{
//     py::object align_module { py::handle<>(py::borrowed(PyImport_AddModule("plask.geometry.align"))) };
//     py::scope().attr("align") = align_module;
//     py::scope scope = align_module;
//
//     scope.attr("__doc__") =
//         "This solver lists available aligners for geometry containers."; //TODO maybe more extensive description
//
//     py::class_<A2, shared_ptr<A2>, boost::noncopyable>("Aligner2D", "Base for all 2D aligners", py::no_init)    //TODO Aligner2D -> AxisAligner
//             .def(py::self & py::other<A2l>())
//     ;
//
//     py::class_<A2l, shared_ptr<A2l>, boost::noncopyable>("AlignerLon", "Base for all longitudal aligners", py::no_init)
//         .def(py::self & py::other<A2>())
//     ;
//
//     py::class_<A3, shared_ptr<A3>, boost::noncopyable>("Aligner3D", "Base for all 3D aligners", py::no_init);
//
//     py::class_<align::ComposeAligner3D<Primitive<3>::DIRECTION_LONG, Primitive<3>::DIRECTION_TRAN>,
//                shared_ptr<align::ComposeAligner3D<Primitive<3>::DIRECTION_LONG, Primitive<3>::DIRECTION_TRAN>>,
//                py::bases<A3>>("ComposedAligner3D", "Three-dimensional aligner composed of two 2D aligners", py::no_init);
//     py::class_<align::ComposeAligner3D<Primitive<3>::DIRECTION_TRAN, Primitive<3>::DIRECTION_LONG>,
//                shared_ptr<align::ComposeAligner3D<Primitive<3>::DIRECTION_TRAN, Primitive<3>::DIRECTION_LONG>>,
//                py::bases<A3>>("ComposedAligner3D", "Three-dimensional  aligner composed of two 2D aligners", py::no_init);
//
//
//     py::class_<align::Tran, shared_ptr<align::Tran>, py::bases<A2>>("Tran",
//                                                      "Two-dimensional aligner with arbitrary child shift\n\n"
//                                                      "Tran(tran=0.0)\n    create aligner with child shifted to shift\n",
//                                                      py::init<double>((py::arg("tran")=0.))
//                                                     );
//
//     py::class_<align::Long, shared_ptr<align::Long>, py::bases<A2l>>("Long",
//                                                      "Aligner with arbitrary longitudal child shift for construction of 3D aligners\n\n"
//                                                      "Lon(lon=0.0)\n    create aligner with child shifted to shift\n",
//                                                      py::init<double>((py::arg("lon")=0.))
//                                                     );
//
//     //TODO aligner for two directions are provided intependend now and use ComposeAligner3D
//     /*py::class_<align::LonTran, shared_ptr<align::LonTran>, py::bases<A3>>("LonTran",
//                                                      "Three-dimensional aligner with arbitrary child shift\n\n"
//                                                      "LonTran(lon_shift=0.0, tran_shift=0.0)\n    create aligner with child shifted to [lon, tran]\n",
//                                                      py::init<double, double>((py::arg("lon")=0., py::arg("tran")=0.))
//                                                     );*/
//
//     //TODO this now requires double argument
//    /* py::class_<align::Left, shared_ptr<align::Left>, py::bases<A2>>("Left", "Two-dimensional aligner: left");
//     py::class_<align::Right, shared_ptr<align::Right>, py::bases<A2>>("Right", "Two-dimensional aligner: right");
//     py::class_<align::Center, shared_ptr<align::Center>, py::bases<A2>> center("Center", "Two-dimensional aligner: center");
//     scope.attr("TranCenter") = center;
//
//     py::class_<align::Front, shared_ptr<align::Front>, py::bases<A2l>>("Front", "Longitudal aligner: front");
//     py::class_<align::Back, shared_ptr<align::Back>, py::bases<A2l>>("Back", "Longitudal aligner: back");
//     py::class_<align::LongCenter, shared_ptr<align::LongCenter>, py::bases<A2l>>("LongCenter", "Longitudal aligner: center");*/
//
//     //TODO aligner for two directions are provided intependend now and use ComposeAligner3D
// /*    py::class_<align::FrontLeft, shared_ptr<align::FrontLeft>, py::bases<A3>>("FrontLeft", "Three-dimesional aligner: front left");
//     py::class_<align::FrontRight, shared_ptr<align::FrontRight>, py::bases<A3>>("FrontRight", "Three-dimesional aligner: front right");
//     py::class_<align::FrontCenter, shared_ptr<align::FrontCenter>, py::bases<A3>>("FrontCenter", "Three-dimesional aligner: front center");
//     py::class_<align::BackLeft, shared_ptr<align::BackLeft>, py::bases<A3>>("BackLeft", "Three-dimesional aligner: back left");
//     py::class_<align::BackRight, shared_ptr<align::BackRight>, py::bases<A3>>("BackRight", "Three-dimesional aligner: back right");
//     py::class_<align::BackCenter, shared_ptr<align::BackCenter>, py::bases<A3>>("BackCenter", "Three-dimesional aligner: back center");
//     py::class_<align::CenterLeft, shared_ptr<align::CenterLeft>, py::bases<A3>>("CenterLeft", "Three-dimesional aligner: center left");
//     py::class_<align::CenterRight, shared_ptr<align::CenterRight>, py::bases<A3>>("CenterRight", "Three-dimesional aligner: center right");
//     py::class_<align::CenterCenter, shared_ptr<align::CenterCenter>, py::bases<A3>>("CenterCenter", "Three-dimesional aligner: center");*/
//
//     // Register string conventers
//     detail::Aligners_from_Python();

}





}} // namespace plask::python
