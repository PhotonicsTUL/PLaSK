#include "geometry.h"

#include <plask/geometry/lattice.h>

namespace plask { namespace python {

template <int dim> inline const char* ArangeName();
template <> inline const char* ArangeName<2>() { return "Arange2D"; }
template <> inline const char* ArangeName<3>() { return "Arange3D"; }

template <int dim> inline const char* ArangeDoc();
template <> inline const char* ArangeDoc<2>() { return
    "Arange2D(item, step, count)\n"
    "Container that repeats its item, shifting each repetition by the specified step.\n\n"
    "Args:\n"
    "    item (GeometryObject2D): Object to repeat.\n"
    "    step (vec): 2D vector, by which each repetition is shifted from the previous\n"
    "                one.\n"
    "    count (int): Number of item repetitions.\n"
    "    warning (bool): Boolean value indicating is overlapping warnings\n"
    "                    are displayed.\n"
    ;
}
template <> inline const char* ArangeDoc<3>() { return
    "Arange3D(item, step, count)\n"
    "Container that repeats its item, shifting each repetition by the specified step.\n\n"
    "Args:\n"
    "    item (GeometryObject3D): Object to repeat.\n"
    "    step (vec): 3D vector, by which each repetition is shifted from the previous\n"
    "                one.\n"
    "    count (int): Number of item repetitions.\n"
    "    warning (bool): Boolean value indicating is overlapping warnings\n"
    "                    are displayed.\n"
    ;
}

template <int dim>
inline static void init_Arange()
{
    py::class_<ArrangeContainer<dim>, shared_ptr<ArrangeContainer<dim>>, py::bases<GeometryObjectTransform<dim>>,
               boost::noncopyable>(ArangeName<dim>(), ArangeDoc<dim>(),
         py::init<const shared_ptr<typename ArrangeContainer<dim>::ChildType>&, const typename ArrangeContainer<dim>::DVec, unsigned>
         ((py::arg("item"), "step", "count", py::arg("warning")=true)))
        .add_property("step", &ArrangeContainer<dim>::getTranslation, &ArrangeContainer<dim>::setTranslation,
                      "Vector, by which each repetition is shifted from the previous one.")
        .add_property("count", &ArrangeContainer<dim>::getRepeatCount, &ArrangeContainer<dim>::setRepeatCount,
                      "Number of item repetitions.")
        .def_readwrite("warning", &ArrangeContainer<dim>::warn_overlapping,
                       "Boolean value indicating is overlapping warnings are displayed.")
        .def("__len__", &ArrangeContainer<dim>::getChildrenCount)
        ;
}

//shared_ptr<GeometryObject> GeometryObject__getitem__(py::object oself, int i);

void lattice_set_segments(py::object self, const py::object& value) {
    std::vector< std::vector<Vec<2, int>> > segments;
    py::stl_input_iterator<py::object> segments_it(value), segments_end_it;
    for ( ; segments_it != segments_end_it; ++segments_it) {
        std::vector<Vec<2, int>> segment;
        py::stl_input_iterator<py::object> points_it(*segments_it), points_end_it;
        for ( ; points_it != points_end_it; ++points_it) {
            py::stl_input_iterator<int> coord_it(*points_it), coord_end_it;
            int a = *coord_it;
            ++coord_it;
            int b = *coord_it;
            ++coord_it;
            if (coord_it != coord_end_it)
                throw IndexError("too many elements in lattice segment boundary patch point (exactly 2 coordinates are required)");;
            segment.emplace_back(a, b);
        }
        segments.push_back(std::move(segment));
    }
    Lattice& l = py::extract<Lattice&>(self);
    l.setSegments(std::move(segments));
}

py::list lattice_get_segments(const Lattice& self) {
    py::list result;
    for (auto segment: self.segments) {
        py::list psegment;
        for (auto point: segment) {
            psegment.append(py::make_tuple(point[0], point[1]));
        }
        result.append(psegment);
    }
    return result;
}

void register_geometry_container_lattice()
{
    init_Arange<2>();
    init_Arange<3>();

    py::class_<Lattice, shared_ptr<Lattice>, py::bases<GeometryObjectTransform<3>>,
            boost::noncopyable>("Lattice", "Lattice container that arranges its children in two-dimensional lattice.",
         py::init<const shared_ptr<typename Lattice::ChildType>&, const typename Lattice::DVec, const typename Lattice::DVec>
         ((py::arg("item"), py::arg("vec0") = plask::Primitive<3>::ZERO_VEC, py::arg("vec1") = plask::Primitive<3>::ZERO_VEC)))
        .def("__len__", &Lattice::getChildrenCount)
        .def("set_segments", lattice_set_segments, "Set polygons specifying lattice boundaries.")  //TODO get_segments - format? tuple of tuple of int pairs(?)
        .add_property("segments", lattice_get_segments, "List of polygons limiting lattice segments.")
        .def_readonly("vec0", &Lattice::vec0, "First lattice vector.")  //TODO allow to set it 
        .def_readonly("vec1", &Lattice::vec1, "Second lattice vector.")  //TODO allow to set it 
        //.def("__getitem__", &GeometryObject__getitem__)   //is in GeometryObject
        ;
}

    
}}
