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

void register_geometry_container_lattice()
{
    init_Arange<2>();
    init_Arange<3>();

    py::class_<Lattice, shared_ptr<Lattice>, py::bases<GeometryObjectTransform<3>>,
            boost::noncopyable>("Lattice", "Lattice container that arranges its children in two-dimensional lattice.",
         py::init<const shared_ptr<typename Lattice::ChildType>&, const typename Lattice::DVec, const typename Lattice::DVec>
         ((py::arg("item"), py::arg("vec0") = plask::Primitive<3>::ZERO_VEC, py::arg("vec1") = plask::Primitive<3>::ZERO_VEC)))
        .def("__len__", &Lattice::getChildrenCount)
        //.def("__getitem__", &GeometryObject__getitem__)   //is in GeometryObject
        ;
}

    
}}
