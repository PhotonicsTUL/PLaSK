#include <plask/geometry/leaf.h>

#include "geometry.h"

namespace plask { namespace python {


// Rectangle constructor wraps
static shared_ptr<Block<2>> Rectangle_constructor(double w, double h, shared_ptr<Material> material) {
    //TODO: check plask.axes to set components properly
    return shared_ptr<Block<2>> ( new Block<2>(Vec<2,double>(w,h), material) );
}
static shared_ptr<Block<2>> Rectangle_constructor(const Vec<2,double>& size, shared_ptr<Material> material) {
    return shared_ptr<Block<2>> ( new Block<2>(size, material) );
}

// Cuboid constructor wraps
static shared_ptr<Block<3>> Cuboid_constructor(double w, double h, double d, shared_ptr<Material> material) {
    //TODO: check plask.axes to set components properly
    return shared_ptr<Block<3>> ( new Block<3>(Vec<3,double>(w,h,d), material) );
}
static shared_ptr<Block<3>> Cuboid_constructor(const Vec<3,double>& size, shared_ptr<Material> material) {
    return shared_ptr<Block<3>> ( new Block<3>(size, material) );
}





void register_geometry_leafs()
{
    py::class_<Block<2>, shared_ptr<Block<2>>>("Rectangle", "Geometry object: a rectangle filled with one material")
//         .def("__init__", py::make_constructor(&Rectangle_constructor)
    ;
}




}} // namespace plask::python
