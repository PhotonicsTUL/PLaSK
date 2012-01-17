#include <boost/python.hpp>
namespace py = boost::python;

#include <config.h>
#include <plask/material/material.h>
#include <plask/geometry/leaf.h>

//// Material /////////////////////////////////////////////////////////////////////////////////////////////////////////

struct MyMaterial : public plask::Material {

    virtual std::string name() const { return "MyMaterial"; }

    virtual double VBO(double T) const { return 0.5*T; }

};

plask::Material* construct_my_material(const std::vector<double>&, plask::MaterialsDB::DOPING_AMOUNT_TYPE, double) {
    return new MyMaterial();
}

void addMyMaterial(plask::MaterialsDB& DB) {
    DB.add("MyMaterial", construct_my_material);
}


std::string materialName(std::string m, plask::MaterialsDB& DB) {
    plask::shared_ptr<plask::Material> mat = DB.get(m);
    return mat->name();
}

double materialVBO(std::string m, plask::MaterialsDB& DB, double T) {
    plask::shared_ptr<plask::Material> mat = DB.get(m);
    return mat->VBO(T);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BOOST_PYTHON_MODULE(plasktest)
{
    py::def("addMyMaterial", &addMyMaterial);

    py::def("materialName", &materialName);
    py::def("materialVBO", &materialVBO);
}
