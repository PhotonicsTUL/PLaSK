#include <cmath>
#include <boost/python.hpp>
namespace py = boost::python;

#include <plask/plask.hpp>

//// Vector ///////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef plask::Vec<2,double> MyVec;

std::vector<MyVec> getVecs() {
    std::vector<MyVec> result;
    result.push_back(MyVec(1,2));
    result.push_back(MyVec(3,4));
    result.push_back(MyVec(5,6));
    result.push_back(MyVec(7,8));
    result.push_back(MyVec(9,10));

    std::cerr << "Vector array: ";
    for (auto p = (double*)&result[0]; p < (double*)&(*result.end()); ++p) std::cerr << *p << " ";
    std::cerr << "\n";

    return result;
}



//// Material /////////////////////////////////////////////////////////////////////////////////////////////////////////

struct MyMaterial : public plask::Material {

    virtual std::string name() const { return "MyMaterial"; }

    virtual double VBO(double T) const { return 0.5*T; }

    virtual double chi(double T, char P) const { std::cerr << "MyMaterial: " << P << "\n"; return 1.0; }

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

double call_chi(plask::shared_ptr<plask::Material> mat, std::string p) {
    return mat->chi(p[0]);
}

void print_ptr(py::object o) {
    std::cerr << "ptr: " << o.ptr() << "\n";
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BOOST_PYTHON_MODULE(plasktest)
{
    py::def("getVecs", &getVecs);



    py::def("addMyMaterial", &addMyMaterial);

    py::def("materialName", &materialName);
    py::def("materialVBO", &materialVBO);

    py::def("call_chi", &call_chi);

    py::def("print_ptr", &print_ptr);

}
