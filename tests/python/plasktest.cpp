#include <plask/plask.hpp>
#include <plask/python.hpp>
namespace py = boost::python;

//// Vector ///////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef plask::Vec<2,double> MyVec;

std::vector<MyVec> getVecs() {
    std::vector<MyVec> result;
    result.push_back(MyVec(1,2));
    result.push_back(MyVec(3,4));
    result.push_back(MyVec(5,6));
    result.push_back(MyVec(7,8));
    result.push_back(MyVec(9,10));

    // std::cerr << "Vector array: ";
    // for (auto p = (double*)&result[0]; p < (double*)&(*result.end()); ++p) std::cerr << *p << " ";
    // std::cerr << "\n";

    return result;
}



//// Material /////////////////////////////////////////////////////////////////////////////////////////////////////////

struct MyMaterial : public plask::Material {

    virtual std::string name() const { return "MyMaterial"; }

    virtual Material::Kind kind() const { return Material::NONE; }

    virtual double VBO(double T) const { return 0.5*T; }

    virtual double chi(double T, char P) const { std::cerr << "MyMaterial: " << P << "\n"; return 1.0; }

};

void addMyMaterial(plask::MaterialsDB& DB) {
    DB.add<MyMaterial, false, false>("MyMaterial");
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

plask::shared_ptr<plask::GeometryElement> getExtrusion(plask::shared_ptr<plask::GeometryElementD<2>>c, double l) {
    return plask::make_shared<plask::Extrusion>(c,l);
}

bool isEmpy(plask::shared_ptr<plask::GeometryElement> p) {
    return !p;
}

std::string materialTypeId(plask::shared_ptr<plask::Material> material) {
    return typeid(*material).name();
}

//// Provider & Receiver /////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ReceiverTest : plask::Module {
    virtual std::string getName() const { return "Receiver Test"; }
    plask::ReceiverFor<plask::EffectiveIndex> inNeff;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BOOST_PYTHON_MODULE(plasktest)
{
    py::def("getVecs", &getVecs);

    py::def("addMyMaterial", &addMyMaterial);

    py::def("materialName", &materialName);
    py::def("materialVBO", &materialVBO);

    py::def("call_chi", &call_chi);

    py::def("print_ptr", &print_ptr);

    py::def("getExtrusion", &getExtrusion);

    py::def("isEmpty", &isEmpy);

    py::def("materialTypeId", &materialTypeId);


    plask::python::RegisterProvider<plask::EffectiveIndex>();
    plask::python::ExportModule<ReceiverTest>("ReceiverTest")
        .add_receiver("inNeff", &ReceiverTest::inNeff, "Test receiver")
    ;
}
