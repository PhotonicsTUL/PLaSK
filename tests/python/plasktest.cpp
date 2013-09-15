#include <plask/plask.hpp>
#include <plask/python.hpp>
#include <boost/concept_check.hpp>
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

    virtual double VBO(double T, double e, char P) const { return 0.5*T; }

    virtual double chi(double T, double e, char P) const { std::cerr << "MyMaterial: " << P << "\n"; return 1.0; }

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

plask::Tensor2<double> materialThermk1(std::string m, plask::MaterialsDB& DB, double T) {
    plask::shared_ptr<plask::Material> mat = DB.get(m);
    return mat->thermk(T);
}

plask::Tensor2<double> materialThermk2(std::string m, plask::MaterialsDB& DB, double T, double t) {
    plask::shared_ptr<plask::Material> mat = DB.get(m);
    return mat->thermk(T, t);
}

double call_chi(plask::shared_ptr<plask::Material> mat, std::string p) {
    return mat->chi(300, p[0]);
}

void print_ptr(py::object o) {
    std::cerr << "ptr: " << o.ptr() << "\n";
}

plask::shared_ptr<plask::GeometryObject> getExtrusion(plask::shared_ptr<plask::GeometryObjectD<2>>c, double l) {
    return plask::make_shared<plask::Extrusion>(c,l);
}

bool isEmpy(plask::shared_ptr<plask::GeometryObject> p) {
    return !p;
}

std::string materialTypeId(plask::shared_ptr<plask::Material> material) {
    return typeid(*material).name();
}

plask::Tensor3<plask::dcomplex> NrTensor(plask::shared_ptr<plask::Material> material) {
    return material->NR(1000., 300.);
}

bool compareMaterials(plask::shared_ptr<plask::Material>  m1, plask::shared_ptr<plask::Material>  m2) {
    return *m1 == *m2;
}

//// Boundary conditions /////////////////////////////////////////////////////////////////////////////////////////////////////////

py::list testBoundary(const plask::RectilinearMesh2D& mesh, const plask::shared_ptr<const plask::GeometryD<2>>& geometry,
                      const typename plask::RectilinearMesh2D::Boundary& boundary) {
    py::list result;
    for(auto i: boundary(mesh, geometry)) {
        result.append(i);
    }
    return result;
}


//// Solver with space /////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SpaceTest : plask::SolverWithMesh<plask::Geometry2DCartesian, plask::RectilinearMesh2D> {
    bool mesh_changed;
    SpaceTest() : mesh_changed(false) {}
    virtual std::string getClassName() const { return "SpaceTest"; }
    void initialize() {
        initCalculation();
    }
    virtual void onMeshChange(const typename plask::RectilinearMesh2D::Event& evt) {
        mesh_changed = true;
    }
    bool getMeshChanged() {
        bool result = mesh_changed;
        mesh_changed = false;
        return result;
    }
};


//// Provider & Receiver /////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SimpleSolver : plask::Solver {
    struct VectorialField: plask::FieldProperty<plask::Vec<2,double>> {};

    virtual std::string getClassName() const { return "SimpleSolver"; }

    plask::ReceiverFor<plask::Temperature, plask::Geometry2DCartesian> inTemperature;

    plask::ReceiverFor<plask::LightIntensity, plask::Geometry2DCartesian> inIntensity;

    plask::ProviderFor<plask::LightIntensity, plask::Geometry2DCartesian>::WithValue<plask::shared_ptr<plask::RegularMesh2D>> outIntensity;

    plask::ReceiverFor<VectorialField, plask::Geometry2DCartesian> inVectors;

    std::string showVectors() {
        plask::RegularMesh2D mesh(plask::RegularAxis(1., 3., 2), plask::RegularAxis(5., 15., 2));
        auto data = inVectors(mesh);
        std::stringstream str;
        for (size_t i = 0; i != 4; i++) {
            str << mesh[i] << ": " << data[i] << "\n";
        }
        return str.str();
    }

    SimpleSolver() :
        outIntensity( plask::make_shared<plask::RegularMesh2D>(plask::RegularAxis(0., 4., 3), plask::RegularAxis(0., 20., 3)) )
    {
        plask::DataVector<double> data(9);
        data[0] = 100.; data[1] = 100.; data[2] = 100.;
        data[3] = 300.; data[4] = 300.; data[5] = 300.;
        data[6] = 500.; data[7] = 500.; data[8] = 500.;
        outIntensity.push_back(data);
    }
};

struct InOutSolver : plask::Solver {
    struct VectorialField: plask::FieldProperty<plask::Vec<2,double>> {};

    virtual std::string getClassName() const { return "InOut"; }

    plask::ReceiverFor<plask::Wavelength> inWavelength;
    plask::ProviderFor<plask::Wavelength>::WithValue outWavelength;

    InOutSolver(const std::string& name="") : Solver(name) {
        inWavelength = 2.;
        outWavelength.push_back(5.);
    }
};


//// Mesh /////////////////////////////////////////////////////////////////////////////////////////////////////////

struct MeshTest {

    plask::shared_ptr<plask::RectilinearMesh2D> rectilinear2d;
    plask::shared_ptr<plask::RectilinearMesh3D> rectilinear3d;
    plask::shared_ptr<plask::RegularMesh2D> regular2d;
    plask::shared_ptr<plask::RegularMesh3D> regular3d;

    bool changed_rectilinear2d, changed_rectilinear3d, changed_regular2d, changed_regular3d;

    void change_rectilinear2d(const plask::Mesh::Event&) { changed_rectilinear2d = true; }
    bool rectilinear2d_changed() { bool r = changed_rectilinear2d; changed_rectilinear2d = false; return r; }

    void change_rectilinear3d(const plask::Mesh::Event&) { changed_rectilinear3d = true; }
    bool rectilinear3d_changed() { bool r = changed_rectilinear3d; changed_rectilinear3d = false; return r; }

    void change_regular2d(const plask::Mesh::Event&) { changed_regular2d = true; }
    bool regular2d_changed() { bool r = changed_regular2d; changed_regular2d = false; return r; }

    void change_regular3d(const plask::Mesh::Event&) { changed_regular3d = true; }
    bool regular3d_changed() { bool r = changed_regular3d; changed_regular3d = false; return r; }

    MeshTest():
        rectilinear2d(plask::make_shared<plask::RectilinearMesh2D>()), rectilinear3d(plask::make_shared<plask::RectilinearMesh3D>()),
        regular2d(plask::make_shared<plask::RegularMesh2D>()), regular3d(plask::make_shared<plask::RegularMesh3D>()),
        changed_rectilinear2d(false), changed_rectilinear3d(false), changed_regular2d(false), changed_regular3d(false) {
        rectilinear2d->changedConnectMethod(this, &MeshTest::change_rectilinear2d);
        rectilinear3d->changedConnectMethod(this, &MeshTest::change_rectilinear3d);
        regular2d->changedConnectMethod(this, &MeshTest::change_regular2d);
        regular3d->changedConnectMethod(this, &MeshTest::change_regular3d);
    }

    ~MeshTest() {
        rectilinear2d->changedDisconnectMethod(this, &MeshTest::change_rectilinear2d);
        rectilinear3d->changedDisconnectMethod(this, &MeshTest::change_rectilinear3d);
        regular2d->changedDisconnectMethod(this, &MeshTest::change_regular2d);
        regular3d->changedDisconnectMethod(this, &MeshTest::change_regular3d);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BOOST_PYTHON_MODULE(plasktest)
{
    py::def("get_vecs", &getVecs);

    py::def("add_my_material", &addMyMaterial);

    py::def("material_name", &materialName);
    py::def("material_VBO", &materialVBO);
    py::def("material_thermk", &materialThermk1);
    py::def("material_thermk", &materialThermk2);

    py::def("call_chi", &call_chi);

    py::def("print_ptr", &print_ptr);

    py::def("get_extrusion", &getExtrusion);

    py::def("is_empty", &isEmpy);

    py::def("material_typeid", &materialTypeId);

    py::def("NR", &NrTensor);

    py::def("compareMaterials", &compareMaterials);

    py::def("test_boundary", &testBoundary);

    plask::python::ExportSolver<SpaceTest>("SpaceTest")
        .def("initialize", &SpaceTest::initialize)
        .add_property("mesh_changed", &SpaceTest::getMeshChanged)
    ;

    plask::python::ExportSolver<SimpleSolver>("SimpleSolver")
        .add_receiver("inTemperature", &SimpleSolver::inTemperature, "Test receiver")
        .add_receiver("inIntensity", &SimpleSolver::inIntensity, "Test multiple receiver")
        .add_provider("outIntensity", &SimpleSolver::outIntensity, "Test provider")
        .add_receiver("inVectors", &SimpleSolver::inVectors, "Test provider")
        .def("show_vectors", &SimpleSolver::showVectors)
    ;

    {
        py::object module { py::handle<>(py::borrowed(PyImport_AddModule("plasktest.solvers"))) };
        py::scope().attr("solvers") = module;
        py::scope scope = module;

        plask::python::ExportSolver<InOutSolver>("InOut", py::init<std::string>())
            .add_receiver("inWavelength", &InOutSolver::inWavelength, "Input Wavelength")
            .add_provider("outWavelength", &InOutSolver::outWavelength, "Output Wavelength")
    ;
    }

    py::class_<MeshTest, boost::noncopyable>("MeshTest")
        .def_readonly("rectilinear2d", &MeshTest::rectilinear2d)
        .add_property("rectilinear2d_changed", &MeshTest::rectilinear2d_changed)
        .def_readonly("rectilinear3d", &MeshTest::rectilinear3d)
        .add_property("rectilinear3d_changed", &MeshTest::rectilinear3d_changed)
        .def_readonly("regular2d", &MeshTest::regular2d)
        .add_property("regular2d_changed", &MeshTest::regular2d_changed)
        .def_readonly("regular3d", &MeshTest::regular3d)
        .add_property("regular3d_changed", &MeshTest::regular3d_changed)
    ;
}
