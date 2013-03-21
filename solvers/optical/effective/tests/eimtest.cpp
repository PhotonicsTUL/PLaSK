#include <plask/plask.hpp>
#include "../eim.h"
using namespace plask;
using namespace plask::solvers::effective;

struct Glass: public Material {
    virtual std::string name() const { return "glass"; }
    virtual Kind kind() const { return Material::DIELECTRIC; }
    virtual dcomplex Nr(double wl, double T) const { return 1.3; }
};

int main() {
    MaterialsDB::getDefault().add<Glass>("glass");

    Manager manager;

    auto solver = make_shared<EffectiveIndex2DSolver>("eim");
    manager.solvers["eim"] = solver;

    manager.loadFromXMLString(
        "<plask>"

//         "<materials>"
//         "  <material name=\"dupa\"/>"
//         "</materials>"

        "<geometry>"
        "  <cartesian2d name=\"main\" axes=\"xy\" left=\"mirror\">"
        "    <stack>"
        "      <container>"
        "        <item ><rectangle dx=\"0.75\" dy=\"0.3\" material=\"glass\" /></item>"
        "        <item ><rectangle dx=\"0.05\" dy=\"0.3\" material=\"air\" /></item>"
        "      </container>"
        "      <rectangle dx=\"0.75\" dy=\"0.2\" material=\"air\" />"
        "    </stack>"
        "  </cartesian2d>"
        "</geometry>"

        "<solvers>"
        "  <optical lib=\"effective\" solver=\"EffectiveIndex2D\" name=\"eim\">"
        "    <geometry ref=\"main\"/>"
        "    <mode polarization=\"TE\" symmetry=\"+1\" wavelength=\"1000\"/>"
        "  </optical>"
        "</solvers>"

        "</plask>"
    );

    // solver->setGeometry(dynamic_pointer_cast<Geometry2DCartesian>(manager.getGeometry("main")));
    // solver->polarization = EffectiveIndex2DSolver::TE;
    // solver->symmetry = EffectiveIndex2DSolver::SYMMETRY_POSITIVE;
    // solver->inWavelength = 1000;

    dcomplex mode = solver->computeMode(1.15);

    writelog(LOG_INFO, "Found mode: %1%", str(mode));

    double right = 1.5,
           top = 1.0,
           left = -right,
           bottom = -0.5;

    size_t N = 5000;

    auto mesh = make_shared<RegularMesh2D>(RegularMesh1D(left, right, N), RegularMesh1D(bottom, top, N));

    auto data = solver->outIntensity(*mesh, DEFAULT_INTERPOLATION);

    return 0;
}
