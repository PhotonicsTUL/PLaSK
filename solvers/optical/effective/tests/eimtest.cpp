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
    manager.loadFromXMLString(
        "<plask>"
        "<geometry>"

        "   <cartesian2d name=\"main\" axes=\"xy\" left=\"mirror\">"
        "        <stack>"
        "            <container>"
        "                <child ><rectangle x=\"0.75\" y=\"0.3\" material=\"glass\" /></child>"
        "                <child ><rectangle x=\"0.05\" y=\"0.3\" material=\"air\" /></child>"
        "            </container>"
        "            <rectangle x=\"0.75\" y=\"0.2\" material=\"air\" />"
        "        </stack>"
        "    </cartesian2d>"
        "</geometry>"
        "</plask>"
    );

    EffectiveIndex2DSolver solver;

    solver.setGeometry(dynamic_pointer_cast<Geometry2DCartesian>(manager.getGeometry("main")));
    solver.polarization = EffectiveIndex2DSolver::TE;
    solver.inWavelength = 1000;
    solver.symmetry = EffectiveIndex2DSolver::SYMMETRY_POSITIVE;

    dcomplex mode = solver.computeMode(1.15);

    writelog(LOG_INFO, "Found mode: %1%", str(mode));

    double right = 1.5,
           top = 1.0,
           left = -right,
           bottom = -0.5;

    size_t N = 50000;

    auto mesh = make_shared<RegularMesh2D>(RegularMesh1D(left, right, N), RegularMesh1D(bottom, top, N));

    auto data = solver.outIntensity(*mesh, DEFAULT_INTERPOLATION);

    return 0;
}
