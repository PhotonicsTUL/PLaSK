#include "reflection_solver.h"

namespace plask { namespace solvers { namespace modal {

FourierReflection2D::FourierReflection2D(const std::string& name): SolverOver<Geometry2DCartesian>(name),
    order(5),
    refine(8),
    outdist(0.1),
    outIntensity(this, &FourierReflection2D::getIntensity)
{
    inTemperature = 300.; // temperature receiver has some sensible value
}


void FourierReflection2D::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd()) {
//         std::string param = reader.getNodeName();
//         if (param == "newton") {
//             newton.tolx = reader.getAttribute<double>("tolx", newton.tolx);
//             newton.tolf = reader.getAttribute<double>("tolf", newton.tolf);
//             newton.maxstep = reader.getAttribute<double>("maxstep", newton.maxstep);
//             reader.requireTagEnd();
//         } else if (param == "wavelength") {
//             std::string = reader.requireTextUntilEnd();
//             inWavelength.setValue(boost::lexical_cast<double>(wavelength));
//         } else
            parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <newton>, or <wavelength>");
    }
}


void FourierReflection2D::setupLayers()
{
    auto mesh = RectilinearMesh2DSimpleGenerator()(geometry->getChild());
    vbounds = mesh->axis1;
    //TODO consider geometry objects non-uniform in vertical direction (step approximation)

    auto points = mesh->getMidpointsMesh();

    struct LayerItem {
        shared_ptr<Material> material;
        std::set<std::string> roles;
        bool operator!=(const LayerItem& other) { return *material != *other.material || roles != other.roles; }
    };

    std::vector<std::vector<LayerItem>> layers;

    // Add layers below bottom boundary and above top one
    points->axis1.addPoint(vbounds[0] - outdist);
    points->axis1.addPoint(vbounds[vbounds.size()-1] + outdist);

    lverts.clear();
    stack.clear();
    stack.reserve(points->axis1.size());

    for (auto v: points->axis1) {
        std::vector<LayerItem> layer(points->axis0.size());
        for (size_t i = 0; i != points->axis0.size(); ++i) {
            Vec<2> p(points->axis0[i],v);
            layer[i].material = this->geometry->getMaterial(p);
            for (const std::string& role: this->geometry->getRolesAt(p))
                if (role.substr(0,3) == "opt" || role == "QW" || role == "QD" || role == "gain") layer[i].roles.insert(role);
        }

        bool unique;
        for (size_t i = 0; i != layers.size(); ++i) {
            unique = false;
            for (size_t j = 0; j != layers[i].size(); ++j) {
                if (layers[i][j] != layer[j]) {
                    unique = true;
                    break;
                }
            }
            if (!unique) {
                lverts[i].addPoint(v);
                stack.push_back(i);
                break;
            }
        }
        if (unique) {
            layers.emplace_back(std::move(layer));
            stack.push_back(lverts.size());
            lverts.emplace_back<std::initializer_list<double>>({v});
        }
    }

    writelog(LOG_INFO, "Detected %1% distinct layers", lverts.size());
}





// void FourierReflection2D::onInitialize() // In this function check if geometry and mesh are set
// {
//     if (!geometry) throw NoGeometryException(getId());
//     if (!mesh) throw NoMeshException(getId());
//     my_data.reset(mesh->size()); // and e.g. allocate memory
// }
//
//
// void FourierReflection2D::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
// {
//     outSingleValue.invalidate(); // clear the value
//     my_data.reset();
//     // Make sure that no provider returns any value.
//     // If this method has been called, before next computations, onInitialize will be called.
// }




const DataVector<const double> FourierReflection2D::getIntensity(const MeshD<2>& dst_mesh, InterpolationMethod method) {
//     if (!outSingleValue.hasValue())  // this is one possible indication that the solver is invalidated
//         throw NoValue(SomeSingleValueProperty::NAME);
//     return interpolate(*mesh, my_data, dst_mesh, defInterpolation<INTERPOLATION_LINEAR>(method)); // interpolate your data to the requested mesh
}


}}} // namespace
