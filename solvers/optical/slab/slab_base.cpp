// #include <plask/filters/factory.h>

#include "slab_base.h"

namespace plask { namespace solvers { namespace slab {

template <typename GeometryT>
SlabSolver<GeometryT>::SlabSolver(const std::string& name): SolverOver<GeometryT>(name),
    outdist(0.1),
    smooth(0.),
    outIntensity(this, &SlabSolver<GeometryT>::getIntensity, &SlabSolver<GeometryT>::nummodes)
{
    inTemperature = 300.; // temperature receiver has some sensible value
}


template <typename GeometryT>
void SlabSolver<GeometryT>::prepareLayers()
{
    vbounds = RectilinearMesh2DSimpleGenerator()(this->geometry->getChild())->vert();
    //TODO consider geometry objects non-uniform in vertical direction (step approximation)
}

template <>
void SlabSolver<Geometry3D>::prepareLayers()
{
    vbounds = RectilinearMesh3DSimpleGenerator()(this->geometry->getChild())->vert();
    //TODO consider geometry objects non-uniform in vertical direction (step approximation)
}

template <typename GeometryT>
void SlabSolver<GeometryT>::setupLayers()
{
    if (!this->geometry) throw NoGeometryException(this->getId());

    if (vbounds.empty()) prepareLayers();

    auto points = RectilinearMesh2DSimpleGenerator()(this->geometry->getChild())->getMidpointsMesh();

    struct LayerItem {
        shared_ptr<Material> material;
        std::set<std::string> roles;
    };

    std::vector<std::vector<LayerItem>> layers;

    // Add layers below bottom boundary and above top one
    points->axis1.addPoint(vbounds[0] - outdist);
    points->axis1.addPoint(vbounds[vbounds.size()-1] + outdist);

    lverts.clear();
    lgained.clear();
    stack.clear();
    stack.reserve(points->axis1.size());

    for (auto v: points->axis1) {
        bool gain = false;

        std::vector<LayerItem> layer(points->axis0.size());
        for (size_t i = 0; i != points->axis0.size(); ++i) {
            Vec<2> p(points->axis0[i], v);
            layer[i].material = this->geometry->getMaterial(p);
            for (const std::string& role: this->geometry->getRolesAt(p)) {
                if (role.substr(0,3) == "opt") layer[i].roles.insert(role);
                if (role == "QW" || role == "QD" || role == "gain") { layer[i].roles.insert(role); gain = true; }
            }
        }

        bool unique = true;
        for (size_t i = 0; i != layers.size(); ++i) {
            unique = false;
            for (size_t j = 0; j != layers[i].size(); ++j) {
                if (*layers[i][j].material != *layer[j].material || layers[i][j].roles != layer[j].roles) {
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
            lgained.push_back(gain);
        }
    }

    this->writelog(LOG_DETAIL, "Detected %1% distinct layers", lverts.size());
}

template <>
void SlabSolver<Geometry3D>::setupLayers()
{
    if (vbounds.empty()) prepareLayers();

    auto points = RectilinearMesh3DSimpleGenerator()(this->geometry->getChild())->getMidpointsMesh();

    struct LayerItem {
        shared_ptr<Material> material;
        std::set<std::string> roles;
        bool operator!=(const LayerItem& other) { return *material != *other.material || roles != other.roles; }
    };

    std::vector<std::vector<LayerItem>> layers;

    // Add layers below bottom boundary and above top one
    points->vert().addPoint(vbounds[0] - outdist);
    points->vert().addPoint(vbounds[vbounds.size()-1] + outdist);

    lverts.clear();
    lgained.clear();
    stack.clear();
    stack.reserve(points->vert().size());

    for (auto v: points->vert()) {
        bool gain = false;

        std::vector<LayerItem> layer(points->axis0.size() * points->axis0.size());
        for (size_t i = 0; i != points->axis0.size(); ++i) {
            for (size_t j = 0; j != points->axis1.size(); ++j) {
                Vec<3> p(points->axis0[i], points->axis1[j], v);
                size_t n = i + points->axis0.size() * j;
                layer[n].material = this->geometry->getMaterial(p);
                for (const std::string& role: this->geometry->getRolesAt(p)) {
                    if (role.substr(0,3) == "opt") layer[n].roles.insert(role);
                    if (role == "QW" || role == "QD" || role == "gain") { layer[n].roles.insert(role); gain = true; }
                }
            }
        }

        bool unique = true;
        for (size_t i = 0; i != layers.size(); ++i) {
            unique = false;
            for (size_t j = 0; j != layers[i].size(); ++j) {
                if (*layers[i][j].material != *layer[j].material || layers[i][j].roles != layer[j].roles) {
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
            lgained.push_back(gain);
        }
    }

    assert(vbounds.size() == stack.size()-1);

    this->writelog(LOG_DETAIL, "Detected %1% distinct layers", lverts.size());
}

template struct SlabSolver<Geometry2DCartesian>;
template struct SlabSolver<Geometry2DCylindrical>;
template struct SlabSolver<Geometry3D>;

// FiltersFactory::RegisterStandard<RefractiveIndex> registerRefractiveIndexFilters;

}}} // namespace
