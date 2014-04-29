#include "slab_base.h"

namespace plask { namespace solvers { namespace slab {

template <typename GeometryT>
SlabSolver<GeometryT>::SlabSolver(const std::string& name): SolverOver<GeometryT>(name),
    interface(1),
    outdist(0.1),
    smooth(0.),
    outLightIntensity(this, &SlabSolver<GeometryT>::getIntensity, &SlabSolver<GeometryT>::nummodes),
    outElectricField(this, &SlabSolver<GeometryT>::getE, &SlabSolver<GeometryT>::nummodes),
    outMagneticField(this, &SlabSolver<GeometryT>::getH, &SlabSolver<GeometryT>::nummodes)
{
    root.tolx = 1.0e-8;
    root.tolf_min = 1.0e-10;
    root.tolf_max = 1.0e-6;
    root.maxstep = 0.1;
    root.maxiter = 500;
    inTemperature = 300.; // temperature receiver has some sensible value
}


template <typename GeometryT>
void SlabSolver<GeometryT>::prepareLayers()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    vbounds = RectilinearMesh2DSimpleGenerator().get<RectangularMesh<2>>(this->geometry->getChild())->vert();
    //TODO consider geometry objects non-uniform in vertical direction (step approximation)
}

template <>
void SlabSolver<Geometry3D>::prepareLayers()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    vbounds = RectilinearMesh3DSimpleGenerator().get<RectangularMesh<2>>(this->geometry->getChild())->vert();
    //TODO consider geometry objects non-uniform in vertical direction (step approximation)
}

template <typename GeometryT>
void SlabSolver<GeometryT>::setupLayers()
{
    if (!this->geometry) throw NoGeometryException(this->getId());

    if (vbounds.empty()) prepareLayers();

    auto points = make_rectilinear_mesh(RectilinearMesh2DSimpleGenerator().get<RectangularMesh<2>>(this->geometry->getChild())->getMidpointsMesh());

    struct LayerItem {
        shared_ptr<Material> material;
        std::set<std::string> roles;
        bool operator==(const LayerItem& other) { return *this->material == *other.material && this->roles == other.roles; }
        bool operator!=(const LayerItem& other) { return !(*this == other); }
    };

    std::vector<std::vector<LayerItem>> layers;

    // Add layers below bottom boundary and above top one
    static_pointer_cast<RectilinearAxis>(points->axis1)->addPoint(vbounds[0] - outdist);
    static_pointer_cast<RectilinearAxis>(points->axis1)->addPoint(vbounds[vbounds.size()-1] + outdist);

    lverts.clear();
    lgained.clear();
    stack.clear();
    stack.reserve(points->axis1->size());

    for (auto v: *points->axis1) {
        bool gain = false;

        std::vector<LayerItem> layer(points->axis0->size());
        for (size_t i = 0; i != points->axis0->size(); ++i) {
            Vec<2> p(points->axis0->at(i), v);
            layer[i].material = this->geometry->getMaterial(p);
            for (const std::string& role: this->geometry->getRolesAt(p)) {
                if (role.substr(0,3) == "opt") layer[i].roles.insert(role);
                else if (role == "QW" || role == "QD" || role == "gain") { layer[i].roles.insert(role); gain = true; }
            }
        }

        bool unique = true;
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
            lgained.push_back(gain);
        }
    }

    this->writelog(LOG_DETAIL, "Detected %1% distinct layers", lverts.size());
}

template <>
void SlabSolver<Geometry3D>::setupLayers()
{
    if (vbounds.empty()) prepareLayers();

    auto points = make_rectilinear_mesh( RectilinearMesh3DSimpleGenerator().get<RectangularMesh<3>>(this->geometry->getChild())->getMidpointsMesh() );

    struct LayerItem {
        shared_ptr<Material> material;
        std::set<std::string> roles;
        bool operator!=(const LayerItem& other) { return *material != *other.material || roles != other.roles; }
    };

    std::vector<std::vector<LayerItem>> layers;

    // Add layers below bottom boundary and above top one
    //static_pointer_cast<RectilinearAxis>(points->vert())->addPoint(vbounds[0] - outdist);
    static_cast<RectilinearAxis&>(points->vert()).addPoint(vbounds[0] - outdist);
    static_cast<RectilinearAxis&>(points->vert()).addPoint(vbounds[vbounds.size()-1] + outdist);

    lverts.clear();
    lgained.clear();
    stack.clear();
    stack.reserve(points->vert().size());

    for (auto v: points->vert()) {
        bool gain = false;

        std::vector<LayerItem> layer(points->axis0->size() * points->axis1->size());
        for (size_t i = 0; i != points->axis1->size(); ++i) {
            size_t offs = i * points->axis0->size();
            for (size_t j = 0; j != points->axis0->size(); ++j) {
                Vec<3> p(points->axis0->at(i), points->axis1->at(j), v);
                size_t n = offs + j;
                layer[n].material = this->geometry->getMaterial(p);
                for (const std::string& role: this->geometry->getRolesAt(p)) {
                    if (role.substr(0,3) == "opt") layer[n].roles.insert(role);
                    else if (role == "QW" || role == "QD" || role == "gain") { layer[n].roles.insert(role); gain = true; }
                }
            }
        }

        bool unique = true;
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
