#include "solver.h"
#include "expansion.h"
#include "meshadapter.h"
#include "muller.h"
#include "broyden.h"
#include "brent.h"
#include "reflection.h"
#include "admittance.h"

namespace plask { namespace optical { namespace slab {

void SlabBase::initTransfer(Expansion& expansion, bool reflection) {
    switch (transfer_method) {
        case Transfer::METHOD_REFLECTION: reflection = true; break;
        case Transfer::METHOD_ADMITTANCE: reflection = false; break;
        default: break;
    }
    if (reflection) {
        if (!this->transfer || !dynamic_cast<ReflectionTransfer*>(this->transfer.get()) ||
            this->transfer->diagonalizer->source() != &expansion)
        this->transfer.reset(new ReflectionTransfer(this, expansion));
    } else {
        if (!this->transfer || !dynamic_cast<AdmittanceTransfer*>(this->transfer.get()) ||
            this->transfer->diagonalizer->source() != &expansion)
        this->transfer.reset(new AdmittanceTransfer(this, expansion));
    }
}


template <typename BaseT>
SlabSolver<BaseT>::SlabSolver(const std::string& name): BaseT(name),
    smooth(0.),
    outRefractiveIndex(this, &SlabSolver<BaseT>::getRefractiveIndexProfile),
    outLightMagnitude(this, &SlabSolver<BaseT>::getMagnitude, &SlabSolver<BaseT>::nummodes),
    outLightE(this, &SlabSolver<BaseT>::getE, &SlabSolver<BaseT>::nummodes),
    outLightH(this, &SlabSolver<BaseT>::getH, &SlabSolver<BaseT>::nummodes)
{
    inTemperature = 300.; // temperature receiver has some sensible value
    this->inTemperature.changedConnectMethod(this, &SlabSolver<BaseT>::onInputChanged);
    this->inGain.changedConnectMethod(this, &SlabSolver<BaseT>::onGainChanged);
}

template <typename BaseT>
SlabSolver<BaseT>::~SlabSolver()
{
    this->inTemperature.changedDisconnectMethod(this, &SlabSolver<BaseT>::onInputChanged);
    this->inGain.changedDisconnectMethod(this, &SlabSolver<BaseT>::onGainChanged);
}


std::unique_ptr<RootDigger> SlabBase::getRootDigger(const RootDigger::function_type& func, const char* name) {
    typedef std::unique_ptr<RootDigger> Res;
    if (root.method == RootDigger::ROOT_MULLER) return Res(new RootMuller(*this, func, root, name));
    else if (root.method == RootDigger::ROOT_BROYDEN) return Res(new RootBroyden(*this, func, root, name));
    else if (root.method == RootDigger::ROOT_BRENT) return Res(new RootBrent(*this, func, root, name));
    throw BadInput(getId(), "Wrong root finding method");
    return Res();
}


template <typename BaseT>
struct LateralMeshAdapter {
    shared_ptr<RectangularMesh<2>> mesh;

    LateralMeshAdapter(const BaseT* solver):
        mesh(makeGeometryGrid(solver->getGeometry())) {}

    void resetMidpoints(const shared_ptr<MeshAxis>& vbounds) {
        mesh = make_shared<RectangularMesh<2>>(mesh->axis0->getMidpointsMesh(),
                                               vbounds, RectangularMesh<2>::ORDER_10);
    }

    void resetMidpoints(const shared_ptr<MeshAxis>& vbounds, double spacing) {
        mesh = make_shared<RectangularMesh<2>>(refineAxis(mesh->axis0, spacing)->getMidpointsMesh(),
                                               vbounds, RectangularMesh<2>::ORDER_10);
    }

    void reset(const shared_ptr<MeshAxis>& verts) {
        mesh = make_shared<RectangularMesh<2>>(mesh->axis0, verts, RectangularMesh<2>::ORDER_10);
    }

    shared_ptr<RectangularMesh<2>> makeMesh(const shared_ptr<MeshAxis>& verts) {
        return make_shared<RectangularMesh<2>>(mesh->axis0, verts, RectangularMesh<2>::ORDER_10);
    }

    shared_ptr<OrderedAxis> vert() {
        return dynamic_pointer_cast<OrderedAxis>(mesh->vert());
    }

    shared_ptr<RectangularMesh<2>> midmesh() const {
        return make_shared<RectangularMesh<2>>(mesh->axis0, mesh->axis1->getMidpointsMesh());
    }

    size_t size() const { return mesh->axis0->size(); }

    size_t idx(size_t i, size_t v) const {
        return mesh->index(i, v);
    }

    Vec<2> at(size_t i, size_t v) const { return mesh->at(i, v); }

    shared_ptr<RectangularMesh<2>> makeLine(size_t i, size_t v, double spacing) const {
        shared_ptr<OrderedAxis> vaxis(new OrderedAxis({mesh->axis1->at(v-1), mesh->axis1->at(v)}));
        vaxis = refineAxis(vaxis, spacing);
        return make_shared<RectangularMesh<2>>(make_shared<OnePointAxis>(mesh->axis0->at(i)), vaxis);
    }
};

template <>
struct LateralMeshAdapter<SolverOver<Geometry3D>> {
  private:
    size_t _size;

  public:
    shared_ptr<RectangularMesh<3>> mesh;

    LateralMeshAdapter(const SolverOver<Geometry3D>* solver):
        mesh(makeGeometryGrid(solver->getGeometry())) {
        _size = mesh->axis0->size() * mesh->axis1->size();
    }

    void resetMidpoints(const shared_ptr<MeshAxis>& vbounds) {
        mesh = make_shared<RectangularMesh<3>>(mesh->axis0->getMidpointsMesh(),
                                               mesh->axis1->getMidpointsMesh(),
                                               vbounds, RectangularMesh<3>::ORDER_201);
        _size = mesh->axis0->size() * mesh->axis1->size();
    }

    void resetMidpoints(const shared_ptr<MeshAxis>& vbounds, double spacing) {
        mesh = make_shared<RectangularMesh<3>>(refineAxis(mesh->axis0, spacing)->getMidpointsMesh(),
                                               refineAxis(mesh->axis1, spacing)->getMidpointsMesh(),
                                               vbounds, RectangularMesh<3>::ORDER_201);
        _size = mesh->axis0->size() * mesh->axis1->size();
    }

    void reset(const shared_ptr<MeshAxis>& verts) {
        mesh = make_shared<RectangularMesh<3>>(mesh->axis0, mesh->axis1, verts, RectangularMesh<3>::ORDER_210);
    }

    shared_ptr<RectangularMesh<3>> makeMesh(const shared_ptr<MeshAxis>& verts) {
        return make_shared<RectangularMesh<3>>(mesh->axis0, mesh->axis1, verts, RectangularMesh<3>::ORDER_210);
    }

    shared_ptr<OrderedAxis> vert() {
        return dynamic_pointer_cast<OrderedAxis>(mesh->vert());
    }

    shared_ptr<RectangularMesh<3>> midmesh() const {
        return make_shared<RectangularMesh<3>>(mesh->axis0, mesh->axis1, mesh->axis2->getMidpointsMesh());
    }

    size_t size() const { return _size; }

    size_t idx(size_t i, size_t v) const {
        return _size * v + i;
    }

    Vec<3> at(size_t i, size_t v) const {
        return mesh->RectilinearMesh3D::at(idx(i, v));
    }

    shared_ptr<RectangularMesh<3>> makeLine(size_t i, size_t v, double spacing) const {
        shared_ptr<OrderedAxis> vaxis(new OrderedAxis({mesh->axis2->at(v-1), mesh->axis2->at(v)}));
        vaxis = refineAxis(vaxis, spacing);
        return make_shared<RectangularMesh<3>>(make_shared<OnePointAxis>(mesh->axis0->at(mesh->index0(i))),
                                               make_shared<OnePointAxis>(mesh->axis1->at(mesh->index1(i))),
                                               vaxis);
    }
};


template <typename BaseT>
void SlabSolver<BaseT>::setupLayers()
{
    if (!this->geometry) throw NoGeometryException(this->getId());

    LateralMeshAdapter<BaseT> adapter(this);

    vbounds = adapter.vert();
    if (this->geometry->isSymmetric(Geometry::DIRECTION_VERT)) {
        std::deque<double> zz;
        for (double z: *vbounds) zz.push_front(-z);
        OrderedAxis::WarningOff nowarn(vbounds);
        vbounds->addOrderedPoints(zz.begin(), zz.end(), zz.size());
    }

    if (inTemperature.hasProvider() &&
        !isnan(max_temp_diff) && !isinf(max_temp_diff) &&
        !isnan(temp_dist) && !isinf(temp_dist) &&
        !isnan(temp_layer) && !isinf(temp_layer))
        adapter.resetMidpoints(vbounds, temp_dist);
    else
        adapter.resetMidpoints(vbounds);

    // Divide layers with too large temperature gradient
    if (inTemperature.hasProvider() &&
        !isnan(max_temp_diff) && !isinf(max_temp_diff) &&
        !isnan(temp_dist) && !isinf(temp_dist) &&
        !isnan(temp_layer) && !isinf(temp_layer)) {
        auto temp = inTemperature(adapter.mesh);
        std::deque<double> refines;
        for (size_t v = 1; v != vbounds->size(); ++v) {
            double mdt = 0.;
            size_t idt;
            for (size_t i = 0; i != adapter.size(); ++i) {
                double dt = abs(temp[adapter.idx(i, v)] - temp[adapter.idx(i, v-1)]);
                if (dt > mdt) {
                    mdt = dt;
                    idt = i;
                }
            }
            if (mdt > max_temp_diff) {
                // We need to divide the layer.
                auto line_mesh = adapter.makeLine(idt, v, temp_layer);
                auto tmp = inTemperature(line_mesh);
                auto line = line_mesh->vert();
                size_t li = 0;
                for (size_t i = 2; i != line->size(); ++i) {
                    if (abs(tmp[i] - tmp[li]) > max_temp_diff) {
                        li = i - 1;
                        refines.push_back(line->at(li));
                    }
                }
            }
        }
        vbounds->addOrderedPoints(refines.begin(), refines.end(), refines.size());
    }

    adapter.reset(vbounds->getMidpointsMesh());

    // Add layers below bottom boundary and above top one
    verts = dynamic_pointer_cast<OrderedAxis>(adapter.mesh->vert());
    OrderedAxis::WarningOff nowarn(verts);
    verts->addPoint(vbounds->at(0) - 2.*OrderedAxis::MIN_DISTANCE);
    verts->addPoint(vbounds->at(vbounds->size()-1) + 2.*OrderedAxis::MIN_DISTANCE);

    lgained.clear();
    stack.clear();
    stack.reserve(verts->size());
    lcount = 0;

    struct LayerItem {
        shared_ptr<Material> material;
        std::set<std::string> roles;
        bool operator==(const LayerItem& other) { return *material == *other.material && roles == other.roles; }
        bool operator!=(const LayerItem& other) { return !(*this == other); }
    };
    std::vector<std::vector<LayerItem>> layers;

    for (size_t v = 0; v != verts->size(); ++v) {
        bool gain = false;
        bool unique = !group_layers || layers.size() == 0;

        std::vector<LayerItem> layer(adapter.size());
        for (size_t i = 0; i != adapter.size(); ++i) {
            auto p(adapter.at(i, v));
            layer[i].material = this->geometry->getMaterial(p);
            for (const std::string& role: this->geometry->getRolesAt(p)) {
                if (role.substr(0,3) == "opt") layer[i].roles.insert(role);
                else if (role == "unique") { layer[i].roles.insert(role); unique = true; }
                else if (role == "QW" || role == "QD" || role == "gain") { layer[i].roles.insert(role); gain = true; }
            }
        }

        if (!unique) {
            for (size_t l = 0; l != layers.size(); ++l) {
                unique = false;
                for (size_t i = 0; i != layers[l].size(); ++i) {
                    if (layers[l][i] != layer[i]) {
                        unique = true;
                        break;
                    }
                }
                if (!unique) {
                    stack.push_back(l);
                    break;
                }
            }
        }
        if (unique) {
            stack.push_back(lcount++);
            layers.emplace_back(std::move(layer));
            lgained.push_back(gain);
        }
    }

    assert(vbounds->size() == stack.size()-1);
    assert(verts->size() == stack.size());
    assert(layers.size() == lcount);

    // Split groups with too large temperature gradient
    // We are using CLINK naive algorithm for this purpose
    if (group_layers && inTemperature.hasProvider() &&
        !isnan(max_temp_diff) && !isinf(max_temp_diff) &&
        !isnan(temp_dist) && !isinf(temp_dist) &&
        !isnan(temp_layer) && !isinf(temp_layer)) {
        auto temp = inTemperature(adapter.mesh);
        size_t nl = lcount;     // number of idependent layers to consider (stays fixed)
        for (size_t l = 0; l != nl; ++l) {
            std::vector<size_t> indices;
            std::list<std::list<size_t>> groups;
            typedef std::list<std::list<size_t>>::iterator ListIterator;
            typedef std::list<size_t>::const_iterator ItemIterator;
            for (size_t i = 0, j = 0; i != stack.size(); ++i) {
                if (stack[i] == l) {
                    indices.push_back(i);
                    groups.emplace_back(std::list<size_t>({j++}));
                }
            }
            size_t n = indices.size();
            // Make distance matrix
            std::unique_ptr<double[]> dists(new double[n * n]);
#define dists_at(a, b) dists[(a)*n+(b)] //TODO develop plask::UniquePtr2D<> and remove this macro
            for (size_t i = 0; i != n; ++i) {
                dists_at(i, i) = INFINITY; // the simplest way to avoid clustering with itself
                for (size_t j = i+1; j != n; ++j) {
                    double mdt = 0.;
                    for (size_t k = 0; k != adapter.size(); ++k) {
                        double dt = abs(temp[adapter.idx(k, indices[i])] - temp[adapter.idx(k, indices[j])]);
                        if (dt > mdt) mdt = dt;
                    }
                    dists_at(i, j) = dists_at(j, i) = mdt;
                }
            }
            // Go and merge groups with the smallest distances
            while(true) {
                double mdt = INFINITY;
                ListIterator mg1, mg2;
                for (ListIterator g1 = groups.begin(); g1 != groups.end(); ++g1) {
                    ListIterator g2 = g1;
                    for (++g2; g2 != groups.end(); ++g2) {
                        double dt = 0.;
                        for (ItemIterator i1 = g1->begin(); i1 != g1->end(); ++i1)
                            for (ItemIterator i2 = g2->begin(); i2 != g2->end(); ++i2)
                                dt = max(dists_at(*i1, *i2), dt);
                        if (dt < mdt) {
                            mg1 = g1;
                            mg2 = g2;
                            mdt = dt;
                        }
                    }
                }
                if (mdt > 0.66667 * max_temp_diff) break;
                for (ItemIterator i2 = mg2->begin(); i2 != mg2->end(); ++i2)
                    mg1->push_back(*i2);
                groups.erase(mg2);
            }
            // Now update the stack
            ListIterator g = groups.begin();
            for (++g; g != groups.end(); ++g) {
                for (ItemIterator i = g->begin(); i != g->end(); ++i)
                    stack[indices[*i]] = lcount;
                ++lcount;
            }
        }
    }

    Solver::writelog(LOG_DETAIL, "Detected {0} {1}layers", lcount, group_layers? "distinct " : "");

    if (!isnan(interface_position)) {
        double pos = interface_position;
        interface = std::lower_bound(vbounds->begin(), vbounds->end(), pos-0.5*OrderedAxis::MIN_DISTANCE) - vbounds->begin() + 1; // OrderedAxis::MIN_DISTANCE to compensate for truncation errors
        if (std::size_t(interface) > vbounds->size()) interface = vbounds->size();
        pos = vbounds->at(interface-1); if (abs(pos) < OrderedAxis::MIN_DISTANCE) pos = 0.;
        Solver::writelog(LOG_DEBUG, "Setting interface at layer {:d} (exact position {:g})", interface, pos);
    } else
        interface = -1;
}


template <typename BaseT>
DataVector<const Tensor3<dcomplex>> SlabSolver<BaseT>::getRefractiveIndexProfile
                                        (const shared_ptr<const MeshD<BaseT::SpaceType::DIM>>& dst_mesh,
                                        InterpolationMethod interp)
{
    this->initCalculation();
    Expansion& expansion = getExpansion();
    setExpansionDefaults(false);
    if (isnan(expansion.lam0) || always_recompute_gain || isnan(expansion.k0))
        expansion.setK0(isnan(k0)? 2e3*M_PI / lam0 : k0);
    initTransfer(expansion, false);
    computeIntegrals();

    //TODO maybe there is a more efficient way to implement this
    DataVector<Tensor3<dcomplex>> result(dst_mesh->size());
    auto levels = makeLevelsAdapter(dst_mesh);

    //std::map<size_t,LazyData<Tensor3<dcomplex>>> cache;
    //while (auto level = levels->yield()) {
    //    double h = level->vpos();
    //    size_t n = getLayerFor(h);
    //    size_t l = stack[n];
    //    LazyData<Tensor3<dcomplex>> data = cache.find(l);
    //    if (data == cache.end()) {
    //        data = transfer->diagonalizer->source()->getMaterialNR(l, level, interp);
    //        cache[l] = data;
    //    }
    //    for (size_t i = 0; i != level->size(); ++i) result[level->index(i)] = data[i];
    //}
    while (auto level = levels->yield()) {
        double h = level->vpos();
        size_t n = getLayerFor(h);
        size_t l = stack[n];
        auto data = transfer->diagonalizer->source()->getMaterialNR(l, level, interp);
        for (size_t i = 0; i != level->size(); ++i) result[level->index(i)] = data[i];
    }

    return result;
}

#ifndef NDEBUG
template <typename BaseT>
void SlabSolver<BaseT>::getMatrices(size_t layer, cmatrix& RE, cmatrix& RH) {
    this->initCalculation();
    computeIntegrals();
    size_t N = this->getExpansion().matrixSize();
    if (std::size_t(RE.cols()) != N || std::size_t(RE.rows()) != N) RE = cmatrix(N, N);
    if (std::size_t(RH.cols()) != N || std::size_t(RH.rows()) != N) RH = cmatrix(N, N);
    this->getExpansion().getMatrices(layer, RE, RH);
}
#endif

template class PLASK_SOLVER_API SlabSolver<SolverOver<Geometry2DCartesian>>;
template class PLASK_SOLVER_API SlabSolver<SolverWithMesh<Geometry2DCylindrical, OrderedAxis>>;
template class PLASK_SOLVER_API SlabSolver<SolverOver<Geometry3D>>;


// FiltersFactory::RegisterStandard<RefractiveIndex> registerRefractiveIndexFilters;

}}} // namespace
