/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "solver.hpp"
#include "diagonalizer.hpp"
#include "expansion.hpp"
#include "meshadapter.hpp"
#include "muller.hpp"
#include "broyden.hpp"
#include "brent.hpp"
#include "reflection.hpp"
#include "admittance.hpp"
#include "impedance.hpp"

namespace plask { namespace optical { namespace slab {

void SlabBase::initTransfer(Expansion& expansion, bool reflection) {
    switch (transfer_method) {
        case Transfer::METHOD_REFLECTION_ADMITTANCE:
        case Transfer::METHOD_REFLECTION_IMPEDANCE:
            reflection = true;
            break;
        case Transfer::METHOD_ADMITTANCE:
        case Transfer::METHOD_IMPEDANCE:
            reflection = false;
            break;
        default:
            break;
    }
    if (reflection) {
        ReflectionTransfer::Matching matching =
            (transfer_method == Transfer::METHOD_REFLECTION_IMPEDANCE)?
                ReflectionTransfer::MATCH_IMPEDANCE :
                ReflectionTransfer::MATCH_ADMITTANCE;
        if (!this->transfer) {
            ReflectionTransfer* transfer = dynamic_cast<ReflectionTransfer*>(this->transfer.get());
            if (!transfer || transfer->diagonalizer->source() != &expansion || transfer->matching != matching)
                this->transfer.reset(new ReflectionTransfer(this, expansion, matching));
        }
    } else {
        if (transfer_method == Transfer::METHOD_IMPEDANCE) {
            if (!this->transfer || !dynamic_cast<ImpedanceTransfer*>(this->transfer.get()) ||
                this->transfer->diagonalizer->source() != &expansion)
                this->transfer.reset(new ImpedanceTransfer(this, expansion));
        } else {
            if (!this->transfer || !dynamic_cast<AdmittanceTransfer*>(this->transfer.get()) ||
                this->transfer->diagonalizer->source() != &expansion)
                this->transfer.reset(new AdmittanceTransfer(this, expansion));
        }
    }
}


template <typename BaseT>
SlabSolver<BaseT>::SlabSolver(const std::string& name): BaseT(name),
    smooth(0.),
    outRefractiveIndex(this, &SlabSolver<BaseT>::getRefractiveIndexProfile),
    outWavelength(this, &SlabSolver<BaseT>::getWavelength, &SlabSolver<BaseT>::nummodes),
    outLightMagnitude(this, &SlabSolver<BaseT>::getLightMagnitude, &SlabSolver<BaseT>::nummodes),
    outLightE(this, &SlabSolver<BaseT>::getLightE<>, &SlabSolver<BaseT>::nummodes),
    outLightH(this, &SlabSolver<BaseT>::getLightH<>, &SlabSolver<BaseT>::nummodes),
    outUpwardsLightE(this, &SlabSolver<BaseT>::getLightE<PROPAGATION_UPWARDS>, &SlabSolver<BaseT>::nummodes),
    outUpwardsLightH(this, &SlabSolver<BaseT>::getLightH<PROPAGATION_UPWARDS>, &SlabSolver<BaseT>::nummodes),
    outDownwardsLightE(this, &SlabSolver<BaseT>::getLightE<PROPAGATION_DOWNWARDS>, &SlabSolver<BaseT>::nummodes),
    outDownwardsLightH(this, &SlabSolver<BaseT>::getLightH<PROPAGATION_DOWNWARDS>, &SlabSolver<BaseT>::nummodes)
{
    inTemperature = 300.; // temperature receiver has some sensible value
    this->inTemperature.changedConnectMethod(this, &SlabSolver<BaseT>::onInputChanged);
    this->inGain.changedConnectMethod(this, &SlabSolver<BaseT>::onGainChanged);
    this->inCarriersConcentration.changedConnectMethod(this, &SlabSolver<BaseT>::onInputChanged);
}

template <typename BaseT>
SlabSolver<BaseT>::~SlabSolver()
{
    this->inTemperature.changedDisconnectMethod(this, &SlabSolver<BaseT>::onInputChanged);
    this->inGain.changedDisconnectMethod(this, &SlabSolver<BaseT>::onGainChanged);
    this->inCarriersConcentration.changedDisconnectMethod(this, &SlabSolver<BaseT>::onInputChanged);
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
        mesh = make_shared<RectangularMesh<2>>(mesh->axis[0]->getMidpointAxis(),
                                               vbounds, RectangularMesh<2>::ORDER_10);
    }

    void resetMidpoints(const shared_ptr<MeshAxis>& vbounds, double spacing) {
        mesh = make_shared<RectangularMesh<2>>(refineAxis(mesh->axis[0], spacing)->getMidpointAxis(),
                                               vbounds, RectangularMesh<2>::ORDER_10);
    }

    void reset(const shared_ptr<MeshAxis>& verts) {
        mesh = make_shared<RectangularMesh<2>>(mesh->axis[0], verts, RectangularMesh<2>::ORDER_10);
    }

    shared_ptr<RectangularMesh<2>> makeMesh(const shared_ptr<MeshAxis>& verts) {
        return make_shared<RectangularMesh<2>>(mesh->axis[0], verts, RectangularMesh<2>::ORDER_10);
    }

    shared_ptr<OrderedAxis> vert() {
        return dynamic_pointer_cast<OrderedAxis>(mesh->vert());
    }

    shared_ptr<RectangularMesh<2>> midmesh() const {
        return make_shared<RectangularMesh<2>>(mesh->axis[0], mesh->axis[1]->getMidpointAxis());
    }

    size_t size() const { return mesh->axis[0]->size(); }

    size_t idx(size_t i, size_t v) const {
        return mesh->index(i, v);
    }

    Vec<2> at(size_t i, size_t v) const { return mesh->at(i, v); }

    shared_ptr<RectangularMesh<2>> makeLine(size_t i, size_t v, double spacing) const {
        shared_ptr<OrderedAxis> vaxis(new OrderedAxis({mesh->axis[1]->at(v-1), mesh->axis[1]->at(v)}));
        vaxis = refineAxis(vaxis, spacing);
        return make_shared<RectangularMesh<2>>(make_shared<OnePointAxis>(mesh->axis[0]->at(i)), vaxis);
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
        // We divide each rectangle into three points to correctly consider circles and triangles
        for (int ax = 0; ax != 2; ++ax) {
            if (mesh->axis[ax]->size() < 2) continue;
            std::vector<double> refines;
            refines.reserve(2 * (mesh->axis[ax]->size()-1));
            double x = mesh->axis[ax]->at(0);
            for (auto it = ++(mesh->axis[ax]->begin()); it != mesh->axis[ax]->end(); ++it) {
                refines.push_back((2. * x + *it) / 3.);
                refines.push_back((x + 2. * *it) / 3.);
                x = *it;
            }
            static_pointer_cast<OrderedAxis>(mesh->axis[ax])->addOrderedPoints(refines.begin(), refines.end());
        }
        _size = mesh->axis[0]->size() * mesh->axis[1]->size();
    }

    void resetMidpoints(const shared_ptr<MeshAxis>& vbounds) {
        mesh = make_shared<RectangularMesh<3>>(mesh->axis[0]->getMidpointAxis(),
                                               mesh->axis[1]->getMidpointAxis(),
                                               vbounds, RectangularMesh<3>::ORDER_201);
        _size = mesh->axis[0]->size() * mesh->axis[1]->size();
    }

    void resetMidpoints(const shared_ptr<MeshAxis>& vbounds, double spacing) {
        mesh = make_shared<RectangularMesh<3>>(refineAxis(mesh->axis[0], spacing)->getMidpointAxis(),
                                               refineAxis(mesh->axis[1], spacing)->getMidpointAxis(),
                                               vbounds, RectangularMesh<3>::ORDER_201);
        _size = mesh->axis[0]->size() * mesh->axis[1]->size();
    }

    void reset(const shared_ptr<MeshAxis>& verts) {
        mesh = make_shared<RectangularMesh<3>>(mesh->axis[0], mesh->axis[1], verts, RectangularMesh<3>::ORDER_210);
    }

    shared_ptr<RectangularMesh<3>> makeMesh(const shared_ptr<MeshAxis>& verts) {
        return make_shared<RectangularMesh<3>>(mesh->axis[0], mesh->axis[1], verts, RectangularMesh<3>::ORDER_210);
    }

    shared_ptr<OrderedAxis> vert() {
        return dynamic_pointer_cast<OrderedAxis>(mesh->vert());
    }

    shared_ptr<RectangularMesh<3>> midmesh() const {
        return make_shared<RectangularMesh<3>>(mesh->axis[0], mesh->axis[1], mesh->axis[2]->getMidpointAxis());
    }

    size_t size() const { return _size; }

    size_t idx(size_t i, size_t v) const {
        return _size * v + i;
    }

    Vec<3> at(size_t i, size_t v) const {
        return mesh->RectilinearMesh3D::at(idx(i, v));
    }

    shared_ptr<RectangularMesh<3>> makeLine(size_t i, size_t v, double spacing) const {
        shared_ptr<OrderedAxis> vaxis(new OrderedAxis({mesh->axis[2]->at(v-1), mesh->axis[2]->at(v)}));
        vaxis = refineAxis(vaxis, spacing);
        return make_shared<RectangularMesh<3>>(make_shared<OnePointAxis>(mesh->axis[0]->at(mesh->index0(i))),
                                               make_shared<OnePointAxis>(mesh->axis[1]->at(mesh->index1(i))),
                                               vaxis);
    }
};




template <typename BaseT>
void SlabSolver<BaseT>::parseCommonSlabConfiguration(XMLReader& reader, Manager& manager) {
    std::string param = reader.getNodeName();
    if (param == "interface") {
        if (reader.hasAttribute("index")) {
            throw XMLException(reader,
                                "Setting interface by layer index is not supported anymore (set it by object or position)");
        } else if (reader.hasAttribute("position")) {
            if (reader.hasAttribute("object")) throw XMLConflictingAttributesException(reader, "index", "object");
            if (reader.hasAttribute("path")) throw XMLConflictingAttributesException(reader, "index", "path");
            setInterfaceAt(reader.requireAttribute<double>("position"));
        } else if (reader.hasAttribute("object")) {
            auto object = manager.requireGeometryObject<GeometryObject>(reader.requireAttribute("object"));
            PathHints path;
            if (auto pathattr = reader.getAttribute("path")) path = manager.requirePathHints(*pathattr);
            setInterfaceOn(object, path);
        } else if (reader.hasAttribute("path")) {
            throw XMLUnexpectedAttrException(reader, "path");
        }
        reader.requireTagEnd();
    } else if (param == "vpml") {
        vpml.factor = reader.getAttribute<dcomplex>("factor", vpml.factor);
        vpml.size = reader.getAttribute<double>("size", vpml.size);
        vpml.dist = reader.getAttribute<double>("dist", vpml.dist);
        if (reader.hasAttribute("order")) { //TODO Remove in the future
            writelog(LOG_WARNING, "XML line {:d} in <vpml>: Attribute 'order' is obsolete, use 'shape' instead", reader.getLineNr());
            vpml.order = reader.requireAttribute<double>("order");
        }
        vpml.order = reader.getAttribute<double>("shape", vpml.order);
        reader.requireTagEnd();
    } else if (param == "transfer") {
        transfer_method = reader.enumAttribute<Transfer::Method>("method")
                                .value("auto", Transfer::METHOD_AUTO)
                                .value("reflection", Transfer::METHOD_REFLECTION_ADMITTANCE)
                                .value("reflection-admittance", Transfer::METHOD_REFLECTION_ADMITTANCE)
                                .value("reflection-impedance", Transfer::METHOD_REFLECTION_IMPEDANCE)
                                .value("admittance", Transfer::METHOD_ADMITTANCE)
                                .value("impedance", Transfer::METHOD_IMPEDANCE)
                                .get(transfer_method);
        determinant_type = reader.enumAttribute<Transfer::Determinant>("determinant")
            .value("eigen", Transfer::DETERMINANT_EIGENVALUE)
            .value("eigenvalue", Transfer::DETERMINANT_EIGENVALUE)
            .value("full", Transfer::DETERMINANT_FULL)
            .get(determinant_type);
        reader.requireTagEnd();
    } else if (param == "root") {
        readRootDiggerConfig(reader);
    } else {
        this->parseStandardConfiguration(reader, manager);
    }
}

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

    adapter.reset(vbounds->getMidpointAxis());

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
#           define dists_at(a, b) dists[(a)*n+(b)] //TODO develop plask::UniquePtr2D<> and remove this macro
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
                lgained.push_back(lgained[l]);
                ++lcount;
            }
        }
    }
    assert(lgained.size() == lcount);

    if (!isnan(interface_position)) {
        interface = std::lower_bound(vbounds->begin(), vbounds->end(),
                                     interface_position - 0.5*OrderedAxis::MIN_DISTANCE) - vbounds->begin() + 1;
                                     // OrderedAxis::MIN_DISTANCE to compensate for truncation errors
        if (std::size_t(interface) > vbounds->size()) interface = vbounds->size();
    } else
        interface = -1;


    // Merge identical adjacent layers
    std::ptrdiff_t i1 = interface - 1;
    for(size_t i = 0; i < vbounds->size(); ++i) {
        if (stack[i] == stack[i+1] && i != i1) {
            stack.erase(stack.begin() + i+1);
            vbounds->removePoint(i);
            verts->removePoints(i, i+2);
            if (i == 0) {
                OrderedAxis::WarningOff nowarn(verts);
                verts->addPoint(vbounds->at(i) - 2.*OrderedAxis::MIN_DISTANCE);
            } else if (i == vbounds->size()) {
                OrderedAxis::WarningOff nowarn(verts);
                verts->addPoint(vbounds->at(i-1) + 2.*OrderedAxis::MIN_DISTANCE);
            } else {
                verts->addPoint(0.5 * (vbounds->at(i-1) + vbounds->at(i)));
            }
            --i;
            if (i < i1) { --interface; --i1; }
            assert(vbounds->size() == stack.size()-1);
            assert(verts->size() == stack.size());
            if (vbounds->size() == 1) break;    // We always have minimum two layers
        }
    }

    Solver::writelog(LOG_DETAIL, "Detected {0} {1}layers", lcount, group_layers? "distinct " : "");

    // DEBUG
    // for (size_t i = 0; i < stack.size(); ++i) {
    //     if (i != 0) std::cerr << "---- " << vbounds->at(i-1) << "\n";
    //     std::cerr << stack[i] << (lgained[stack[i]]? "*\n" : "\n");
    // }

    if (interface >= 0) {
        double pos = vbounds->at(interface-1); if (abs(pos) < OrderedAxis::MIN_DISTANCE) pos = 0.;
        Solver::writelog(LOG_DEBUG, "Interface is at layer {:d} (exact position {:g}um)", interface, pos);
    } else {
        interface = -1;
    }
}


template <typename BaseT>
DataVector<const Tensor3<dcomplex>> SlabSolver<BaseT>::getRefractiveIndexProfile
                                        (const shared_ptr<const MeshD<BaseT::SpaceType::DIM>>& dst_mesh,
                                        InterpolationMethod interp)
{
    Solver::initCalculation();
    Expansion& expansion = getExpansion();
    setExpansionDefaults(false);
    if (isnan(expansion.lam0) || always_recompute_gain || isnan(expansion.k0))
        expansion.setK0(isnan(k0)? 2e3*PI / lam0 : k0);
    // initTransfer(expansion, false);
    expansion.beforeGetRefractiveIndex();

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
    //        data = expansion.getMaterialNR(l, level, interp);
    //        cache[l] = data;
    //    }
    //    for (size_t i = 0; i != level->size(); ++i) result[level->index(i)] = data[i];
    //}
    while (auto level = levels->yield()) {
        double h = level->vpos();
        size_t n = getLayerFor(h);
        size_t l = stack[n];
        auto data = expansion.getMaterialNR(l, level, interp);
        for (size_t i = 0; i != level->size(); ++i) result[level->index(i)] = data[i];
    }

    expansion.afterGetRefractiveIndex();

    return result;
}

#ifndef NDEBUG
void SlabBase::getMatrices(size_t layer, cmatrix& RE, cmatrix& RH) {
    initCalculation();
    computeIntegrals();
    size_t N = this->getExpansion().matrixSize();
    if (std::size_t(RE.cols()) != N || std::size_t(RE.rows()) != N) RE.reset(N, N);
    if (std::size_t(RH.cols()) != N || std::size_t(RH.rows()) != N) RH.reset(N, N);
    this->getExpansion().getMatrices(layer, RE, RH);
}
#endif


template <typename BaseT>
size_t SlabSolver<BaseT>::initIncidence(Transfer::IncidentDirection side, dcomplex lam) {
    Expansion& expansion = getExpansion();
    bool changed = Solver::initCalculation() || setExpansionDefaults(isnan(lam));
    if (!isnan(lam)) {
        dcomplex k0 = 2e3*M_PI / lam;
        if (!is_zero(k0 - expansion.getK0())) {
            expansion.setK0(k0);
            changed = true;
        }
    }
    size_t layer = stack[(side == Transfer::INCIDENCE_BOTTOM)? 0 : stack.size()-1];
    if (!transfer) {
        initTransfer(expansion, true);
        changed = true;
    }
    if (changed) {
        transfer->initDiagonalization();
        transfer->diagonalizer->diagonalizeLayer(layer);
    } else if (!transfer->diagonalizer->isDiagonalized(layer))
        transfer->diagonalizer->diagonalizeLayer(layer);
    return layer;
}

template <typename BaseT>
cvector SlabSolver<BaseT>::incidentVector(Transfer::IncidentDirection side, size_t idx, dcomplex lam) {
    size_t layer = initIncidence(side, lam);
    if (idx >= transfer->diagonalizer->matrixSize()) throw BadInput(getId(), "Wrong incident eignenmode index");
    cvector incident(transfer->diagonalizer->matrixSize(), 0.);
    incident[idx] = 1.;

    scaleIncidentVector(incident, layer);
    return incident;
}

template <typename BaseT>
cvector SlabSolver<BaseT>::incidentVector(Transfer::IncidentDirection side, const cvector& incident, dcomplex lam) {
    size_t layer = initIncidence(side, lam);
    if (incident.size() != transfer->diagonalizer->matrixSize()) throw BadInput(getId(), "Wrong incident vector size");
    cvector result = incident.claim();
    scaleIncidentVector(result, layer);
    return result;
}

void SlabBase::scaleIncidentVector(cvector& incident, size_t layer, double size_factor) {
    double norm2 = 0.;
    size_t N = transfer->diagonalizer->matrixSize();
    for (size_t i = 0; i != N; ++i) {
        double P = real(incident[i] * conj(incident[i]));
        if (P != 0.) norm2 += P * getExpansion().getModeFlux(i, transfer->diagonalizer->TE(layer), transfer->diagonalizer->TH(layer));
    }

    double norm = size_factor / sqrt(abs(norm2));
    for (size_t i = 0; i != N; ++i) incident[i] *= norm;
}

template <>
void SlabSolver<SolverWithMesh<Geometry2DCartesian, MeshAxis>>::scaleIncidentVector(cvector& incident, size_t layer) {
    SlabBase::scaleIncidentVector(incident, layer, 1e-3); // sqrt(µm -> m)
}

template <>
void SlabSolver<SolverWithMesh<Geometry2DCylindrical, MeshAxis>>::scaleIncidentVector(cvector& incident, size_t layer) {
    throw NotImplemented(getId(), "CylindicalSolver::incidentVector");
}

template <>
void SlabSolver<SolverOver<Geometry3D>>::scaleIncidentVector(cvector& incident, size_t layer) {
    SlabBase::scaleIncidentVector(incident, layer, 1e-6); // sqrt(µm² -> m²)
}


dvector SlabBase::getIncidentFluxes(const cvector& incident, Transfer::IncidentDirection side)
{
    initCalculation();
    if (!transfer) initTransfer(getExpansion(), true);

    dvector result(incident.size());

    size_t n = (side == Transfer::INCIDENCE_BOTTOM)? 0 : stack.size()-1;
    size_t l = stack[n];

    size_t N = transfer->diagonalizer->matrixSize();

    Expansion& expansion = getExpansion();

    double input_flux = 0.;
    for (size_t i = 0; i != N; ++i) {
        double P = real(incident[i] * conj(incident[i]));
        if (P != 0.) {
            result[i] = P * expansion.getModeFlux(i, transfer->diagonalizer->TE(l), transfer->diagonalizer->TH(l));
            input_flux += result[i];
        } else
            result[i] = 0.;
    }

    result /= input_flux;

    return result;
}

dvector SlabBase::getReflectedFluxes(const cvector& incident, Transfer::IncidentDirection side)
{
    cvector reflected = getReflectedCoefficients(incident, side);
    dvector result(reflected.size());

    size_t n = (side == Transfer::INCIDENCE_BOTTOM)? 0 : stack.size()-1;
    size_t l = stack[n];

    size_t N = transfer->diagonalizer->matrixSize();

    Expansion& expansion = getExpansion();

    double input_flux = 0.;
    for (size_t i = 0; i != N; ++i) {
        double P = real(incident[i] * conj(incident[i]));
        if (P != 0.)
            input_flux += P * expansion.getModeFlux(i, transfer->diagonalizer->TE(l), transfer->diagonalizer->TH(l));
    }

    for (size_t i = 0; i != N; ++i) {
        double R = real(reflected[i] * conj(reflected[i]));
        if (R != 0.)
            result[i] = R * expansion.getModeFlux(i, transfer->diagonalizer->TE(l), transfer->diagonalizer->TH(l)) / input_flux;
        else
            result[i] = 0.;
    }

    return result;
}


dvector SlabBase::getTransmittedFluxes(const cvector& incident, Transfer::IncidentDirection side)
{
    cvector transmitted = getTransmittedCoefficients(incident, side);
    dvector result(transmitted.size());

    size_t ni = (side == Transfer::INCIDENCE_BOTTOM)? 0 : stack.size()-1;
    size_t li = stack[ni];

    size_t nt = stack.size()-1 - ni;    // opposite side than ni
    size_t lt = stack[nt];

    size_t N = transfer->diagonalizer->matrixSize();

    Expansion& expansion = getExpansion();

    double input_flux = 0.;
    for (size_t i = 0; i != N; ++i) {
        double P = real(incident[i] * conj(incident[i]));
        if (P != 0.)
            input_flux += P * expansion.getModeFlux(i, transfer->diagonalizer->TE(li), transfer->diagonalizer->TH(li));
    }

    for (size_t i = 0; i != N; ++i) {
        double T = real(transmitted[i] * conj(transmitted[i]));
        if (T != 0.)
            result[i] = T * expansion.getModeFlux(i, transfer->diagonalizer->TE(lt), transfer->diagonalizer->TH(lt)) / input_flux;
        else
            result[i] = 0.;
    }

    return result;
}


cvector SlabBase::getReflectedCoefficients(const cvector& incident, Transfer::IncidentDirection side)
{
    initCalculation();
    if (!transfer) initTransfer(getExpansion(), true);

    return transfer->getReflectionVector(incident, side);
}

cvector SlabBase::getTransmittedCoefficients(const cvector& incident, Transfer::IncidentDirection side)
{
    initCalculation();
    if (!transfer) initTransfer(getExpansion(), true);

    return transfer->getTransmissionVector(incident, side);
}


template <typename BaseT>
template <PropagationDirection part>
LazyData<Vec<3,dcomplex>> SlabSolver<BaseT>::getLightE(size_t num, shared_ptr<const MeshD<BaseT::SpaceType::DIM>> dst_mesh, InterpolationMethod method)
{
    assert(transfer);
    double power = applyMode(num);
    return transfer->getFieldE(power, dst_mesh, method, part);
}


template <typename BaseT>
template <PropagationDirection part>
LazyData<Vec<3,dcomplex>> SlabSolver<BaseT>::getLightH(size_t num, shared_ptr<const MeshD<BaseT::SpaceType::DIM>> dst_mesh, InterpolationMethod method)
{
    assert(transfer);
    double power = applyMode(num);
    return transfer->getFieldH(power, dst_mesh, method, part);
}


template <typename BaseT>
LazyData<double> SlabSolver<BaseT>::getLightMagnitude(size_t num, shared_ptr<const MeshD<BaseT::SpaceType::DIM>> dst_mesh, InterpolationMethod method)
{
    assert(transfer);
    double power = applyMode(num);
    return transfer->getFieldMagnitude(power, dst_mesh, method);
}





// template class PLASK_SOLVER_API SlabSolver<SolverOver<Geometry2DCartesian>>;
template class PLASK_SOLVER_API SlabSolver<SolverWithMesh<Geometry2DCartesian, MeshAxis>>;
template class PLASK_SOLVER_API SlabSolver<SolverWithMesh<Geometry2DCylindrical, MeshAxis>>;
template class PLASK_SOLVER_API SlabSolver<SolverOver<Geometry3D>>;


// FiltersFactory::RegisterStandard<RefractiveIndex> registerRefractiveIndexFilters;

}}} // namespace
