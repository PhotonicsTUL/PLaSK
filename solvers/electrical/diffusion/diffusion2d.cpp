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
 * GNU General Public License for more details.tutorial3-20230720-2358.txt
 */
#include "diffusion2d.hpp"

namespace plask { namespace electrical { namespace diffusion {

constexpr double inv_hc = 1.0e-9 / (phys::c * phys::h_J);
using phys::Z0;

template <typename Geometry2DType>
Diffusion2DSolver<Geometry2DType>::Diffusion2DSolver(const std::string& name)
    : FemSolverWithMesh<Geometry2DType, RectangularMesh<2>>(name),
      loopno(0),
      maxerr(0.05),
      outCarriersConcentration(this, &Diffusion2DSolver<Geometry2DType>::getConcentration) {
    inTemperature = 300.;
}

template <typename Geometry2DType> void Diffusion2DSolver<Geometry2DType>::loadConfiguration(XMLReader& source, Manager& manager) {
    while (source.requireTagOrEnd()) parseConfiguration(source, manager);
}

template <typename Geometry2DType> void Diffusion2DSolver<Geometry2DType>::parseConfiguration(XMLReader& source, Manager& manager) {
    std::string param = source.getNodeName();

    if (param == "loop") {
        maxerr = source.getAttribute<double>("maxerr", maxerr);
        source.requireTagEnd();
    }

    else if (param == "mesh") {
        auto name = source.getAttribute("ref");
        if (!name)
            name.reset(source.requireTextInCurrentTag());
        else
            source.requireTagEnd();
        auto found = manager.meshes.find(*name);
        if (found != manager.meshes.end()) {
            if (shared_ptr<RectangularMesh<2>> mesh = dynamic_pointer_cast<RectangularMesh<2>>(found->second)) {
                this->setMesh(mesh);
            } else if (shared_ptr<MeshGeneratorD<2>> generator = dynamic_pointer_cast<MeshGeneratorD<2>>(found->second)) {
                this->setMesh(generator);
            } else if (shared_ptr<MeshD<1>> mesh = dynamic_pointer_cast<MeshD<1>>(found->second)) {
                this->setMesh(mesh);
            } else if (shared_ptr<MeshGeneratorD<1>> generator = dynamic_pointer_cast<MeshGeneratorD<1>>(found->second)) {
                this->setMesh(generator);
            }
        }
    }

    else if (!this->parseFemConfiguration(source, manager)) {
        this->parseStandardConfiguration(source, manager);
    }
}

template <typename Geometry2DType> Diffusion2DSolver<Geometry2DType>::~Diffusion2DSolver() {}

template <typename Geometry2DType> void Diffusion2DSolver<Geometry2DType>::setActiveRegions() {
    if (!this->geometry || !this->mesh) return;

    auto points = this->mesh->getElementMesh();

    std::vector<typename ActiveRegion::Region> regions;

    for (size_t r = 0; r < points->vert()->size(); ++r) {
        size_t prev = 0;
        shared_ptr<Material> material;
        for (size_t c = 0; c < points->tran()->size(); ++c) {  // In the (possible) active region
            auto point = points->at(c, r);
            size_t num = isActive(point);

            if (num) {  // here we are inside the active region
                if (regions.size() >= num && regions[num - 1].warn) {
                    if (!material)
                        material = this->geometry->getMaterial(points->at(c, r));
                    else if (*material != *this->geometry->getMaterial(points->at(c, r))) {
                        writelog(LOG_WARNING, "Active region {} is laterally non-uniform", num - 1);
                        regions[num - 1].warn = false;
                    }
                }
                regions.resize(max(regions.size(), num));
                auto& reg = regions[num - 1];
                if (prev != num) {  // this region starts in the current row
                    if (reg.top < r) {
                        throw Exception("{0}: Active region {1} is disjoint", this->getId(), num - 1);
                    }
                    if (reg.bottom >= r)
                        reg.bottom = r;  // first row
                    else if (reg.rowr <= c)
                        throw Exception("{0}: Active region {1} is disjoint", this->getId(), num - 1);
                    reg.top = r + 1;
                    reg.rowl = c;
                    if (reg.left > reg.rowl) reg.left = reg.rowl;
                }
            }
            if (prev && prev != num) {  // previous region ended
                auto& reg = regions[prev - 1];
                if (reg.bottom < r && reg.rowl >= c) throw Exception("{0}: Active region {1} is disjoint", this->getId(), prev - 1);
                reg.rowr = c;
                if (reg.right < reg.rowr) reg.right = reg.rowr;
            }
            prev = num;
        }
        if (prev)  // junction reached the edge
            regions[prev - 1].rowr = regions[prev - 1].right = points->tran()->size();
    }

    active.clear();
    active.reserve(regions.size());
    size_t act = 0;
    for (auto& reg : regions) {
        if (reg.bottom == size_t(-1)) continue;
        // Detect quantum wells in the active region
        std::vector<double> QWz;
        std::vector<std::pair<size_t, size_t>> QWbt;
        double QWheight = 0.;
        std::vector<bool> isQW;
        isQW.reserve(reg.top - reg.bottom);
        for (size_t c = reg.left; c < reg.right; ++c) {
            shared_ptr<Material> material;
            for (size_t r = reg.bottom, j = 0; r < reg.top; ++r, ++j) {
                auto point = points->at(c, r);
                auto tags = this->geometry->getRolesAt(point);
                bool QW = tags.find("QW") != tags.end() || tags.find("QD") != tags.end();
                if (c == reg.left) {
                    isQW.push_back(QW);
                    if (QW) {
                        if (QWbt.empty() || QWbt.back().second != r)
                            QWbt.emplace_back(r, r + 1);
                        else
                            QWbt.back().second = r + 1;
                        QWz.push_back(point.c1);
                        QWheight += this->mesh->vert()->at(r + 1) - this->mesh->vert()->at(r);
                    }
                } else if (isQW[j] != QW) {
                    throw Exception("{}: Quantum wells in active region {} are not identical", this->getId(), act);
                }
                if (QW) {
                    if (!material)
                        material = this->geometry->getMaterial(point);
                    else if (*material != *this->geometry->getMaterial(point)) {
                        throw Exception("{}: Quantum wells in active region {} are not identical", this->getId(), act);
                    }
                }
            }
        }
        if (QWz.empty()) {
            throw Exception("{}: Active region {} does not contain quantum wells", this->getId(), act);
        }
        active.emplace_back(this, reg.left, reg.right, reg.bottom, reg.top, QWheight, std::move(QWz), std::move(QWbt));
        this->writelog(LOG_DETAIL, "Total QWs thickness in active region {}: {}nm", act++, 1e3 * active.back().QWheight);
    }
}

template <typename Geometry2DType> void Diffusion2DSolver<Geometry2DType>::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) {
        auto mesh1 = makeGeometryGrid(this->geometry);
        this->mesh = make_shared<RectangularMesh<2>>(refineAxis(mesh1->tran(), DEFAULT_MESH_SPACING), mesh1->vert());
        writelog(LOG_DETAIL, "{}: Setting up default mesh [{}]", this->getId(), this->mesh->tran()->size());
    }
    setActiveRegions();
    loopno = 0;
}

template <typename Geometry2DType> void Diffusion2DSolver<Geometry2DType>::onInvalidate() { active.clear(); }

// clang-format off
template <>
inline void Diffusion2DSolver<Geometry2DCartesian>::setLocalMatrix(
    const double, const double L, const double L2, const double L3, const double L4, const double L5, const double L6,
    const double A, const double B, const double C, const double D,
    const double* U, const double* J,
    double& K00, double& K01, double& K02, double& K03, double& K11,
    double& K12, double& K13, double& K22, double& K23, double& K33,
    double& F0, double& F1, double& F2, double& F3)
{
    K00 += (1.0/180180.0)*(216216*D + L2*(66924*A + 13871*B*L*U[1] - 6149*B*L*U[3] + 110682*B*U[0] + 23166*B*U[2] + 2520*C*L2*U[1]*U[1] - 2331*C*L2*U[1]*U[3] + 804*C*L2*U[3]*U[3] + 33012*C*L*U[0]*U[1] - 12033*C*L*U[0]*U[3] + 8601*C*L*U[1]*U[2] - 6414*C*L*U[2]*U[3] + 144396*C*U[0]*U[0] + 43254*C*U[0]*U[2] + 13122*C*U[2]*U[2]))/L;
    K01 += (11.0/210.0)*A*L2 + (4.0/315.0)*B*L3*U[1] - 1.0/140.0*B*L3*U[3] + (97.0/1260.0)*B*L2*U[0] + (1.0/36.0)*B*L2*U[2] + (7.0/2860.0)*C*L4*U[1]*U[1] - 23.0/8580.0*C*L4*U[1]*U[3] + (25.0/24024.0)*C*L4*U[3]*U[3] + (4.0/143.0)*C*L3*U[0]*U[1] - 37.0/2860.0*C*L3*U[0]*U[3] + (152.0/15015.0)*C*L3*U[1]*U[2] - 17.0/2002.0*C*L3*U[2]*U[3] + (131.0/1430.0)*C*L2*U[0]*U[0] + (2867.0/60060.0)*C*L2*U[0]*U[2] + (1069.0/60060.0)*C*L2*U[2]*U[2] + (1.0/10.0)*D;
    K02 += (1.0/180180.0)*(-216216*D + L2*(23166*A + 5005*B*L*U[1] - 5005*B*L*U[3] + 23166*B*U[0] + 23166*B*U[2] + 912*C*L2*U[1]*U[1] - 1530*C*L2*U[1]*U[3] + 912*C*L2*U[3]*U[3] + 8601*C*L*U[0]*U[1] - 6414*C*L*U[0]*U[3] + 6414*C*L*U[1]*U[2] - 8601*C*L*U[2]*U[3] + 21627*C*U[0]*U[0] + 26244*C*U[0]*U[2] + 21627*C*U[2]*U[2]))/L;
    K03 += -13.0/420.0*A*L2 - 1.0/140.0*B*L3*U[1] + (2.0/315.0)*B*L3*U[3] - 43.0/1260.0*B*L2*U[0] - 1.0/36.0*B*L2*U[2] - 23.0/17160.0*C*L4*U[1]*U[1] + (25.0/12012.0)*C*L4*U[1]*U[3] - 9.0/8008.0*C*L4*U[3]*U[3] - 37.0/2860.0*C*L3*U[0]*U[1] + (134.0/15015.0)*C*L3*U[0]*U[3] - 17.0/2002.0*C*L3*U[1]*U[2] + (152.0/15015.0)*C*L3*U[2]*U[3] - 191.0/5720.0*C*L2*U[0]*U[0] - 1069.0/30030.0*C*L2*U[0]*U[2] - 2867.0/120120.0*C*L2*U[2]*U[2] + (1.0/10.0)*D;
    K11 += (1.0/180180.0)*L*(1716*A*L2 + 429*B*L3*U[1] - 286*B*L3*U[3] + 2288*B*L2*U[0] + 1144*B*L2*U[2] + 84*C*L4*U[1]*U[1] - 105*C*L4*U[1]*U[3] + 45*C*L4*U[3]*U[3] + 882*C*L3*U[0]*U[1] - 483*C*L3*U[0]*U[3] + 405*C*L3*U[1]*U[2] - 375*C*L3*U[2]*U[3] + 2520*C*L2*U[0]*U[0] + 1824*C*L2*U[0]*U[2] + 804*C*L2*U[2]*U[2] + 24024*D);
    K12 += (13.0/420.0)*A*L2 + (2.0/315.0)*B*L3*U[1] - 1.0/140.0*B*L3*U[3] + (1.0/36.0)*B*L2*U[0] + (43.0/1260.0)*B*L2*U[2] + (9.0/8008.0)*C*L4*U[1]*U[1] - 25.0/12012.0*C*L4*U[1]*U[3] + (23.0/17160.0)*C*L4*U[3]*U[3] + (152.0/15015.0)*C*L3*U[0]*U[1] - 17.0/2002.0*C*L3*U[0]*U[3] + (134.0/15015.0)*C*L3*U[1]*U[2] - 37.0/2860.0*C*L3*U[2]*U[3] + (2867.0/120120.0)*C*L2*U[0]*U[0] + (1069.0/30030.0)*C*L2*U[0]*U[2] + (191.0/5720.0)*C*L2*U[2]*U[2] - 1.0/10.0*D;
    K13 += (1.0/360360.0)*L*(-2574*A*L2 - 572*B*L3*U[1] + 572*B*L3*U[3] - 2574*B*L2*U[0] - 2574*B*L2*U[2] - 105*C*L4*U[1]*U[1] + 180*C*L4*U[1]*U[3] - 105*C*L4*U[3]*U[3] - 966*C*L3*U[0]*U[1] + 750*C*L3*U[0]*U[3] - 750*C*L3*U[1]*U[2] + 966*C*L3*U[2]*U[3] - 2331*C*L2*U[0]*U[0] - 3060*C*L2*U[0]*U[2] - 2331*C*L2*U[2]*U[2] - 12012*D);
    K22 += (1.0/180180.0)*(216216*D + L2*(66924*A + 6149*B*L*U[1] - 13871*B*L*U[3] + 23166*B*U[0] + 110682*B*U[2] + 804*C*L2*U[1]*U[1] - 2331*C*L2*U[1]*U[3] + 2520*C*L2*U[3]*U[3] + 6414*C*L*U[0]*U[1] - 8601*C*L*U[0]*U[3] + 12033*C*L*U[1]*U[2] - 33012*C*L*U[2]*U[3] + 13122*C*U[0]*U[0] + 43254*C*U[0]*U[2] + 144396*C*U[2]*U[2]))/L;
    K23 += -11.0/210.0*A*L2 - 1.0/140.0*B*L3*U[1] + (4.0/315.0)*B*L3*U[3] - 1.0/36.0*B*L2*U[0] - 97.0/1260.0*B*L2*U[2] - 25.0/24024.0*C*L4*U[1]*U[1] + (23.0/8580.0)*C*L4*U[1]*U[3] - 7.0/2860.0*C*L4*U[3]*U[3] - 17.0/2002.0*C*L3*U[0]*U[1] + (152.0/15015.0)*C*L3*U[0]*U[3] - 37.0/2860.0*C*L3*U[1]*U[2] + (4.0/143.0)*C*L3*U[2]*U[3] - 1069.0/60060.0*C*L2*U[0]*U[0] - 2867.0/60060.0*C*L2*U[0]*U[2] - 131.0/1430.0*C*L2*U[2]*U[2] - 1.0/10.0*D;
    K33 += (1.0/180180.0)*L*(1716*A*L2 + 286*B*L3*U[1] - 429*B*L3*U[3] + 1144*B*L2*U[0] + 2288*B*L2*U[2] + 45*C*L4*U[1]*U[1] - 105*C*L4*U[1]*U[3] + 84*C*L4*U[3]*U[3] + 375*C*L3*U[0]*U[1] - 405*C*L3*U[0]*U[3] + 483*C*L3*U[1]*U[2] - 882*C*L3*U[2]*U[3] + 804*C*L2*U[0]*U[0] + 1824*C*L2*U[0]*U[2] + 2520*C*L2*U[2]*U[2] + 24024*D);
    F0 += (1.0/180180.0)*L*(1144*B*L2*U[1]*U[1] - 1287*B*L2*U[1]*U[3] + 572*B*L2*U[3]*U[3] + 13871*B*L*U[0]*U[1] - 6149*B*L*U[0]*U[3] + 5005*B*L*U[1]*U[2] - 5005*B*L*U[2]*U[3] + 55341*B*U[0]*U[0] + 23166*B*U[0]*U[2] + 11583*B*U[2]*U[2] + 294*C*L3*U[1]*U[1]*U[1] - 483*C*L3*U[1]*U[1]*U[3] + 375*C*L3*U[1]*U[3]*U[3] - 135*C*L3*U[3]*U[3]*U[3] + 5040*C*L2*U[0]*U[1]*U[1] - 4662*C*L2*U[0]*U[1]*U[3] + 1608*C*L2*U[0]*U[3]*U[3] + 1824*C*L2*U[1]*U[1]*U[2] - 3060*C*L2*U[1]*U[2]*U[3] + 1824*C*L2*U[2]*U[3]*U[3] + 33012*C*L*U[0]*U[0]*U[1] - 12033*C*L*U[0]*U[0]*U[3] + 17202*C*L*U[0]*U[1]*U[2] - 12828*C*L*U[0]*U[2]*U[3] + 6414*C*L*U[1]*U[2]*U[2] - 8601*C*L*U[2]*U[2]*U[3] + 96264*C*U[0]*U[0]*U[0] + 43254*C*U[0]*U[0]*U[2] + 26244*C*U[0]*U[2]*U[2] + 14418*C*U[2]*U[2]*U[2] + 63063*J[0] + 27027*J[1]);
    F1 += (1.0/360360.0)*L2*(429*B*L2*U[1]*U[1] - 572*B*L2*U[1]*U[3] + 286*B*L2*U[3]*U[3] + 4576*B*L*U[0]*U[1] - 2574*B*L*U[0]*U[3] + 2288*B*L*U[1]*U[2] - 2574*B*L*U[2]*U[3] + 13871*B*U[0]*U[0] + 10010*B*U[0]*U[2] + 6149*B*U[2]*U[2] + 112*C*L3*U[1]*U[1]*U[1] - 210*C*L3*U[1]*U[1]*U[3] + 180*C*L3*U[1]*U[3]*U[3] - 70*C*L3*U[3]*U[3]*U[3] + 1764*C*L2*U[0]*U[1]*U[1] - 1932*C*L2*U[0]*U[1]*U[3] + 750*C*L2*U[0]*U[3]*U[3] + 810*C*L2*U[1]*U[1]*U[2] - 1500*C*L2*U[1]*U[2]*U[3] + 966*C*L2*U[2]*U[3]*U[3] + 10080*C*L*U[0]*U[0]*U[1] - 4662*C*L*U[0]*U[0]*U[3] + 7296*C*L*U[0]*U[1]*U[2] - 6120*C*L*U[0]*U[2]*U[3] + 3216*C*L*U[1]*U[2]*U[2] - 4662*C*L*U[2]*U[2]*U[3] + 22008*C*U[0]*U[0]*U[0] + 17202*C*U[0]*U[0]*U[2] + 12828*C*U[0]*U[2]*U[2] + 8022*C*U[2]*U[2]*U[2] + 18018*J[0] + 12012*J[1]);
    F2 += (1.0/180180.0)*L*(572*B*L2*U[1]*U[1] - 1287*B*L2*U[1]*U[3] + 1144*B*L2*U[3]*U[3] + 5005*B*L*U[0]*U[1] - 5005*B*L*U[0]*U[3] + 6149*B*L*U[1]*U[2] - 13871*B*L*U[2]*U[3] + 11583*B*U[0]*U[0] + 23166*B*U[0]*U[2] + 55341*B*U[2]*U[2] + 135*C*L3*U[1]*U[1]*U[1] - 375*C*L3*U[1]*U[1]*U[3] + 483*C*L3*U[1]*U[3]*U[3] - 294*C*L3*U[3]*U[3]*U[3] + 1824*C*L2*U[0]*U[1]*U[1] - 3060*C*L2*U[0]*U[1]*U[3] + 1824*C*L2*U[0]*U[3]*U[3] + 1608*C*L2*U[1]*U[1]*U[2] - 4662*C*L2*U[1]*U[2]*U[3] + 5040*C*L2*U[2]*U[3]*U[3] + 8601*C*L*U[0]*U[0]*U[1] - 6414*C*L*U[0]*U[0]*U[3] + 12828*C*L*U[0]*U[1]*U[2] - 17202*C*L*U[0]*U[2]*U[3] + 12033*C*L*U[1]*U[2]*U[2] - 33012*C*L*U[2]*U[2]*U[3] + 14418*C*U[0]*U[0]*U[0] + 26244*C*U[0]*U[0]*U[2] + 43254*C*U[0]*U[2]*U[2] + 96264*C*U[2]*U[2]*U[2] + 27027*J[0] + 63063*J[1]);
    F3 += (1.0/360360.0)*L2*(-286*B*L2*U[1]*U[1] + 572*B*L2*U[1]*U[3] - 429*B*L2*U[3]*U[3] - 2574*B*L*U[0]*U[1] + 2288*B*L*U[0]*U[3] - 2574*B*L*U[1]*U[2] + 4576*B*L*U[2]*U[3] - 6149*B*U[0]*U[0] - 10010*B*U[0]*U[2] - 13871*B*U[2]*U[2] - 70*C*L3*U[1]*U[1]*U[1] + 180*C*L3*U[1]*U[1]*U[3] - 210*C*L3*U[1]*U[3]*U[3] + 112*C*L3*U[3]*U[3]*U[3] - 966*C*L2*U[0]*U[1]*U[1] + 1500*C*L2*U[0]*U[1]*U[3] - 810*C*L2*U[0]*U[3]*U[3] - 750*C*L2*U[1]*U[1]*U[2] + 1932*C*L2*U[1]*U[2]*U[3] - 1764*C*L2*U[2]*U[3]*U[3] - 4662*C*L*U[0]*U[0]*U[1] + 3216*C*L*U[0]*U[0]*U[3] - 6120*C*L*U[0]*U[1]*U[2] + 7296*C*L*U[0]*U[2]*U[3] - 4662*C*L*U[1]*U[2]*U[2] + 10080*C*L*U[2]*U[2]*U[3] - 8022*C*U[0]*U[0]*U[0] - 12828*C*U[0]*U[0]*U[2] - 17202*C*U[0]*U[2]*U[2] - 22008*C*U[2]*U[2]*U[2] - 12012*J[0] - 18018*J[1]);
}

template <>
inline void Diffusion2DSolver<Geometry2DCylindrical>::setLocalMatrix(
    const double R, const double L, const double L2, const double L3, const double L4, const double L5, const double L6,
    const double A, const double B, const double C, const double D,
    const double* U, const double* J,
    double& K00, double& K01, double& K02, double& K03, double& K11,
    double& K12, double& K13, double& K22, double& K23, double& K33,
    double& F0, double& F1, double& F2, double& F3)
{
    K00 += (1.0/360360.0)*(432432*D*R + L*(30888*A*L2 + 133848*A*L*R + 7540*B*L3*U[1] - 4758*B*L3*U[3] + 27742*B*L2*R*U[1] - 12298*B*L2*R*U[3] + 42822*B*L2*U[0] + 18954*B*L2*U[2] + 221364*B*L*R*U[0] + 46332*B*L*R*U[2] + 1458*C*L4*U[1]*U[1] - 1746*C*L4*U[1]*U[3] + 735*C*L4*U[3]*U[3] + 5040*C*L3*R*U[1]*U[1] - 4662*C*L3*R*U[1]*U[3] + 1608*C*L3*R*U[3]*U[3] + 15912*C*L3*U[0]*U[1] - 8154*C*L3*U[0]*U[3] + 6708*C*L3*U[1]*U[2] - 6120*C*L3*U[2]*U[3] + 66024*C*L2*R*U[0]*U[1] - 24066*C*L2*R*U[0]*U[3] + 17202*C*L2*R*U[1]*U[2] - 12828*C*L2*R*U[2]*U[3] + 48924*C*L2*U[0]*U[0] + 30618*C*L2*U[0]*U[2] + 13122*C*L2*U[2]*U[2] + 288792*C*L*R*U[0]*U[0] + 86508*C*L*R*U[0]*U[2] + 26244*C*L*R*U[2]*U[2] + 216216*D))/L;
    K01 += (1.0/60.0)*A*L3 + (11.0/210.0)*A*L2*R + (19.0/4620.0)*B*L4*U[1] - 1.0/330.0*B*L4*U[3] + (4.0/315.0)*B*L3*R*U[1] - 1.0/140.0*B*L3*R*U[3] + (29.0/1386.0)*B*L3*U[0] + (43.0/3465.0)*B*L3*U[2] + (97.0/1260.0)*B*L2*R*U[0] + (1.0/36.0)*B*L2*R*U[2] + (4.0/5005.0)*C*L5*U[1]*U[1] - 1.0/924.0*C*L5*U[1]*U[3] + (1.0/2002.0)*C*L5*U[3]*U[3] + (7.0/2860.0)*C*L4*R*U[1]*U[1] - 23.0/8580.0*C*L4*R*U[1]*U[3] + (25.0/24024.0)*C*L4*R*U[3]*U[3] + (81.0/10010.0)*C*L4*U[0]*U[1] - 97.0/20020.0*C*L4*U[0]*U[3] + (17.0/4004.0)*C*L4*U[1]*U[2] - 17.0/4004.0*C*L4*U[2]*U[3] + (4.0/143.0)*C*L3*R*U[0]*U[1] - 37.0/2860.0*C*L3*R*U[0]*U[3] + (152.0/15015.0)*C*L3*R*U[1]*U[2] - 17.0/2002.0*C*L3*R*U[2]*U[3] + (17.0/770.0)*C*L3*U[0]*U[0] + (43.0/2310.0)*C*L3*U[0]*U[2] + (43.0/4620.0)*C*L3*U[2]*U[2] + (131.0/1430.0)*C*L2*R*U[0]*U[0] + (2867.0/60060.0)*C*L2*R*U[0]*U[2] + (1069.0/60060.0)*C*L2*R*U[2]*U[2] + (1.0/10.0)*D*L + (1.0/10.0)*D*R;
    K02 += (1.0/360360.0)*(-432432*D*R + L*(23166*A*L2 + 46332*A*L*R + 4472*B*L3*U[1] - 5538*B*L3*U[3] + 10010*B*L2*R*U[1] - 10010*B*L2*R*U[3] + 18954*B*L2*U[0] + 27378*B*L2*U[2] + 46332*B*L*R*U[0] + 46332*B*L*R*U[2] + 765*C*L4*U[1]*U[1] - 1530*C*L4*U[1]*U[3] + 1059*C*L4*U[3]*U[3] + 1824*C*L3*R*U[1]*U[1] - 3060*C*L3*R*U[1]*U[3] + 1824*C*L3*R*U[3]*U[3] + 6708*C*L3*U[0]*U[1] - 6120*C*L3*U[0]*U[3] + 6708*C*L3*U[1]*U[2] - 10494*C*L3*U[2]*U[3] + 17202*C*L2*R*U[0]*U[1] - 12828*C*L2*R*U[0]*U[3] + 12828*C*L2*R*U[1]*U[2] - 17202*C*L2*R*U[2]*U[3] + 15309*C*L2*U[0]*U[0] + 26244*C*L2*U[0]*U[2] + 27945*C*L2*U[2]*U[2] + 43254*C*L*R*U[0]*U[0] + 52488*C*L*R*U[0]*U[2] + 43254*C*L*R*U[2]*U[2] - 216216*D))/L;
    K03 += -1.0/70.0*A*L3 - 13.0/420.0*A*L2*R - 1.0/330.0*B*L4*U[1] + (23.0/6930.0)*B*L4*U[3] - 1.0/140.0*B*L3*R*U[1] + (2.0/315.0)*B*L3*R*U[3] - 61.0/4620.0*B*L3*U[0] - 71.0/4620.0*B*L3*U[2] - 43.0/1260.0*B*L2*R*U[0] - 1.0/36.0*B*L2*R*U[2] - 1.0/1848.0*C*L5*U[1]*U[1] + (1.0/1001.0)*C*L5*U[1]*U[3] - 5.0/8008.0*C*L5*U[3]*U[3] - 23.0/17160.0*C*L4*R*U[1]*U[1] + (25.0/12012.0)*C*L4*R*U[1]*U[3] - 9.0/8008.0*C*L4*R*U[3]*U[3] - 97.0/20020.0*C*L4*U[0]*U[1] + (7.0/1716.0)*C*L4*U[0]*U[3] - 17.0/4004.0*C*L4*U[1]*U[2] + (353.0/60060.0)*C*L4*U[2]*U[3] - 37.0/2860.0*C*L3*R*U[0]*U[1] + (134.0/15015.0)*C*L3*R*U[0]*U[3] - 17.0/2002.0*C*L3*R*U[1]*U[2] + (152.0/15015.0)*C*L3*R*U[2]*U[3] - 453.0/40040.0*C*L3*U[0]*U[0] - 17.0/1001.0*C*L3*U[0]*U[2] - 53.0/3640.0*C*L3*U[2]*U[2] - 191.0/5720.0*C*L2*R*U[0]*U[0] - 1069.0/30030.0*C*L2*R*U[0]*U[2] - 2867.0/120120.0*C*L2*R*U[2]*U[2] + (1.0/10.0)*D*R;
    K11 += (1.0/360360.0)*L*(1287*A*L3 + 3432*A*L2*R + 312*B*L4*U[1] - 260*B*L4*U[3] + 858*B*L3*R*U[1] - 572*B*L3*R*U[3] + 1482*B*L3*U[0] + 1092*B*L3*U[2] + 4576*B*L2*R*U[0] + 2288*B*L2*R*U[2] + 60*C*L5*U[1]*U[1] - 90*C*L5*U[1]*U[3] + 45*C*L5*U[3]*U[3] + 168*C*L4*R*U[1]*U[1] - 210*C*L4*R*U[1]*U[3] + 90*C*L4*R*U[3]*U[3] + 576*C*L4*U[0]*U[1] - 390*C*L4*U[0]*U[3] + 360*C*L4*U[1]*U[2] - 390*C*L4*U[2]*U[3] + 1764*C*L3*R*U[0]*U[1] - 966*C*L3*R*U[0]*U[3] + 810*C*L3*R*U[1]*U[2] - 750*C*L3*R*U[2]*U[3] + 1458*C*L3*U[0]*U[0] + 1530*C*L3*U[0]*U[2] + 873*C*L3*U[2]*U[2] + 5040*C*L2*R*U[0]*U[0] + 3648*C*L2*R*U[0]*U[2] + 1608*C*L2*R*U[2]*U[2] + 12012*D*L + 48048*D*R);
    K12 += (1.0/60.0)*A*L3 + (13.0/420.0)*A*L2*R + (1.0/330.0)*B*L4*U[1] - 19.0/4620.0*B*L4*U[3] + (2.0/315.0)*B*L3*R*U[1] - 1.0/140.0*B*L3*R*U[3] + (43.0/3465.0)*B*L3*U[0] + (29.0/1386.0)*B*L3*U[2] + (1.0/36.0)*B*L2*R*U[0] + (43.0/1260.0)*B*L2*R*U[2] + (1.0/2002.0)*C*L5*U[1]*U[1] - 1.0/924.0*C*L5*U[1]*U[3] + (4.0/5005.0)*C*L5*U[3]*U[3] + (9.0/8008.0)*C*L4*R*U[1]*U[1] - 25.0/12012.0*C*L4*R*U[1]*U[3] + (23.0/17160.0)*C*L4*R*U[3]*U[3] + (17.0/4004.0)*C*L4*U[0]*U[1] - 17.0/4004.0*C*L4*U[0]*U[3] + (97.0/20020.0)*C*L4*U[1]*U[2] - 81.0/10010.0*C*L4*U[2]*U[3] + (152.0/15015.0)*C*L3*R*U[0]*U[1] - 17.0/2002.0*C*L3*R*U[0]*U[3] + (134.0/15015.0)*C*L3*R*U[1]*U[2] - 37.0/2860.0*C*L3*R*U[2]*U[3] + (43.0/4620.0)*C*L3*U[0]*U[0] + (43.0/2310.0)*C*L3*U[0]*U[2] + (17.0/770.0)*C*L3*U[2]*U[2] + (2867.0/120120.0)*C*L2*R*U[0]*U[0] + (1069.0/30030.0)*C*L2*R*U[0]*U[2] + (191.0/5720.0)*C*L2*R*U[2]*U[2] - 1.0/10.0*D*L - 1.0/10.0*D*R;
    K13 += (1.0/360360.0)*L*(-1287*A*L3 - 2574*A*L2*R - 260*B*L4*U[1] + 312*B*L4*U[3] - 572*B*L3*R*U[1] + 572*B*L3*R*U[3] - 1092*B*L3*U[0] - 1482*B*L3*U[2] - 2574*B*L2*R*U[0] - 2574*B*L2*R*U[2] - 45*C*L5*U[1]*U[1] + 90*C*L5*U[1]*U[3] - 60*C*L5*U[3]*U[3] - 105*C*L4*R*U[1]*U[1] + 180*C*L4*R*U[1]*U[3] - 105*C*L4*R*U[3]*U[3] - 390*C*L4*U[0]*U[1] + 360*C*L4*U[0]*U[3] - 390*C*L4*U[1]*U[2] + 576*C*L4*U[2]*U[3] - 966*C*L3*R*U[0]*U[1] + 750*C*L3*R*U[0]*U[3] - 750*C*L3*R*U[1]*U[2] + 966*C*L3*R*U[2]*U[3] - 873*C*L3*U[0]*U[0] - 1530*C*L3*U[0]*U[2] - 1458*C*L3*U[2]*U[2] - 2331*C*L2*R*U[0]*U[0] - 3060*C*L2*R*U[0]*U[2] - 2331*C*L2*R*U[2]*U[2] - 6006*D*L - 12012*D*R);
    K22 += (1.0/360360.0)*(432432*D*R + L*(102960*A*L2 + 133848*A*L*R + 7540*B*L3*U[1] - 20202*B*L3*U[3] + 12298*B*L2*R*U[1] - 27742*B*L2*R*U[3] + 27378*B*L2*U[0] + 178542*B*L2*U[2] + 46332*B*L*R*U[0] + 221364*B*L*R*U[2] + 873*C*L4*U[1]*U[1] - 2916*C*L4*U[1]*U[3] + 3582*C*L4*U[3]*U[3] + 1608*C*L3*R*U[1]*U[1] - 4662*C*L3*R*U[1]*U[3] + 5040*C*L3*R*U[3]*U[3] + 6708*C*L3*U[0]*U[1] - 10494*C*L3*U[0]*U[3] + 15912*C*L3*U[1]*U[2] - 50112*C*L3*U[2]*U[3] + 12828*C*L2*R*U[0]*U[1] - 17202*C*L2*R*U[0]*U[3] + 24066*C*L2*R*U[1]*U[2] - 66024*C*L2*R*U[2]*U[3] + 13122*C*L2*U[0]*U[0] + 55890*C*L2*U[0]*U[2] + 239868*C*L2*U[2]*U[2] + 26244*C*L*R*U[0]*U[0] + 86508*C*L*R*U[0]*U[2] + 288792*C*L*R*U[2]*U[2] + 216216*D))/L;
    K23 += -1.0/28.0*A*L3 - 11.0/210.0*A*L2*R - 19.0/4620.0*B*L4*U[1] + (17.0/1980.0)*B*L4*U[3] - 1.0/140.0*B*L3*R*U[1] + (4.0/315.0)*B*L3*R*U[3] - 71.0/4620.0*B*L3*U[0] - 37.0/660.0*B*L3*U[2] - 1.0/36.0*B*L2*R*U[0] - 97.0/1260.0*B*L2*R*U[2] - 1.0/1848.0*C*L5*U[1]*U[1] + (8.0/5005.0)*C*L5*U[1]*U[3] - 3.0/1820.0*C*L5*U[3]*U[3] - 25.0/24024.0*C*L4*R*U[1]*U[1] + (23.0/8580.0)*C*L4*R*U[1]*U[3] - 7.0/2860.0*C*L4*R*U[3]*U[3] - 17.0/4004.0*C*L4*U[0]*U[1] + (353.0/60060.0)*C*L4*U[0]*U[3] - 81.0/10010.0*C*L4*U[1]*U[2] + (199.0/10010.0)*C*L4*U[2]*U[3] - 17.0/2002.0*C*L3*R*U[0]*U[1] + (152.0/15015.0)*C*L3*R*U[0]*U[3] - 37.0/2860.0*C*L3*R*U[1]*U[2] + (4.0/143.0)*C*L3*R*U[2]*U[3] - 17.0/2002.0*C*L3*U[0]*U[0] - 53.0/1820.0*C*L3*U[0]*U[2] - 348.0/5005.0*C*L3*U[2]*U[2] - 1069.0/60060.0*C*L2*R*U[0]*U[0] - 2867.0/60060.0*C*L2*R*U[0]*U[2] - 131.0/1430.0*C*L2*R*U[2]*U[2] - 1.0/10.0*D*R;
    K33 += (1.0/360360.0)*L*(2145*A*L3 + 3432*A*L2*R + 312*B*L4*U[1] - 546*B*L4*U[3] + 572*B*L3*R*U[1] - 858*B*L3*R*U[3] + 1196*B*L3*U[0] + 3094*B*L3*U[2] + 2288*B*L2*R*U[0] + 4576*B*L2*R*U[2] + 45*C*L5*U[1]*U[1] - 120*C*L5*U[1]*U[3] + 108*C*L5*U[3]*U[3] + 90*C*L4*R*U[1]*U[1] - 210*C*L4*R*U[1]*U[3] + 168*C*L4*R*U[3]*U[3] + 360*C*L4*U[0]*U[1] - 450*C*L4*U[0]*U[3] + 576*C*L4*U[1]*U[2] - 1188*C*L4*U[2]*U[3] + 750*C*L3*R*U[0]*U[1] - 810*C*L3*R*U[0]*U[3] + 966*C*L3*R*U[1]*U[2] - 1764*C*L3*R*U[2]*U[3] + 735*C*L3*U[0]*U[0] + 2118*C*L3*U[0]*U[2] + 3582*C*L3*U[2]*U[2] + 1608*C*L2*R*U[0]*U[0] + 3648*C*L2*R*U[0]*U[2] + 5040*C*L2*R*U[2]*U[2] + 36036*D*L + 48048*D*R);
    F0 += (1.0/360360.0)*L*(741*B*L3*U[1]*U[1] - 1092*B*L3*U[1]*U[3] + 598*B*L3*U[3]*U[3] + 2288*B*L2*R*U[1]*U[1] - 2574*B*L2*R*U[1]*U[3] + 1144*B*L2*R*U[3]*U[3] + 7540*B*L2*U[0]*U[1] - 4758*B*L2*U[0]*U[3] + 4472*B*L2*U[1]*U[2] - 5538*B*L2*U[2]*U[3] + 27742*B*L*R*U[0]*U[1] - 12298*B*L*R*U[0]*U[3] + 10010*B*L*R*U[1]*U[2] - 10010*B*L*R*U[2]*U[3] + 21411*B*L*U[0]*U[0] + 18954*B*L*U[0]*U[2] + 13689*B*L*U[2]*U[2] + 110682*B*R*U[0]*U[0] + 46332*B*R*U[0]*U[2] + 23166*B*R*U[2]*U[2] + 192*C*L4*U[1]*U[1]*U[1] - 390*C*L4*U[1]*U[1]*U[3] + 360*C*L4*U[1]*U[3]*U[3] - 150*C*L4*U[3]*U[3]*U[3] + 588*C*L3*R*U[1]*U[1]*U[1] - 966*C*L3*R*U[1]*U[1]*U[3] + 750*C*L3*R*U[1]*U[3]*U[3] - 270*C*L3*R*U[3]*U[3]*U[3] + 2916*C*L3*U[0]*U[1]*U[1] - 3492*C*L3*U[0]*U[1]*U[3] + 1470*C*L3*U[0]*U[3]*U[3] + 1530*C*L3*U[1]*U[1]*U[2] - 3060*C*L3*U[1]*U[2]*U[3] + 2118*C*L3*U[2]*U[3]*U[3] + 10080*C*L2*R*U[0]*U[1]*U[1] - 9324*C*L2*R*U[0]*U[1]*U[3] + 3216*C*L2*R*U[0]*U[3]*U[3] + 3648*C*L2*R*U[1]*U[1]*U[2] - 6120*C*L2*R*U[1]*U[2]*U[3] + 3648*C*L2*R*U[2]*U[3]*U[3] + 15912*C*L2*U[0]*U[0]*U[1] - 8154*C*L2*U[0]*U[0]*U[3] + 13416*C*L2*U[0]*U[1]*U[2] - 12240*C*L2*U[0]*U[2]*U[3] + 6708*C*L2*U[1]*U[2]*U[2] - 10494*C*L2*U[2]*U[2]*U[3] + 66024*C*L*R*U[0]*U[0]*U[1] - 24066*C*L*R*U[0]*U[0]*U[3] + 34404*C*L*R*U[0]*U[1]*U[2] - 25656*C*L*R*U[0]*U[2]*U[3] + 12828*C*L*R*U[1]*U[2]*U[2] - 17202*C*L*R*U[2]*U[2]*U[3] + 32616*C*L*U[0]*U[0]*U[0] + 30618*C*L*U[0]*U[0]*U[2] + 26244*C*L*U[0]*U[2]*U[2] + 18630*C*L*U[2]*U[2]*U[2] + 192528*C*R*U[0]*U[0]*U[0] + 86508*C*R*U[0]*U[0]*U[2] + 52488*C*R*U[0]*U[2]*U[2] + 28836*C*R*U[2]*U[2]*U[2] + 30030*L*J[0] + 24024*L*J[1] + 126126*R*J[0] + 54054*R*J[1]);
    F1 += (1.0/360360.0)*L2*(156*B*L3*U[1]*U[1] - 260*B*L3*U[1]*U[3] + 156*B*L3*U[3]*U[3] + 429*B*L2*R*U[1]*U[1] - 572*B*L2*R*U[1]*U[3] + 286*B*L2*R*U[3]*U[3] + 1482*B*L2*U[0]*U[1] - 1092*B*L2*U[0]*U[3] + 1092*B*L2*U[1]*U[2] - 1482*B*L2*U[2]*U[3] + 4576*B*L*R*U[0]*U[1] - 2574*B*L*R*U[0]*U[3] + 2288*B*L*R*U[1]*U[2] - 2574*B*L*R*U[2]*U[3] + 3770*B*L*U[0]*U[0] + 4472*B*L*U[0]*U[2] + 3770*B*L*U[2]*U[2] + 13871*B*R*U[0]*U[0] + 10010*B*R*U[0]*U[2] + 6149*B*R*U[2]*U[2] + 40*C*L4*U[1]*U[1]*U[1] - 90*C*L4*U[1]*U[1]*U[3] + 90*C*L4*U[1]*U[3]*U[3] - 40*C*L4*U[3]*U[3]*U[3] + 112*C*L3*R*U[1]*U[1]*U[1] - 210*C*L3*R*U[1]*U[1]*U[3] + 180*C*L3*R*U[1]*U[3]*U[3] - 70*C*L3*R*U[3]*U[3]*U[3] + 576*C*L3*U[0]*U[1]*U[1] - 780*C*L3*U[0]*U[1]*U[3] + 360*C*L3*U[0]*U[3]*U[3] + 360*C*L3*U[1]*U[1]*U[2] - 780*C*L3*U[1]*U[2]*U[3] + 576*C*L3*U[2]*U[3]*U[3] + 1764*C*L2*R*U[0]*U[1]*U[1] - 1932*C*L2*R*U[0]*U[1]*U[3] + 750*C*L2*R*U[0]*U[3]*U[3] + 810*C*L2*R*U[1]*U[1]*U[2] - 1500*C*L2*R*U[1]*U[2]*U[3] + 966*C*L2*R*U[2]*U[3]*U[3] + 2916*C*L2*U[0]*U[0]*U[1] - 1746*C*L2*U[0]*U[0]*U[3] + 3060*C*L2*U[0]*U[1]*U[2] - 3060*C*L2*U[0]*U[2]*U[3] + 1746*C*L2*U[1]*U[2]*U[2] - 2916*C*L2*U[2]*U[2]*U[3] + 10080*C*L*R*U[0]*U[0]*U[1] - 4662*C*L*R*U[0]*U[0]*U[3] + 7296*C*L*R*U[0]*U[1]*U[2] - 6120*C*L*R*U[0]*U[2]*U[3] + 3216*C*L*R*U[1]*U[2]*U[2] - 4662*C*L*R*U[2]*U[2]*U[3] + 5304*C*L*U[0]*U[0]*U[0] + 6708*C*L*U[0]*U[0]*U[2] + 6708*C*L*U[0]*U[2]*U[2] + 5304*C*L*U[2]*U[2]*U[2] + 22008*C*R*U[0]*U[0]*U[0] + 17202*C*R*U[0]*U[0]*U[2] + 12828*C*R*U[0]*U[2]*U[2] + 8022*C*R*U[2]*U[2]*U[2] + 6006*L*J[0] + 6006*L*J[1] + 18018*R*J[0] + 12012*R*J[1]);
    F2 += (1.0/360360.0)*L*(546*B*L3*U[1]*U[1] - 1482*B*L3*U[1]*U[3] + 1547*B*L3*U[3]*U[3] + 1144*B*L2*R*U[1]*U[1] - 2574*B*L2*R*U[1]*U[3] + 2288*B*L2*R*U[3]*U[3] + 4472*B*L2*U[0]*U[1] - 5538*B*L2*U[0]*U[3] + 7540*B*L2*U[1]*U[2] - 20202*B*L2*U[2]*U[3] + 10010*B*L*R*U[0]*U[1] - 10010*B*L*R*U[0]*U[3] + 12298*B*L*R*U[1]*U[2] - 27742*B*L*R*U[2]*U[3] + 9477*B*L*U[0]*U[0] + 27378*B*L*U[0]*U[2] + 89271*B*L*U[2]*U[2] + 23166*B*R*U[0]*U[0] + 46332*B*R*U[0]*U[2] + 110682*B*R*U[2]*U[2] + 120*C*L4*U[1]*U[1]*U[1] - 390*C*L4*U[1]*U[1]*U[3] + 576*C*L4*U[1]*U[3]*U[3] - 396*C*L4*U[3]*U[3]*U[3] + 270*C*L3*R*U[1]*U[1]*U[1] - 750*C*L3*R*U[1]*U[1]*U[3] + 966*C*L3*R*U[1]*U[3]*U[3] - 588*C*L3*R*U[3]*U[3]*U[3] + 1530*C*L3*U[0]*U[1]*U[1] - 3060*C*L3*U[0]*U[1]*U[3] + 2118*C*L3*U[0]*U[3]*U[3] + 1746*C*L3*U[1]*U[1]*U[2] - 5832*C*L3*U[1]*U[2]*U[3] + 7164*C*L3*U[2]*U[3]*U[3] + 3648*C*L2*R*U[0]*U[1]*U[1] - 6120*C*L2*R*U[0]*U[1]*U[3] + 3648*C*L2*R*U[0]*U[3]*U[3] + 3216*C*L2*R*U[1]*U[1]*U[2] - 9324*C*L2*R*U[1]*U[2]*U[3] + 10080*C*L2*R*U[2]*U[3]*U[3] + 6708*C*L2*U[0]*U[0]*U[1] - 6120*C*L2*U[0]*U[0]*U[3] + 13416*C*L2*U[0]*U[1]*U[2] - 20988*C*L2*U[0]*U[2]*U[3] + 15912*C*L2*U[1]*U[2]*U[2] - 50112*C*L2*U[2]*U[2]*U[3] + 17202*C*L*R*U[0]*U[0]*U[1] - 12828*C*L*R*U[0]*U[0]*U[3] + 25656*C*L*R*U[0]*U[1]*U[2] - 34404*C*L*R*U[0]*U[2]*U[3] + 24066*C*L*R*U[1]*U[2]*U[2] - 66024*C*L*R*U[2]*U[2]*U[3] + 10206*C*L*U[0]*U[0]*U[0] + 26244*C*L*U[0]*U[0]*U[2] + 55890*C*L*U[0]*U[2]*U[2] + 159912*C*L*U[2]*U[2]*U[2] + 28836*C*R*U[0]*U[0]*U[0] + 52488*C*R*U[0]*U[0]*U[2] + 86508*C*R*U[0]*U[2]*U[2] + 192528*C*R*U[2]*U[2]*U[2] + 30030*L*J[0] + 96096*L*J[1] + 54054*R*J[0] + 126126*R*J[1]);
    F3 += (1.0/360360.0)*L2*(-130*B*L3*U[1]*U[1] + 312*B*L3*U[1]*U[3] - 273*B*L3*U[3]*U[3] - 286*B*L2*R*U[1]*U[1] + 572*B*L2*R*U[1]*U[3] - 429*B*L2*R*U[3]*U[3] - 1092*B*L2*U[0]*U[1] + 1196*B*L2*U[0]*U[3] - 1482*B*L2*U[1]*U[2] + 3094*B*L2*U[2]*U[3] - 2574*B*L*R*U[0]*U[1] + 2288*B*L*R*U[0]*U[3] - 2574*B*L*R*U[1]*U[2] + 4576*B*L*R*U[2]*U[3] - 2379*B*L*U[0]*U[0] - 5538*B*L*U[0]*U[2] - 10101*B*L*U[2]*U[2] - 6149*B*R*U[0]*U[0] - 10010*B*R*U[0]*U[2] - 13871*B*R*U[2]*U[2] - 30*C*L4*U[1]*U[1]*U[1] + 90*C*L4*U[1]*U[1]*U[3] - 120*C*L4*U[1]*U[3]*U[3] + 72*C*L4*U[3]*U[3]*U[3] - 70*C*L3*R*U[1]*U[1]*U[1] + 180*C*L3*R*U[1]*U[1]*U[3] - 210*C*L3*R*U[1]*U[3]*U[3] + 112*C*L3*R*U[3]*U[3]*U[3] - 390*C*L3*U[0]*U[1]*U[1] + 720*C*L3*U[0]*U[1]*U[3] - 450*C*L3*U[0]*U[3]*U[3] - 390*C*L3*U[1]*U[1]*U[2] + 1152*C*L3*U[1]*U[2]*U[3] - 1188*C*L3*U[2]*U[3]*U[3] - 966*C*L2*R*U[0]*U[1]*U[1] + 1500*C*L2*R*U[0]*U[1]*U[3] - 810*C*L2*R*U[0]*U[3]*U[3] - 750*C*L2*R*U[1]*U[1]*U[2] + 1932*C*L2*R*U[1]*U[2]*U[3] - 1764*C*L2*R*U[2]*U[3]*U[3] - 1746*C*L2*U[0]*U[0]*U[1] + 1470*C*L2*U[0]*U[0]*U[3] - 3060*C*L2*U[0]*U[1]*U[2] + 4236*C*L2*U[0]*U[2]*U[3] - 2916*C*L2*U[1]*U[2]*U[2] + 7164*C*L2*U[2]*U[2]*U[3] - 4662*C*L*R*U[0]*U[0]*U[1] + 3216*C*L*R*U[0]*U[0]*U[3] - 6120*C*L*R*U[0]*U[1]*U[2] + 7296*C*L*R*U[0]*U[2]*U[3] - 4662*C*L*R*U[1]*U[2]*U[2] + 10080*C*L*R*U[2]*U[2]*U[3] - 2718*C*L*U[0]*U[0]*U[0] - 6120*C*L*U[0]*U[0]*U[2] - 10494*C*L*U[0]*U[2]*U[2] - 16704*C*L*U[2]*U[2]*U[2] - 8022*C*R*U[0]*U[0]*U[0] - 12828*C*R*U[0]*U[0]*U[2] - 17202*C*R*U[0]*U[2]*U[2] - 22008*C*R*U[2]*U[2]*U[2] - 6006*L*J[0] - 12012*L*J[1] - 12012*R*J[0] - 18018*R*J[1]);
}

template <>
inline void Diffusion2DSolver<Geometry2DCartesian>::addLocalBurningMatrix(
    const double, const double L, const double L2, const double L3,
    const double* P, const double* g, const double* dg, const double ug,
    double& K00, double& K01, double& K02, double& K03, double& K11,
    double& K12, double& K13, double& K22, double& K23, double& K33, double& F0,
    double& F1, double& F2, double& F3)
{
    K00 += (1.0/35.0)*L*(10*P[0]*dg[0] + 10*P[1]*dg[1] + 3*P[2]*dg[0] + 3*P[3]*dg[1]);
    K01 += L2*((1.0/28.0)*P[0]*dg[0] + (1.0/28.0)*P[1]*dg[1] + (1.0/60.0)*P[2]*dg[0] + (1.0/60.0)*P[3]*dg[1]);
    K02 += (9.0/140.0)*L*(P[0]*dg[0] + P[1]*dg[1] + P[2]*dg[0] + P[3]*dg[1]);
    K03 += L2*(-1.0/60.0*P[0]*dg[0] - 1.0/60.0*P[1]*dg[1] - 1.0/70.0*P[2]*dg[0] - 1.0/70.0*P[3]*dg[1]);
    K11 += L3*((1.0/168.0)*P[0]*dg[0] + (1.0/168.0)*P[1]*dg[1] + (1.0/280.0)*P[2]*dg[0] + (1.0/280.0)*P[3]*dg[1]);
    K12 += L2*((1.0/70.0)*P[0]*dg[0] + (1.0/70.0)*P[1]*dg[1] + (1.0/60.0)*P[2]*dg[0] + (1.0/60.0)*P[3]*dg[1]);
    K13 += (1.0/280.0)*L3*(-P[0]*dg[0] - P[1]*dg[1] - P[2]*dg[0] - P[3]*dg[1]);
    K22 += (1.0/35.0)*L*(3*P[0]*dg[0] + 3*P[1]*dg[1] + 10*P[2]*dg[0] + 10*P[3]*dg[1]);
    K23 += L2*(-1.0/60.0*P[0]*dg[0] - 1.0/60.0*P[1]*dg[1] - 1.0/28.0*P[2]*dg[0] - 1.0/28.0*P[3]*dg[1]);
    K33 += L3*((1.0/280.0)*P[0]*dg[0] + (1.0/280.0)*P[1]*dg[1] + (1.0/168.0)*P[2]*dg[0] + (1.0/168.0)*P[3]*dg[1]);
    F0 += (1.0/20.0)*L*(7*ug*P[0]*dg[0] + 7*ug*P[1]*dg[1] + 3*ug*P[2]*dg[0] + 3*ug*P[3]*dg[1] - 7*P[0]*g[0] - 7*P[1]*g[1] - 3*P[2]*g[0] - 3*P[3]*g[1]);
    F1 += L2*((1.0/20.0)*ug*P[0]*dg[0] + (1.0/20.0)*ug*P[1]*dg[1] + (1.0/30.0)*ug*P[2]*dg[0] + (1.0/30.0)*ug*P[3]*dg[1] - 1.0/20.0*P[0]*g[0] - 1.0/20.0*P[1]*g[1] - 1.0/30.0*P[2]*g[0] - 1.0/30.0*P[3]*g[1]);
    F2 += (1.0/20.0)*L*(3*ug*P[0]*dg[0] + 3*ug*P[1]*dg[1] + 7*ug*P[2]*dg[0] + 7*ug*P[3]*dg[1] - 3*P[0]*g[0] - 3*P[1]*g[1] - 7*P[2]*g[0] - 7*P[3]*g[1]);
    F3 += L2*(-1.0/30.0*ug*P[0]*dg[0] - 1.0/30.0*ug*P[1]*dg[1] - 1.0/20.0*ug*P[2]*dg[0] - 1.0/20.0*ug*P[3]*dg[1] + (1.0/30.0)*P[0]*g[0] + (1.0/30.0)*P[1]*g[1] + (1.0/20.0)*P[2]*g[0] + (1.0/20.0)*P[3]*g[1]);
}

template <>
inline void Diffusion2DSolver<Geometry2DCylindrical>::addLocalBurningMatrix(
    const double R, const double L, const double L2, const double L3,
    const double* P, const double* g, const double* dg, const double ug,
    double& K00, double& K01, double& K02, double& K03, double& K11,
    double& K12, double& K13, double& K22, double& K23, double& K33, double& F0,
    double& F1, double& F2, double& F3)
{
    K00 += (1.0/630.0)*L*(35*L*P[0]*dg[0] + 35*L*P[1]*dg[1] + 19*L*P[2]*dg[0] + 19*L*P[3]*dg[1] + 180*R*P[0]*dg[0] + 180*R*P[1]*dg[1] + 54*R*P[2]*dg[0] + 54*R*P[3]*dg[1]);
    K01 += (1.0/2520.0)*L2*(25*L*P[0]*dg[0] + 25*L*P[1]*dg[1] + 17*L*P[2]*dg[0] + 17*L*P[3]*dg[1] + 90*R*P[0]*dg[0] + 90*R*P[1]*dg[1] + 42*R*P[2]*dg[0] + 42*R*P[3]*dg[1]);
    K02 += (1.0/1260.0)*L*(35*L*P[0]*dg[0] + 35*L*P[1]*dg[1] + 46*L*P[2]*dg[0] + 46*L*P[3]*dg[1] + 81*R*P[0]*dg[0] + 81*R*P[1]*dg[1] + 81*R*P[2]*dg[0] + 81*R*P[3]*dg[1]);
    K03 += (1.0/2520.0)*L2*(-17*L*P[0]*dg[0] - 17*L*P[1]*dg[1] - 19*L*P[2]*dg[0] - 19*L*P[3]*dg[1] - 42*R*P[0]*dg[0] - 42*R*P[1]*dg[1] - 36*R*P[2]*dg[0] - 36*R*P[3]*dg[1]);
    K11 += L3*((1.0/504.0)*L*P[0]*dg[0] + (1.0/504.0)*L*P[1]*dg[1] + (1.0/630.0)*L*P[2]*dg[0] + (1.0/630.0)*L*P[3]*dg[1] + (1.0/168.0)*R*P[0]*dg[0] + (1.0/168.0)*R*P[1]*dg[1] + (1.0/280.0)*R*P[2]*dg[0] + (1.0/280.0)*R*P[3]*dg[1]);
    K12 += (1.0/2520.0)*L2*(17*L*P[0]*dg[0] + 17*L*P[1]*dg[1] + 25*L*P[2]*dg[0] + 25*L*P[3]*dg[1] + 36*R*P[0]*dg[0] + 36*R*P[1]*dg[1] + 42*R*P[2]*dg[0] + 42*R*P[3]*dg[1]);
    K13 += L3*(-1.0/630.0*L*P[0]*dg[0] - 1.0/630.0*L*P[1]*dg[1] - 1.0/504.0*L*P[2]*dg[0] - 1.0/504.0*L*P[3]*dg[1] - 1.0/280.0*R*P[0]*dg[0] - 1.0/280.0*R*P[1]*dg[1] - 1.0/280.0*R*P[2]*dg[0] - 1.0/280.0*R*P[3]*dg[1]);
    K22 += (1.0/630.0)*L*(35*L*P[0]*dg[0] + 35*L*P[1]*dg[1] + 145*L*P[2]*dg[0] + 145*L*P[3]*dg[1] + 54*R*P[0]*dg[0] + 54*R*P[1]*dg[1] + 180*R*P[2]*dg[0] + 180*R*P[3]*dg[1]);
    K23 += (1.0/2520.0)*L2*(-25*L*P[0]*dg[0] - 25*L*P[1]*dg[1] - 65*L*P[2]*dg[0] - 65*L*P[3]*dg[1] - 42*R*P[0]*dg[0] - 42*R*P[1]*dg[1] - 90*R*P[2]*dg[0] - 90*R*P[3]*dg[1]);
    K33 += L3*((1.0/504.0)*L*P[0]*dg[0] + (1.0/504.0)*L*P[1]*dg[1] + (1.0/252.0)*L*P[2]*dg[0] + (1.0/252.0)*L*P[3]*dg[1] + (1.0/280.0)*R*P[0]*dg[0] + (1.0/280.0)*R*P[1]*dg[1] + (1.0/168.0)*R*P[2]*dg[0] + (1.0/168.0)*R*P[3]*dg[1]);
    F0 += (1.0/60.0)*L*(5*L*ug*P[0]*dg[0] + 5*L*ug*P[1]*dg[1] + 4*L*ug*P[2]*dg[0] + 4*L*ug*P[3]*dg[1] - 5*L*P[0]*g[0] - 5*L*P[1]*g[1] - 4*L*P[2]*g[0] - 4*L*P[3]*g[1] + 21*R*ug*P[0]*dg[0] + 21*R*ug*P[1]*dg[1] + 9*R*ug*P[2]*dg[0] + 9*R*ug*P[3]*dg[1] - 21*R*P[0]*g[0] - 21*R*P[1]*g[1] - 9*R*P[2]*g[0] - 9*R*P[3]*g[1]);
    F1 += (1.0/60.0)*L2*(L*ug*P[0]*dg[0] + L*ug*P[1]*dg[1] + L*ug*P[2]*dg[0] + L*ug*P[3]*dg[1] - L*P[0]*g[0] - L*P[1]*g[1] - L*P[2]*g[0] - L*P[3]*g[1] + 3*R*ug*P[0]*dg[0] + 3*R*ug*P[1]*dg[1] + 2*R*ug*P[2]*dg[0] + 2*R*ug*P[3]*dg[1] - 3*R*P[0]*g[0] - 3*R*P[1]*g[1] - 2*R*P[2]*g[0] - 2*R*P[3]*g[1]);
    F2 += (1.0/60.0)*L*(5*L*ug*P[0]*dg[0] + 5*L*ug*P[1]*dg[1] + 16*L*ug*P[2]*dg[0] + 16*L*ug*P[3]*dg[1] - 5*L*P[0]*g[0] - 5*L*P[1]*g[1] - 16*L*P[2]*g[0] - 16*L*P[3]*g[1] + 9*R*ug*P[0]*dg[0] + 9*R*ug*P[1]*dg[1] + 21*R*ug*P[2]*dg[0] + 21*R*ug*P[3]*dg[1] - 9*R*P[0]*g[0] - 9*R*P[1]*g[1] - 21*R*P[2]*g[0] - 21*R*P[3]*g[1]);
    F3 += (1.0/60.0)*L2*(-L*ug*P[0]*dg[0] - L*ug*P[1]*dg[1] - 2*L*ug*P[2]*dg[0] - 2*L*ug*P[3]*dg[1] + L*P[0]*g[0] + L*P[1]*g[1] + 2*L*P[2]*g[0] + 2*L*P[3]*g[1] - 2*R*ug*P[0]*dg[0] - 2*R*ug*P[1]*dg[1] - 3*R*ug*P[2]*dg[0] - 3*R*ug*P[3]*dg[1] + 2*R*P[0]*g[0] + 2*R*P[1]*g[1] + 3*R*P[2]*g[0] + 3*R*P[3]*g[1]);
}
// clang-format on

template <>
template <typename T>
inline T Diffusion2DSolver<Geometry2DCartesian>::integrateLinear(const double R, const double L, const T* P) {
    T res = 0.5 * (P[0] + P[1]) * L;
    if (this->geometry->getExtrusion()->getLength()) res *= this->geometry->getExtrusion()->getLength();
    return res;
}

template <>
template <typename T>
inline T Diffusion2DSolver<Geometry2DCylindrical>::integrateLinear(const double R, const double L, const T* P) {
    return (PI / 6.) * L * (L * (P[0] + 2 * P[1]) + 3 * R * (P[0] + P[1]));
}

template <typename Geometry2DType> double Diffusion2DSolver<Geometry2DType>::compute(unsigned loops, bool shb, size_t act) {
    this->initCalculation();

    auto& active = this->active[act];
    double z = active.vert();

    auto mesh = active.mesh();
    size_t nn = mesh->size();
    size_t N = 2 * nn, ne = nn - 1;
    assert(active.mesh1->size() == nn);
    assert(active.emesh1->size() == ne);

    size_t nmodes = 0;

    if (!active.U) active.U.reset(N, 0.);

    DataVector<double> A(ne), B(ne), C(ne), D(ne);

    auto temperature = active.verticallyAverage(inTemperature, active.emesh2, InterpolationMethod::INTERPOLATION_SPLINE);
    for (size_t i = 0; i != ne; ++i) {
        auto material = this->geometry->getMaterial(active.emesh1->at(i));
        double T = temperature[i];
        A[i] = material->A(T);
        B[i] = material->B(T);
        C[i] = material->C(T);
        D[i] = 1e8 * material->D(T);  // cm²/s -> µm²/s
    }

    DataVector<double> J(nn);
    double js = 1e7 / (phys::qe * active.QWheight);
    size_t i = 0;
    for (auto j : inCurrentDensity(active.mesh1, InterpolationMethod::INTERPOLATION_SPLINE)) {
        J[i] = abs(js * j.c1);
        ++i;
    }

    std::vector<DataVector<Tensor2<double>>> Ps;
    std::vector<DataVector<double>> nrs;

    this->writelog(LOG_INFO, "Running diffusion calculations");

    if (shb) {
        nmodes = inWavelength.size();

        if (inLightE.size() != nmodes)
            throw BadInput(this->getId(), "Number of modes in inWavelength ({}) and inLightE ({}) differ", inWavelength.size(),
                           inLightE.size());

        active.modesP.assign(inWavelength.size(), 0.);

        Ps.reserve(nmodes);
        nrs.reserve(nmodes);
        for (size_t i = 0; i != nmodes; ++i) {
            Ps.emplace_back(nn);
            nrs.emplace_back(ne);
        }

        for (size_t i = 0; i != ne; ++i) {
            auto material = this->geometry->getMaterial(active.emesh1->at(i));
            for (size_t n = 0; n != nmodes; ++n) nrs[n][i] = material->Nr(real(inWavelength(n)), temperature[i]).real();
        }

        for (size_t n = 0; n != nmodes; ++n) {
            double wavelength = real(inWavelength(n));
            writelog(LOG_DEBUG, "Mode {} wavelength: {} nm", n, wavelength);
            auto P = active.verticallyAverage(inLightE, active.mesh2, InterpolationMethod::INTERPOLATION_SPLINE);
            for (size_t i = 0; i != nn; ++i) {
                Ps[n][i].c00 = (0.5 / Z0) * real(P[i].c0 * conj(P[i].c0) + P[i].c1 * conj(P[i].c1));
                Ps[n][i].c11 = (0.5 / Z0) * real(P[i].c2 * conj(P[i].c2));
            }
        }
    }

    unsigned loop = 0;

    std::unique_ptr<FemMatrix> K;

    toterr = 0.;

    DataVector<double> F(N);
    DataVector<double> resid(N);

    switch (this->algorithm) {
        case ALGORITHM_CHOLESKY: K.reset(new DpbMatrix(this, N, 3)); break;
        case ALGORITHM_GAUSS: K.reset(new DgbMatrix(this, N, 3)); break;
        case ALGORITHM_ITERATIVE: K.reset(new SparseBandMatrix(this, N, 3)); break;
    }

    while (true) {
        // Set stiffness matrix and load vector
        this->writelog(LOG_DETAIL, "Setting up matrix system (size={})", K->size);
        K->clear();
        F.fill(0.);
        for (size_t ie = 0; ie < ne; ++ie) {
            double x0 = mesh->at(ie), x1 = mesh->at(ie + 1);
            size_t i = 2 * ie;
            // clang-format off
            const double L = x1 - x0;
            const double L2 = L * L;
            const double L3 = L2 * L;
            const double L4 = L2 * L2, L5 = L3 * L2, L6 = L3 * L3;
            setLocalMatrix(x0, L, L2, L3, L4, L5, L6, A[ie], B[ie], C[ie], D[ie], active.U.data() + i, J.data() + ie,
                           (*K)(i,i), (*K)(i,i+1), (*K)(i,i+2), (*K)(i,i+3), (*K)(i+1,i+1),
                           (*K)(i+1,i+2), (*K)(i+1,i+3), (*K)(i+2,i+2), (*K)(i+2,i+3), (*K)(i+3,i+3),
                           F[i], F[i+1], F[i+2], F[i+3]);
            // clang-format on
        }

        // Add SHB
        if (shb) {
            std::fill(active.modesP.begin(), active.modesP.end(), 0.);
            for (size_t n = 0; n != nmodes; ++n) {
                double wavelength = real(inWavelength(n));
                double factor = 1e-4 * inv_hc * wavelength;
                auto gain = inGain(active.emesh1, wavelength, InterpolationMethod::INTERPOLATION_SPLINE);
                auto dgdn = inGain(Gain::DGDN, active.emesh1, wavelength, InterpolationMethod::INTERPOLATION_SPLINE);
                const Tensor2<double>* Pdata = Ps[n].data();
                for (size_t ie = 0; ie < ne; ++ie) {
                    double x0 = mesh->at(ie), x1 = mesh->at(ie + 1);
                    size_t i = 2 * ie;
                    const double L = x1 - x0;
                    // clang-format off
                    const double L2 = L * L;
                    const double L3 = L2 * L;
                    Tensor2<double> g = nrs[n][ie] * gain[ie];
                    Tensor2<double> dg = nrs[n][ie] * dgdn[ie];
                    Tensor2<double> p = integrateLinear(x0, L, Pdata + ie);
                    active.modesP[n] += p.c00 * g.c00 + p.c11 * g.c11;
                    g *= factor;
                    dg *= factor;
                    double ug = 0.5 * active.U[i] + 0.125 * active.U[i+1] + 0.5 * active.U[i+2] - 0.125 * active.U[i+3];
                    addLocalBurningMatrix(x0, L, L2, L3, reinterpret_cast<const double*>(Pdata + ie), &g.c00, &dg.c00, ug,
                                          (*K)(i,i), (*K)(i,i+1), (*K)(i,i+2), (*K)(i,i+3), (*K)(i+1,i+1),
                                          (*K)(i+1,i+2), (*K)(i+1,i+3), (*K)(i+2,i+2), (*K)(i+2,i+3), (*K)(i+3,i+3),
                                          F[i], F[i+1], F[i+2], F[i+3]);
                    // clang-format ons
                }
                active.modesP[n] *= 1e-1 * active.QWheight;
                // 10⁻¹ from µm to cm conversion and conversion to mW (r dr), (...) - photon energy
            }
        }

        // Set derivatives to 0 at the edges
        K->setBC(F, 1, 0.);
        K->setBC(F, K->size - 1, 0.);

#ifndef NDEBUG
        double* kend = K->data + K->size * K->kd;
        for (double* pk = K->data; pk != kend; ++pk) {
            if (isnan(*pk) || isinf(*pk))
                throw ComputationError(this->getId(), "Error in stiffness matrix at position {0} ({1})", pk - K->data,
                                       isnan(*pk) ? "nan" : "inf");
        }
        for (auto f = F.begin(); f != F.end(); ++f) {
            if (isnan(*f) || isinf(*f))
                throw ComputationError(this->getId(), "Error in load vector at position {0} ({1})", f - F.begin(),
                                       isnan(*f) ? "nan" : "inf");
        }
#endif

        // Compute current error
        for (auto f = F.begin(), r = resid.begin(); f != F.end(); ++f, ++r) *r = -*f;
        K->addmult(active.U, resid);

        double err = 0.;
        for (auto r = resid.begin(); r != resid.end(); ++r) err += *r * *r;
        double denorm = 0.;
        for (auto f = F.begin(); f != F.end(); ++f) denorm += *f * *f;
        err = 100. * sqrt(err / denorm);

        // Do next calculation step
        if (loop != 0) this->writelog(LOG_RESULT, "Loop {:d}({:d}) @ active region {}: error = {:g}%", loop, loopno, act, err);
        ++loopno;
        ++loop;
        if (err < maxerr || ((loops != 0 && loop >= loops))) break;

        // TODO add linear mixing with the previous solution
        K->solve(F, active.U);
    }

    outCarriersConcentration.fireChanged();

    return toterr;
}

template <typename Geometry2DType>
Diffusion2DSolver<Geometry2DType>::ConcentrationDataImpl::ConcentrationDataImpl(const Diffusion2DSolver* solver,
                                                                                shared_ptr<const plask::MeshD<2>> dest_mesh,
                                                                                InterpolationMethod interp)
    : solver(solver), destination_mesh(dest_mesh), interpolationFlags(InterpolationFlags(solver->geometry)) {
    concentrations.reserve(solver->active.size());

    if (interp == InterpolationMethod::INTERPOLATION_DEFAULT || interp == InterpolationMethod::INTERPOLATION_SPLINE) {
        for (const auto& active : solver->active) {
            auto src_mesh = active.mesh();
            if (!active.U) throw NoValue("Carriers concentration");
            assert(src_mesh->size() == active.U.size() / 2);
            concentrations.emplace_back(LazyData<double>(dest_mesh->size(), [this, active, src_mesh](size_t i) -> double {
                double x = interpolationFlags.wrap(0, destination_mesh->at(i).c0);
                assert(src_mesh->at(0) <= x && x <= src_mesh->at(src_mesh->size() - 1));
                size_t idx = src_mesh->findIndex(x);
                if (idx == 0) return active.U[0];
                const double x0 = src_mesh->at(idx - 1);
                const double L = src_mesh->at(idx) - x0;
                x -= x0;
                double L2 = L * L;
                double L3 = L2 * L;
                double x2 = x * x;
                double x3 = x2 * x;
                idx *= 2;
                // U[idx - 2] <- U[idx - 1].value
                // U[idx]     <- U[idx].value
                // U[idx - 1] <- U[idx - 1].derivative
                // U[idx + 1] <- U[idx].derivative
                return (1 - 3 * x2 / L2 + 2 * x3 / L3) * active.U[idx - 2] + (3 * x2 / L2 - 2 * x3 / L3) * active.U[idx] +
                       (x - 2 * x2 / L + x3 / L2) * active.U[idx - 1] + (-x2 / L + x3 / L2) * active.U[idx + 1];
            }));
        }

    } else {
        for (const auto& active : solver->active) {
            if (!active.U) throw NoValue("Carriers concentration");
            shared_ptr<RectangularMesh<2>> mesh(
                new RectangularMesh<2>(active.mesh(), shared_ptr<OnePointAxis>(new OnePointAxis(active.vert()))));
            DataVector<double> conc(active.U.size() / 2);
            DataVector<double>::iterator c = conc.begin();
            for (auto u = active.U.begin(); u < active.U.end(); u += 2, ++c) *c = *u;
            concentrations.emplace_back(interpolate(mesh, conc, dest_mesh, interp, interpolationFlags));
        }
    }
}

template <typename Geometry2DType> double Diffusion2DSolver<Geometry2DType>::ConcentrationDataImpl::at(size_t i) const {
    auto point = interpolationFlags.wrap(destination_mesh->at(i));
    bool found = false;
    size_t an = 0;
    for (const auto& active : solver->active) {
        if (solver->mesh->vert()->at(active.bottom) <= point.c1 && point.c1 <= solver->mesh->vert()->at(active.top)) {
            // Make sure we have concentration only in the quantum wells
            // TODO maybe more optimal approach would be reasonable?
            if (solver->mesh->tran()->at(active.left) <= point.c0 && point.c0 <= solver->mesh->tran()->at(active.right))
                for (auto qw : active.QWs)
                    if (qw.first <= point.c1 && point.c1 < qw.second) {
                        found = true;
                        break;
                    }
            break;
        }
        ++an;
    }
    if (!found) return 0.;
    return concentrations[an][i];
}

template <typename Geometry2DType>
const LazyData<double> Diffusion2DSolver<Geometry2DType>::getConcentration(CarriersConcentration::EnumType what,
                                                                           shared_ptr<const plask::MeshD<2>> dest_mesh,
                                                                           InterpolationMethod interpolation) const {
    if (what != CarriersConcentration::MAJORITY && what != CarriersConcentration::PAIRS) {
        return LazyData<double>(dest_mesh->size(), NAN);
    }
    return LazyData<double>(new Diffusion2DSolver<Geometry2DType>::ConcentrationDataImpl(this, dest_mesh, interpolation));
}

template <> std::string Diffusion2DSolver<Geometry2DCartesian>::getClassName() const { return "electrical.Diffusion2D"; }
template <> std::string Diffusion2DSolver<Geometry2DCylindrical>::getClassName() const { return "electrical.DiffusionCyl"; }

template struct PLASK_SOLVER_API Diffusion2DSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API Diffusion2DSolver<Geometry2DCylindrical>;

}}}  // namespace plask::electrical::diffusion
