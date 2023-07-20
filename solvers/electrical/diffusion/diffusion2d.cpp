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
#include "diffusion2d.hpp"

namespace plask { namespace electrical { namespace diffusion {

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

    for (size_t r = 0; r < points->axis[1]->size(); ++r) {
        size_t prev = 0;
        shared_ptr<Material> material;
        for (size_t c = 0; c < points->axis[0]->size(); ++c) {  // In the (possible) active region
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
            regions[prev - 1].rowr = regions[prev - 1].right = points->axis[0]->size();
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
                        QWheight += this->mesh->axis[1]->at(r + 1) - this->mesh->axis[1]->at(r);
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
        this->mesh = make_shared<RectangularMesh<2>>(refineAxis(mesh1->axis[0], DEFAULT_MESH_SPACING), mesh1->axis[1]);
        writelog(LOG_DETAIL, "{}: Setting up default mesh [{}]", this->getId(), this->mesh->axis[0]->size());
    }
    setActiveRegions();
    loopno = 0;
}

template <typename Geometry2DType> void Diffusion2DSolver<Geometry2DType>::onInvalidate() { active.clear(); }

// clang-format off
template <>
inline void Diffusion2DSolver<Geometry2DCartesian>::setLocalMatrix(
    const double R, const double L, const double A, const double B, const double C, const double D,
    const double* U, const double* J,
    double& K00, double& K01, double& K02, double& K03, double& K11,
    double& K12, double& K13, double& K22, double& K23, double& K33,
    double& F0, double& F1, double& F2, double& F3)
{
    K00 += (13.0/35.0)*A*L + (1.0/1260.0)*B*L*(97*L*U[1] - 43*L*U[3] + 774*U[0] + 162*U[2]) + (1.0/60060.0)*C*L*(840*L*L*U[1]*U[1] - 777*L*L*U[1]*U[3] + 268*L*L*U[3]*U[3] + 11004*L*U[0]*U[1] - 4011*L*U[0]*U[3] + 2867*L*U[1]*U[2] - 2138*L*U[2]*U[3] + 48132*U[0]*U[0] + 14418*U[0]*U[2] + 4374*U[2]*U[2]) + (6.0/5.0)*D/L;
    K01 += (11.0/210.0)*A*L*L + (1.0/1260.0)*B*L*L*(16*L*U[1] - 9*L*U[3] + 97*U[0] + 35*U[2]) + (1.0/120120.0)*C*L*L*(294*L*L*U[1]*U[1] - 322*L*L*U[1]*U[3] + 125*L*L*U[3]*U[3] + 3360*L*U[0]*U[1] - 1554*L*U[0]*U[3] + 1216*L*U[1]*U[2] - 1020*L*U[2]*U[3] + 11004*U[0]*U[0] + 5734*U[0]*U[2] + 2138*U[2]*U[2]) + (1.0/10.0)*D;
    K02 += (9.0/70.0)*A*L + (1.0/1260.0)*B*L*(35*L*U[1] - 35*L*U[3] + 162*U[0] + 162*U[2]) + (1.0/60060.0)*C*L*(304*L*L*U[1]*U[1] - 510*L*L*U[1]*U[3] + 304*L*L*U[3]*U[3] + 2867*L*U[0]*U[1] - 2138*L*U[0]*U[3] + 2138*L*U[1]*U[2] - 2867*L*U[2]*U[3] + 7209*U[0]*U[0] + 8748*U[0]*U[2] + 7209*U[2]*U[2]) - 6.0/5.0*D/L;
    K03 += -13.0/420.0*A*L*L + (1.0/1260.0)*B*L*L*(-9*L*U[1] + 8*L*U[3] - 43*U[0] - 35*U[2]) + (1.0/120120.0)*C*L*L*(-161*L*L*U[1]*U[1] + 250*L*L*U[1]*U[3] - 135*L*L*U[3]*U[3] - 1554*L*U[0]*U[1] + 1072*L*U[0]*U[3] - 1020*L*U[1]*U[2] + 1216*L*U[2]*U[3] - 4011*U[0]*U[0] - 4276*U[0]*U[2] - 2867*U[2]*U[2]) + (1.0/10.0)*D;
    K11 += (1.0/105.0)*A*L*L*L + (1.0/1260.0)*B*L*L*L*(3*L*U[1] - 2*L*U[3] + 16*U[0] + 8*U[2]) + (1.0/60060.0)*C*L*L*L*(28*L*L*U[1]*U[1] - 35*L*L*U[1]*U[3] + 15*L*L*U[3]*U[3] + 294*L*U[0]*U[1] - 161*L*U[0]*U[3] + 135*L*U[1]*U[2] - 125*L*U[2]*U[3] + 840*U[0]*U[0] + 608*U[0]*U[2] + 268*U[2]*U[2]) + (2.0/15.0)*D*L;
    K12 += (13.0/420.0)*A*L*L + (1.0/1260.0)*B*L*L*(8*L*U[1] - 9*L*U[3] + 35*U[0] + 43*U[2]) + (1.0/120120.0)*C*L*L*(135*L*L*U[1]*U[1] - 250*L*L*U[1]*U[3] + 161*L*L*U[3]*U[3] + 1216*L*U[0]*U[1] - 1020*L*U[0]*U[3] + 1072*L*U[1]*U[2] - 1554*L*U[2]*U[3] + 2867*U[0]*U[0] + 4276*U[0]*U[2] + 4011*U[2]*U[2]) - 1.0/10.0*D;
    K13 += -1.0/140.0*A*L*L*L + (1.0/1260.0)*B*L*L*L*(-2*L*U[1] + 2*L*U[3] - 9*U[0] - 9*U[2]) + (1.0/120120.0)*C*L*L*L*(-35*L*L*U[1]*U[1] + 60*L*L*U[1]*U[3] - 35*L*L*U[3]*U[3] - 322*L*U[0]*U[1] + 250*L*U[0]*U[3] - 250*L*U[1]*U[2] + 322*L*U[2]*U[3] - 777*U[0]*U[0] - 1020*U[0]*U[2] - 777*U[2]*U[2]) - 1.0/30.0*D*L;
    K22 += (13.0/35.0)*A*L + (1.0/1260.0)*B*L*(43*L*U[1] - 97*L*U[3] + 162*U[0] + 774*U[2]) + (1.0/60060.0)*C*L*(268*L*L*U[1]*U[1] - 777*L*L*U[1]*U[3] + 840*L*L*U[3]*U[3] + 2138*L*U[0]*U[1] - 2867*L*U[0]*U[3] + 4011*L*U[1]*U[2] - 11004*L*U[2]*U[3] + 4374*U[0]*U[0] + 14418*U[0]*U[2] + 48132*U[2]*U[2]) + (6.0/5.0)*D/L;
    K23 += -11.0/210.0*A*L*L + (1.0/1260.0)*B*L*L*(-9*L*U[1] + 16*L*U[3] - 35*U[0] - 97*U[2]) + (1.0/120120.0)*C*L*L*(-125*L*L*U[1]*U[1] + 322*L*L*U[1]*U[3] - 294*L*L*U[3]*U[3] - 1020*L*U[0]*U[1] + 1216*L*U[0]*U[3] - 1554*L*U[1]*U[2] + 3360*L*U[2]*U[3] - 2138*U[0]*U[0] - 5734*U[0]*U[2] - 11004*U[2]*U[2]) - 1.0/10.0*D;
    K33 += (1.0/105.0)*A*L*L*L + (1.0/1260.0)*B*L*L*L*(2*L*U[1] - 3*L*U[3] + 8*U[0] + 16*U[2]) + (1.0/60060.0)*C*L*L*L*(15*L*L*U[1]*U[1] - 35*L*L*U[1]*U[3] + 28*L*L*U[3]*U[3] + 125*L*U[0]*U[1] - 135*L*U[0]*U[3] + 161*L*U[1]*U[2] - 294*L*U[2]*U[3] + 268*U[0]*U[0] + 608*U[0]*U[2] + 840*U[2]*U[2]) + (2.0/15.0)*D*L;
    F0 += (1.0/1260.0)*B*L*(8*L*L*U[1]*U[1] - 9*L*L*U[1]*U[3] + 4*L*L*U[3]*U[3] + 97*L*U[0]*U[1] - 43*L*U[0]*U[3] + 35*L*U[1]*U[2] - 35*L*U[2]*U[3] + 387*U[0]*U[0] + 162*U[0]*U[2] + 81*U[2]*U[2]) + (1.0/60060.0)*C*L*(98*L*L*L*U[1]*U[1]*U[1] - 161*L*L*L*U[1]*U[1]*U[3] + 125*L*L*L*U[1]*U[3]*U[3] - 45*L*L*L*U[3]*U[3]*U[3] + 1680*L*L*U[0]*U[1]*U[1] - 1554*L*L*U[0]*U[1]*U[3] + 536*L*L*U[0]*U[3]*U[3] + 608*L*L*U[1]*U[1]*U[2] - 1020*L*L*U[1]*U[2]*U[3] + 608*L*L*U[2]*U[3]*U[3] + 11004*L*U[0]*U[0]*U[1] - 4011*L*U[0]*U[0]*U[3] + 5734*L*U[0]*U[1]*U[2] - 4276*L*U[0]*U[2]*U[3] + 2138*L*U[1]*U[2]*U[2] - 2867*L*U[2]*U[2]*U[3] + 32088*U[0]*U[0]*U[0] + 14418*U[0]*U[0]*U[2] + 8748*U[0]*U[2]*U[2] + 4806*U[2]*U[2]*U[2]) + (1.0/20.0)*L*(7*J[0] + 3*J[1]);
    F1 += (1.0/2520.0)*B*L*L*(3*L*L*U[1]*U[1] - 4*L*L*U[1]*U[3] + 2*L*L*U[3]*U[3] + 32*L*U[0]*U[1] - 18*L*U[0]*U[3] + 16*L*U[1]*U[2] - 18*L*U[2]*U[3] + 97*U[0]*U[0] + 70*U[0]*U[2] + 43*U[2]*U[2]) + (1.0/180180.0)*C*L*L*(56*L*L*L*U[1]*U[1]*U[1] - 105*L*L*L*U[1]*U[1]*U[3] + 90*L*L*L*U[1]*U[3]*U[3] - 35*L*L*L*U[3]*U[3]*U[3] + 882*L*L*U[0]*U[1]*U[1] - 966*L*L*U[0]*U[1]*U[3] + 375*L*L*U[0]*U[3]*U[3] + 405*L*L*U[1]*U[1]*U[2] - 750*L*L*U[1]*U[2]*U[3] + 483*L*L*U[2]*U[3]*U[3] + 5040*L*U[0]*U[0]*U[1] - 2331*L*U[0]*U[0]*U[3] + 3648*L*U[0]*U[1]*U[2] - 3060*L*U[0]*U[2]*U[3] + 1608*L*U[1]*U[2]*U[2] - 2331*L*U[2]*U[2]*U[3] + 11004*U[0]*U[0]*U[0] + 8601*U[0]*U[0]*U[2] + 6414*U[0]*U[2]*U[2] + 4011*U[2]*U[2]*U[2]) + L*L*((1.0/20.0)*J[0] + (1.0/30.0)*J[1]);
    F2 += (1.0/1260.0)*B*L*(4*L*L*U[1]*U[1] - 9*L*L*U[1]*U[3] + 8*L*L*U[3]*U[3] + 35*L*U[0]*U[1] - 35*L*U[0]*U[3] + 43*L*U[1]*U[2] - 97*L*U[2]*U[3] + 81*U[0]*U[0] + 162*U[0]*U[2] + 387*U[2]*U[2]) + (1.0/60060.0)*C*L*(45*L*L*L*U[1]*U[1]*U[1] - 125*L*L*L*U[1]*U[1]*U[3] + 161*L*L*L*U[1]*U[3]*U[3] - 98*L*L*L*U[3]*U[3]*U[3] + 608*L*L*U[0]*U[1]*U[1] - 1020*L*L*U[0]*U[1]*U[3] + 608*L*L*U[0]*U[3]*U[3] + 536*L*L*U[1]*U[1]*U[2] - 1554*L*L*U[1]*U[2]*U[3] + 1680*L*L*U[2]*U[3]*U[3] + 2867*L*U[0]*U[0]*U[1] - 2138*L*U[0]*U[0]*U[3] + 4276*L*U[0]*U[1]*U[2] - 5734*L*U[0]*U[2]*U[3] + 4011*L*U[1]*U[2]*U[2] - 11004*L*U[2]*U[2]*U[3] + 4806*U[0]*U[0]*U[0] + 8748*U[0]*U[0]*U[2] + 14418*U[0]*U[2]*U[2] + 32088*U[2]*U[2]*U[2]) + (1.0/20.0)*L*(3*J[0] + 7*J[1]);
    F3 += (1.0/2520.0)*B*L*L*(-2*L*L*U[1]*U[1] + 4*L*L*U[1]*U[3] - 3*L*L*U[3]*U[3] - 18*L*U[0]*U[1] + 16*L*U[0]*U[3] - 18*L*U[1]*U[2] + 32*L*U[2]*U[3] - 43*U[0]*U[0] - 70*U[0]*U[2] - 97*U[2]*U[2]) + (1.0/180180.0)*C*L*L*(-35*L*L*L*U[1]*U[1]*U[1] + 90*L*L*L*U[1]*U[1]*U[3] - 105*L*L*L*U[1]*U[3]*U[3] + 56*L*L*L*U[3]*U[3]*U[3] - 483*L*L*U[0]*U[1]*U[1] + 750*L*L*U[0]*U[1]*U[3] - 405*L*L*U[0]*U[3]*U[3] - 375*L*L*U[1]*U[1]*U[2] + 966*L*L*U[1]*U[2]*U[3] - 882*L*L*U[2]*U[3]*U[3] - 2331*L*U[0]*U[0]*U[1] + 1608*L*U[0]*U[0]*U[3] - 3060*L*U[0]*U[1]*U[2] + 3648*L*U[0]*U[2]*U[3] - 2331*L*U[1]*U[2]*U[2] + 5040*L*U[2]*U[2]*U[3] - 4011*U[0]*U[0]*U[0] - 6414*U[0]*U[0]*U[2] - 8601*U[0]*U[2]*U[2] - 11004*U[2]*U[2]*U[2]) + L*L*(-1.0/30.0*J[0] - 1.0/20.0*J[1]);
}

template <>
inline void Diffusion2DSolver<Geometry2DCylindrical>::setLocalMatrix(
    const double R, const double L, const double A, const double B, const double C, const double D,
    const double* U, const double* J,
    double& K00, double& K01, double& K02, double& K03, double& K11,
    double& K12, double& K13, double& K22, double& K23, double& K33,
    double& F0, double& F1, double& F2, double& F3)
{
    K00 += (1.0/360360.0)*(432432*D*R + 2*L*L*R*(66924*A + 13871*B*L*U[1] - 6149*B*L*U[3] + 110682*B*U[0] + 23166*B*U[2] + 2520*C*L*L*U[1]*U[1] - 2331*C*L*L*U[1]*U[3] + 804*C*L*L*U[3]*U[3] + 33012*C*L*U[0]*U[1] - 12033*C*L*U[0]*U[3] + 8601*C*L*U[1]*U[2] - 6414*C*L*U[2]*U[3] + 144396*C*U[0]*U[0] + 43254*C*U[0]*U[2] + 13122*C*U[2]*U[2]) + L*(30888*A*L*L + 7540*B*L*L*L*U[1] - 4758*B*L*L*L*U[3] + 42822*B*L*L*U[0] + 18954*B*L*L*U[2] + 1458*C*std::pow(L,4)*U[1]*U[1] - 1746*C*std::pow(L,4)*U[1]*U[3] + 735*C*std::pow(L,4)*U[3]*U[3] + 15912*C*L*L*L*U[0]*U[1] - 8154*C*L*L*L*U[0]*U[3] + 6708*C*L*L*L*U[1]*U[2] - 6120*C*L*L*L*U[2]*U[3] + 48924*C*L*L*U[0]*U[0] + 30618*C*L*L*U[0]*U[2] + 13122*C*L*L*U[2]*U[2] + 216216*D))/L;
    K01 += (1.0/60.0)*A*L*L*L + (19.0/4620.0)*B*std::pow(L,4)*U[1] - 1.0/330.0*B*std::pow(L,4)*U[3] + (29.0/1386.0)*B*L*L*L*U[0] + (43.0/3465.0)*B*L*L*L*U[2] + (4.0/5005.0)*C*std::pow(L,5)*U[1]*U[1] - 1.0/924.0*C*std::pow(L,5)*U[1]*U[3] + (1.0/2002.0)*C*std::pow(L,5)*U[3]*U[3] + (81.0/10010.0)*C*std::pow(L,4)*U[0]*U[1] - 97.0/20020.0*C*std::pow(L,4)*U[0]*U[3] + (17.0/4004.0)*C*std::pow(L,4)*U[1]*U[2] - 17.0/4004.0*C*std::pow(L,4)*U[2]*U[3] + (17.0/770.0)*C*L*L*L*U[0]*U[0] + (43.0/2310.0)*C*L*L*L*U[0]*U[2] + (43.0/4620.0)*C*L*L*L*U[2]*U[2] + (1.0/10.0)*D*L + (1.0/360360.0)*R*(18876*A*L*L + 4576*B*L*L*L*U[1] - 2574*B*L*L*L*U[3] + 27742*B*L*L*U[0] + 10010*B*L*L*U[2] + 882*C*std::pow(L,4)*U[1]*U[1] - 966*C*std::pow(L,4)*U[1]*U[3] + 375*C*std::pow(L,4)*U[3]*U[3] + 10080*C*L*L*L*U[0]*U[1] - 4662*C*L*L*L*U[0]*U[3] + 3648*C*L*L*L*U[1]*U[2] - 3060*C*L*L*L*U[2]*U[3] + 33012*C*L*L*U[0]*U[0] + 17202*C*L*L*U[0]*U[2] + 6414*C*L*L*U[2]*U[2] + 36036*D);
    K02 += (1.0/360360.0)*(-432432*D*R + 2*L*L*R*(23166*A + 5005*B*L*U[1] - 5005*B*L*U[3] + 23166*B*U[0] + 23166*B*U[2] + 912*C*L*L*U[1]*U[1] - 1530*C*L*L*U[1]*U[3] + 912*C*L*L*U[3]*U[3] + 8601*C*L*U[0]*U[1] - 6414*C*L*U[0]*U[3] + 6414*C*L*U[1]*U[2] - 8601*C*L*U[2]*U[3] + 21627*C*U[0]*U[0] + 26244*C*U[0]*U[2] + 21627*C*U[2]*U[2]) + L*(23166*A*L*L + 4472*B*L*L*L*U[1] - 5538*B*L*L*L*U[3] + 18954*B*L*L*U[0] + 27378*B*L*L*U[2] + 765*C*std::pow(L,4)*U[1]*U[1] - 1530*C*std::pow(L,4)*U[1]*U[3] + 1059*C*std::pow(L,4)*U[3]*U[3] + 6708*C*L*L*L*U[0]*U[1] - 6120*C*L*L*L*U[0]*U[3] + 6708*C*L*L*L*U[1]*U[2] - 10494*C*L*L*L*U[2]*U[3] + 15309*C*L*L*U[0]*U[0] + 26244*C*L*L*U[0]*U[2] + 27945*C*L*L*U[2]*U[2] - 216216*D))/L;
    K03 += -1.0/70.0*A*L*L*L - 1.0/330.0*B*std::pow(L,4)*U[1] + (23.0/6930.0)*B*std::pow(L,4)*U[3] - 61.0/4620.0*B*L*L*L*U[0] - 71.0/4620.0*B*L*L*L*U[2] - 1.0/1848.0*C*std::pow(L,5)*U[1]*U[1] + (1.0/1001.0)*C*std::pow(L,5)*U[1]*U[3] - 5.0/8008.0*C*std::pow(L,5)*U[3]*U[3] - 97.0/20020.0*C*std::pow(L,4)*U[0]*U[1] + (7.0/1716.0)*C*std::pow(L,4)*U[0]*U[3] - 17.0/4004.0*C*std::pow(L,4)*U[1]*U[2] + (353.0/60060.0)*C*std::pow(L,4)*U[2]*U[3] - 453.0/40040.0*C*L*L*L*U[0]*U[0] - 17.0/1001.0*C*L*L*L*U[0]*U[2] - 53.0/3640.0*C*L*L*L*U[2]*U[2] + (1.0/360360.0)*R*(-11154*A*L*L - 2574*B*L*L*L*U[1] + 2288*B*L*L*L*U[3] - 12298*B*L*L*U[0] - 10010*B*L*L*U[2] - 483*C*std::pow(L,4)*U[1]*U[1] + 750*C*std::pow(L,4)*U[1]*U[3] - 405*C*std::pow(L,4)*U[3]*U[3] - 4662*C*L*L*L*U[0]*U[1] + 3216*C*L*L*L*U[0]*U[3] - 3060*C*L*L*L*U[1]*U[2] + 3648*C*L*L*L*U[2]*U[3] - 12033*C*L*L*U[0]*U[0] - 12828*C*L*L*U[0]*U[2] - 8601*C*L*L*U[2]*U[2] + 36036*D);
    K11 += (1.0/360360.0)*L*(1287*A*L*L*L + 312*B*std::pow(L,4)*U[1] - 260*B*std::pow(L,4)*U[3] + 1482*B*L*L*L*U[0] + 1092*B*L*L*L*U[2] + 60*C*std::pow(L,5)*U[1]*U[1] - 90*C*std::pow(L,5)*U[1]*U[3] + 45*C*std::pow(L,5)*U[3]*U[3] + 576*C*std::pow(L,4)*U[0]*U[1] - 390*C*std::pow(L,4)*U[0]*U[3] + 360*C*std::pow(L,4)*U[1]*U[2] - 390*C*std::pow(L,4)*U[2]*U[3] + 1458*C*L*L*L*U[0]*U[0] + 1530*C*L*L*L*U[0]*U[2] + 873*C*L*L*L*U[2]*U[2] + 12012*D*L + 2*R*(1716*A*L*L + 429*B*L*L*L*U[1] - 286*B*L*L*L*U[3] + 2288*B*L*L*U[0] + 1144*B*L*L*U[2] + 84*C*std::pow(L,4)*U[1]*U[1] - 105*C*std::pow(L,4)*U[1]*U[3] + 45*C*std::pow(L,4)*U[3]*U[3] + 882*C*L*L*L*U[0]*U[1] - 483*C*L*L*L*U[0]*U[3] + 405*C*L*L*L*U[1]*U[2] - 375*C*L*L*L*U[2]*U[3] + 2520*C*L*L*U[0]*U[0] + 1824*C*L*L*U[0]*U[2] + 804*C*L*L*U[2]*U[2] + 24024*D));
    K12 += (1.0/60.0)*A*L*L*L + (1.0/330.0)*B*std::pow(L,4)*U[1] - 19.0/4620.0*B*std::pow(L,4)*U[3] + (43.0/3465.0)*B*L*L*L*U[0] + (29.0/1386.0)*B*L*L*L*U[2] + (1.0/2002.0)*C*std::pow(L,5)*U[1]*U[1] - 1.0/924.0*C*std::pow(L,5)*U[1]*U[3] + (4.0/5005.0)*C*std::pow(L,5)*U[3]*U[3] + (17.0/4004.0)*C*std::pow(L,4)*U[0]*U[1] - 17.0/4004.0*C*std::pow(L,4)*U[0]*U[3] + (97.0/20020.0)*C*std::pow(L,4)*U[1]*U[2] - 81.0/10010.0*C*std::pow(L,4)*U[2]*U[3] + (43.0/4620.0)*C*L*L*L*U[0]*U[0] + (43.0/2310.0)*C*L*L*L*U[0]*U[2] + (17.0/770.0)*C*L*L*L*U[2]*U[2] - 1.0/10.0*D*L + (1.0/360360.0)*R*(11154*A*L*L + 2288*B*L*L*L*U[1] - 2574*B*L*L*L*U[3] + 10010*B*L*L*U[0] + 12298*B*L*L*U[2] + 405*C*std::pow(L,4)*U[1]*U[1] - 750*C*std::pow(L,4)*U[1]*U[3] + 483*C*std::pow(L,4)*U[3]*U[3] + 3648*C*L*L*L*U[0]*U[1] - 3060*C*L*L*L*U[0]*U[3] + 3216*C*L*L*L*U[1]*U[2] - 4662*C*L*L*L*U[2]*U[3] + 8601*C*L*L*U[0]*U[0] + 12828*C*L*L*U[0]*U[2] + 12033*C*L*L*U[2]*U[2] - 36036*D);
    K13 += (1.0/360360.0)*L*(-1287*A*L*L*L - 260*B*std::pow(L,4)*U[1] + 312*B*std::pow(L,4)*U[3] - 1092*B*L*L*L*U[0] - 1482*B*L*L*L*U[2] - 45*C*std::pow(L,5)*U[1]*U[1] + 90*C*std::pow(L,5)*U[1]*U[3] - 60*C*std::pow(L,5)*U[3]*U[3] - 390*C*std::pow(L,4)*U[0]*U[1] + 360*C*std::pow(L,4)*U[0]*U[3] - 390*C*std::pow(L,4)*U[1]*U[2] + 576*C*std::pow(L,4)*U[2]*U[3] - 873*C*L*L*L*U[0]*U[0] - 1530*C*L*L*L*U[0]*U[2] - 1458*C*L*L*L*U[2]*U[2] - 6006*D*L + R*(-2574*A*L*L - 572*B*L*L*L*U[1] + 572*B*L*L*L*U[3] - 2574*B*L*L*U[0] - 2574*B*L*L*U[2] - 105*C*std::pow(L,4)*U[1]*U[1] + 180*C*std::pow(L,4)*U[1]*U[3] - 105*C*std::pow(L,4)*U[3]*U[3] - 966*C*L*L*L*U[0]*U[1] + 750*C*L*L*L*U[0]*U[3] - 750*C*L*L*L*U[1]*U[2] + 966*C*L*L*L*U[2]*U[3] - 2331*C*L*L*U[0]*U[0] - 3060*C*L*L*U[0]*U[2] - 2331*C*L*L*U[2]*U[2] - 12012*D));
    K22 += (1.0/360360.0)*(432432*D*R + 2*L*L*R*(66924*A + 6149*B*L*U[1] - 13871*B*L*U[3] + 23166*B*U[0] + 110682*B*U[2] + 804*C*L*L*U[1]*U[1] - 2331*C*L*L*U[1]*U[3] + 2520*C*L*L*U[3]*U[3] + 6414*C*L*U[0]*U[1] - 8601*C*L*U[0]*U[3] + 12033*C*L*U[1]*U[2] - 33012*C*L*U[2]*U[3] + 13122*C*U[0]*U[0] + 43254*C*U[0]*U[2] + 144396*C*U[2]*U[2]) + L*(102960*A*L*L + 7540*B*L*L*L*U[1] - 20202*B*L*L*L*U[3] + 27378*B*L*L*U[0] + 178542*B*L*L*U[2] + 873*C*std::pow(L,4)*U[1]*U[1] - 2916*C*std::pow(L,4)*U[1]*U[3] + 3582*C*std::pow(L,4)*U[3]*U[3] + 6708*C*L*L*L*U[0]*U[1] - 10494*C*L*L*L*U[0]*U[3] + 15912*C*L*L*L*U[1]*U[2] - 50112*C*L*L*L*U[2]*U[3] + 13122*C*L*L*U[0]*U[0] + 55890*C*L*L*U[0]*U[2] + 239868*C*L*L*U[2]*U[2] + 216216*D))/L;
    K23 += -1.0/28.0*A*L*L*L - 19.0/4620.0*B*std::pow(L,4)*U[1] + (17.0/1980.0)*B*std::pow(L,4)*U[3] - 71.0/4620.0*B*L*L*L*U[0] - 37.0/660.0*B*L*L*L*U[2] - 1.0/1848.0*C*std::pow(L,5)*U[1]*U[1] + (8.0/5005.0)*C*std::pow(L,5)*U[1]*U[3] - 3.0/1820.0*C*std::pow(L,5)*U[3]*U[3] - 17.0/4004.0*C*std::pow(L,4)*U[0]*U[1] + (353.0/60060.0)*C*std::pow(L,4)*U[0]*U[3] - 81.0/10010.0*C*std::pow(L,4)*U[1]*U[2] + (199.0/10010.0)*C*std::pow(L,4)*U[2]*U[3] - 17.0/2002.0*C*L*L*L*U[0]*U[0] - 53.0/1820.0*C*L*L*L*U[0]*U[2] - 348.0/5005.0*C*L*L*L*U[2]*U[2] + (1.0/360360.0)*R*(-18876*A*L*L - 2574*B*L*L*L*U[1] + 4576*B*L*L*L*U[3] - 10010*B*L*L*U[0] - 27742*B*L*L*U[2] - 375*C*std::pow(L,4)*U[1]*U[1] + 966*C*std::pow(L,4)*U[1]*U[3] - 882*C*std::pow(L,4)*U[3]*U[3] - 3060*C*L*L*L*U[0]*U[1] + 3648*C*L*L*L*U[0]*U[3] - 4662*C*L*L*L*U[1]*U[2] + 10080*C*L*L*L*U[2]*U[3] - 6414*C*L*L*U[0]*U[0] - 17202*C*L*L*U[0]*U[2] - 33012*C*L*L*U[2]*U[2] - 36036*D);
    K33 += (1.0/360360.0)*L*(2145*A*L*L*L + 312*B*std::pow(L,4)*U[1] - 546*B*std::pow(L,4)*U[3] + 1196*B*L*L*L*U[0] + 3094*B*L*L*L*U[2] + 45*C*std::pow(L,5)*U[1]*U[1] - 120*C*std::pow(L,5)*U[1]*U[3] + 108*C*std::pow(L,5)*U[3]*U[3] + 360*C*std::pow(L,4)*U[0]*U[1] - 450*C*std::pow(L,4)*U[0]*U[3] + 576*C*std::pow(L,4)*U[1]*U[2] - 1188*C*std::pow(L,4)*U[2]*U[3] + 735*C*L*L*L*U[0]*U[0] + 2118*C*L*L*L*U[0]*U[2] + 3582*C*L*L*L*U[2]*U[2] + 36036*D*L + 2*R*(1716*A*L*L + 286*B*L*L*L*U[1] - 429*B*L*L*L*U[3] + 1144*B*L*L*U[0] + 2288*B*L*L*U[2] + 45*C*std::pow(L,4)*U[1]*U[1] - 105*C*std::pow(L,4)*U[1]*U[3] + 84*C*std::pow(L,4)*U[3]*U[3] + 375*C*L*L*L*U[0]*U[1] - 405*C*L*L*L*U[0]*U[3] + 483*C*L*L*L*U[1]*U[2] - 882*C*L*L*L*U[2]*U[3] + 804*C*L*L*U[0]*U[0] + 1824*C*L*L*U[0]*U[2] + 2520*C*L*L*U[2]*U[2] + 24024*D));
    F0 += (1.0/360360.0)*L*(741*B*L*L*L*U[1]*U[1] - 1092*B*L*L*L*U[1]*U[3] + 598*B*L*L*L*U[3]*U[3] + 7540*B*L*L*U[0]*U[1] - 4758*B*L*L*U[0]*U[3] + 4472*B*L*L*U[1]*U[2] - 5538*B*L*L*U[2]*U[3] + 21411*B*L*U[0]*U[0] + 18954*B*L*U[0]*U[2] + 13689*B*L*U[2]*U[2] + 192*C*std::pow(L,4)*U[1]*U[1]*U[1] - 390*C*std::pow(L,4)*U[1]*U[1]*U[3] + 360*C*std::pow(L,4)*U[1]*U[3]*U[3] - 150*C*std::pow(L,4)*U[3]*U[3]*U[3] + 2916*C*L*L*L*U[0]*U[1]*U[1] - 3492*C*L*L*L*U[0]*U[1]*U[3] + 1470*C*L*L*L*U[0]*U[3]*U[3] + 1530*C*L*L*L*U[1]*U[1]*U[2] - 3060*C*L*L*L*U[1]*U[2]*U[3] + 2118*C*L*L*L*U[2]*U[3]*U[3] + 15912*C*L*L*U[0]*U[0]*U[1] - 8154*C*L*L*U[0]*U[0]*U[3] + 13416*C*L*L*U[0]*U[1]*U[2] - 12240*C*L*L*U[0]*U[2]*U[3] + 6708*C*L*L*U[1]*U[2]*U[2] - 10494*C*L*L*U[2]*U[2]*U[3] + 32616*C*L*U[0]*U[0]*U[0] + 30618*C*L*U[0]*U[0]*U[2] + 26244*C*L*U[0]*U[2]*U[2] + 18630*C*L*U[2]*U[2]*U[2] + 30030*L*J[0] + 24024*L*J[1] + 2*R*(1144*B*L*L*U[1]*U[1] - 1287*B*L*L*U[1]*U[3] + 572*B*L*L*U[3]*U[3] + 13871*B*L*U[0]*U[1] - 6149*B*L*U[0]*U[3] + 5005*B*L*U[1]*U[2] - 5005*B*L*U[2]*U[3] + 55341*B*U[0]*U[0] + 23166*B*U[0]*U[2] + 11583*B*U[2]*U[2] + 294*C*L*L*L*U[1]*U[1]*U[1] - 483*C*L*L*L*U[1]*U[1]*U[3] + 375*C*L*L*L*U[1]*U[3]*U[3] - 135*C*L*L*L*U[3]*U[3]*U[3] + 5040*C*L*L*U[0]*U[1]*U[1] - 4662*C*L*L*U[0]*U[1]*U[3] + 1608*C*L*L*U[0]*U[3]*U[3] + 1824*C*L*L*U[1]*U[1]*U[2] - 3060*C*L*L*U[1]*U[2]*U[3] + 1824*C*L*L*U[2]*U[3]*U[3] + 33012*C*L*U[0]*U[0]*U[1] - 12033*C*L*U[0]*U[0]*U[3] + 17202*C*L*U[0]*U[1]*U[2] - 12828*C*L*U[0]*U[2]*U[3] + 6414*C*L*U[1]*U[2]*U[2] - 8601*C*L*U[2]*U[2]*U[3] + 96264*C*U[0]*U[0]*U[0] + 43254*C*U[0]*U[0]*U[2] + 26244*C*U[0]*U[2]*U[2] + 14418*C*U[2]*U[2]*U[2] + 63063*J[0] + 27027*J[1]));
    F1 += (1.0/360360.0)*L*L*(156*B*L*L*L*U[1]*U[1] - 260*B*L*L*L*U[1]*U[3] + 156*B*L*L*L*U[3]*U[3] + 1482*B*L*L*U[0]*U[1] - 1092*B*L*L*U[0]*U[3] + 1092*B*L*L*U[1]*U[2] - 1482*B*L*L*U[2]*U[3] + 3770*B*L*U[0]*U[0] + 4472*B*L*U[0]*U[2] + 3770*B*L*U[2]*U[2] + 40*C*std::pow(L,4)*U[1]*U[1]*U[1] - 90*C*std::pow(L,4)*U[1]*U[1]*U[3] + 90*C*std::pow(L,4)*U[1]*U[3]*U[3] - 40*C*std::pow(L,4)*U[3]*U[3]*U[3] + 576*C*L*L*L*U[0]*U[1]*U[1] - 780*C*L*L*L*U[0]*U[1]*U[3] + 360*C*L*L*L*U[0]*U[3]*U[3] + 360*C*L*L*L*U[1]*U[1]*U[2] - 780*C*L*L*L*U[1]*U[2]*U[3] + 576*C*L*L*L*U[2]*U[3]*U[3] + 2916*C*L*L*U[0]*U[0]*U[1] - 1746*C*L*L*U[0]*U[0]*U[3] + 3060*C*L*L*U[0]*U[1]*U[2] - 3060*C*L*L*U[0]*U[2]*U[3] + 1746*C*L*L*U[1]*U[2]*U[2] - 2916*C*L*L*U[2]*U[2]*U[3] + 5304*C*L*U[0]*U[0]*U[0] + 6708*C*L*U[0]*U[0]*U[2] + 6708*C*L*U[0]*U[2]*U[2] + 5304*C*L*U[2]*U[2]*U[2] + 6006*L*J[0] + 6006*L*J[1] + R*(429*B*L*L*U[1]*U[1] - 572*B*L*L*U[1]*U[3] + 286*B*L*L*U[3]*U[3] + 4576*B*L*U[0]*U[1] - 2574*B*L*U[0]*U[3] + 2288*B*L*U[1]*U[2] - 2574*B*L*U[2]*U[3] + 13871*B*U[0]*U[0] + 10010*B*U[0]*U[2] + 6149*B*U[2]*U[2] + 112*C*L*L*L*U[1]*U[1]*U[1] - 210*C*L*L*L*U[1]*U[1]*U[3] + 180*C*L*L*L*U[1]*U[3]*U[3] - 70*C*L*L*L*U[3]*U[3]*U[3] + 1764*C*L*L*U[0]*U[1]*U[1] - 1932*C*L*L*U[0]*U[1]*U[3] + 750*C*L*L*U[0]*U[3]*U[3] + 810*C*L*L*U[1]*U[1]*U[2] - 1500*C*L*L*U[1]*U[2]*U[3] + 966*C*L*L*U[2]*U[3]*U[3] + 10080*C*L*U[0]*U[0]*U[1] - 4662*C*L*U[0]*U[0]*U[3] + 7296*C*L*U[0]*U[1]*U[2] - 6120*C*L*U[0]*U[2]*U[3] + 3216*C*L*U[1]*U[2]*U[2] - 4662*C*L*U[2]*U[2]*U[3] + 22008*C*U[0]*U[0]*U[0] + 17202*C*U[0]*U[0]*U[2] + 12828*C*U[0]*U[2]*U[2] + 8022*C*U[2]*U[2]*U[2] + 18018*J[0] + 12012*J[1]));
    F2 += (1.0/360360.0)*L*(546*B*L*L*L*U[1]*U[1] - 1482*B*L*L*L*U[1]*U[3] + 1547*B*L*L*L*U[3]*U[3] + 4472*B*L*L*U[0]*U[1] - 5538*B*L*L*U[0]*U[3] + 7540*B*L*L*U[1]*U[2] - 20202*B*L*L*U[2]*U[3] + 9477*B*L*U[0]*U[0] + 27378*B*L*U[0]*U[2] + 89271*B*L*U[2]*U[2] + 120*C*std::pow(L,4)*U[1]*U[1]*U[1] - 390*C*std::pow(L,4)*U[1]*U[1]*U[3] + 576*C*std::pow(L,4)*U[1]*U[3]*U[3] - 396*C*std::pow(L,4)*U[3]*U[3]*U[3] + 1530*C*L*L*L*U[0]*U[1]*U[1] - 3060*C*L*L*L*U[0]*U[1]*U[3] + 2118*C*L*L*L*U[0]*U[3]*U[3] + 1746*C*L*L*L*U[1]*U[1]*U[2] - 5832*C*L*L*L*U[1]*U[2]*U[3] + 7164*C*L*L*L*U[2]*U[3]*U[3] + 6708*C*L*L*U[0]*U[0]*U[1] - 6120*C*L*L*U[0]*U[0]*U[3] + 13416*C*L*L*U[0]*U[1]*U[2] - 20988*C*L*L*U[0]*U[2]*U[3] + 15912*C*L*L*U[1]*U[2]*U[2] - 50112*C*L*L*U[2]*U[2]*U[3] + 10206*C*L*U[0]*U[0]*U[0] + 26244*C*L*U[0]*U[0]*U[2] + 55890*C*L*U[0]*U[2]*U[2] + 159912*C*L*U[2]*U[2]*U[2] + 30030*L*J[0] + 96096*L*J[1] + 2*R*(572*B*L*L*U[1]*U[1] - 1287*B*L*L*U[1]*U[3] + 1144*B*L*L*U[3]*U[3] + 5005*B*L*U[0]*U[1] - 5005*B*L*U[0]*U[3] + 6149*B*L*U[1]*U[2] - 13871*B*L*U[2]*U[3] + 11583*B*U[0]*U[0] + 23166*B*U[0]*U[2] + 55341*B*U[2]*U[2] + 135*C*L*L*L*U[1]*U[1]*U[1] - 375*C*L*L*L*U[1]*U[1]*U[3] + 483*C*L*L*L*U[1]*U[3]*U[3] - 294*C*L*L*L*U[3]*U[3]*U[3] + 1824*C*L*L*U[0]*U[1]*U[1] - 3060*C*L*L*U[0]*U[1]*U[3] + 1824*C*L*L*U[0]*U[3]*U[3] + 1608*C*L*L*U[1]*U[1]*U[2] - 4662*C*L*L*U[1]*U[2]*U[3] + 5040*C*L*L*U[2]*U[3]*U[3] + 8601*C*L*U[0]*U[0]*U[1] - 6414*C*L*U[0]*U[0]*U[3] + 12828*C*L*U[0]*U[1]*U[2] - 17202*C*L*U[0]*U[2]*U[3] + 12033*C*L*U[1]*U[2]*U[2] - 33012*C*L*U[2]*U[2]*U[3] + 14418*C*U[0]*U[0]*U[0] + 26244*C*U[0]*U[0]*U[2] + 43254*C*U[0]*U[2]*U[2] + 96264*C*U[2]*U[2]*U[2] + 27027*J[0] + 63063*J[1]));
    F3 += (1.0/360360.0)*L*L*(-130*B*L*L*L*U[1]*U[1] + 312*B*L*L*L*U[1]*U[3] - 273*B*L*L*L*U[3]*U[3] - 1092*B*L*L*U[0]*U[1] + 1196*B*L*L*U[0]*U[3] - 1482*B*L*L*U[1]*U[2] + 3094*B*L*L*U[2]*U[3] - 2379*B*L*U[0]*U[0] - 5538*B*L*U[0]*U[2] - 10101*B*L*U[2]*U[2] - 30*C*std::pow(L,4)*U[1]*U[1]*U[1] + 90*C*std::pow(L,4)*U[1]*U[1]*U[3] - 120*C*std::pow(L,4)*U[1]*U[3]*U[3] + 72*C*std::pow(L,4)*U[3]*U[3]*U[3] - 390*C*L*L*L*U[0]*U[1]*U[1] + 720*C*L*L*L*U[0]*U[1]*U[3] - 450*C*L*L*L*U[0]*U[3]*U[3] - 390*C*L*L*L*U[1]*U[1]*U[2] + 1152*C*L*L*L*U[1]*U[2]*U[3] - 1188*C*L*L*L*U[2]*U[3]*U[3] - 1746*C*L*L*U[0]*U[0]*U[1] + 1470*C*L*L*U[0]*U[0]*U[3] - 3060*C*L*L*U[0]*U[1]*U[2] + 4236*C*L*L*U[0]*U[2]*U[3] - 2916*C*L*L*U[1]*U[2]*U[2] + 7164*C*L*L*U[2]*U[2]*U[3] - 2718*C*L*U[0]*U[0]*U[0] - 6120*C*L*U[0]*U[0]*U[2] - 10494*C*L*U[0]*U[2]*U[2] - 16704*C*L*U[2]*U[2]*U[2] - 6006*L*J[0] - 12012*L*J[1] + R*(-286*B*L*L*U[1]*U[1] + 572*B*L*L*U[1]*U[3] - 429*B*L*L*U[3]*U[3] - 2574*B*L*U[0]*U[1] + 2288*B*L*U[0]*U[3] - 2574*B*L*U[1]*U[2] + 4576*B*L*U[2]*U[3] - 6149*B*U[0]*U[0] - 10010*B*U[0]*U[2] - 13871*B*U[2]*U[2] - 70*C*L*L*L*U[1]*U[1]*U[1] + 180*C*L*L*L*U[1]*U[1]*U[3] - 210*C*L*L*L*U[1]*U[3]*U[3] + 112*C*L*L*L*U[3]*U[3]*U[3] - 966*C*L*L*U[0]*U[1]*U[1] + 1500*C*L*L*U[0]*U[1]*U[3] - 810*C*L*L*U[0]*U[3]*U[3] - 750*C*L*L*U[1]*U[1]*U[2] + 1932*C*L*L*U[1]*U[2]*U[3] - 1764*C*L*L*U[2]*U[3]*U[3] - 4662*C*L*U[0]*U[0]*U[1] + 3216*C*L*U[0]*U[0]*U[3] - 6120*C*L*U[0]*U[1]*U[2] + 7296*C*L*U[0]*U[2]*U[3] - 4662*C*L*U[1]*U[2]*U[2] + 10080*C*L*U[2]*U[2]*U[3] - 8022*C*U[0]*U[0]*U[0] - 12828*C*U[0]*U[0]*U[2] - 17202*C*U[0]*U[2]*U[2] - 22008*C*U[2]*U[2]*U[2] - 12012*J[0] - 18018*J[1]));
}
// clang-format on

/// Set stiffness matrix + load vector
template <typename Geometry2DType>
void Diffusion2DSolver<Geometry2DType>::setMatrix(FemMatrix& K,
                                                  DataVector<double>& F,
                                                  const DataVector<double>& U0,
                                                  const shared_ptr<OrderedAxis> mesh,
                                                  double z,
                                                  const LazyData<double>& temp,
                                                  const DataVector<double>& J) {
    this->writelog(LOG_DETAIL, "Setting up matrix system (size={})", K.size);

    K.clear();
    F.fill(0.);

    // Set stiffness matrix and load vector
    for (size_t ie = 0, end = mesh->size() - 1; ie < end; ++ie) {
        double x0 = mesh->at(ie), x1 = mesh->at(ie + 1);
        size_t i = 2 * ie;

        auto material = this->geometry->getMaterial(Vec<2>(0.5 * (x0 + x1), z));
        double T = temp[ie];
        double A = material->A(T);
        double B = material->B(T);
        double C = material->C(T);
        double D = 1e8 * material->D(T);  // cm²/s -> µm²/s

        // clang-format off
        setLocalMatrix(x0, x1 - x0, A, B, C, D, U0.data() + i, J.data() + ie,
                       K(i, i), K(i, i+1), K(i, i+2), K(i, i+3), K(i+1, i+1),
                       K(i+1, i+2), K(i+1, i+3), K(i+2, i+2), K(i+2, i+3), K(i+3, i+3),
                       F[i], F[i+1], F[i+2], F[i+3]);
        // clang-format on
    }

    // Set derivatives to 0 at the edges
    K.setBC(F, 1, 0.);
    K.setBC(F, K.size - 1, 0.);

#ifndef NDEBUG
    double* kend = K.data + K.size * K.kd;
    for (double* pk = K.data; pk != kend; ++pk) {
        if (isnan(*pk) || isinf(*pk))
            throw ComputationError(this->getId(), "Error in stiffness matrix at position {0} ({1})", pk - K.data,
                                   isnan(*pk) ? "nan" : "inf");
    }
    for (auto f = F.begin(); f != F.end(); ++f) {
        if (isnan(*f) || isinf(*f))
            throw ComputationError(this->getId(), "Error in load vector at position {0} ({1})", f - F.begin(),
                                   isnan(*f) ? "nan" : "inf");
    }
#endif
}

// template <typename Geometry2DType> void Diffusion2DSolver<Geometry2DType>::computeInitial(ActiveRegion& active) {
//     this->writelog(LOG_INFO, "Computing initial concentration");
//
//     auto mesh = active.mesh()->getMidpointAxis();
//
//     auto& conc = active.conc;
//
//     shared_ptr<RectangularMesh<2>> emesh(new RectangularMesh<2>(mesh, SinglePointMesh(active.vert())));
//
//     auto temperature = inTemperature(emesh, InterpolationMethod::INTERPOLATION_SPLINE);
//     auto current = inCurrentDensity(emesh, InterpolationMethod::INTERPOLATION_SPLINE);
//
//     #pragma omp parallel for
//     for (size_t i = 0; i < conc.size(); ++i) {
//         T = temperature[i];
//         J = current[i];
//
//     }
// }

template <typename Geometry2DType> double Diffusion2DSolver<Geometry2DType>::compute(unsigned loops, size_t act) {
    this->initCalculation();

    auto& active = this->active[act];
    double z = active.vert();

    auto mesh = active.mesh();
    size_t N = 2 * mesh->size();

    if (!active.conc) {
        active.conc.reset(mesh->size(), 0.);
        active.dconc.reset(mesh->size(), 0.);
    }

    auto& conc = active.conc;
    auto& dconc = active.dconc;

    auto temperature = inTemperature(active.emesh, InterpolationMethod::INTERPOLATION_SPLINE);
    DataVector<double> J(active.jmesh->size());
    double js = 1e7 / (phys::qe * active.QWheight);
    size_t i = 0;
    for (auto j : inCurrentDensity(active.jmesh, InterpolationMethod::INTERPOLATION_SPLINE)) {
        J[i] = abs(js * j.c1);
        ++i;
    }

    this->writelog(LOG_INFO, "Running diffusion calculations");

    unsigned loop = 0;

    std::unique_ptr<FemMatrix> K;

    toterr = 0.;

    DataVector<double> U(N);
    DataVector<double> F(N);
    DataVector<double> resid(N);

    switch (this->algorithm) {
        case ALGORITHM_CHOLESKY: K.reset(new DpbMatrix(this, N, 3)); break;
        case ALGORITHM_GAUSS: K.reset(new DgbMatrix(this, N, 3)); break;
        case ALGORITHM_ITERATIVE: K.reset(new SparseBandMatrix(this, N, 3)); break;
    }

    // Setup initial values — this is useful for iterative algorithms
    for (auto u = U.begin(), c = conc.begin(); u < U.end(); u += 2, ++c) *u = *c;
    for (auto u = U.begin() + 1, d = dconc.begin(); u < U.end(); u += 2, ++d) *u = *d;

    while (true) {
        setMatrix(*K, F, U, mesh, z, temperature, J);

        // Compute current error
        for (auto f = F.begin(), r = resid.begin(); f != F.end(); ++f, ++r) *r = -*f;
        K->addmult(U, resid);

        double err = 0.;
        for (auto r = resid.begin(); r != resid.end(); ++r) err += *r * *r;
        double denorm = 0.;
        for (auto f = F.begin(); f != F.end(); ++f) denorm += *f * *f;
        err = 100. * sqrt(err / denorm);

        if (loop != 0) this->writelog(LOG_RESULT, "Loop {:d}({:d}) @ active region {}: error = {:g}%", loop, loopno, act, err);
        ++loopno;
        ++loop;
        if (err < maxerr || ((loops != 0 && loop >= loops))) break;

        // TODO add linear mixing with the previous solution
        K->solve(F, U);
    }

    for (auto u = U.begin(), c = conc.begin(); u < U.end(); u += 2, ++c) *c = *u;
    for (auto u = U.begin() + 1, d = dconc.begin(); u < U.end(); u += 2, ++d) *d = *u;

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
            concentrations.emplace_back(LazyData<double>(dest_mesh->size(), [this, active, src_mesh](size_t i) -> double {
                double x = interpolationFlags.wrap(0, destination_mesh->at(i).c0);
                assert(src_mesh->at(0) <= x && x <= src_mesh->at(src_mesh->size() - 1));
                size_t idx = src_mesh->findIndex(x);
                if (idx == 0) return active.conc[0];
                const double x0 = src_mesh->at(idx - 1);
                const double L = src_mesh->at(idx) - x0;
                x -= x0;
                double L2 = L * L;
                double L3 = L2 * L;
                double x2 = x * x;
                double x3 = x2 * x;
                return (1 - 3 * x2 / L2 + 2 * x3 / L3) * active.conc[idx - 1] + (3 * x2 / L2 - 2 * x3 / L3) * active.conc[idx] +
                       (x - 2 * x2 / L + x3 / L2) * active.dconc[idx - 1] + (-x2 / L + x3 / L2) * active.dconc[idx];
            }));
        }

    } else {
        for (const auto& active : solver->active) {
            shared_ptr<RectangularMesh<2>> mesh(
                new RectangularMesh<2>(active.mesh(), shared_ptr<OnePointAxis>(new OnePointAxis(active.vert()))));
            concentrations.emplace_back(interpolate(mesh, active.conc, dest_mesh,
                                                    getInterpolationMethod<INTERPOLATION_SPLINE>(interp), interpolationFlags));
        }
    }
}

template <typename Geometry2DType> double Diffusion2DSolver<Geometry2DType>::ConcentrationDataImpl::at(size_t i) const {
    auto point = interpolationFlags.wrap(destination_mesh->at(i));
    bool found = false;
    size_t an = 0;
    for (const auto& active : solver->active) {
        if (solver->mesh->axis[1]->at(active.bottom) <= point.c1 && point.c1 <= solver->mesh->axis[1]->at(active.top)) {
            // Make sure we have concentration only in the quantum wells
            // TODO maybe more optimal approach would be reasonable?
            if (solver->mesh->axis[0]->at(active.left) <= point.c0 && point.c0 <= solver->mesh->axis[0]->at(active.right))
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
