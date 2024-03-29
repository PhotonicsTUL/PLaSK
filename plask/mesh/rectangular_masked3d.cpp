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
#include "rectangular_masked3d.hpp"

namespace plask {

void RectangularMaskedMesh3D::reset(const RectangularMaskedMesh3D::Predicate &predicate) {
    RectangularMaskedMeshBase<3>::reset();
    initNodesAndElements(predicate);
}

RectangularMaskedMesh3D::RectangularMaskedMesh3D(const RectangularMesh<3> &rectangularMesh, const RectangularMaskedMesh3D::Predicate &predicate, bool clone_axes)
    : RectangularMaskedMeshBase(rectangularMesh, clone_axes)
{
    initNodesAndElements(predicate);
}

void RectangularMaskedMesh3D::reset(const RectangularMesh<3> &rectangularMesh, const RectangularMaskedMesh3D::Predicate &predicate, bool clone_axes)
{
    this->fullMesh.reset(rectangularMesh, clone_axes);
    reset(predicate);
}

RectangularMaskedMesh3D::RectangularMaskedMesh3D(const RectangularMesh<DIM> &rectangularMesh, RectangularMaskedMeshBase::Set nodeSet, bool clone_axes)
    : RectangularMaskedMeshBase(rectangularMesh, std::move(nodeSet), clone_axes)
{
}

void RectangularMaskedMesh3D::initNodesAndElements(const RectangularMaskedMesh3D::Predicate &predicate)
{
    for (auto el_it = this->fullMesh.elements().begin(); el_it != this->fullMesh.elements().end(); ++el_it)
        if (predicate(*el_it)) {
            elementSet.push_back(el_it.index);
            nodeSet.insert(el_it->getLoLoLoIndex());

            nodeSet.insert(el_it->getUpLoLoIndex());
            nodeSet.insert(el_it->getLoUpLoIndex());
            nodeSet.insert(el_it->getLoLoUpIndex());

            nodeSet.insert(el_it->getLoUpUpIndex());
            nodeSet.insert(el_it->getUpLoUpIndex());
            nodeSet.insert(el_it->getUpUpLoIndex());

            nodeSet.push_back(el_it->getUpUpUpIndex());
            /*boundaryIndex[0].improveLo(el_it->getLowerIndex0());
            boundaryIndex[0].improveUp(el_it->getUpperIndex0());
            boundaryIndex[1].improveLo(el_it->getLowerIndex1());
            boundaryIndex[1].improveUp(el_it->getUpperIndex1());
            boundaryIndex[2].improveLo(el_it->getLowerIndex2());
            boundaryIndex[2].improveUp(el_it->getUpperIndex2());*/  // this is initialized lazy
        }
    nodeSet.shrink_to_fit();
    elementSet.shrink_to_fit();
    elementSetInitialized = true;
}

bool RectangularMaskedMesh3D::prepareInterpolation(const Vec<3> &point, Vec<3> &wrapped_point,
                                                     std::size_t& index0_lo, std::size_t& index0_hi,
                                                     std::size_t& index1_lo, std::size_t& index1_hi,
                                                     std::size_t& index2_lo, std::size_t& index2_hi,
                                                     const InterpolationFlags &flags) const {
    wrapped_point = flags.wrap(point);

    if (!canBeIncluded(wrapped_point)) return false;

    findIndexes(*fullMesh.axis[0], wrapped_point.c0, index0_lo, index0_hi);
    findIndexes(*fullMesh.axis[1], wrapped_point.c1, index1_lo, index1_hi);
    findIndexes(*fullMesh.axis[2], wrapped_point.c2, index2_lo, index2_hi);
    assert(index0_hi == index0_lo + 1);
    assert(index1_hi == index1_lo + 1);
    assert(index2_hi == index2_lo + 1);

    double lo0 = fullMesh.axis[0]->at(index0_lo), hi0 = fullMesh.axis[0]->at(index0_hi),
           lo1 = fullMesh.axis[1]->at(index1_lo), hi1 = fullMesh.axis[1]->at(index1_hi),
           lo2 = fullMesh.axis[2]->at(index2_lo), hi2 = fullMesh.axis[2]->at(index2_hi);

    ensureHasElements();
    for (char i2 = 0; i2 < 2; ++i2) {
        for (char i1 = 0; i1 < 2; ++i1) {
            for (char i0 = 0; i0 < 2; ++i0) {
                if (elementSet.includes(fullMesh.getElementIndexFromLowIndexes(index0_lo, index1_lo, index2_lo))) {
                    index0_hi = index0_lo + 1; index1_hi = index1_lo + 1; index2_hi = index2_lo + 1;
                    return true;
                }
                if (index0_lo > 0 && lo0 <= wrapped_point.c0 && wrapped_point.c0 < lo0+MIN_DISTANCE) index0_lo = index0_hi-2;
                else if (index0_lo < fullMesh.axis[0]->size()-2 && hi0-MIN_DISTANCE < wrapped_point.c0 && wrapped_point.c0 <= hi0) index0_lo = index0_hi;
                else break;
            }
            index0_lo = index0_hi - 1;
            if (index1_lo > 0 && lo1 <= wrapped_point.c1 && wrapped_point.c1 < lo1+MIN_DISTANCE) index1_lo = index1_hi-2;
            else if (index1_lo < fullMesh.axis[1]->size()-2 && hi1-MIN_DISTANCE < wrapped_point.c1 && wrapped_point.c1 <= hi1) index1_lo = index1_hi;
            else break;
        }
        index1_lo = index1_hi - 1;
        if (index2_lo > 0 && lo2 <= wrapped_point.c2 && wrapped_point.c2 < lo2+MIN_DISTANCE) index2_lo = index2_hi-2;
        else if (index2_lo < fullMesh.axis[2]->size()-2 && hi2-MIN_DISTANCE < wrapped_point.c2 && wrapped_point.c2 <= hi1) index2_lo = index2_hi;
        else break;
    }

    return false;
}

BoundaryNodeSet RectangularMaskedMesh3D::createIndex0BoundaryAtLine(std::size_t line_nr_axis0, std::size_t index1Begin, std::size_t index1End, std::size_t index2Begin, std::size_t index2End) const
{
    if (this->fullMesh.isChangeSlower(1, 2))
        return new BoundaryNodeSetImpl<1, 2>(*this, line_nr_axis0, index1Begin, index2Begin, index1End, index2End);
    else
        return new BoundaryNodeSetImpl<2, 1>(*this, line_nr_axis0, index1Begin, index2Begin, index2End, index1End);
}

BoundaryNodeSet RectangularMaskedMesh3D::createIndex0BoundaryAtLine(std::size_t line_nr_axis0) const {
    ensureHasBoundaryIndex();
    return createIndex0BoundaryAtLine(line_nr_axis0, boundaryIndex[1].lo, boundaryIndex[1].up+1, boundaryIndex[2].lo, boundaryIndex[2].up+1);
}

BoundaryNodeSet RectangularMaskedMesh3D::createIndex1BoundaryAtLine(std::size_t line_nr_axis1, std::size_t index0Begin, std::size_t index0End, std::size_t index2Begin, std::size_t index2End) const
{
    if (this->fullMesh.isChangeSlower(0, 2))
        return new BoundaryNodeSetImpl<0, 2>(*this, index0Begin, line_nr_axis1, index2Begin, index0End, index2End);
    else
        return new BoundaryNodeSetImpl<2, 0>(*this, index0Begin, line_nr_axis1, index2Begin, index2End, index0End);
}

BoundaryNodeSet RectangularMaskedMesh3D::createIndex1BoundaryAtLine(std::size_t line_nr_axis1) const {
    ensureHasBoundaryIndex();
    return createIndex1BoundaryAtLine(line_nr_axis1, boundaryIndex[0].lo, boundaryIndex[0].up+1, boundaryIndex[2].lo, boundaryIndex[2].up+1);
}

BoundaryNodeSet RectangularMaskedMesh3D::createIndex2BoundaryAtLine(std::size_t line_nr_axis2, std::size_t index0Begin, std::size_t index0End, std::size_t index1Begin, std::size_t index1End) const
{
    if (this->fullMesh.isChangeSlower(0, 1))
        return new BoundaryNodeSetImpl<0, 1>(*this, index0Begin, index1Begin, line_nr_axis2, index0End, index1End);
    else
        return new BoundaryNodeSetImpl<1, 0>(*this, index0Begin, index1Begin, line_nr_axis2, index1End, index0End);
}

BoundaryNodeSet RectangularMaskedMesh3D::createIndex2BoundaryAtLine(std::size_t line_nr_axis2) const {
    ensureHasBoundaryIndex();
    return createIndex2BoundaryAtLine(line_nr_axis2, boundaryIndex[0].lo, boundaryIndex[0].up+1, boundaryIndex[1].lo, boundaryIndex[1].up+1);
}

BoundaryNodeSet RectangularMaskedMesh3D::createBackBoundary() const {
    return createIndex0BoundaryAtLine(ensureHasBoundaryIndex()[0].lo);
}

BoundaryNodeSet RectangularMaskedMesh3D::createFrontBoundary() const {
    return createIndex0BoundaryAtLine(ensureHasBoundaryIndex()[0].up);
}

BoundaryNodeSet RectangularMaskedMesh3D::createLeftBoundary() const {
    return createIndex1BoundaryAtLine(ensureHasBoundaryIndex()[1].lo);
}

BoundaryNodeSet RectangularMaskedMesh3D::createRightBoundary() const {
    return createIndex1BoundaryAtLine(ensureHasBoundaryIndex()[1].up);
}

BoundaryNodeSet RectangularMaskedMesh3D::createBottomBoundary() const {
    return createIndex2BoundaryAtLine(ensureHasBoundaryIndex()[2].lo);
}

BoundaryNodeSet RectangularMaskedMesh3D::createTopBoundary() const {
    return createIndex2BoundaryAtLine(ensureHasBoundaryIndex()[2].up);
}

BoundaryNodeSet RectangularMaskedMesh3D::createBackOfBoundary(const Box3D &box) const {
    std::size_t line, begInd1, endInd1, begInd2, endInd2;
    if (details::getLineLo(line, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *fullMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd2, endInd2, *fullMesh.axis[2], box.lower.c2, box.upper.c2))
        return createIndex0BoundaryAtLine(line, begInd1, endInd1, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularMaskedMesh3D::createFrontOfBoundary(const Box3D &box) const {
    std::size_t line, begInd1, endInd1, begInd2, endInd2;
    if (details::getLineHi(line, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *fullMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd2, endInd2, *fullMesh.axis[2], box.lower.c2, box.upper.c2))
        return createIndex0BoundaryAtLine(line, begInd1, endInd1, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularMaskedMesh3D::createLeftOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd2, endInd2;
    if (details::getLineLo(line, *fullMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd0, endInd0, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd2, endInd2, *fullMesh.axis[2], box.lower.c2, box.upper.c2))
        return createIndex1BoundaryAtLine(line, begInd0, endInd0, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularMaskedMesh3D::createRightOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd2, endInd2;
    if (details::getLineHi(line, *fullMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd0, endInd0, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd2, endInd2, *fullMesh.axis[2], box.lower.c2, box.upper.c2))
        return createIndex1BoundaryAtLine(line, begInd0, endInd0, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularMaskedMesh3D::createBottomOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd1, endInd1;
    if (details::getLineLo(line, *fullMesh.axis[2], box.lower.c2, box.upper.c2) &&
            details::getIndexesInBounds(begInd0, endInd0, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *fullMesh.axis[1], box.lower.c1, box.upper.c1))
        return createIndex2BoundaryAtLine(line, begInd0, endInd0, begInd1, endInd1);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularMaskedMesh3D::createTopOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd1, endInd1;
    if (details::getLineHi(line, *fullMesh.axis[2], box.lower.c2, box.upper.c2) &&
            details::getIndexesInBounds(begInd0, endInd0, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *fullMesh.axis[1], box.lower.c1, box.upper.c1))
        return createIndex2BoundaryAtLine(line, begInd0, endInd0, begInd1, endInd1);
    else
        return new EmptyBoundaryImpl();
}

}   // namespace plask
