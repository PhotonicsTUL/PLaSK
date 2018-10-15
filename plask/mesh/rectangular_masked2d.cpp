#include "rectangular_masked2d.h"

namespace plask {

void RectangularMaskedMesh2D::reset(const RectangularMaskedMesh2D::Predicate &predicate) {
    RectangularMaskedMeshBase<2>::reset();
    initNodesAndElements(predicate);
}

RectangularMaskedMesh2D::RectangularMaskedMesh2D(const RectangularMesh<2> &rectangularMesh, const RectangularMaskedMesh2D::Predicate &predicate, bool clone_axes)
    : RectangularMaskedMeshBase(rectangularMesh, clone_axes)
{
    initNodesAndElements(predicate);
}

void RectangularMaskedMesh2D::reset(const RectangularMesh<2> &rectangularMesh, const RectangularMaskedMesh2D::Predicate &predicate, bool clone_axes) {
    this->fullMesh.reset(rectangularMesh, clone_axes);
    reset(predicate);
}

RectangularMaskedMesh2D::RectangularMaskedMesh2D(const RectangularMesh<DIM> &rectangularMesh, RectangularMaskedMeshBase::Set nodeSet, bool clone_axes)
    : RectangularMaskedMeshBase(rectangularMesh, std::move(nodeSet), clone_axes)
{
}

void RectangularMaskedMesh2D::initNodesAndElements(const RectangularMaskedMesh2D::Predicate &predicate)
{
    for (auto el_it = this->fullMesh.elements().begin(); el_it != this->fullMesh.elements().end(); ++el_it)
        if (predicate(*el_it)) {
            elementSet.push_back(el_it.index);
            nodeSet.insert(el_it->getLoLoIndex());
            nodeSet.insert(el_it->getLoUpIndex());
            nodeSet.insert(el_it->getUpLoIndex());
            nodeSet.push_back(el_it->getUpUpIndex());  //this is safe also for 10 axis order
            /*boundaryIndex[0].improveLo(el_it->getLowerIndex0());
            boundaryIndex[0].improveUp(el_it->getUpperIndex0());
            boundaryIndex[1].improveLo(el_it->getLowerIndex1());
            boundaryIndex[1].improveUp(el_it->getUpperIndex1());*/  // this is initilized lazy
        }
    nodeSet.shrink_to_fit();
    elementSet.shrink_to_fit();
    elementSetInitialized = true;
}

bool RectangularMaskedMesh2D::prepareInterpolation(const Vec<2> &point, Vec<2> &wrapped_point, std::size_t &index0_lo, std::size_t &index0_hi, std::size_t &index1_lo, std::size_t &index1_hi, const InterpolationFlags &flags) const {
    wrapped_point = flags.wrap(point);

    if (!canBeIncluded(wrapped_point)) return false;

    findIndexes(*fullMesh.axis[0], wrapped_point.c0, index0_lo, index0_hi);
    findIndexes(*fullMesh.axis[1], wrapped_point.c1, index1_lo, index1_hi);
    assert(index0_hi == index0_lo + 1);
    assert(index1_hi == index1_lo + 1);

    double lo0 = fullMesh.axis[0]->at(index0_lo), hi0 = fullMesh.axis[0]->at(index0_hi),
           lo1 = fullMesh.axis[1]->at(index1_lo), hi1 = fullMesh.axis[1]->at(index1_hi);

    ensureHasElements();
    for (char i1 = 0; i1 < 2; ++i1) {
        for (char i0 = 0; i0 < 2; ++i0) {
            if (elementSet.includes(fullMesh.getElementIndexFromLowIndexes(index0_lo, index1_lo))) {
                index0_hi = index0_lo + 1; index1_hi = index1_lo + 1;
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

    return false;
}




BoundaryNodeSet RectangularMaskedMesh2D::createVerticalBoundaryAtLine(std::size_t line_nr_axis0) const {
    ensureHasBoundaryIndex();
    return createVerticalBoundaryAtLine(line_nr_axis0, boundaryIndex[1].lo, boundaryIndex[1].up+1);
}

BoundaryNodeSet RectangularMaskedMesh2D::createVerticalBoundaryAtLine(std::size_t line_nr_axis0, std::size_t indexBegin, std::size_t indexEnd) const {
    return new BoundaryNodeSetImpl<1>(*this, line_nr_axis0, indexBegin, indexEnd);
}

BoundaryNodeSet RectangularMaskedMesh2D::createVerticalBoundaryNear(double axis0_coord) const {
    return createVerticalBoundaryAtLine(fullMesh.axis[0]->findNearestIndex(axis0_coord));
}

BoundaryNodeSet RectangularMaskedMesh2D::createVerticalBoundaryNear(double axis0_coord, double from, double to) const {
    std::size_t begInd, endInd;
    if (!details::getIndexesInBoundsExt(begInd, endInd, *fullMesh.axis[1], from, to))
        return new EmptyBoundaryImpl();
    return createVerticalBoundaryAtLine(fullMesh.axis[0]->findNearestIndex(axis0_coord), begInd, endInd);
}

BoundaryNodeSet RectangularMaskedMesh2D::createLeftBoundary() const {
    return createVerticalBoundaryAtLine(ensureHasBoundaryIndex()[0].lo);
}

BoundaryNodeSet RectangularMaskedMesh2D::createRightBoundary() const {
    return createVerticalBoundaryAtLine(ensureHasBoundaryIndex()[0].up);
}

BoundaryNodeSet RectangularMaskedMesh2D::createLeftOfBoundary(const Box2D &box) const {
    std::size_t line, begInd, endInd;
    if (details::getLineLo(line, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd, endInd, *fullMesh.axis[1], box.lower.c1, box.upper.c1))
        return createVerticalBoundaryAtLine(line, begInd, endInd);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularMaskedMesh2D::createRightOfBoundary(const Box2D &box) const {
    std::size_t line, begInd, endInd;
    if (details::getLineHi(line, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd, endInd, *fullMesh.axis[1], box.lower.c1, box.upper.c1))
        return createVerticalBoundaryAtLine(line, begInd, endInd);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularMaskedMesh2D::createBottomOfBoundary(const Box2D &box) const {
    std::size_t line, begInd, endInd;
    if (details::getLineLo(line, *fullMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd, endInd, *fullMesh.axis[0], box.lower.c0, box.upper.c0))
        return createHorizontalBoundaryAtLine(line, begInd, endInd);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularMaskedMesh2D::createTopOfBoundary(const Box2D &box) const {
    std::size_t line, begInd, endInd;
    if (details::getLineHi(line, *fullMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd, endInd, *fullMesh.axis[0], box.lower.c0, box.upper.c0))
        return createHorizontalBoundaryAtLine(line, begInd, endInd);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularMaskedMesh2D::createHorizontalBoundaryAtLine(std::size_t line_nr_axis1) const {
    ensureHasBoundaryIndex();
    return createHorizontalBoundaryAtLine(line_nr_axis1, boundaryIndex[0].lo, boundaryIndex[0].up+1);
}

BoundaryNodeSet RectangularMaskedMesh2D::createHorizontalBoundaryAtLine(std::size_t line_nr_axis1, std::size_t indexBegin, std::size_t indexEnd) const {
    return new BoundaryNodeSetImpl<0>(*this, indexBegin, line_nr_axis1, indexEnd);
}

BoundaryNodeSet RectangularMaskedMesh2D::createHorizontalBoundaryNear(double axis1_coord) const {
    return createHorizontalBoundaryAtLine(fullMesh.axis[1]->findNearestIndex(axis1_coord));
}

BoundaryNodeSet RectangularMaskedMesh2D::createHorizontalBoundaryNear(double axis1_coord, double from, double to) const {
    std::size_t begInd, endInd;
    if (!details::getIndexesInBoundsExt(begInd, endInd, *fullMesh.axis[0], from, to))
        return new EmptyBoundaryImpl();
    return createHorizontalBoundaryAtLine(fullMesh.axis[1]->findNearestIndex(axis1_coord), begInd, endInd);
}

BoundaryNodeSet RectangularMaskedMesh2D::createTopBoundary() const {
    return createHorizontalBoundaryAtLine(ensureHasBoundaryIndex()[1].up);
}

BoundaryNodeSet RectangularMaskedMesh2D::createBottomBoundary() const {
    return createHorizontalBoundaryAtLine(ensureHasBoundaryIndex()[1].lo);
}

}   // namespace plask
