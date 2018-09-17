#include "rectangular_filtered2d.h"

namespace plask {

void RectangularFilteredMesh2D::reset(const RectangularFilteredMesh2D::Predicate &predicate) {
    RectangularFilteredMeshBase<2>::reset();
    initNodesAndElements(predicate);
}

RectangularFilteredMesh2D::RectangularFilteredMesh2D(const RectangularMesh<2> &rectangularMesh, const RectangularFilteredMesh2D::Predicate &predicate, bool clone_axes)
    : RectangularFilteredMeshBase(rectangularMesh, clone_axes)
{
    initNodesAndElements(predicate);
}

void RectangularFilteredMesh2D::reset(const RectangularMesh<2> &rectangularMesh, const RectangularFilteredMesh2D::Predicate &predicate, bool clone_axes) {
    this->fullMesh.reset(rectangularMesh, clone_axes);
    reset(predicate);
}

void RectangularFilteredMesh2D::initNodesAndElements(const RectangularFilteredMesh2D::Predicate &predicate)
{
    for (auto el_it = this->fullMesh.elements().begin(); el_it != this->fullMesh.elements().end(); ++el_it)
        if (predicate(*el_it)) {
            elementSet.push_back(el_it.index);
            nodeSet.insert(el_it->getLoLoIndex());
            nodeSet.insert(el_it->getLoUpIndex());
            nodeSet.insert(el_it->getUpLoIndex());
            nodeSet.push_back(el_it->getUpUpIndex());  //this is safe also for 10 axis order
            if (el_it->getLowerIndex0() < boundaryIndex[0].lo) boundaryIndex[0].lo = el_it->getLowerIndex0();
            if (el_it->getUpperIndex0() > boundaryIndex[0].up) boundaryIndex[0].up = el_it->getUpperIndex0();
            if (el_it->getLowerIndex1() < boundaryIndex[1].lo) boundaryIndex[1].lo = el_it->getLowerIndex1();
            if (el_it->getUpperIndex1() > boundaryIndex[1].up) boundaryIndex[1].up = el_it->getUpperIndex1();
        }
    nodeSet.shrink_to_fit();
    elementSet.shrink_to_fit();
}

bool RectangularFilteredMesh2D::prepareInterpolation(const Vec<2> &point, Vec<2> &wrapped_point, std::size_t &index0_lo, std::size_t &index0_hi, std::size_t &index1_lo, std::size_t &index1_hi, std::size_t &rectmesh_index_lo, const InterpolationFlags &flags) const {
    wrapped_point = flags.wrap(point);

    if (!canBeIncluded(wrapped_point)) return false;

    findIndexes(*fullMesh.axis[0], wrapped_point.c0, index0_lo, index0_hi);
    findIndexes(*fullMesh.axis[1], wrapped_point.c1, index1_lo, index1_hi);

    rectmesh_index_lo = fullMesh.index(index0_lo, index1_lo);
    return elementSet.includes(fullMesh.getElementIndexFromLowIndex(rectmesh_index_lo));
}

BoundaryNodeSet RectangularFilteredMesh2D::createVerticalBoundaryAtLine(std::size_t line_nr_axis0) const {
    return createVerticalBoundaryAtLine(line_nr_axis0, boundaryIndex[1].lo, boundaryIndex[1].up+1);
}

BoundaryNodeSet RectangularFilteredMesh2D::createVerticalBoundaryAtLine(std::size_t line_nr_axis0, std::size_t indexBegin, std::size_t indexEnd) const {
    return new BoundaryNodeSetImpl<1>(*this, line_nr_axis0, indexBegin, indexEnd);
}

BoundaryNodeSet RectangularFilteredMesh2D::createVerticalBoundaryNear(double axis0_coord) const {
    return createVerticalBoundaryAtLine(fullMesh.axis[0]->findNearestIndex(axis0_coord));
}

BoundaryNodeSet RectangularFilteredMesh2D::createVerticalBoundaryNear(double axis0_coord, double from, double to) const {
    std::size_t begInd, endInd;
    if (!details::getIndexesInBoundsExt(begInd, endInd, *fullMesh.axis[1], from, to))
        return new EmptyBoundaryImpl();
    return createVerticalBoundaryAtLine(fullMesh.axis[0]->findNearestIndex(axis0_coord), begInd, endInd);
}

BoundaryNodeSet RectangularFilteredMesh2D::createLeftBoundary() const {
    return createVerticalBoundaryAtLine(boundaryIndex[0].lo);
}

BoundaryNodeSet RectangularFilteredMesh2D::createRightBoundary() const {
    return createVerticalBoundaryAtLine(boundaryIndex[0].up);
}

BoundaryNodeSet RectangularFilteredMesh2D::createLeftOfBoundary(const Box2D &box) const {
    std::size_t line, begInd, endInd;
    if (details::getLineLo(line, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd, endInd, *fullMesh.axis[1], box.lower.c1, box.upper.c1))
        return createVerticalBoundaryAtLine(line, begInd, endInd);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularFilteredMesh2D::createRightOfBoundary(const Box2D &box) const {
    std::size_t line, begInd, endInd;
    if (details::getLineHi(line, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd, endInd, *fullMesh.axis[1], box.lower.c1, box.upper.c1))
        return createVerticalBoundaryAtLine(line, begInd, endInd);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularFilteredMesh2D::createBottomOfBoundary(const Box2D &box) const {
    std::size_t line, begInd, endInd;
    if (details::getLineLo(line, *fullMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd, endInd, *fullMesh.axis[0], box.lower.c0, box.upper.c0))
        return createHorizontalBoundaryAtLine(line, begInd, endInd);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularFilteredMesh2D::createTopOfBoundary(const Box2D &box) const {
    std::size_t line, begInd, endInd;
    if (details::getLineHi(line, *fullMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd, endInd, *fullMesh.axis[0], box.lower.c0, box.upper.c0))
        return createHorizontalBoundaryAtLine(line, begInd, endInd);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularFilteredMesh2D::createHorizontalBoundaryAtLine(std::size_t line_nr_axis1) const {
    return createHorizontalBoundaryAtLine(line_nr_axis1, boundaryIndex[0].lo, boundaryIndex[0].up+1);
}

BoundaryNodeSet RectangularFilteredMesh2D::createHorizontalBoundaryAtLine(std::size_t line_nr_axis1, std::size_t indexBegin, std::size_t indexEnd) const {
    return new BoundaryNodeSetImpl<0>(*this, indexBegin, line_nr_axis1, indexEnd);
}

BoundaryNodeSet RectangularFilteredMesh2D::createHorizontalBoundaryNear(double axis1_coord) const {
    return createHorizontalBoundaryAtLine(fullMesh.axis[1]->findNearestIndex(axis1_coord));
}

BoundaryNodeSet RectangularFilteredMesh2D::createHorizontalBoundaryNear(double axis1_coord, double from, double to) const {
    std::size_t begInd, endInd;
    if (!details::getIndexesInBoundsExt(begInd, endInd, *fullMesh.axis[0], from, to))
        return new EmptyBoundaryImpl();
    return createHorizontalBoundaryAtLine(fullMesh.axis[1]->findNearestIndex(axis1_coord), begInd, endInd);
}

BoundaryNodeSet RectangularFilteredMesh2D::createTopBoundary() const {
    return createHorizontalBoundaryAtLine(boundaryIndex[1].up);
}

BoundaryNodeSet RectangularFilteredMesh2D::createBottomBoundary() const {
    return createHorizontalBoundaryAtLine(boundaryIndex[1].lo);
}

}   // namespace plask
