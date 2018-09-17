#include "rectangular_filtered3d.h"

namespace plask {

void RectangularFilteredMesh3D::reset(const RectangularFilteredMesh3D::Predicate &predicate) {
    RectangularFilteredMeshBase<3>::reset();
    initNodesAndElements(predicate);
}

RectangularFilteredMesh3D::RectangularFilteredMesh3D(const RectangularMesh<3> &rectangularMesh, const RectangularFilteredMesh3D::Predicate &predicate, bool clone_axes)
    : RectangularFilteredMeshBase(rectangularMesh, clone_axes)
{
    initNodesAndElements(predicate);
}

void RectangularFilteredMesh3D::reset(const RectangularMesh<3> &rectangularMesh, const RectangularFilteredMesh3D::Predicate &predicate, bool clone_axes)
{
    this->fullMesh.reset(rectangularMesh, clone_axes);
    reset(predicate);
}

void RectangularFilteredMesh3D::initNodesAndElements(const RectangularFilteredMesh3D::Predicate &predicate)
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
            if (el_it->getLowerIndex0() < boundaryIndex[0].lo) boundaryIndex[0].lo = el_it->getLowerIndex0();
            if (el_it->getUpperIndex0() > boundaryIndex[0].up) boundaryIndex[0].up = el_it->getUpperIndex0();
            if (el_it->getLowerIndex1() < boundaryIndex[1].lo) boundaryIndex[1].lo = el_it->getLowerIndex1();
            if (el_it->getUpperIndex1() > boundaryIndex[1].up) boundaryIndex[1].up = el_it->getUpperIndex1();
            if (el_it->getLowerIndex2() < boundaryIndex[2].lo) boundaryIndex[2].lo = el_it->getLowerIndex2();
            if (el_it->getUpperIndex2() > boundaryIndex[2].up) boundaryIndex[2].up = el_it->getUpperIndex2();
        }
    nodeSet.shrink_to_fit();
    elementSet.shrink_to_fit();
}

bool RectangularFilteredMesh3D::prepareInterpolation(const Vec<3> &point, Vec<3> &wrapped_point,
                                                     std::size_t& index0_lo, std::size_t& index0_hi,
                                                     std::size_t& index1_lo, std::size_t& index1_hi,
                                                     std::size_t& index2_lo, std::size_t& index2_hi,
                                                     std::size_t &rectmesh_index_lo, const InterpolationFlags &flags) const {
    wrapped_point = flags.wrap(point);

    if (!canBeIncluded(wrapped_point)) return false;

    findIndexes(*fullMesh.axis[0], wrapped_point.c0, index0_lo, index0_hi);
    findIndexes(*fullMesh.axis[1], wrapped_point.c1, index1_lo, index1_hi);
    findIndexes(*fullMesh.axis[2], wrapped_point.c2, index2_lo, index2_hi);

    rectmesh_index_lo = fullMesh.index(index0_lo, index1_lo, index2_lo);
    return elementSet.includes(fullMesh.getElementIndexFromLowIndex(rectmesh_index_lo));
}

BoundaryNodeSet RectangularFilteredMesh3D::createIndex0BoundaryAtLine(std::size_t line_nr_axis0, std::size_t index1Begin, std::size_t index1End, std::size_t index2Begin, std::size_t index2End) const
{
    return new BoundaryNodeSetImpl<1, 2>(*this, line_nr_axis0, index1Begin, index2Begin, index1End, index2End);
}

BoundaryNodeSet RectangularFilteredMesh3D::createIndex0BoundaryAtLine(std::size_t line_nr_axis0) const {
    return createIndex0BoundaryAtLine(line_nr_axis0, boundaryIndex[1].lo, boundaryIndex[1].up+1, boundaryIndex[2].lo, boundaryIndex[2].up+1);
}

BoundaryNodeSet RectangularFilteredMesh3D::createIndex1BoundaryAtLine(std::size_t line_nr_axis1, std::size_t index0Begin, std::size_t index0End, std::size_t index2Begin, std::size_t index2End) const
{
    return new BoundaryNodeSetImpl<0, 2>(*this, index0Begin, line_nr_axis1, index2Begin, index0End, index2End);
}

BoundaryNodeSet RectangularFilteredMesh3D::createIndex1BoundaryAtLine(std::size_t line_nr_axis1) const {
    return createIndex1BoundaryAtLine(line_nr_axis1, boundaryIndex[0].lo, boundaryIndex[0].up+1, boundaryIndex[2].lo, boundaryIndex[2].up+1);
}

BoundaryNodeSet RectangularFilteredMesh3D::createIndex2BoundaryAtLine(std::size_t line_nr_axis2, std::size_t index0Begin, std::size_t index0End, std::size_t index1Begin, std::size_t index1End) const
{
    return new BoundaryNodeSetImpl<0, 1>(*this, index0Begin, index1Begin, line_nr_axis2, index0End, index1End);
}

BoundaryNodeSet RectangularFilteredMesh3D::createIndex2BoundaryAtLine(std::size_t line_nr_axis2) const {
    return createIndex2BoundaryAtLine(line_nr_axis2, boundaryIndex[0].lo, boundaryIndex[0].up+1, boundaryIndex[1].lo, boundaryIndex[1].up+1);
}

BoundaryNodeSet RectangularFilteredMesh3D::createBackBoundary() const {
    return createIndex0BoundaryAtLine(boundaryIndex[0].lo);
}

BoundaryNodeSet RectangularFilteredMesh3D::createFrontBoundary() const {
    return createIndex0BoundaryAtLine(boundaryIndex[0].up);
}

BoundaryNodeSet RectangularFilteredMesh3D::createLeftBoundary() const {
    return createIndex1BoundaryAtLine(boundaryIndex[1].lo);
}

BoundaryNodeSet RectangularFilteredMesh3D::createRightBoundary() const {
    return createIndex1BoundaryAtLine(boundaryIndex[1].up);
}

BoundaryNodeSet RectangularFilteredMesh3D::createBottomBoundary() const {
    return createIndex2BoundaryAtLine(boundaryIndex[2].lo);
}

BoundaryNodeSet RectangularFilteredMesh3D::createTopBoundary() const {
    return createIndex2BoundaryAtLine(boundaryIndex[2].up);
}

BoundaryNodeSet RectangularFilteredMesh3D::createBackOfBoundary(const Box3D &box) const {
    std::size_t line, begInd1, endInd1, begInd2, endInd2;
    if (details::getLineLo(line, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *fullMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd2, endInd2, *fullMesh.axis[2], box.lower.c2, box.upper.c2))
        return createIndex0BoundaryAtLine(line, begInd1, endInd1, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularFilteredMesh3D::createFrontOfBoundary(const Box3D &box) const {
    std::size_t line, begInd1, endInd1, begInd2, endInd2;
    if (details::getLineHi(line, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *fullMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd2, endInd2, *fullMesh.axis[2], box.lower.c2, box.upper.c2))
        return createIndex0BoundaryAtLine(line, begInd1, endInd1, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularFilteredMesh3D::createLeftOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd2, endInd2;
    if (details::getLineLo(line, *fullMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd0, endInd0, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd2, endInd2, *fullMesh.axis[2], box.lower.c2, box.upper.c2))
        return createIndex1BoundaryAtLine(line, begInd0, endInd0, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularFilteredMesh3D::createRightOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd2, endInd2;
    if (details::getLineHi(line, *fullMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd0, endInd0, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd2, endInd2, *fullMesh.axis[2], box.lower.c2, box.upper.c2))
        return createIndex1BoundaryAtLine(line, begInd0, endInd0, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularFilteredMesh3D::createBottomOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd1, endInd1;
    if (details::getLineLo(line, *fullMesh.axis[2], box.lower.c2, box.upper.c2) &&
            details::getIndexesInBounds(begInd0, endInd0, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *fullMesh.axis[1], box.lower.c1, box.upper.c1))
        return createIndex2BoundaryAtLine(line, begInd0, endInd0, begInd1, endInd1);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularFilteredMesh3D::createTopOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd1, endInd1;
    if (details::getLineHi(line, *fullMesh.axis[2], box.lower.c2, box.upper.c2) &&
            details::getIndexesInBounds(begInd0, endInd0, *fullMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *fullMesh.axis[1], box.lower.c1, box.upper.c1))
        return createIndex2BoundaryAtLine(line, begInd0, endInd0, begInd1, endInd1);
    else
        return new EmptyBoundaryImpl();
}

}   // namespace plask
