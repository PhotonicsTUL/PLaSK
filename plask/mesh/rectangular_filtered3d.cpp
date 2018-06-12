#include "rectangular_filtered3d.h"

namespace plask {

RectangularFilteredMesh3D::RectangularFilteredMesh3D(const RectangularMesh<3> &rectangularMesh, const RectangularFilteredMesh3D::Predicate &predicate, bool clone_axes)
    : RectangularFilteredMeshBase(rectangularMesh, clone_axes)
{
    for (auto el_it = this->rectangularMesh.elements().begin(); el_it != this->rectangularMesh.elements().end(); ++el_it)
        if (predicate(*el_it)) {
            elementsSet.push_back(el_it.index);
            nodesSet.insert(el_it->getLoLoLoIndex());

            nodesSet.insert(el_it->getUpLoLoIndex());
            nodesSet.insert(el_it->getLoUpLoIndex());
            nodesSet.insert(el_it->getLoLoUpIndex());

            nodesSet.insert(el_it->getLoUpUpIndex());
            nodesSet.insert(el_it->getUpLoUpIndex());
            nodesSet.insert(el_it->getUpUpLoIndex());

            nodesSet.push_back(el_it->getUpUpUpIndex());
            if (el_it->getLowerIndex0() < boundaryIndex[0].lo) boundaryIndex[0].lo = el_it->getLowerIndex0();
            if (el_it->getUpperIndex0() > boundaryIndex[0].up) boundaryIndex[0].up = el_it->getUpperIndex0();
            if (el_it->getLowerIndex1() < boundaryIndex[1].lo) boundaryIndex[1].lo = el_it->getLowerIndex1();
            if (el_it->getUpperIndex1() > boundaryIndex[1].up) boundaryIndex[1].up = el_it->getUpperIndex1();
            if (el_it->getLowerIndex2() < boundaryIndex[2].lo) boundaryIndex[2].lo = el_it->getLowerIndex2();
            if (el_it->getUpperIndex2() > boundaryIndex[2].up) boundaryIndex[2].up = el_it->getUpperIndex2();
        }
    nodesSet.shrink_to_fit();
    elementsSet.shrink_to_fit();
}

bool RectangularFilteredMesh3D::prepareInterpolation(const Vec<3> &point, Vec<3> &wrapped_point,
                                                     std::size_t& index0_lo, std::size_t& index0_hi,
                                                     std::size_t& index1_lo, std::size_t& index1_hi,
                                                     std::size_t& index2_lo, std::size_t& index2_hi,
                                                     std::size_t &rectmesh_index_lo, const InterpolationFlags &flags) const {
    wrapped_point = flags.wrap(point);

    if (!canBeIncluded(wrapped_point)) return false;

    findIndexes(*rectangularMesh.axis[0], wrapped_point.c0, index0_lo, index0_hi);
    findIndexes(*rectangularMesh.axis[1], wrapped_point.c1, index1_lo, index1_hi);
    findIndexes(*rectangularMesh.axis[2], wrapped_point.c2, index2_lo, index2_hi);

    rectmesh_index_lo = rectangularMesh.index(index0_lo, index1_lo, index2_lo);
    return elementsSet.includes(rectangularMesh.getElementIndexFromLowIndex(rectmesh_index_lo));
}

BoundaryNodeSet RectangularFilteredMesh3D::createIndex0BoundaryAtLine(std::size_t line_nr_axis0, std::size_t index1Begin, std::size_t index1End, std::size_t index2Begin, std::size_t index2End) const
{
    return new BoundaryNodeSetImpl<1, 2>(*this, line_nr_axis0, index1Begin, index1End, index2Begin, index2End);
}

BoundaryNodeSet RectangularFilteredMesh3D::createIndex0BoundaryAtLine(std::size_t line_nr_axis0) const {
    return createIndex0BoundaryAtLine(line_nr_axis0, boundaryIndex[1].lo, boundaryIndex[1].up+1, boundaryIndex[2].lo, boundaryIndex[2].up+1);
}

BoundaryNodeSet RectangularFilteredMesh3D::createIndex1BoundaryAtLine(std::size_t line_nr_axis1, std::size_t index0Begin, std::size_t index0End, std::size_t index2Begin, std::size_t index2End) const
{
    return new BoundaryNodeSetImpl<0, 2>(*this, line_nr_axis1, index0Begin, index0End, index2Begin, index2End);
}

BoundaryNodeSet RectangularFilteredMesh3D::createIndex1BoundaryAtLine(std::size_t line_nr_axis1) const {
    return createIndex1BoundaryAtLine(line_nr_axis1, boundaryIndex[0].lo, boundaryIndex[0].up+1, boundaryIndex[2].lo, boundaryIndex[2].up+1);
}

BoundaryNodeSet RectangularFilteredMesh3D::createIndex2BoundaryAtLine(std::size_t line_nr_axis2, std::size_t index0Begin, std::size_t index0End, std::size_t index1Begin, std::size_t index1End) const
{
    return new BoundaryNodeSetImpl<0, 1>(*this, line_nr_axis2, index0Begin, index0End, index1Begin, index1End);
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
    if (details::getLineLo(line, *rectangularMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *rectangularMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd2, endInd2, *rectangularMesh.axis[2], box.lower.c2, box.upper.c2))
        return createIndex0BoundaryAtLine(line, begInd1, endInd1, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularFilteredMesh3D::createFrontOfBoundary(const Box3D &box) const {
    std::size_t line, begInd1, endInd1, begInd2, endInd2;
    if (details::getLineHi(line, *rectangularMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *rectangularMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd2, endInd2, *rectangularMesh.axis[2], box.lower.c2, box.upper.c2))
        return createIndex0BoundaryAtLine(line, begInd1, endInd1, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularFilteredMesh3D::createLeftOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd2, endInd2;
    if (details::getLineLo(line, *rectangularMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd0, endInd0, *rectangularMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd2, endInd2, *rectangularMesh.axis[2], box.lower.c2, box.upper.c2))
        return createIndex1BoundaryAtLine(line, begInd0, endInd0, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularFilteredMesh3D::createRightOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd2, endInd2;
    if (details::getLineHi(line, *rectangularMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd0, endInd0, *rectangularMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd2, endInd2, *rectangularMesh.axis[2], box.lower.c2, box.upper.c2))
        return createIndex1BoundaryAtLine(line, begInd0, endInd0, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularFilteredMesh3D::createBottomOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd1, endInd1;
    if (details::getLineLo(line, *rectangularMesh.axis[2], box.lower.c2, box.upper.c2) &&
            details::getIndexesInBounds(begInd0, endInd0, *rectangularMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *rectangularMesh.axis[1], box.lower.c1, box.upper.c1))
        return createIndex2BoundaryAtLine(line, begInd0, endInd0, begInd1, endInd1);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularFilteredMesh3D::createTopOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd1, endInd1;
    if (details::getLineHi(line, *rectangularMesh.axis[2], box.lower.c2, box.upper.c2) &&
            details::getIndexesInBounds(begInd0, endInd0, *rectangularMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *rectangularMesh.axis[1], box.lower.c1, box.upper.c1))
        return createIndex2BoundaryAtLine(line, begInd0, endInd0, begInd1, endInd1);
    else
        return new EmptyBoundaryImpl();
}

}   // namespace plask
