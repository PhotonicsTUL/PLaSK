#include "rectangular_common.h"

namespace plask {

BoundaryNodeSet RectangularMeshBase2D::createVerticalBoundaryAtLine(std::size_t PLASK_UNUSED(line_nr_axis0)) const {
    throw NotImplemented("createVerticalBoundaryAtLine(line_nr_axis0)");
}

BoundaryNodeSet RectangularMeshBase2D::createVerticalBoundaryAtLine(std::size_t PLASK_UNUSED(line_nr_axis0), std::size_t PLASK_UNUSED(indexBegin), std::size_t PLASK_UNUSED(indexEnd)) const {
    throw NotImplemented("createVerticalBoundaryAtLine(line_nr_axis0, indexBegin, indexEnd)");
}

BoundaryNodeSet RectangularMeshBase2D::createVerticalBoundaryNear(double PLASK_UNUSED(axis0_coord)) const {
    throw NotImplemented("createVerticalBoundaryNear(axis0_coord)");
}

BoundaryNodeSet RectangularMeshBase2D::createVerticalBoundaryNear(double PLASK_UNUSED(axis0_coord), double PLASK_UNUSED(from), double PLASK_UNUSED(to)) const {
    throw NotImplemented("createVerticalBoundaryNear(axis0_coord, from, to)");
}

BoundaryNodeSet RectangularMeshBase2D::createLeftBoundary() const {
    throw NotImplemented("createLeftBoundary()");
}

BoundaryNodeSet RectangularMeshBase2D::createRightBoundary() const {
    throw NotImplemented("createRightBoundary()");
}

BoundaryNodeSet RectangularMeshBase2D::createLeftOfBoundary(const Box2D &PLASK_UNUSED(box)) const {
    throw NotImplemented("createLeftOfBoundary(box)");
}

BoundaryNodeSet RectangularMeshBase2D::createRightOfBoundary(const Box2D &PLASK_UNUSED(box)) const {
    throw NotImplemented("createRightOfBoundary(box)");
}

BoundaryNodeSet RectangularMeshBase2D::createBottomOfBoundary(const Box2D &PLASK_UNUSED(box)) const {
    throw NotImplemented("createBottomOfBoundary(box)");
}

BoundaryNodeSet RectangularMeshBase2D::createTopOfBoundary(const Box2D &PLASK_UNUSED(box)) const {
    throw NotImplemented("createTopOfBoundary(box)");
}

BoundaryNodeSet RectangularMeshBase2D::createHorizontalBoundaryAtLine(std::size_t PLASK_UNUSED(line_nr_axis1)) const {
    throw NotImplemented("createHorizontalBoundaryAtLine(line_nr_axis1)");
}

BoundaryNodeSet RectangularMeshBase2D::createHorizontalBoundaryAtLine(std::size_t PLASK_UNUSED(line_nr_axis1), std::size_t PLASK_UNUSED(indexBegin), std::size_t PLASK_UNUSED(indexEnd)) const {
    throw NotImplemented("createHorizontalBoundaryAtLine(line_nr_axis1, indexBegin, indexEnd)");
}

BoundaryNodeSet RectangularMeshBase2D::createHorizontalBoundaryNear(double PLASK_UNUSED(axis1_coord)) const {
    throw NotImplemented("createHorizontalBoundaryNear(axis1_coord)");
}

BoundaryNodeSet RectangularMeshBase2D::createHorizontalBoundaryNear(double PLASK_UNUSED(axis1_coord), double PLASK_UNUSED(from), double PLASK_UNUSED(to)) const {
    throw NotImplemented("createHorizontalBoundaryNear(axis1_coord, from, to)");
}

BoundaryNodeSet RectangularMeshBase2D::createTopBoundary() const {
    throw NotImplemented("createTopBoundary()");
}

BoundaryNodeSet RectangularMeshBase2D::createBottomBoundary() const {
    throw NotImplemented("createBottomBoundary()");
}

RectangularMeshBase2D::Boundary RectangularMeshBase2D::getBoundary(const std::string &boundary_desc) {
    if (boundary_desc == "bottom") return getBottomBoundary();
    if (boundary_desc == "left") return getLeftBoundary();
    if (boundary_desc == "right") return getRightBoundary();
    if (boundary_desc == "top") return getTopBoundary();
    return Boundary();
}

RectangularMeshBase2D::Boundary RectangularMeshBase2D::getBoundary(XMLReader &boundary_desc, Manager &manager) {
    auto side = boundary_desc.getAttribute("side");
    auto line = boundary_desc.getAttribute("line");
    if (side && line) {
        throw XMLConflictingAttributesException(boundary_desc, "side", "line");
    } else if (side) {
        if (*side == "bottom")
            return details::parseBoundaryFromXML<Boundary, 2>(boundary_desc, manager, &getBottomBoundary, &getBottomOfBoundary);
        if (*side == "left")
            return details::parseBoundaryFromXML<Boundary, 2>(boundary_desc, manager, &getLeftBoundary, &getLeftOfBoundary);
        if (*side == "right")
            return details::parseBoundaryFromXML<Boundary, 2>(boundary_desc, manager, &getRightBoundary, &getRightOfBoundary);
        if (*side == "top")
            return details::parseBoundaryFromXML<Boundary, 2>(boundary_desc, manager, &getTopBoundary, &getTopOfBoundary);
        throw XMLBadAttrException(boundary_desc, "side", *side);
    } else if (line) {
        double at = boundary_desc.requireAttribute<double>("at"),
                start = boundary_desc.requireAttribute<double>("start"),
                stop = boundary_desc.requireAttribute<double>("stop");
        boundary_desc.requireTagEnd();
        if (*line == "vertical")
            return getVerticalBoundaryNear(at, start, stop);
        if (*line == "horizontal")
            return getHorizontalBoundaryNear(at, start, stop);
        throw XMLBadAttrException(boundary_desc, "line", *line);
    }
    return Boundary();
}


}   // namespace plask
