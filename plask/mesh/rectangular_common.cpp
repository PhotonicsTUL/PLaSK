#include "rectangular_common.h"

namespace plask {

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



RectangularMeshBase3D::Boundary RectangularMeshBase3D::getBoundary(const std::string &boundary_desc) {
    if (boundary_desc == "back") return getBackBoundary();
    if (boundary_desc == "front") return getFrontBoundary();
    if (boundary_desc == "left") return getLeftBoundary();
    if (boundary_desc == "right") return getRightBoundary();
    if (boundary_desc == "bottom") return getBottomBoundary();
    if (boundary_desc == "top") return getTopBoundary();
    return Boundary();
}

RectangularMeshBase3D::Boundary RectangularMeshBase3D::getBoundary(plask::XMLReader &boundary_desc, plask::Manager &manager) {
    auto side = boundary_desc.requireAttribute("side");
    /* auto side = boundary_desc.getAttribute("side");
        auto line = boundary_desc.getAttribute("line");
        if (side && line) {
            throw XMLConflictingAttributesException(boundary_desc, "size", "line");
        } else if (side)*/ {
        if (side == "back")
            return details::parseBoundaryFromXML<Boundary, 3>(boundary_desc, manager, &getBackBoundary, &getBackOfBoundary);
        if (side == "front")
            return details::parseBoundaryFromXML<Boundary, 3>(boundary_desc, manager, &getFrontBoundary, &getFrontOfBoundary);
        if (side == "left")
            return details::parseBoundaryFromXML<Boundary, 3>(boundary_desc, manager, &getLeftBoundary, &getLeftOfBoundary);
        if (side == "right")
            return details::parseBoundaryFromXML<Boundary, 3>(boundary_desc, manager, &getRightBoundary, &getRightOfBoundary);
        if (side == "bottom")
            return details::parseBoundaryFromXML<Boundary, 3>(boundary_desc, manager, &getBottomBoundary, &getBottomOfBoundary);
        if (side == "top")
            return details::parseBoundaryFromXML<Boundary, 3>(boundary_desc, manager, &getTopBoundary, &getTopOfBoundary);
        throw XMLBadAttrException(boundary_desc, "side", side);
    } /*else if (line) {
            double at = boundary_desc.requireAttribute<double>("at"),
                   start = boundary_desc.requireAttribute<double>("start"),
                   stop = boundary_desc.requireAttribute<double>("stop");
            boundary_desc.requireTagEnd();
            if (*line == "vertical")
                return getVerticalBoundaryNear(at, start, stop);
            if (*line == "horizontal")
                return getHorizontalBoundaryNear(at, start, stop);
            throw XMLBadAttrException(boundary_desc, "line", *line);
        }*/
    return Boundary();
}


}   // namespace plask
