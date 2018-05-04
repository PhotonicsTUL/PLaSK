#include "rectangular3d.h"

#include "regular1d.h"
#include "ordered1d.h"

namespace plask {


shared_ptr<RectangularMesh3D> RectangularMesh3D::getMidpointsMesh() {
    return plask::make_shared<RectangularMesh3D>(axis[0]->getMidpointsMesh(), axis[1]->getMidpointsMesh(), axis[2]->getMidpointsMesh(), getIterationOrder());
}

RectangularMesh3D::RectangularMesh3D(IterationOrder iterationOrder): RectilinearMesh3D(iterationOrder) {}

RectangularMesh3D::RectangularMesh3D(shared_ptr<MeshAxis> mesh0, shared_ptr<MeshAxis> mesh1, shared_ptr<MeshAxis> mesh2, IterationOrder iterationOrder):
    RectilinearMesh3D(std::move(mesh0), std::move(mesh1), std::move(mesh2), iterationOrder) {}

RectangularMesh3D::RectangularMesh3D(const RectangularMesh3D& src, bool clone_axes): RectilinearMesh3D(src, clone_axes) {}

void RectangularMesh3D::writeXML(XMLElement& object) const {
    object.attr("type", "rectangular3d");
    { auto a = object.addTag("axis0"); axis[0]->writeXML(a); }
    { auto a = object.addTag("axis1"); axis[1]->writeXML(a); }
    { auto a = object.addTag("axis2"); axis[2]->writeXML(a); }
}

RectangularMesh3D::Boundary RectangularMesh3D::getBoundary(const std::string &boundary_desc) {
    if (boundary_desc == "back") return getBackBoundary();
    if (boundary_desc == "front") return getFrontBoundary();
    if (boundary_desc == "left") return getLeftBoundary();
    if (boundary_desc == "right") return getRightBoundary();
    if (boundary_desc == "bottom") return getBottomBoundary();
    if (boundary_desc == "top") return getTopBoundary();
    return Boundary();
}

RectangularMesh3D::Boundary RectangularMesh3D::getBoundary(plask::XMLReader &boundary_desc, plask::Manager &manager) {
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

shared_ptr<RectangularMesh3D> make_rectangular_mesh(const RectangularMesh3D &to_copy) {
    return plask::make_shared<RectangularMesh3D>(
        plask::make_shared<OrderedAxis>(*to_copy.axis[0]),
        plask::make_shared<OrderedAxis>(*to_copy.axis[1]),
        plask::make_shared<OrderedAxis>(*to_copy.axis[2]),
        to_copy.getIterationOrder()
    );
}

static shared_ptr<Mesh> readRectangularMesh3D(XMLReader& reader) {
    shared_ptr<MeshAxis> axis[3];
    XMLReader::CheckTagDuplication dub_check;
    for (int i = 0; i < 3; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();
        if (node != "axis0" && node != "axis1" && node != "axis2") throw XMLUnexpectedElementException(reader, "<axis0>, <axis1> or <axis2>");
        dub_check(std::string("<mesh>"), node);
        plask::optional<std::string> type = reader.getAttribute("type");
        if (type) {
            if (*type == "regular") axis[node[4]-'0'] = readRegularMeshAxis(reader);
            else if (*type == "ordered") axis[node[4]-'0'] = readRectilinearMeshAxis(reader);
            else throw XMLBadAttrException(reader, "type", *type, "\"regular\" or \"ordered\"");
        } else {
            if (reader.hasAttribute("start")) axis[node[4]-'0'] = readRegularMeshAxis(reader);
            else axis[node[4]-'0'] = readRectilinearMeshAxis(reader);
        }
    }
    reader.requireTagEnd();
    return plask::make_shared<RectangularMesh3D>(std::move(axis[0]), std::move(axis[1]), std::move(axis[2]));
}

static RegisterMeshReader rectangular3d_reader("rectangular3d", readRectangularMesh3D);

} // namespace plask




