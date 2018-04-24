#include "rectangular2d.h"

#include "regular1d.h"
#include "ordered1d.h"

namespace plask {

static std::size_t normal_index(const RectangularMesh<2>* mesh, std::size_t index0, std::size_t index1) {
    return index0 + mesh->axis[0]->size() * index1;
}
static std::size_t normal_index0(const RectangularMesh<2>* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->axis[0]->size();
}
static std::size_t normal_index1(const RectangularMesh<2>* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->axis[0]->size();
}

static std::size_t transposed_index(const RectangularMesh<2>* mesh, std::size_t index0, std::size_t index1) {
    return mesh->axis[1]->size() * index0 + index1;
}
static std::size_t transposed_index0(const RectangularMesh<2>* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->axis[1]->size();
}
static std::size_t transposed_index1(const RectangularMesh<2>* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->axis[1]->size();
}

void RectangularMesh<2>::setIterationOrder(IterationOrder iterationOrder) {
    if (iterationOrder == ORDER_01) {
        index_f = transposed_index;
        index0_f = transposed_index0;
        index1_f = transposed_index1;
        minor_axis = &axis[1];
        major_axis = &axis[0];
    } else {
        index_f = normal_index;
        index0_f = normal_index0;
        index1_f = normal_index1;
        minor_axis = &axis[0];
        major_axis = &axis[1];
    }
    this->fireChanged();
}

typename RectangularMesh<2>::IterationOrder RectangularMesh<2>::getIterationOrder() const {
    return (index_f == &transposed_index)? ORDER_01 : ORDER_10;
}

void RectangularMesh<2>::setAxis(const shared_ptr<MeshAxis> &axis, shared_ptr<MeshAxis> new_val) {
    if (axis == new_val) return;
    unsetChangeSignal(axis);
    const_cast<shared_ptr<MeshAxis>&>(axis) = new_val;
    setChangeSignal(axis);
    fireResized();
}

void RectangularMesh<2>::onAxisChanged(Mesh::Event &e) {
    assert(!e.isDelete());
    this->fireChanged(e.flags());
}

RectangularMesh<2>::RectangularMesh(IterationOrder iterationOrder)
    : axis{ plask::make_shared<OrderedAxis>(), plask::make_shared<OrderedAxis>() } {
    setIterationOrder(iterationOrder);
    setChangeSignal(this->axis[0]);
    setChangeSignal(this->axis[1]);
}

RectangularMesh<2>::RectangularMesh(shared_ptr<MeshAxis> axis0, shared_ptr<MeshAxis> axis1, IterationOrder iterationOrder)
    : axis{std::move(axis0), std::move(axis1)} {
    setIterationOrder(iterationOrder);
    setChangeSignal(this->axis[0]);
    setChangeSignal(this->axis[1]);
}

RectangularMesh<2>::RectangularMesh(const RectangularMesh<2> &src, bool clone_axes):
    MeshD<2>(src),
    axis {clone_axes ? src.axis[0]->clone() : src.axis[0],
          clone_axes ? src.axis[1]->clone() : src.axis[1]}
{
    setIterationOrder(src.getIterationOrder());
    setChangeSignal(this->axis[0]);
    setChangeSignal(this->axis[1]);
}

RectangularMesh<2>::~RectangularMesh() {
    unsetChangeSignal(this->axis[0]);
    unsetChangeSignal(this->axis[1]);
}

shared_ptr<RectangularMesh<2> > RectangularMesh<2>::getMidpointsMesh() {
    return plask::make_shared<RectangularMesh<2>>(axis[0]->getMidpointsMesh(), axis[1]->getMidpointsMesh(), getIterationOrder());
}

void RectangularMesh<2>::writeXML(XMLElement& object) const {
    object.attr("type", "rectangular2d");
    { auto a = object.addTag("axis0"); axis[0]->writeXML(a); }
    { auto a = object.addTag("axis1"); axis[1]->writeXML(a); }
}

RectangularMesh<2>::Boundary RectangularMesh<2>::getBoundary(const std::string &boundary_desc) {
    if (boundary_desc == "bottom") return getBottomBoundary();
    if (boundary_desc == "left") return getLeftBoundary();
    if (boundary_desc == "right") return getRightBoundary();
    if (boundary_desc == "top") return getTopBoundary();
    return Boundary();
}

RectangularMesh<2>::Boundary RectangularMesh<2>::getBoundary(XMLReader &boundary_desc, Manager &manager) {
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

shared_ptr<RectangularMesh<2> > make_rectangular_mesh(const RectangularMesh<2> &to_copy) {
   return plask::make_shared<RectangularMesh<2>>(plask::make_shared<OrderedAxis>(*to_copy.axis[0]), plask::make_shared<OrderedAxis>(*to_copy.axis[1]), to_copy.getIterationOrder());
}

static shared_ptr<Mesh> readRectangularMesh2D(XMLReader& reader) {
    shared_ptr<MeshAxis> axis[2];
    XMLReader::CheckTagDuplication dub_check;
    for (int i = 0; i < 2; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();
        if (node != "axis0" && node != "axis1") throw XMLUnexpectedElementException(reader, "<axis0> or <axis1>");
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
    return plask::make_shared<RectangularMesh<2>>(std::move(axis[0]), std::move(axis[1]));
}

static RegisterMeshReader rectangular2d_reader("rectangular2d", readRectangularMesh2D);

// obsolete:
static shared_ptr<Mesh> readRectangularMesh2D_obsolete(XMLReader& reader) {
        writelog(LOG_WARNING, "Mesh type \"{0}\" is obsolete (will not work in future versions of PLaSK), use \"rectangular2d\" instead.", reader.requireAttribute("type"));
        return readRectangularMesh2D(reader);
}
static RegisterMeshReader regularmesh2d_reader("regular2d", readRectangularMesh2D_obsolete);
static RegisterMeshReader rectilinear2d_reader("rectilinear2d", readRectangularMesh2D_obsolete);

template class PLASK_API RectangularMesh<2>;

} // namespace plask









