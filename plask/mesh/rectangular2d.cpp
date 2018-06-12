#include "rectangular2d.h"

#include "regular1d.h"
#include "ordered1d.h"

namespace plask {

static std::size_t normal_index(const RectangularMesh2D* mesh, std::size_t index0, std::size_t index1) {
    return index0 + mesh->axis[0]->size() * index1;
}
static std::size_t normal_index0(const RectangularMesh2D* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->axis[0]->size();
}
static std::size_t normal_index1(const RectangularMesh2D* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->axis[0]->size();
}

static std::size_t transposed_index(const RectangularMesh2D* mesh, std::size_t index0, std::size_t index1) {
    return mesh->axis[1]->size() * index0 + index1;
}
static std::size_t transposed_index0(const RectangularMesh2D* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->axis[1]->size();
}
static std::size_t transposed_index1(const RectangularMesh2D* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->axis[1]->size();
}

void RectangularMesh2D::setIterationOrder(IterationOrder iterationOrder) {
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

typename RectangularMesh2D::IterationOrder RectangularMesh2D::getIterationOrder() const {
    return (index_f == &transposed_index)? ORDER_01 : ORDER_10;
}

void RectangularMesh2D::setAxis(const shared_ptr<MeshAxis> &axis, shared_ptr<MeshAxis> new_val) {
    if (axis == new_val) return;
    unsetChangeSignal(axis);
    const_cast<shared_ptr<MeshAxis>&>(axis) = new_val;
    setChangeSignal(axis);
    fireResized();
}

void RectangularMesh2D::onAxisChanged(Mesh::Event &e) {
    assert(!e.isDelete());
    this->fireChanged(e.flags());
}

RectangularMesh2D::RectangularMesh2D(IterationOrder iterationOrder)
    : axis{ plask::make_shared<OrderedAxis>(), plask::make_shared<OrderedAxis>() } {
    setIterationOrder(iterationOrder);
    setChangeSignal(this->axis[0]);
    setChangeSignal(this->axis[1]);
}

RectangularMesh2D::RectangularMesh2D(shared_ptr<MeshAxis> axis0, shared_ptr<MeshAxis> axis1, IterationOrder iterationOrder)
    : axis{ std::move(axis0), std::move(axis1) } {
    setIterationOrder(iterationOrder);
    setChangeSignal(this->axis[0]);
    setChangeSignal(this->axis[1]);
}

RectangularMesh2D::RectangularMesh2D(const RectangularMesh2D &src, bool clone_axes):
    RectangularMeshBase2D(src),
    axis {clone_axes ? src.axis[0]->clone() : src.axis[0],
          clone_axes ? src.axis[1]->clone() : src.axis[1]}
{
    setIterationOrder(src.getIterationOrder());
    setChangeSignal(this->axis[0]);
    setChangeSignal(this->axis[1]);
}

RectangularMesh2D::~RectangularMesh2D() {
    unsetChangeSignal(this->axis[0]);
    unsetChangeSignal(this->axis[1]);
}

shared_ptr<RectangularMesh2D > RectangularMesh2D::getMidpointsMesh() {
    return plask::make_shared<RectangularMesh2D>(axis[0]->getMidpointsMesh(), axis[1]->getMidpointsMesh(), getIterationOrder());
}

BoundaryNodeSet RectangularMesh2D::createVerticalBoundaryAtLine(std::size_t line_nr_axis0) const {
    return new VerticalBoundary(*this, line_nr_axis0);
}

BoundaryNodeSet RectangularMesh2D::createVerticalBoundaryAtLine(std::size_t line_nr_axis0, std::size_t indexBegin, std::size_t indexEnd) const {
    return new VerticalBoundaryInRange(*this, line_nr_axis0, indexBegin, indexEnd);
}

BoundaryNodeSet RectangularMesh2D::createVerticalBoundaryNear(double axis0_coord) const {
    return new VerticalBoundary(*this, axis[0]->findNearestIndex(axis0_coord));
}

BoundaryNodeSet RectangularMesh2D::createVerticalBoundaryNear(double axis0_coord, double from, double to) const {
    std::size_t begInd, endInd;
    if (!details::getIndexesInBoundsExt(begInd, endInd, *axis[1], from, to))
        return new EmptyBoundaryImpl();
    return new VerticalBoundaryInRange(*this, axis[0]->findNearestIndex(axis0_coord), begInd, endInd);
}

BoundaryNodeSet RectangularMesh2D::createLeftBoundary() const {
    return new VerticalBoundary(*this, 0);
}

BoundaryNodeSet RectangularMesh2D::createRightBoundary() const {
    return new VerticalBoundary(*this, axis[0]->size()-1);
}

BoundaryNodeSet RectangularMesh2D::createLeftOfBoundary(const Box2D &box) const {
    std::size_t line, begInd, endInd;
    if (details::getLineLo(line, *axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd, endInd, *axis[1], box.lower.c1, box.upper.c1))
        return new VerticalBoundaryInRange(*this, line, begInd, endInd);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularMesh2D::createRightOfBoundary(const Box2D &box) const {
    std::size_t line, begInd, endInd;
    if (details::getLineHi(line, *axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd, endInd, *axis[1], box.lower.c1, box.upper.c1))
        return new VerticalBoundaryInRange(*this, line, begInd, endInd);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularMesh2D::createBottomOfBoundary(const Box2D &box) const {
    std::size_t line, begInd, endInd;
    if (details::getLineLo(line, *axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd, endInd, *axis[0], box.lower.c0, box.upper.c0))
        return new HorizontalBoundaryInRange(*this, line, begInd, endInd);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularMesh2D::createTopOfBoundary(const Box2D &box) const {
    std::size_t line, begInd, endInd;
    if (details::getLineHi(line, *axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd, endInd, *axis[0], box.lower.c0, box.upper.c0))
        return new HorizontalBoundaryInRange(*this, line, begInd, endInd);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectangularMesh2D::createHorizontalBoundaryAtLine(std::size_t line_nr_axis1) const {
    return new HorizontalBoundary(*this, line_nr_axis1);
}

BoundaryNodeSet RectangularMesh2D::createHorizontalBoundaryAtLine(std::size_t line_nr_axis1, std::size_t indexBegin, std::size_t indexEnd) const {
    return new HorizontalBoundaryInRange(*this, line_nr_axis1, indexBegin, indexEnd);
}

BoundaryNodeSet RectangularMesh2D::createHorizontalBoundaryNear(double axis1_coord) const {
    return new HorizontalBoundary(*this, axis[1]->findNearestIndex(axis1_coord));
}

BoundaryNodeSet RectangularMesh2D::createHorizontalBoundaryNear(double axis1_coord, double from, double to) const {
    std::size_t begInd, endInd;
    if (!details::getIndexesInBoundsExt(begInd, endInd, *axis[0], from, to))
        return new EmptyBoundaryImpl();
    return new HorizontalBoundaryInRange(*this, axis[1]->findNearestIndex(axis1_coord), begInd, endInd);
}

BoundaryNodeSet RectangularMesh2D::createTopBoundary() const {
    return new HorizontalBoundary(*this, axis[1]->size()-1);
}

BoundaryNodeSet RectangularMesh2D::createBottomBoundary() const {
    return new HorizontalBoundary(*this, 0);
}

void RectangularMesh2D::writeXML(XMLElement& object) const {
    object.attr("type", "rectangular2d");
    { auto a = object.addTag("axis0"); axis[0]->writeXML(a); }
    { auto a = object.addTag("axis1"); axis[1]->writeXML(a); }
}

shared_ptr<RectangularMesh2D > make_rectangular_mesh(const RectangularMesh2D &to_copy) {
    return plask::make_shared<RectangularMesh2D>(plask::make_shared<OrderedAxis>(*to_copy.axis[0]), plask::make_shared<OrderedAxis>(*to_copy.axis[1]), to_copy.getIterationOrder());
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
    return plask::make_shared<RectangularMesh2D>(std::move(axis[0]), std::move(axis[1]));
}

static RegisterMeshReader rectangular2d_reader("rectangular2d", readRectangularMesh2D);

// obsolete:
static shared_ptr<Mesh> readRectangularMesh2D_obsolete(XMLReader& reader) {
        writelog(LOG_WARNING, "Mesh type \"{0}\" is obsolete (will not work in future versions of PLaSK), use \"rectangular2d\" instead.", reader.requireAttribute("type"));
        return readRectangularMesh2D(reader);
}
static RegisterMeshReader regularmesh2d_reader("regular2d", readRectangularMesh2D_obsolete);
static RegisterMeshReader rectilinear2d_reader("rectilinear2d", readRectangularMesh2D_obsolete);

} // namespace plask









