#include "rectangular2d.h"

#include "regular1d.h"
#include "rectilinear1d.h"

namespace plask {

static std::size_t normal_index(const RectangularMesh<2>* mesh, std::size_t index0, std::size_t index1) {
    return index0 + mesh->axis0->size() * index1;
}
static std::size_t normal_index0(const RectangularMesh<2>* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->axis0->size();
}
static std::size_t normal_index1(const RectangularMesh<2>* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->axis0->size();
}

static std::size_t transposed_index(const RectangularMesh<2>* mesh, std::size_t index0, std::size_t index1) {
    return mesh->axis1->size() * index0 + index1;
}
static std::size_t transposed_index0(const RectangularMesh<2>* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->axis1->size();
}
static std::size_t transposed_index1(const RectangularMesh<2>* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->axis1->size();
}

void RectangularMesh<2>::setIterationOrder(IterationOrder iterationOrder) {
    if (iterationOrder == ORDER_01) {
        index_f = transposed_index;
        index0_f = transposed_index0;
        index1_f = transposed_index1;
        minor_axis = &axis1;
        major_axis = &axis0;
    } else {
        index_f = normal_index;
        index0_f = normal_index0;
        index1_f = normal_index1;
        minor_axis = &axis0;
        major_axis = &axis1;
    }
    this->fireChanged();
}

typename RectangularMesh<2>::IterationOrder RectangularMesh<2>::getIterationOrder() const {
    return (index_f == &transposed_index)? ORDER_01 : ORDER_10;
}

void RectangularMesh<2>::setAxis(const shared_ptr<RectangularAxis> &axis, shared_ptr<RectangularAxis> new_val) {
    if (axis == new_val) return;
    unsetChangeSignal(axis);
    const_cast<shared_ptr<RectangularAxis>&>(axis) = new_val;
    setChangeSignal(axis);
    fireResized();
}

void RectangularMesh<2>::onAxisChanged(Mesh::Event &e) {
    assert(!e.isDelete());
    this->fireChanged(e.flags());
}

RectangularMesh<2>::~RectangularMesh() {
    unsetChangeSignal(axis0);
    unsetChangeSignal(axis1);
}

shared_ptr<RectangularMesh<2> > RectangularMesh<2>::getMidpointsMesh() {
    return make_shared<RectangularMesh<2>>(axis0->getMidpointsMesh(), axis1->getMidpointsMesh(), getIterationOrder());
}

void RectangularMesh<2>::writeXML(XMLElement& object) const {
    object.attr("type", "rectangular2d");
    { auto a = object.addTag("axis0"); axis0->writeXML(a); }
    { auto a = object.addTag("axis1"); axis1->writeXML(a); }
}


// Particular instantations
template class RectangularMesh<2>;

shared_ptr<RectangularMesh<2> > make_rectilinear_mesh(const RectangularMesh<2> &to_copy) {
   return make_shared<RectangularMesh<2>>(make_shared<RectilinearAxis>(*to_copy.axis0), make_shared<RectilinearAxis>(*to_copy.axis1), to_copy.getIterationOrder());
}

static shared_ptr<Mesh> readRectangularMesh2D(XMLReader& reader) {
    shared_ptr<RectangularAxis> axis[2];
    XMLReader::CheckTagDuplication dub_check;
    for (int i = 0; i < 2; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();
        if (node != "axis0" && node != "axis1") throw XMLUnexpectedElementException(reader, "<axis0> or <axis1>");
        dub_check(std::string("<mesh>"), node);
        boost::optional<std::string> type = reader.getAttribute("type");
        if (type) {
            if (*type == "regular") axis[node[4]-'0'] = readRegularMeshAxis(reader);
            else if (*type == "rectilinear") axis[node[4]-'0'] = readRectilinearMeshAxis(reader);
            else throw XMLBadAttrException(reader, "type", *type, "\"regular\" or \"rectilinear\"");
        } else {
            if (reader.hasAttribute("start")) axis[node[4]-'0'] = readRegularMeshAxis(reader);
            else axis[node[4]-'0'] = readRectilinearMeshAxis(reader);
        }
    }
    reader.requireTagEnd();
    return make_shared<RectangularMesh<2>>(std::move(axis[0]), std::move(axis[1]));
}


static RegisterMeshReader rectangular2d_reader("rectangular2d", readRectangularMesh2D);

// deprecated:
static shared_ptr<Mesh> readRectangularMesh2D_deprecated(XMLReader& reader) {
        writelog(LOG_WARNING, "Mesh type \"%1%\" is deprecated (will not work in future versions of PLaSK), use \"rectangular2d\" instead.", reader.requireAttribute("type"));
        return readRectangularMesh2D(reader);
}
static RegisterMeshReader regularmesh2d_reader("regular2d", readRectangularMesh2D_deprecated);
static RegisterMeshReader rectilinear2d_reader("rectilinear2d", readRectangularMesh2D_deprecated);


} // namespace plask



