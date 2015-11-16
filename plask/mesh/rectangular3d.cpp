#include "rectangular3d.h"

#include "regular1d.h"
#include "ordered1d.h"

namespace plask {

#define RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(first, second, third) \
    static std::size_t index_##first##second##third(const RectangularMesh<3>* mesh, std::size_t index0, std::size_t index1, std::size_t index2) { \
        return index##third + mesh->axis##third->size() * (index##second + mesh->axis##second->size() * index##first); \
    } \
    static std::size_t index##first##_##first##second##third(const RectangularMesh<3>* mesh, std::size_t mesh_index) { \
        return mesh_index / mesh->axis##third->size() / mesh->axis##second->size(); \
    } \
    static std::size_t index##second##_##first##second##third(const RectangularMesh<3>* mesh, std::size_t mesh_index) { \
        return (mesh_index / mesh->axis##third->size()) % mesh->axis##second->size(); \
    } \
    static std::size_t index##third##_##first##second##third(const RectangularMesh<3>* mesh, std::size_t mesh_index) { \
        return mesh_index % mesh->axis##third->size(); \
    }

RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(0,1,2)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(0,2,1)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(1,0,2)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(1,2,0)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(2,0,1)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(2,1,0)


void RectangularMesh<3>::setIterationOrder(IterationOrder iterationOrder) {
#   define RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(o1,o2,o3) \
        case ORDER_##o1##o2##o3: \
            index_f = index_##o1##o2##o3; index0_f = index0_##o1##o2##o3; \
            index1_f = index1_##o1##o2##o3; index2_f = index2_##o1##o2##o3; \
            major_axis = &axis##o1; medium_axis = &axis##o2; minor_axis = &axis##o3; \
            break;
    switch (iterationOrder) {
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(0,1,2)
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(0,2,1)
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(1,0,2)
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(1,2,0)
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(2,0,1)
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(2,1,0)
        default:
            index_f = index_210; index0_f = index0_210;  index1_f = index1_210; index2_f = index2_210;
            major_axis = &axis2; medium_axis = &axis1; minor_axis = &axis0;
    }
    this->fireChanged();
}


typename RectangularMesh<3>::IterationOrder RectangularMesh<3>::getIterationOrder() const {
    return this->index_f == decltype(this->index_f)(index_012) ? ORDER_012 :
           this->index_f == decltype(this->index_f)(index_021) ? ORDER_021 :
           this->index_f == decltype(this->index_f)(index_102) ? ORDER_102 :
           this->index_f == decltype(this->index_f)(index_120) ? ORDER_120 :
           this->index_f == decltype(this->index_f)(index_201) ? ORDER_201 :
                                                                 ORDER_210;
}

void RectangularMesh<3>::setOptimalIterationOrder() {
#   define RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(first, second, third) \
        if (this->axis##third->size() <= this->axis##second->size() && this->axis##second->size() <= this->axis##first->size()) { \
            setIterationOrder(ORDER_##first##second##third); return; \
        }
    RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(0,1,2)
    RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(0,2,1)
    RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(1,0,2)
    RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(1,2,0)
    RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(2,0,1)
    RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(2,1,0)
}

shared_ptr<RectangularMesh<3> > RectangularMesh<3>::getMidpointsMesh() {
    return plask::make_shared<RectangularMesh<3>>(axis0->getMidpointsMesh(), axis1->getMidpointsMesh(), axis2->getMidpointsMesh(), getIterationOrder());
}

void RectangularMesh<3>::setAxis(const shared_ptr<RectangularAxis> &axis, shared_ptr<RectangularAxis> new_val) {
    if (axis == new_val) return;
    unsetChangeSignal(axis);
    const_cast<shared_ptr<RectangularAxis>&>(axis) = new_val;
    setChangeSignal(axis);
    fireResized();
}

void RectangularMesh<3>::onAxisChanged(Mesh::Event &e) {
    assert(!e.isDelete());
    this->fireChanged(e.flags());
}

RectangularMesh<3>::RectangularMesh(IterationOrder iterationOrder)
    : axis0(plask::make_shared<OrderedAxis>()), axis1(plask::make_shared<OrderedAxis>()), axis2(plask::make_shared<OrderedAxis>()), elements(this) {
    setIterationOrder(iterationOrder);
    setChangeSignal(this->axis0);
    setChangeSignal(this->axis1);
    setChangeSignal(this->axis2);
}

RectangularMesh<3>::RectangularMesh(shared_ptr<RectangularAxis> mesh0, shared_ptr<RectangularAxis> mesh1, shared_ptr<RectangularAxis> mesh2, IterationOrder iterationOrder) :
    axis0(std::move(mesh0)),
    axis1(std::move(mesh1)),
    axis2(std::move(mesh2)),
    elements(this)
{
    setIterationOrder(iterationOrder);
    setChangeSignal(this->axis0);
    setChangeSignal(this->axis1);
    setChangeSignal(this->axis2);
}

RectangularMesh<3>::RectangularMesh(const RectangularMesh<3> &src): axis0(src.axis0), axis1(src.axis1), axis2(src.axis2), elements(this) {    //->clone()
    setIterationOrder(src.getIterationOrder());
    setChangeSignal(this->axis0);
    setChangeSignal(this->axis1);
    setChangeSignal(this->axis2);
}

RectangularMesh<3>::~RectangularMesh() {
    unsetChangeSignal(this->axis0);
    unsetChangeSignal(this->axis1);
    unsetChangeSignal(this->axis2);
}

void RectangularMesh<3>::writeXML(XMLElement& object) const {
    object.attr("type", "rectangular3d");
    { auto a = object.addTag("axis0"); axis0->writeXML(a); }
    { auto a = object.addTag("axis1"); axis1->writeXML(a); }
    { auto a = object.addTag("axis2"); axis2->writeXML(a); }
}

RectangularMesh<3>::Boundary RectangularMesh<3>::getBoundary(const std::string &boundary_desc) {
    if (boundary_desc == "back") return getBackBoundary();
    if (boundary_desc == "front") return getFrontBoundary();
    if (boundary_desc == "left") return getLeftBoundary();
    if (boundary_desc == "right") return getRightBoundary();
    if (boundary_desc == "bottom") return getBottomBoundary();
    if (boundary_desc == "top") return getTopBoundary();
    return Boundary();
}

RectangularMesh<3>::Boundary RectangularMesh<3>::getBoundary(plask::XMLReader &boundary_desc, plask::Manager &manager) {
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

shared_ptr<RectangularMesh<3> > make_rectilinear_mesh(const RectangularMesh<3> &to_copy) {
   return plask::make_shared<RectangularMesh<3>>(plask::make_shared<OrderedAxis>(*to_copy.axis0), plask::make_shared<OrderedAxis>(*to_copy.axis1), plask::make_shared<OrderedAxis>(*to_copy.axis2), to_copy.getIterationOrder());
}

static shared_ptr<Mesh> readRectangularMesh3D(XMLReader& reader) {
    shared_ptr<RectangularAxis> axis[3];
    XMLReader::CheckTagDuplication dub_check;
    for (int i = 0; i < 3; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();
        if (node != "axis0" && node != "axis1" && node != "axis2") throw XMLUnexpectedElementException(reader, "<axis0>, <axis1> or <axis2>");
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
    return plask::make_shared<RectangularMesh<3>>(std::move(axis[0]), std::move(axis[1]), std::move(axis[2]));
}

static RegisterMeshReader rectangular3d_reader("rectangular3d", readRectangularMesh3D);

// obsolete:
static shared_ptr<Mesh> readRectangularMesh3D_obsolete(XMLReader& reader) {
        writelog(LOG_WARNING, "Mesh type \"%1%\" is obsolete (will not work in future versions of PLaSK), use \"rectangular3d\" instead.", reader.requireAttribute("type"));
        return readRectangularMesh3D(reader);
}
static RegisterMeshReader regularmesh3d_reader("regular3d", readRectangularMesh3D_obsolete);
static RegisterMeshReader rectilinear3d_reader("rectilinear3d", readRectangularMesh3D_obsolete);

template class PLASK_API RectangularMesh<3>;

} // namespace plask




