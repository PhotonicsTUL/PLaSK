#include "rectangular3d.h"

#include "regular1d.h"
#include "rectilinear1d.h"

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
    return make_shared<RectangularMesh<3>>(axis0->getMidpointsMesh(), axis1->getMidpointsMesh(), axis2->getMidpointsMesh(), getIterationOrder());
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

RectangularMesh<3>::~RectangularMesh() {
    unsetChangeSignal(axis0);
    unsetChangeSignal(axis1);
    unsetChangeSignal(axis2);
}

void RectangularMesh<3>::writeXML(XMLElement& object) const {
    object.attr("type", "rectangular3d");
    { auto a = object.addTag("axis0"); axis0->writeXML(a); }
    { auto a = object.addTag("axis1"); axis1->writeXML(a); }
    { auto a = object.addTag("axis2"); axis2->writeXML(a); }
}

// Particular instantations
template class RectangularMesh<3>;

shared_ptr<RectangularMesh<3> > make_rectilinear_mesh(const RectangularMesh<3> &to_copy) {
   return make_shared<RectangularMesh<3>>(make_shared<RectilinearAxis>(*to_copy.axis0), make_shared<RectilinearAxis>(*to_copy.axis1), make_shared<RectilinearAxis>(*to_copy.axis2), to_copy.getIterationOrder());
}

static shared_ptr<Mesh> readRectangularMesh3D(XMLReader& reader) {
    shared_ptr<RectangularAxis> axis[3];
    XMLReader::CheckTagDuplication dub_check;
    for (int i = 0; i < 3; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();
        if (node != "axis0" && node != "axis1" && node != "axis3") throw XMLUnexpectedElementException(reader, "<axis0>, <axis1> or <axis2>");
        dub_check(std::string("<mesh>"), node);
        boost::optional<std::string> type = reader.getAttribute("type");
        if (type) {
            if (*type == "regular") axis[node[4]-'0'] = readRegularMesh1D(reader);
            else if (*type == "rectilinear") axis[node[4]-'0'] = readRectilinearMesh1D(reader);
            else throw XMLBadAttrException(reader, "type", *type, "\"regular\" or \"rectilinear\"");
        } else {
            if (reader.hasAttribute("start")) axis[node[4]-'0'] = readRegularMesh1D(reader);
            else axis[node[4]-'0'] = readRectilinearMesh1D(reader);
        }
    }
    reader.requireTagEnd();
    return make_shared<RectangularMesh<3>>(std::move(axis[0]), std::move(axis[1]), std::move(axis[2]));
}

static RegisterMeshReader rectangular3d_reader("rectangular3d", readRectangularMesh3D);

// deprecated:
static RegisterMeshReader regularmesh3d_reader("regular3d", readRectangularMesh3D);
static RegisterMeshReader rectilinear3d_reader("rectilinear3d", readRectangularMesh3D);

} // namespace plask
