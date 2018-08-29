#include "rectilinear3d.h"

#include "regular1d.h"
#include "ordered1d.h"

namespace plask {

#define RECTILINEAR_MESH_3D_DECLARE_ITERATION_ORDER(first, second, third) \
    static std::size_t index_##first##second##third(const RectilinearMesh3D* mesh, std::size_t index0, std::size_t index1, std::size_t index2) { \
        return index##third + mesh->axis[third]->size() * (index##second + mesh->axis[second]->size() * index##first); \
    } \
    static std::size_t index##first##_##first##second##third(const RectilinearMesh3D* mesh, std::size_t mesh_index) { \
        return mesh_index / mesh->axis[third]->size() / mesh->axis[second]->size(); \
    } \
    static std::size_t index##second##_##first##second##third(const RectilinearMesh3D* mesh, std::size_t mesh_index) { \
        return (mesh_index / mesh->axis[third]->size()) % mesh->axis[second]->size(); \
    } \
    static std::size_t index##third##_##first##second##third(const RectilinearMesh3D* mesh, std::size_t mesh_index) { \
        return mesh_index % mesh->axis[third]->size(); \
    }

RECTILINEAR_MESH_3D_DECLARE_ITERATION_ORDER(0,1,2)
RECTILINEAR_MESH_3D_DECLARE_ITERATION_ORDER(0,2,1)
RECTILINEAR_MESH_3D_DECLARE_ITERATION_ORDER(1,0,2)
RECTILINEAR_MESH_3D_DECLARE_ITERATION_ORDER(1,2,0)
RECTILINEAR_MESH_3D_DECLARE_ITERATION_ORDER(2,0,1)
RECTILINEAR_MESH_3D_DECLARE_ITERATION_ORDER(2,1,0)


void RectilinearMesh3D::setIterationOrder(IterationOrder iterationOrder) {
#   define RECTILINEAR_MESH_3D_CASE_ITERATION_ORDER(o1,o2,o3) \
        case ORDER_##o1##o2##o3: \
            index_f = index_##o1##o2##o3; index0_f = index0_##o1##o2##o3; \
            index1_f = index1_##o1##o2##o3; index2_f = index2_##o1##o2##o3; \
            major_axis = &axis[o1]; medium_axis = &axis[o2]; minor_axis = &axis[o3]; \
            break;
    switch (iterationOrder) {
        RECTILINEAR_MESH_3D_CASE_ITERATION_ORDER(0,1,2)
        RECTILINEAR_MESH_3D_CASE_ITERATION_ORDER(0,2,1)
        RECTILINEAR_MESH_3D_CASE_ITERATION_ORDER(1,0,2)
        RECTILINEAR_MESH_3D_CASE_ITERATION_ORDER(1,2,0)
        RECTILINEAR_MESH_3D_CASE_ITERATION_ORDER(2,0,1)
        RECTILINEAR_MESH_3D_CASE_ITERATION_ORDER(2,1,0)
        default:
            index_f = index_210; index0_f = index0_210;  index1_f = index1_210; index2_f = index2_210;
            major_axis = &axis[2]; medium_axis = &axis[1]; minor_axis = &axis[0];
    }
    fireChanged();
}


typename RectilinearMesh3D::IterationOrder RectilinearMesh3D::getIterationOrder() const {
    return this->index_f == decltype(this->index_f)(index_012) ? ORDER_012 :
           this->index_f == decltype(this->index_f)(index_021) ? ORDER_021 :
           this->index_f == decltype(this->index_f)(index_102) ? ORDER_102 :
           this->index_f == decltype(this->index_f)(index_120) ? ORDER_120 :
           this->index_f == decltype(this->index_f)(index_201) ? ORDER_201 :
                                                                 ORDER_210;
}

void RectilinearMesh3D::setOptimalIterationOrder() {
#   define RECTILINEAR_MESH_3D_DETERMINE_ITERATION_ORDER(first, second, third) \
        if (this->axis[third]->size() <= this->axis[second]->size() && this->axis[second]->size() <= this->axis[first]->size()) { \
            setIterationOrder(ORDER_##first##second##third); return; \
        }
    RECTILINEAR_MESH_3D_DETERMINE_ITERATION_ORDER(0,1,2)
    RECTILINEAR_MESH_3D_DETERMINE_ITERATION_ORDER(0,2,1)
    RECTILINEAR_MESH_3D_DETERMINE_ITERATION_ORDER(1,0,2)
    RECTILINEAR_MESH_3D_DETERMINE_ITERATION_ORDER(1,2,0)
    RECTILINEAR_MESH_3D_DETERMINE_ITERATION_ORDER(2,0,1)
    RECTILINEAR_MESH_3D_DETERMINE_ITERATION_ORDER(2,1,0)
}

void RectilinearMesh3D::onAxisChanged(Mesh::Event &e) {
    assert(!e.isDelete());
    fireChanged(e.flags());
}

RectilinearMesh3D::RectilinearMesh3D(IterationOrder iterationOrder)
    : axis{plask::make_shared<OrderedAxis>(), plask::make_shared<OrderedAxis>(), plask::make_shared<OrderedAxis>()} {
    setIterationOrder(iterationOrder);
    setChangeSignal(this->axis[0]);
    setChangeSignal(this->axis[1]);
    setChangeSignal(this->axis[2]);
}

RectilinearMesh3D::RectilinearMesh3D(shared_ptr<MeshAxis> mesh0, shared_ptr<MeshAxis> mesh1, shared_ptr<MeshAxis> mesh2, IterationOrder iterationOrder)
    : axis{std::move(mesh0), std::move(mesh1), std::move(mesh2)}
{
    setIterationOrder(iterationOrder);
    setChangeSignal(this->axis[0]);
    setChangeSignal(this->axis[1]);
    setChangeSignal(this->axis[2]);
}

void RectilinearMesh3D::reset(shared_ptr<MeshAxis> axis0, shared_ptr<MeshAxis> axis1, shared_ptr<MeshAxis> axis2, RectilinearMesh3D::IterationOrder iterationOrder) {
    setAxis(0, std::move(axis0), false);
    setAxis(1, std::move(axis1), false);
    setAxis(2, std::move(axis2), false);
    setIterationOrder(iterationOrder);
}

RectilinearMesh3D::RectilinearMesh3D(const RectilinearMesh3D &src, bool clone_axes)
: RectangularMeshBase3D(src),
  axis{clone_axes ? src.axis[0]->clone() : src.axis[0],
       clone_axes ? src.axis[1]->clone() : src.axis[1],
       clone_axes ? src.axis[2]->clone() : src.axis[2]}
{
    setIterationOrder(src.getIterationOrder());
    setChangeSignal(this->axis[0]);
    setChangeSignal(this->axis[1]);
    setChangeSignal(this->axis[2]);
}

void RectilinearMesh3D::reset(const RectilinearMesh3D &src, bool clone_axes)
{
    if (clone_axes)
        reset(src.axis[0]->clone(), src.axis[1]->clone(), src.axis[2]->clone(), src.getIterationOrder());
    else
        reset(src.axis[0], src.axis[1], src.axis[2], src.getIterationOrder());
}

RectilinearMesh3D::~RectilinearMesh3D() {
    unsetChangeSignal(this->axis[0]);
    unsetChangeSignal(this->axis[1]);
    unsetChangeSignal(this->axis[2]);
}

void RectilinearMesh3D::setAxis(std::size_t axis_nr, shared_ptr<MeshAxis> new_val, bool fireResized) {
    if (axis[axis_nr] == new_val) return;
    unsetChangeSignal(axis[axis_nr]);
    const_cast<shared_ptr<MeshAxis>&>(axis[axis_nr]) = new_val;
    setChangeSignal(axis[axis_nr]);
    if (fireResized) this->fireResized();
}

BoundaryNodeSet RectilinearMesh3D::createIndex0BoundaryAtLine(std::size_t line_nr_axis0) const {
    return new FixedIndex0Boundary(*this, line_nr_axis0);
}

BoundaryNodeSet RectilinearMesh3D::createBackBoundary() const {
    return createIndex0BoundaryAtLine(0);
}

BoundaryNodeSet RectilinearMesh3D::createFrontBoundary() const {
    return createIndex0BoundaryAtLine(axis[0]->size()-1);
}

BoundaryNodeSet RectilinearMesh3D::createIndex1BoundaryAtLine(std::size_t line_nr_axis1) const {
    return new FixedIndex1Boundary(*this, line_nr_axis1);
}

BoundaryNodeSet RectilinearMesh3D::createLeftBoundary() const {
    return createIndex1BoundaryAtLine(0);
}

BoundaryNodeSet RectilinearMesh3D::createRightBoundary() const {
    return createIndex1BoundaryAtLine(axis[1]->size()-1);
}

BoundaryNodeSet RectilinearMesh3D::createIndex2BoundaryAtLine(std::size_t line_nr_axis2) const {
    return new FixedIndex2Boundary(*this, line_nr_axis2);
}

BoundaryNodeSet RectilinearMesh3D::createBottomBoundary() const {
    return createIndex2BoundaryAtLine(0);
}

BoundaryNodeSet RectilinearMesh3D::createTopBoundary() const {
    return createIndex2BoundaryAtLine(axis[2]->size()-1);
}

BoundaryNodeSet RectilinearMesh3D::createIndex0BoundaryAtLine(std::size_t line_nr_axis0, std::size_t index1Begin, std::size_t index1End, std::size_t index2Begin, std::size_t index2End) const
{
    if (index1Begin < index1End && index2Begin < index2End)
        return new FixedIndex0BoundaryInRange(*this, line_nr_axis0, index1Begin, index1End, index2Begin, index2End);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectilinearMesh3D::createBackOfBoundary(const Box3D &box) const {
    std::size_t line, begInd1, endInd1, begInd2, endInd2;
    if (details::getLineLo(line, *axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd2, endInd2, *axis[2], box.lower.c2, box.upper.c2))
        return new FixedIndex0BoundaryInRange(*this, line, begInd1, endInd1, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectilinearMesh3D::createFrontOfBoundary(const Box3D &box) const {
    std::size_t line, begInd1, endInd1, begInd2, endInd2;
    if (details::getLineHi(line, *axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd2, endInd2, *axis[2], box.lower.c2, box.upper.c2))
        return new FixedIndex0BoundaryInRange(*this, line, begInd1, endInd1, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectilinearMesh3D::createIndex1BoundaryAtLine(std::size_t line_nr_axis1, std::size_t index0Begin, std::size_t index0End, std::size_t index2Begin, std::size_t index2End) const
{
    if (index0Begin < index0End && index2Begin < index2End)
        return new FixedIndex1BoundaryInRange(*this, line_nr_axis1, index0Begin, index0End, index2Begin, index2End);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectilinearMesh3D::createLeftOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd2, endInd2;
    if (details::getLineLo(line, *axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd0, endInd0, *axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd2, endInd2, *axis[2], box.lower.c2, box.upper.c2))
        return new FixedIndex1BoundaryInRange(*this, line, begInd0, endInd0, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectilinearMesh3D::createRightOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd2, endInd2;
    if (details::getLineHi(line, *axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd0, endInd0, *axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd2, endInd2, *axis[2], box.lower.c2, box.upper.c2))
        return new FixedIndex1BoundaryInRange(*this, line, begInd0, endInd0, begInd2, endInd2);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectilinearMesh3D::createIndex2BoundaryAtLine(std::size_t line_nr_axis2, std::size_t index0Begin, std::size_t index0End, std::size_t index1Begin, std::size_t index1End) const
{
    if (index0Begin < index0End && index1Begin < index1End)
        return new FixedIndex2BoundaryInRange(*this, line_nr_axis2, index0Begin, index0End, index1Begin, index1End);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectilinearMesh3D::createBottomOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd1, endInd1;
    if (details::getLineLo(line, *axis[2], box.lower.c2, box.upper.c2) &&
            details::getIndexesInBounds(begInd0, endInd0, *axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *axis[1], box.lower.c1, box.upper.c1))
        return new FixedIndex2BoundaryInRange(*this, line, begInd0, endInd0, begInd1, endInd1);
    else
        return new EmptyBoundaryImpl();
}

BoundaryNodeSet RectilinearMesh3D::createTopOfBoundary(const Box3D &box) const {
    std::size_t line, begInd0, endInd0, begInd1, endInd1;
    if (details::getLineHi(line, *axis[2], box.lower.c2, box.upper.c2) &&
            details::getIndexesInBounds(begInd0, endInd0, *axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd1, endInd1, *axis[1], box.lower.c1, box.upper.c1))
        return new FixedIndex2BoundaryInRange(*this, line, begInd0, endInd0, begInd1, endInd1);
    else
        return new EmptyBoundaryImpl();
}

} // namespace plask




