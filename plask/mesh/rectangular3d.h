#ifndef PLASK__RECTANGULAR3D_H
#define PLASK__RECTANGULAR3D_H

/** @file
This file contains rectangular mesh for 3D space.
*/

#include "rectangular2d.h"
#include "rectilinear3d.h"

namespace plask {

/**
 * Rectangular mesh in 3D space.
 *
 * Includes three 1d rectilinear meshes:
 * - axis0 (alternative names: lon(), ee_z(), rad_r())
 * - axis1 (alternative names: tran(), ee_x(), rad_phi())
 * - axis2 (alternative names: vert(), ee_y(), rad_z())
 * Represent all points (x, y, z) such that x is in axis0, y is in axis1, z is in axis2.
 */
template<>
class PLASK_API RectangularMesh<3>: public RectilinearMesh3D {

  public:

    /// Boundary type.
    typedef ::plask::Boundary<RectangularMesh<3>> Boundary;

    /**
     * Construct mesh which has all axes of type OrderedAxis and all are empty.
     * @param iterationOrder iteration order
     */
    explicit RectangularMesh(IterationOrder iterationOrder = ORDER_012);

    /**
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     * @param mesh2 mesh for the third coordinate
     * @param iterationOrder iteration order
     */
    RectangularMesh(shared_ptr<MeshAxis> mesh0, shared_ptr<MeshAxis> mesh1, shared_ptr<MeshAxis> mesh2, IterationOrder iterationOrder = ORDER_012);

    /// Copy constructor
    RectangularMesh(const RectangularMesh<3>& src);

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<MeshAxis>& lon() const { return axis0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<MeshAxis>& tran() const { return axis1; }

    /**
     * Get third coordinate of points in this mesh.
     * @return axis2
     */
    const shared_ptr<MeshAxis>& vert() const { return axis2; }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<MeshAxis>& ee_z() const { return axis0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<MeshAxis>& ee_x() const { return axis1; }

    /**
     * Get third coordinate of points in this mesh.
     * @return axis2
     */
    const shared_ptr<MeshAxis>& ee_y() const { return axis2; }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<MeshAxis>& rad_r() const { return axis0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<MeshAxis>& rad_phi() const { return axis1; }

    /**
     * Get thirs coordinate of points in this mesh.
     * @return axis2
     */
    const shared_ptr<MeshAxis>& rad_z() const { return axis2; }

    /**
     * Write mesh to XML
     * \param object XML object to write to
     */
    void writeXML(XMLElement& object) const override;


    /**
     * Get point with given mesh indices.
     * @param index0 index of point in axis0
     * @param index1 index of point in axis1
     * @param index2 index of point in axis2
     * @return point with given @p index
     */
    Vec<3, double> at(std::size_t index0, std::size_t index1, std::size_t index2) const override {
        return Vec<3, double>(axis0->at(index0), axis1->at(index1), axis2->at(index2));
    }

    /**
     * Return a mesh that enables iterating over middle points of the cuboids
     * \return new rectangular mesh with points in the middles of original cuboids
     */
    shared_ptr<RectangularMesh> getMidpointsMesh();

    /**
     * Get area of given element.
     * @param index0, index1, index2 axis 0, 1 and 2 indexes of element
     * @return area of elements with given index
     */
    double getElementArea(std::size_t index0, std::size_t index1, std::size_t index2) const {
        return (axis0->at(index0+1) - axis0->at(index0)) * (axis1->at(index1+1) - axis1->at(index1)) * (axis2->at(index2+1) - axis2->at(index2));
    }

    /**
     * Get area of given element.
     * @param element_index index of element
     * @return area of elements with given index
     */
    double getElementArea(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return getElementArea(index0(bl_index), index1(bl_index), index2(bl_index));
    }

    /**
     * Get point in center of Elements.
     * @param index0, index1, index2 index of Elements
     * @return point in center of element with given index
     */
    Vec<3, double> getElementMidpoint(std::size_t index0, std::size_t index1, std::size_t index2) const override {
        return vec(getElementMidpoint0(index0), getElementMidpoint1(index1), getElementMidpoint2(index2));
    }

    /**
     * Get point in center of Elements.
     * @param element_index index of Elements
     * @return point in center of element with given index
     */
    Vec<3, double> getElementMidpoint(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return getElementMidpoint(index0(bl_index), index1(bl_index), index2(bl_index));
    }

    /**
     * Get element as cuboid.
     * @param index0, index1, index2 index of Elements
     * @return box of elements with given index
     */
    Box3D getElementBox(std::size_t index0, std::size_t index1, std::size_t index2) const {
        return Box3D(axis0->at(index0), axis1->at(index1), axis2->at(index2), axis0->at(index0+1), axis1->at(index1+1), axis2->at(index2+1));
    }

    /**
     * Get element as cuboid.
     * @param element_index index of element
     * @return box of elements with given index
     */
    Box3D getElementBox(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return getElementBox(index0(bl_index), index1(bl_index), index2(bl_index));
    }

  private:

    // Common code for: left, right, bottom, top, front, back boundaries:
    struct BoundaryIteratorImpl: public BoundaryLogicImpl::IteratorImpl {

        const RectangularMesh &mesh;

        const std::size_t level;

        std::size_t index_f, index_s;

        const std::size_t index_f_begin, index_f_end;

        BoundaryIteratorImpl(const RectangularMesh& mesh, std::size_t level,
                             std::size_t index_f, std::size_t index_f_begin, std::size_t index_f_end,
                             std::size_t index_s)
            : mesh(mesh), level(level), index_f(index_f), index_s(index_s), index_f_begin(index_f_begin), index_f_end(index_f_end) {
        }

        virtual void increment() override {
            ++index_f;
            if (index_f == index_f_end) {
                index_f = index_f_begin;
                ++index_s;
            }
        }

        virtual bool equal(const typename BoundaryLogicImpl::IteratorImpl& other) const override {
            return index_f == static_cast<const BoundaryIteratorImpl&>(other).index_f && index_s == static_cast<const BoundaryIteratorImpl&>(other).index_s;
        }

    };

    // iterator with fixed first coordinate
    struct FixedIndex0IteratorImpl: public BoundaryIteratorImpl {

        FixedIndex0IteratorImpl(const RectangularMesh& mesh, std::size_t level_index0, std::size_t index_1, std::size_t index_1_begin, std::size_t index_1_end, std::size_t index_2)
            : BoundaryIteratorImpl(mesh, level_index0, index_1, index_1_begin, index_1_end, index_2) {}

        virtual std::size_t dereference() const override { return this->mesh.index(this->level, this->index_f, this->index_s); }

        virtual typename BoundaryLogicImpl::IteratorImpl* clone() const override {
            return new FixedIndex0IteratorImpl(*this);
        }
    };

    // iterator with fixed second coordinate
    struct FixedIndex1IteratorImpl: public BoundaryIteratorImpl {

        FixedIndex1IteratorImpl(const RectangularMesh& mesh, std::size_t level_index1, std::size_t index_0, std::size_t index_0_begin, std::size_t index_0_end, std::size_t index_2)
            : BoundaryIteratorImpl(mesh, level_index1, index_0, index_0_begin, index_0_end, index_2) {}

        virtual std::size_t dereference() const override { return this->mesh.index(this->index_f, this->level, this->index_s); }

        virtual typename BoundaryLogicImpl::IteratorImpl* clone() const override {
            return new FixedIndex1IteratorImpl(*this);
        }
    };

    // iterator with fixed third coordinate
    struct FixedIndex2IteratorImpl: public BoundaryIteratorImpl {

        FixedIndex2IteratorImpl(const RectangularMesh& mesh, std::size_t level_index2, std::size_t index_0, std::size_t index_0_begin, std::size_t index_0_end, std::size_t index_1)
            : BoundaryIteratorImpl(mesh, level_index2, index_0, index_0_begin, index_0_end, index_1) {}

        virtual std::size_t dereference() const override { return this->mesh.index(this->index_f, this->index_s, this->level); }

        virtual typename BoundaryLogicImpl::IteratorImpl* clone() const override {
            return new FixedIndex2IteratorImpl(*this);
        }
    };

    struct FixedIndex0Boundary: public BoundaryWithMeshLogicImpl<RectangularMesh<3>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t level_axis0;

        FixedIndex0Boundary(const RectangularMesh<3>& mesh, std::size_t level_axis0): BoundaryWithMeshLogicImpl<RectangularMesh<3>>(mesh), level_axis0(level_axis0) {}

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index0(mesh_index) == level_axis0;
        }

        Iterator begin() const override {
            return Iterator(new FixedIndex0IteratorImpl(this->mesh, level_axis0, 0, 0, this->mesh.axis1->size(), 0));
        }

        Iterator end() const override {
            return Iterator(new FixedIndex0IteratorImpl(this->mesh, level_axis0, 0, 0, this->mesh.axis1->size(), this->mesh.axis2->size()));
        }

        std::size_t size() const override {
            return this->mesh.axis1->size() * this->mesh.axis2->size();
        }
    };

    struct FixedIndex0BoundaryInRange: public BoundaryWithMeshLogicImpl<RectangularMesh<3>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t level_axis0, beginAxis1, endAxis1, beginAxis2, endAxis2;

        FixedIndex0BoundaryInRange(const RectangularMesh<3>& mesh, std::size_t level_axis0, std::size_t beginAxis1, std::size_t endAxis1, std::size_t beginAxis2, std::size_t endAxis2)
            : BoundaryWithMeshLogicImpl<RectangularMesh<3>>(mesh), level_axis0(level_axis0),
              beginAxis1(beginAxis1), endAxis1(endAxis1), beginAxis2(beginAxis2), endAxis2(endAxis2)
              {}

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index0(mesh_index) == level_axis0
                    && in_range(this->mesh.index1(mesh_index), beginAxis1, endAxis1)
                    && in_range(this->mesh.index2(mesh_index), beginAxis2, endAxis2);
        }

        Iterator begin() const override {
            return Iterator(new FixedIndex0IteratorImpl(this->mesh, level_axis0, beginAxis1, beginAxis1, endAxis1, beginAxis2));
        }

        Iterator end() const override {
            return Iterator(new FixedIndex0IteratorImpl(this->mesh, level_axis0, beginAxis1, beginAxis1, endAxis1, endAxis2));
        }

        std::size_t size() const override {
            return (endAxis1 - beginAxis1) * (endAxis2 - beginAxis2);
        }

        bool empty() const override {
            return beginAxis1 == endAxis1 || beginAxis2 == endAxis2;
        }
    };

    struct FixedIndex1Boundary: public BoundaryWithMeshLogicImpl<RectangularMesh<3>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t level_axis1;

        FixedIndex1Boundary(const RectangularMesh<3>& mesh, std::size_t level_axis1): BoundaryWithMeshLogicImpl<RectangularMesh<3>>(mesh), level_axis1(level_axis1) {}

        //virtual LeftBoundary* clone() const { return new LeftBoundary(); }

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index1(mesh_index) == level_axis1;
        }

        Iterator begin() const override {
            return Iterator(new FixedIndex1IteratorImpl(this->mesh, level_axis1, 0, 0, this->mesh.axis0->size(), 0));
        }

        Iterator end() const override {
            return Iterator(new FixedIndex1IteratorImpl(this->mesh, level_axis1, 0, 0, this->mesh.axis0->size(), this->mesh.axis2->size()));
        }

        std::size_t size() const override {
            return this->mesh.axis0->size() * this->mesh.axis2->size();
        }
    };

    struct FixedIndex1BoundaryInRange: public BoundaryWithMeshLogicImpl<RectangularMesh<3>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t level_axis1, beginAxis0, endAxis0, beginAxis2, endAxis2;

        FixedIndex1BoundaryInRange(const RectangularMesh<3>& mesh, std::size_t level_axis1, std::size_t beginAxis0, std::size_t endAxis0, std::size_t beginAxis2, std::size_t endAxis2)
            : BoundaryWithMeshLogicImpl<RectangularMesh<3>>(mesh), level_axis1(level_axis1),
              beginAxis0(beginAxis0), endAxis0(endAxis0), beginAxis2(beginAxis2), endAxis2(endAxis2)
              {}

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index1(mesh_index) == level_axis1
                    && in_range(this->mesh.index0(mesh_index), beginAxis0, endAxis0)
                    && in_range(this->mesh.index2(mesh_index), beginAxis2, endAxis2);
        }

        Iterator begin() const override {
            return Iterator(new FixedIndex1IteratorImpl(this->mesh, level_axis1, beginAxis0, beginAxis0, endAxis0, beginAxis2));
        }

        Iterator end() const override {
            return Iterator(new FixedIndex1IteratorImpl(this->mesh, level_axis1, beginAxis0, beginAxis0, endAxis0, endAxis2));
        }

        std::size_t size() const override {
            return (endAxis0 - beginAxis0) * (endAxis2 - beginAxis2);
        }

        bool empty() const override {
            return beginAxis0 == endAxis0 || beginAxis2 == endAxis2;
        }
    };


    struct FixedIndex2Boundary: public BoundaryWithMeshLogicImpl<RectangularMesh<3>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t level_axis2;

        FixedIndex2Boundary(const RectangularMesh<3>& mesh, std::size_t level_axis2): BoundaryWithMeshLogicImpl<RectangularMesh<3>>(mesh), level_axis2(level_axis2) {}

        //virtual LeftBoundary* clone() const { return new LeftBoundary(); }

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index2(mesh_index) == level_axis2;
        }

        Iterator begin() const override {
            return Iterator(new FixedIndex2IteratorImpl(this->mesh, level_axis2, 0, 0, this->mesh.axis0->size(), 0));
        }

        Iterator end() const override {
            return Iterator(new FixedIndex2IteratorImpl(this->mesh, level_axis2, 0, 0, this->mesh.axis0->size(), this->mesh.axis1->size()));
        }

        std::size_t size() const override {
            return this->mesh.axis0->size() * this->mesh.axis1->size();
        }
    };

    struct FixedIndex2BoundaryInRange: public BoundaryWithMeshLogicImpl<RectangularMesh<3>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t level_axis2, beginAxis0, endAxis0, beginAxis1, endAxis1;

        FixedIndex2BoundaryInRange(const RectangularMesh<3>& mesh, std::size_t level_axis2, std::size_t beginAxis0, std::size_t endAxis0, std::size_t beginAxis1, std::size_t endAxis1)
            : BoundaryWithMeshLogicImpl<RectangularMesh<3>>(mesh), level_axis2(level_axis2),
              beginAxis0(beginAxis0), endAxis0(endAxis0), beginAxis1(beginAxis1), endAxis1(endAxis1)
              {
            }

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index2(mesh_index) == level_axis2
                    && in_range(this->mesh.index0(mesh_index), beginAxis0, endAxis0)
                    && in_range(this->mesh.index1(mesh_index), beginAxis1, endAxis1);
        }

        Iterator begin() const override {
            return Iterator(new FixedIndex2IteratorImpl(this->mesh, level_axis2, beginAxis0, beginAxis0, endAxis0, beginAxis1));
        }

        Iterator end() const override {
            return Iterator(new FixedIndex2IteratorImpl(this->mesh, level_axis2, beginAxis0, beginAxis0, endAxis0, endAxis1));
        }

        std::size_t size() const override {
            return (endAxis0 - beginAxis0) * (endAxis1 - beginAxis1);
        }

        bool empty() const override {
            return beginAxis0 == endAxis0 || beginAxis1 == endAxis1;
        }
    };


    public:
    /**
     * Get boundary which shows one plane in mesh, which has 0 coordinate equals to axis0[0] (back of mesh).
     * @return boundary which show plane in mesh
     */
    static Boundary getBackBoundary() {
        return Boundary( [](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) {
            return new FixedIndex0Boundary(mesh, 0);
        } );
    }

    /**
     * Get boundary which shows one plane in mesh, which has 0 coordinate equals to axis0[axis0->size()-1] (front of mesh).
     * @return boundary which show plane in mesh
     */
    static Boundary getFrontBoundary() {
        return Boundary( [](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) {
            return new FixedIndex0Boundary(mesh, mesh.axis0->size()-1);
        } );
    }

    /**
     * Get boundary which shows one plane in mesh, which has 0 coordinate equals to @p line_nr_axis0->
     * @param line_nr_axis0 index of axis0 mesh
     * @return boundary which show plane in mesh
     */
    static Boundary getIndex0BoundaryAtLine(std::size_t line_nr_axis0) {
        return Boundary( [line_nr_axis0](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) {
            return new FixedIndex0Boundary(mesh, line_nr_axis0);
        } );
    }

    /**
     * Get boundary which shows one plane in mesh, which has 1 coordinate equals to axis1[0] (left of mesh).
     * @return boundary which show plane in mesh
     */
    static Boundary getLeftBoundary() {
        return Boundary( [](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) {
            return new FixedIndex1Boundary(mesh, 0);
        } );
    }

    /**
     * Get boundary which shows one plane in mesh, which has 1 coordinate equals to axis1[axis1->size()-1] (right of mesh).
     * @return boundary which show plane in mesh
     */
    static Boundary getRightBoundary() {
        return Boundary( [](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) {
            return new FixedIndex1Boundary(mesh, mesh.axis1->size()-1);
        } );
    }

    /**
     * Get boundary which shows one plane in mesh, which has 1 coordinate equals to @p line_nr_axis1->
     * @param line_nr_axis1 index of axis1 mesh
     * @return boundary which show plane in mesh
     */
    static Boundary getIndex1BoundaryAtLine(std::size_t line_nr_axis1) {
        return Boundary( [line_nr_axis1](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) {
            return new FixedIndex1Boundary(mesh, line_nr_axis1);
        } );
    }

    /**
     * Get boundary which shows one plane in mesh, which has 2 coordinate equals to axis2[0] (bottom of mesh).
     * @return boundary which show plane in mesh
     */
    static Boundary getBottomBoundary() {
        return Boundary( [](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) {
            return new FixedIndex2Boundary(mesh, 0);
        } );
    }

    /**
     * Get boundary which shows one plane in mesh, which has 2nd coordinate equals to axis2[axis2->size()-1] (top of mesh).
     * @return boundary which show plane in mesh
     */
    static Boundary getTopBoundary() {
        return Boundary( [](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) {
            return new FixedIndex2Boundary(mesh, mesh.axis2->size()-1);
        } );
    }

    /**
     * Get boundary which shows one plane in mesh, which has 2 coordinate equals to @p line_nr_axis2->
     * @param line_nr_axis2 index of axis2 mesh
     * @return boundary which show plane in mesh
     */
    static Boundary getIndex2BoundaryAtLine(std::size_t line_nr_axis2) {
        return Boundary( [line_nr_axis2](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) {
            return new FixedIndex2Boundary(mesh, line_nr_axis2);
        } );
    }

    /**
     * GGet boundary which has fixed index at axis 0 direction and lies on back of the @p box (at nearest lines inside the box).
     * @param box box in which boundary should lie
     * @return boundary which has fixed index at axis 0 direction and lies on lower face of the @p box
     */
    static Boundary getBackOfBoundary(const Box3D& box) {
        if (!box.isValid()) return makeEmptyBoundary<RectangularMesh<3>>();
        return Boundary( [=](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) -> BoundaryLogicImpl* {
            std::size_t line, begInd1, endInd1, begInd2, endInd2;
            if (details::getLineLo(line, *mesh.axis0, box.lower.c0, box.upper.c0) &&
                details::getIndexesInBounds(begInd1, endInd1, *mesh.axis1, box.lower.c1, box.upper.c1) &&
                details::getIndexesInBounds(begInd2, endInd2, *mesh.axis2, box.lower.c2, box.upper.c2))
                return new FixedIndex0BoundaryInRange(mesh, line, begInd1, endInd1, begInd2, endInd2);
            else
                return new EmptyBoundaryImpl();
        } );
    }

    /**
     * Get boundary which has fixed index at axis 0 direction and lies on front of the @p box (at nearest lines inside the box).
     * @param box box in which boundary should lie
     * @return boundary which has fixed index at axis 0 direction and lies on higher face of the @p box
     */
    static Boundary getFrontOfBoundary(const Box3D& box) {
        if (!box.isValid()) return makeEmptyBoundary<RectangularMesh<3>>();
        return Boundary( [=](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) -> BoundaryLogicImpl* {
            std::size_t line, begInd1, endInd1, begInd2, endInd2;
            if (details::getLineHi(line, *mesh.axis0, box.lower.c0, box.upper.c0) &&
                details::getIndexesInBounds(begInd1, endInd1, *mesh.axis1, box.lower.c1, box.upper.c1) &&
                details::getIndexesInBounds(begInd2, endInd2, *mesh.axis2, box.lower.c2, box.upper.c2))
                return new FixedIndex0BoundaryInRange(mesh, line, begInd1, endInd1, begInd2, endInd2);
            else
                return new EmptyBoundaryImpl();
        } );
    }

    /**
     * Get boundary which has fixed index at axis 1 direction and lies on left of the @p box (at nearest lines inside the box).
     * @param box box in which boundary should lie
     * @return boundary which has fixed index at axis 1 direction and lies on lower face of the @p box
     */
    static Boundary getLeftOfBoundary(const Box3D& box) {
        if (!box.isValid()) return makeEmptyBoundary<RectangularMesh<3>>();
        return Boundary( [=](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) -> BoundaryLogicImpl* {
            std::size_t line, begInd0, endInd0, begInd2, endInd2;
            if (details::getLineLo(line, *mesh.axis1, box.lower.c1, box.upper.c1) &&
                details::getIndexesInBounds(begInd0, endInd0, *mesh.axis0, box.lower.c0, box.upper.c0) &&
                details::getIndexesInBounds(begInd2, endInd2, *mesh.axis2, box.lower.c2, box.upper.c2))
                return new FixedIndex1BoundaryInRange(mesh, line, begInd0, endInd0, begInd2, endInd2);
            else
                return new EmptyBoundaryImpl();
        } );
    }

    /**
     * Get boundary which has fixed index at axis 1 direction and lies on right of the @p box (at nearest lines inside the box).
     * @param box box in which boundary should lie
     * @return boundary which has fixed index at axis 1 direction and lies on higher face of the @p box
     */
    static Boundary getRightOfBoundary(const Box3D& box) {
        if (!box.isValid()) return makeEmptyBoundary<RectangularMesh<3>>();
        return Boundary( [=](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) -> BoundaryLogicImpl* {
            std::size_t line, begInd0, endInd0, begInd2, endInd2;
            if (details::getLineHi(line, *mesh.axis1, box.lower.c1, box.upper.c1) &&
                details::getIndexesInBounds(begInd0, endInd0, *mesh.axis0, box.lower.c0, box.upper.c0) &&
                details::getIndexesInBounds(begInd2, endInd2, *mesh.axis2, box.lower.c2, box.upper.c2))
                return new FixedIndex1BoundaryInRange(mesh, line, begInd0, endInd0, begInd2, endInd2);
            else
                return new EmptyBoundaryImpl();
        } );
    }

    /**
     * Get boundary which has fixed index at axis 1 direction and lies on bottom of the @p box (at nearest lines inside the box).
     * @param box box in which boundary should lie
     * @return boundary which has fixed index at axis 1 direction and lies on lower face of the @p box
     */
    static Boundary getBottomOfBoundary(const Box3D& box) {
        if (!box.isValid()) return makeEmptyBoundary<RectangularMesh<3>>();
        return Boundary( [=](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) -> BoundaryLogicImpl* {
            std::size_t line, begInd0, endInd0, begInd1, endInd1;
            if (details::getLineLo(line, *mesh.axis2, box.lower.c2, box.upper.c2) &&
                details::getIndexesInBounds(begInd0, endInd0, *mesh.axis0, box.lower.c0, box.upper.c0) &&
                details::getIndexesInBounds(begInd1, endInd1, *mesh.axis1, box.lower.c1, box.upper.c1))
                return new FixedIndex2BoundaryInRange(mesh, line, begInd0, endInd0, begInd1, endInd1);
            else
                return new EmptyBoundaryImpl();
        } );
    }

    /**
     * Get boundary which has fixed index at axis 2 direction and lies on top of the @p box (at nearest lines inside the box).
     * @param box box in which boundary should lie
     * @return boundary which has fixed index at axis 2 direction and lies on higher face of the @p box
     */
    static Boundary getTopOfBoundary(const Box3D& box) {
        if (!box.isValid()) return makeEmptyBoundary<RectangularMesh<3>>();
        return Boundary( [=](const RectangularMesh<3>& mesh, const shared_ptr<const GeometryD<3>>&) -> BoundaryLogicImpl* {
            std::size_t line, begInd0, endInd0, begInd1, endInd1;
            if (details::getLineHi(line, *mesh.axis2, box.lower.c2, box.upper.c2) &&
                details::getIndexesInBounds(begInd0, endInd0, *mesh.axis0, box.lower.c0, box.upper.c0) &&
                details::getIndexesInBounds(begInd1, endInd1, *mesh.axis1, box.lower.c1, box.upper.c1))
                return new FixedIndex2BoundaryInRange(mesh, line, begInd0, endInd0, begInd1, endInd1);
            else
                return new EmptyBoundaryImpl();
        } );
    }

    /**
     * Get boundary which lies on back faces of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geometry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geometry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries on faces of @p object's bounding-boxes
     */
    static Boundary getBackOfBoundary(shared_ptr<const GeometryObject> object, const PathHints& path) {
        return details::getBoundaryForBoxes< RectangularMesh<3> >(
            [=](const shared_ptr<const GeometryD<3>>& geometry) { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box3D& box) { return RectangularMesh<3>::getBackOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on back faces of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geometry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geometry
     * @param path (optional) hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries on faces of @p object's bounding-boxes
     */
    static Boundary getBackOfBoundary(shared_ptr<const GeometryObject> object, const PathHints* path = nullptr) {
        if (path) return getBackOfBoundary(object, *path);
        return details::getBoundaryForBoxes< RectangularMesh<3> >(
            [=](const shared_ptr<const GeometryD<3>>& geometry) { return geometry->getObjectBoundingBoxes(object); },
            [](const Box3D& box) { return RectangularMesh<3>::getBackOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on front faces of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geometry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geometry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries on faces of @p object's bounding-boxes
     */
    static Boundary getFrontOfBoundary(shared_ptr<const GeometryObject> object, const PathHints& path) {
        return details::getBoundaryForBoxes< RectangularMesh<3> >(
            [=](const shared_ptr<const GeometryD<3>>& geometry) { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box3D& box) { return RectangularMesh<3>::getFrontOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on front faces of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geometry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geometry
     * @param path (optional) hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries on faces of @p object's bounding-boxes
     */
    static Boundary getFrontOfBoundary(shared_ptr<const GeometryObject> object, const PathHints* path = nullptr) {
        if (path) return getFrontOfBoundary(object, *path);
        return details::getBoundaryForBoxes< RectangularMesh<3> >(
            [=](const shared_ptr<const GeometryD<3>>& geometry) { return geometry->getObjectBoundingBoxes(object); },
            [](const Box3D& box) { return RectangularMesh<3>::getFrontOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on left faces of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geometry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geometry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries on faces of @p object's bounding-boxes
     */
    static Boundary getLeftOfBoundary(shared_ptr<const GeometryObject> object, const PathHints& path) {
        return details::getBoundaryForBoxes< RectangularMesh<3> >(
            [=](const shared_ptr<const GeometryD<3>>& geometry) { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box3D& box) { return RectangularMesh<3>::getLeftOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on left faces of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geometry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geometry
     * @param path (optional) hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries on faces of @p object's bounding-boxes
     */
    static Boundary getLeftOfBoundary(shared_ptr<const GeometryObject> object, const PathHints* path = nullptr) {
        if (path) return getLeftOfBoundary(object, *path);
        return details::getBoundaryForBoxes< RectangularMesh<3> >(
            [=](const shared_ptr<const GeometryD<3>>& geometry) { return geometry->getObjectBoundingBoxes(object); },
            [](const Box3D& box) { return RectangularMesh<3>::getLeftOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on right faces of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geometry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geometry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries on faces of @p object's bounding-boxes
     */
    static Boundary getRightOfBoundary(shared_ptr<const GeometryObject> object, const PathHints& path) {
        return details::getBoundaryForBoxes< RectangularMesh<3> >(
            [=](const shared_ptr<const GeometryD<3>>& geometry) { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box3D& box) { return RectangularMesh<3>::getRightOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on right faces of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geometry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geometry
     * @param path (optional) hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries on faces of @p object's bounding-boxes
     */
    static Boundary getRightOfBoundary(shared_ptr<const GeometryObject> object, const PathHints* path = nullptr) {
        if (path) return getRightOfBoundary(object, *path);
        return details::getBoundaryForBoxes< RectangularMesh<3> >(
            [=](const shared_ptr<const GeometryD<3>>& geometry) { return geometry->getObjectBoundingBoxes(object); },
            [](const Box3D& box) { return RectangularMesh<3>::getRightOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on bottom faces of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geometry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geometry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries on faces of @p object's bounding-boxes
     */
    static Boundary getBottomOfBoundary(shared_ptr<const GeometryObject> object, const PathHints& path) {
        return details::getBoundaryForBoxes< RectangularMesh<3> >(
            [=](const shared_ptr<const GeometryD<3>>& geometry) { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box3D& box) { return RectangularMesh<3>::getBottomOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on bottom faces of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geometry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geometry
     * @param path (optional) hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries on faces of @p object's bounding-boxes
     */
    static Boundary getBottomOfBoundary(shared_ptr<const GeometryObject> object, const PathHints* path = nullptr) {
        if (path) return getBottomOfBoundary(object, *path);
        return details::getBoundaryForBoxes< RectangularMesh<3> >(
            [=](const shared_ptr<const GeometryD<3>>& geometry) { return geometry->getObjectBoundingBoxes(object); },
            [](const Box3D& box) { return RectangularMesh<3>::getBottomOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on top faces (higher faces with fixed axis 2 coordinate) of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geometry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geometry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries on faces of @p object's bounding-boxes
     */
    static Boundary getTopOfBoundary(shared_ptr<const GeometryObject> object, const PathHints& path) {
        return details::getBoundaryForBoxes< RectangularMesh<3> >(
            [=](const shared_ptr<const GeometryD<3>>& geometry) { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box3D& box) { return RectangularMesh<3>::getTopOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on top faces (higher faces with fixed axis 2 coordinate) of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geometry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geometry
     * @param path (optional) hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries on faces of @p object's bounding-boxes
     */
    static Boundary getTopOfBoundary(shared_ptr<const GeometryObject> object, const PathHints* path = nullptr) {
        if (path) return getTopOfBoundary(object, *path);
        return details::getBoundaryForBoxes< RectangularMesh<3> >(
            [=](const shared_ptr<const GeometryD<3>>& geometry) { return geometry->getObjectBoundingBoxes(object); },
            [](const Box3D& box) { return RectangularMesh<3>::getTopOfBoundary(box); }
        );
    }

    static Boundary getBoundary(const std::string& boundary_desc);

    static Boundary getBoundary(plask::XMLReader& boundary_desc, plask::Manager& manager);
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh<3>, SrcT, DstT, INTERPOLATION_LINEAR> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh<3>>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->axis0->size() == 0 || src_mesh->axis1->size() == 0 || src_mesh->axis2->size() == 0)
            throw BadMesh("interpolate", "Source mesh empty");
        return new LinearInterpolatedLazyDataImpl<DstT, RectilinearMesh3D, SrcT>(src_mesh, src_vec, dst_mesh, flags);
    }
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh<3>, SrcT, DstT, INTERPOLATION_NEAREST> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh<3>>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->axis0->size() == 0 || src_mesh->axis1->size() == 0 || src_mesh->axis2->size() == 0)
            throw BadMesh("interpolate", "Source mesh empty");
        return new NearestNeighborInterpolatedLazyDataImpl<DstT, RectilinearMesh3D, SrcT>(src_mesh, src_vec, dst_mesh, flags);
    }
};


/**
 * Copy @p to_copy mesh using OrderedAxis to represent each axis in returned mesh.
 * @param to_copy mesh to copy
 * @return mesh with each axis of type OrderedAxis
 */
PLASK_API shared_ptr<RectangularMesh<3> > make_rectangular_mesh(const RectangularMesh<3> &to_copy);
inline shared_ptr<RectangularMesh<3>> make_rectangular_mesh(shared_ptr<const RectangularMesh<3>> to_copy) { return make_rectangular_mesh(*to_copy); }

template <>
inline Boundary<RectangularMesh<3>> parseBoundary<RectangularMesh<3>>(const std::string& boundary_desc, plask::Manager&) { return RectangularMesh<3>::getBoundary(boundary_desc); }

template <>
inline Boundary<RectangularMesh<3>> parseBoundary<RectangularMesh<3>>(XMLReader& boundary_desc, Manager& env) { return RectangularMesh<3>::getBoundary(boundary_desc, env); }

PLASK_API_EXTERN_TEMPLATE_CLASS(RectangularMesh<3>)

}   // namespace plask

#endif // PLASK__RECTANGULAR3D_H
