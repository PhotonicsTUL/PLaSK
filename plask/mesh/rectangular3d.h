#ifndef PLASK__RECTANGULAR3D_H
#define PLASK__RECTANGULAR3D_H

/** @file
This file contains rectangular mesh for 3D space.
*/

#include "rectangular_common.h"
#include "rectilinear3d.h"
#include "../optional.h"

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
class PLASK_API RectangularMesh3D: public RectilinearMesh3D {

  public:

    /// Boundary type.
    typedef ::plask::Boundary<RectangularMesh3D> Boundary;

    /**
     * Construct mesh which has all axes of type OrderedAxis and all are empty.
     * @param iterationOrder iteration order
     */
    explicit RectangularMesh3D(IterationOrder iterationOrder = ORDER_012);

    /**
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     * @param mesh2 mesh for the third coordinate
     * @param iterationOrder iteration order
     */
    RectangularMesh3D(shared_ptr<MeshAxis> mesh0, shared_ptr<MeshAxis> mesh1, shared_ptr<MeshAxis> mesh2, IterationOrder iterationOrder = ORDER_012);

    /**
     * Copy constructor.
     * @param src mesh to copy
     * @param clone_axes whether axes of the @p src should be cloned (if true) or shared (if false; default)
     */
    RectangularMesh3D(const RectangularMesh3D& src, bool clone_axes = false);

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<MeshAxis>& lon() const { return axis[0]; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<MeshAxis>& tran() const { return axis[1]; }

    /**
     * Get third coordinate of points in this mesh.
     * @return axis2
     */
    const shared_ptr<MeshAxis>& vert() const { return axis[2]; }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<MeshAxis>& ee_z() const { return axis[0]; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<MeshAxis>& ee_x() const { return axis[1]; }

    /**
     * Get third coordinate of points in this mesh.
     * @return axis2
     */
    const shared_ptr<MeshAxis>& ee_y() const { return axis[2]; }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<MeshAxis>& rad_r() const { return axis[0]; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<MeshAxis>& rad_phi() const { return axis[1]; }

    /**
     * Get thirs coordinate of points in this mesh.
     * @return axis2
     */
    const shared_ptr<MeshAxis>& rad_z() const { return axis[2]; }

    /**
     * Write mesh to XML
     * \param object XML object to write to
     */
    void writeXML(XMLElement& object) const override;

    using RectilinearMesh3D::at;    // MSVC needs this

    /**
     * Get point with given mesh indices.
     * @param index0 index of point in axis0
     * @param index1 index of point in axis1
     * @param index2 index of point in axis2
     * @return point with given @p index
     */
    Vec<3, double> at(std::size_t index0, std::size_t index1, std::size_t index2) const override {
        return Vec<3, double>(axis[0]->at(index0), axis[1]->at(index1), axis[2]->at(index2));
    }

    /**
     * Return a mesh that enables iterating over middle points of the cuboids
     * \return new rectangular mesh with points in the middles of original cuboids
     */
    shared_ptr<RectangularMesh3D> getMidpointsMesh();

    /**
     * Get area of given element.
     * @param index0, index1, index2 axis 0, 1 and 2 indexes of element
     * @return area of elements with given index
     */
    double getElementArea(std::size_t index0, std::size_t index1, std::size_t index2) const {
        return (axis[0]->at(index0+1) - axis[0]->at(index0)) * (axis[1]->at(index1+1) - axis[1]->at(index1)) * (axis[2]->at(index2+1) - axis[2]->at(index2));
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
        return Box3D(axis[0]->at(index0), axis[1]->at(index1), axis[2]->at(index2), axis[0]->at(index0+1), axis[1]->at(index1+1), axis[2]->at(index2+1));
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
    struct BoundaryIteratorImpl: public BoundaryNodeSetImpl::IteratorImpl {

        const RectangularMesh3D &mesh;

        const std::size_t level;

        std::size_t index_f, index_s;

        const std::size_t index_f_begin, index_f_end;

        BoundaryIteratorImpl(const RectangularMesh3D& mesh, std::size_t level,
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

        virtual bool equal(const typename BoundaryNodeSetImpl::IteratorImpl& other) const override {
            return index_f == static_cast<const BoundaryIteratorImpl&>(other).index_f && index_s == static_cast<const BoundaryIteratorImpl&>(other).index_s;
        }

    };

    // iterator with fixed first coordinate
    struct FixedIndex0IteratorImpl: public BoundaryIteratorImpl {

        FixedIndex0IteratorImpl(const RectangularMesh3D& mesh, std::size_t level_index0, std::size_t index_1, std::size_t index_1_begin, std::size_t index_1_end, std::size_t index_2)
            : BoundaryIteratorImpl(mesh, level_index0, index_1, index_1_begin, index_1_end, index_2) {}

        virtual std::size_t dereference() const override { return this->mesh.index(this->level, this->index_f, this->index_s); }

        virtual typename BoundaryNodeSetImpl::IteratorImpl* clone() const override {
            return new FixedIndex0IteratorImpl(*this);
        }
    };

    // iterator with fixed second coordinate
    struct FixedIndex1IteratorImpl: public BoundaryIteratorImpl {

        FixedIndex1IteratorImpl(const RectangularMesh3D& mesh, std::size_t level_index1, std::size_t index_0, std::size_t index_0_begin, std::size_t index_0_end, std::size_t index_2)
            : BoundaryIteratorImpl(mesh, level_index1, index_0, index_0_begin, index_0_end, index_2) {}

        virtual std::size_t dereference() const override { return this->mesh.index(this->index_f, this->level, this->index_s); }

        virtual typename BoundaryNodeSetImpl::IteratorImpl* clone() const override {
            return new FixedIndex1IteratorImpl(*this);
        }
    };

    // iterator with fixed third coordinate
    struct FixedIndex2IteratorImpl: public BoundaryIteratorImpl {

        FixedIndex2IteratorImpl(const RectangularMesh3D& mesh, std::size_t level_index2, std::size_t index_0, std::size_t index_0_begin, std::size_t index_0_end, std::size_t index_1)
            : BoundaryIteratorImpl(mesh, level_index2, index_0, index_0_begin, index_0_end, index_1) {}

        virtual std::size_t dereference() const override { return this->mesh.index(this->index_f, this->index_s, this->level); }

        virtual typename BoundaryNodeSetImpl::IteratorImpl* clone() const override {
            return new FixedIndex2IteratorImpl(*this);
        }
    };

    struct FixedIndex0Boundary: public BoundaryNodeSetWithMeshImpl<RectangularMesh3D> {

        typedef typename BoundaryNodeSetImpl::Iterator Iterator;

        std::size_t level_axis0;

        FixedIndex0Boundary(const RectangularMesh3D& mesh, std::size_t level_axis0): BoundaryNodeSetWithMeshImpl<RectangularMesh3D>(mesh), level_axis0(level_axis0) {}

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index0(mesh_index) == level_axis0;
        }

        Iterator begin() const override {
            return Iterator(new FixedIndex0IteratorImpl(this->mesh, level_axis0, 0, 0, this->mesh.axis[1]->size(), 0));
        }

        Iterator end() const override {
            return Iterator(new FixedIndex0IteratorImpl(this->mesh, level_axis0, 0, 0, this->mesh.axis[1]->size(), this->mesh.axis[2]->size()));
        }

        std::size_t size() const override {
            return this->mesh.axis[1]->size() * this->mesh.axis[2]->size();
        }
    };

    struct FixedIndex0BoundaryInRange: public BoundaryNodeSetWithMeshImpl<RectangularMesh3D> {

        typedef typename BoundaryNodeSetImpl::Iterator Iterator;

        std::size_t level_axis0, beginAxis1, endAxis1, beginAxis2, endAxis2;

        FixedIndex0BoundaryInRange(const RectangularMesh3D& mesh, std::size_t level_axis0, std::size_t beginAxis1, std::size_t endAxis1, std::size_t beginAxis2, std::size_t endAxis2)
            : BoundaryNodeSetWithMeshImpl<RectangularMesh3D>(mesh), level_axis0(level_axis0),
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

    struct FixedIndex1Boundary: public BoundaryNodeSetWithMeshImpl<RectangularMesh3D> {

        typedef typename BoundaryNodeSetImpl::Iterator Iterator;

        std::size_t level_axis1;

        FixedIndex1Boundary(const RectangularMesh3D& mesh, std::size_t level_axis1): BoundaryNodeSetWithMeshImpl<RectangularMesh3D>(mesh), level_axis1(level_axis1) {}

        //virtual LeftBoundary* clone() const { return new LeftBoundary(); }

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index1(mesh_index) == level_axis1;
        }

        Iterator begin() const override {
            return Iterator(new FixedIndex1IteratorImpl(this->mesh, level_axis1, 0, 0, this->mesh.axis[0]->size(), 0));
        }

        Iterator end() const override {
            return Iterator(new FixedIndex1IteratorImpl(this->mesh, level_axis1, 0, 0, this->mesh.axis[0]->size(), this->mesh.axis[2]->size()));
        }

        std::size_t size() const override {
            return this->mesh.axis[0]->size() * this->mesh.axis[2]->size();
        }
    };

    struct FixedIndex1BoundaryInRange: public BoundaryNodeSetWithMeshImpl<RectangularMesh3D> {

        typedef typename BoundaryNodeSetImpl::Iterator Iterator;

        std::size_t level_axis1, beginAxis0, endAxis0, beginAxis2, endAxis2;

        FixedIndex1BoundaryInRange(const RectangularMesh3D& mesh, std::size_t level_axis1, std::size_t beginAxis0, std::size_t endAxis0, std::size_t beginAxis2, std::size_t endAxis2)
            : BoundaryNodeSetWithMeshImpl<RectangularMesh3D>(mesh), level_axis1(level_axis1),
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


    struct FixedIndex2Boundary: public BoundaryNodeSetWithMeshImpl<RectangularMesh3D> {

        typedef typename BoundaryNodeSetImpl::Iterator Iterator;

        std::size_t level_axis2;

        FixedIndex2Boundary(const RectangularMesh3D& mesh, std::size_t level_axis2): BoundaryNodeSetWithMeshImpl<RectangularMesh3D>(mesh), level_axis2(level_axis2) {}

        //virtual LeftBoundary* clone() const { return new LeftBoundary(); }

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index2(mesh_index) == level_axis2;
        }

        Iterator begin() const override {
            return Iterator(new FixedIndex2IteratorImpl(this->mesh, level_axis2, 0, 0, this->mesh.axis[0]->size(), 0));
        }

        Iterator end() const override {
            return Iterator(new FixedIndex2IteratorImpl(this->mesh, level_axis2, 0, 0, this->mesh.axis[0]->size(), this->mesh.axis[1]->size()));
        }

        std::size_t size() const override {
            return this->mesh.axis[0]->size() * this->mesh.axis[1]->size();
        }
    };

    struct FixedIndex2BoundaryInRange: public BoundaryNodeSetWithMeshImpl<RectangularMesh3D> {

        typedef typename BoundaryNodeSetImpl::Iterator Iterator;

        std::size_t level_axis2, beginAxis0, endAxis0, beginAxis1, endAxis1;

        FixedIndex2BoundaryInRange(const RectangularMesh3D& mesh, std::size_t level_axis2, std::size_t beginAxis0, std::size_t endAxis0, std::size_t beginAxis1, std::size_t endAxis1)
            : BoundaryNodeSetWithMeshImpl<RectangularMesh3D>(mesh), level_axis2(level_axis2),
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

    BoundaryNodeSet createIndex0BoundaryAtLine(std::size_t line_nr_axis0) const override {
        return new FixedIndex0Boundary(*this, line_nr_axis0);
    }

    BoundaryNodeSet createBackBoundary() const override {
        return createIndex0BoundaryAtLine(0);
    }

    BoundaryNodeSet createFrontBoundary() const override {
        return createIndex0BoundaryAtLine(axis[0]->size()-1);
    }

    BoundaryNodeSet createIndex1BoundaryAtLine(std::size_t line_nr_axis1) const override {
        return new FixedIndex1Boundary(*this, line_nr_axis1);
    }

    BoundaryNodeSet createLeftBoundary() const override {
        return createIndex1BoundaryAtLine(0);
    }

    BoundaryNodeSet createRightBoundary() const override {
        return createIndex1BoundaryAtLine(axis[1]->size()-1);
    }

    BoundaryNodeSet createIndex2BoundaryAtLine(std::size_t line_nr_axis2) const override {
        return new FixedIndex2Boundary(*this, line_nr_axis2);
    }

    BoundaryNodeSet createBottomBoundary() const override {
        return createIndex2BoundaryAtLine(0);
    }

    BoundaryNodeSet createTopBoundary() const override {
        return createIndex2BoundaryAtLine(axis[2]->size()-1);
    }

    BoundaryNodeSet createIndex0BoundaryAtLine(std::size_t line_nr_axis0,
                                               std::size_t index1Begin, std::size_t index1End,
                                               std::size_t index2Begin, std::size_t index2End) const override
    {
        if (index1Begin < index1End && index2Begin < index2End)
            return new FixedIndex0BoundaryInRange(*this, line_nr_axis0, index1Begin, index1End, index2Begin, index2End);
        else
            return new EmptyBoundaryImpl();
    }

    BoundaryNodeSet createBackOfBoundary(const Box3D& box) const override {
        std::size_t line, begInd1, endInd1, begInd2, endInd2;
        if (details::getLineLo(line, *axis[0], box.lower.c0, box.upper.c0) &&
                details::getIndexesInBounds(begInd1, endInd1, *axis[1], box.lower.c1, box.upper.c1) &&
                details::getIndexesInBounds(begInd2, endInd2, *axis[2], box.lower.c2, box.upper.c2))
                return new FixedIndex0BoundaryInRange(*this, line, begInd1, endInd1, begInd2, endInd2);
        else
                return new EmptyBoundaryImpl();
    }

    BoundaryNodeSet createFrontOfBoundary(const Box3D& box) const override {
            std::size_t line, begInd1, endInd1, begInd2, endInd2;
            if (details::getLineHi(line, *axis[0], box.lower.c0, box.upper.c0) &&
                details::getIndexesInBounds(begInd1, endInd1, *axis[1], box.lower.c1, box.upper.c1) &&
                details::getIndexesInBounds(begInd2, endInd2, *axis[2], box.lower.c2, box.upper.c2))
                return new FixedIndex0BoundaryInRange(*this, line, begInd1, endInd1, begInd2, endInd2);
            else
                return new EmptyBoundaryImpl();
    }

    BoundaryNodeSet createIndex1BoundaryAtLine(std::size_t line_nr_axis1,
                                               std::size_t index0Begin, std::size_t index0End,
                                               std::size_t index2Begin, std::size_t index2End) const override
    {
        if (index0Begin < index0End && index2Begin < index2End)
            return new FixedIndex1BoundaryInRange(*this, line_nr_axis1, index0Begin, index0End, index2Begin, index2End);
        else
            return new EmptyBoundaryImpl();
    }

    BoundaryNodeSet createLeftOfBoundary(const Box3D& box) const override {
            std::size_t line, begInd0, endInd0, begInd2, endInd2;
            if (details::getLineLo(line, *axis[1], box.lower.c1, box.upper.c1) &&
                details::getIndexesInBounds(begInd0, endInd0, *axis[0], box.lower.c0, box.upper.c0) &&
                details::getIndexesInBounds(begInd2, endInd2, *axis[2], box.lower.c2, box.upper.c2))
                return new FixedIndex1BoundaryInRange(*this, line, begInd0, endInd0, begInd2, endInd2);
            else
                return new EmptyBoundaryImpl();
    }

    BoundaryNodeSet createRightOfBoundary(const Box3D& box) const override {
            std::size_t line, begInd0, endInd0, begInd2, endInd2;
            if (details::getLineHi(line, *axis[1], box.lower.c1, box.upper.c1) &&
                details::getIndexesInBounds(begInd0, endInd0, *axis[0], box.lower.c0, box.upper.c0) &&
                details::getIndexesInBounds(begInd2, endInd2, *axis[2], box.lower.c2, box.upper.c2))
                return new FixedIndex1BoundaryInRange(*this, line, begInd0, endInd0, begInd2, endInd2);
            else
                return new EmptyBoundaryImpl();
    }

    BoundaryNodeSet createIndex2BoundaryAtLine(std::size_t line_nr_axis2,
                                               std::size_t index0Begin, std::size_t index0End,
                                               std::size_t index1Begin, std::size_t index1End) const override
    {
        if (index0Begin < index0End && index1Begin < index1End)
            return new FixedIndex2BoundaryInRange(*this, line_nr_axis2, index0Begin, index0End, index1Begin, index1End);
        else
            return new EmptyBoundaryImpl();
    }

    BoundaryNodeSet createBottomOfBoundary(const Box3D& box) const override {
            std::size_t line, begInd0, endInd0, begInd1, endInd1;
            if (details::getLineLo(line, *axis[2], box.lower.c2, box.upper.c2) &&
                details::getIndexesInBounds(begInd0, endInd0, *axis[0], box.lower.c0, box.upper.c0) &&
                details::getIndexesInBounds(begInd1, endInd1, *axis[1], box.lower.c1, box.upper.c1))
                return new FixedIndex2BoundaryInRange(*this, line, begInd0, endInd0, begInd1, endInd1);
            else
                return new EmptyBoundaryImpl();
    }

    BoundaryNodeSet createTopOfBoundary(const Box3D& box) const override {
            std::size_t line, begInd0, endInd0, begInd1, endInd1;
            if (details::getLineHi(line, *axis[2], box.lower.c2, box.upper.c2) &&
                details::getIndexesInBounds(begInd0, endInd0, *axis[0], box.lower.c0, box.upper.c0) &&
                details::getIndexesInBounds(begInd1, endInd1, *axis[1], box.lower.c1, box.upper.c1))
                return new FixedIndex2BoundaryInRange(*this, line, begInd0, endInd0, begInd1, endInd1);
            else
                return new EmptyBoundaryImpl();
    }
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh3D, SrcT, DstT, INTERPOLATION_LINEAR> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh3D>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->axis[0]->size() == 0 || src_mesh->axis[1]->size() == 0 || src_mesh->axis[2]->size() == 0)
            throw BadMesh("interpolate", "Source mesh empty");
        return new LinearInterpolatedLazyDataImpl<DstT, RectilinearMesh3D, SrcT>(src_mesh, src_vec, dst_mesh, flags);
    }
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh3D, SrcT, DstT, INTERPOLATION_NEAREST> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh3D>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->axis[0]->size() == 0 || src_mesh->axis[1]->size() == 0 || src_mesh->axis[2]->size() == 0)
            throw BadMesh("interpolate", "Source mesh empty");
        return new NearestNeighborInterpolatedLazyDataImpl<DstT, RectilinearMesh3D, SrcT>(src_mesh, src_vec, dst_mesh, flags);
    }
};


/**
 * Copy @p to_copy mesh using OrderedAxis to represent each axis in returned mesh.
 * @param to_copy mesh to copy
 * @return mesh with each axis of type OrderedAxis
 */
PLASK_API shared_ptr<RectangularMesh3D > make_rectangular_mesh(const RectangularMesh3D &to_copy);
inline shared_ptr<RectangularMesh3D> make_rectangular_mesh(shared_ptr<const RectangularMesh3D> to_copy) { return make_rectangular_mesh(*to_copy); }

}   // namespace plask

#endif // PLASK__RECTANGULAR3D_H
