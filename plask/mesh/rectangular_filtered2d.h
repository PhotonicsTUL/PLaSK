#ifndef PLASK__RECTANGULAR_FILTERED2D_H
#define PLASK__RECTANGULAR_FILTERED2D_H

#include "rectangular_filtered_common.h"

namespace plask {

struct PLASK_API RectangularFilteredMesh2D: public RectangularFilteredMeshBase<2> {

    typedef std::function<bool(const RectangularMesh2D::Element&)> Predicate;

    class PLASK_API Element {

        const RectangularFilteredMesh2D& filteredMesh;

        //std::uint32_t elementNumber;    ///< index of element in oryginal mesh
        std::size_t index0, index1; // probably this form allows to do most operation fastest in average, low indexes of element corner or just element indexes

        /// Index of element. If it equals to UNKONOWN_ELEMENT_INDEX, it will be calculated on-demand from index0 and index1.
        mutable std::size_t elementIndex;

        const RectangularMesh<2>& rectangularMesh() const { return filteredMesh.rectangularMesh; }

    public:

        enum: std::size_t { UNKONOWN_ELEMENT_INDEX = std::numeric_limits<std::size_t>::max() };

        Element(const RectangularFilteredMesh2D& filteredMesh, std::size_t elementIndex, std::size_t index0, std::size_t index1)
            : filteredMesh(filteredMesh), index0(index0), index1(index1), elementIndex(elementIndex)
        {
        }

        Element(const RectangularFilteredMesh2D& filteredMesh, std::size_t elementIndex, std::size_t elementIndexOfFullMesh)
            : filteredMesh(filteredMesh), elementIndex(elementIndex)
        {
            const std::size_t v = rectangularMesh().getElementMeshLowIndex(elementIndexOfFullMesh);
            index0 = rectangularMesh().index0(v);
            index1 = rectangularMesh().index1(v);
        }

        Element(const RectangularFilteredMesh2D& filteredMesh, std::size_t elementIndex)
            : Element(filteredMesh, elementIndex, filteredMesh.elementsSet.at(elementIndex))
        {}

        /// \return tran index of the element
        inline std::size_t getIndex0() const { return index0; }

        /// \return vert index of the element
        inline std::size_t getIndex1() const { return index1; }

        /// \return tran index of the left edge of the element
        inline std::size_t getLowerIndex0() const { return index0; }

        /// \return vert index of the bottom edge of the element
        inline std::size_t getLowerIndex1() const { return index1; }

        /// \return tran coordinate of the left edge of the element
        inline double getLower0() const { return rectangularMesh().axis[0]->at(index0); }

        /// \return vert coordinate of the bottom edge of the element
        inline double getLower1() const { return rectangularMesh().axis[1]->at(index1); }

        /// \return tran index of the right edge of the element
        inline std::size_t getUpperIndex0() const { return index0+1; }

        /// \return vert index of the top edge of the element
        inline std::size_t getUpperIndex1() const { return index1+1; }

        /// \return tran coordinate of the right edge of the element
        inline double getUpper0() const { return rectangularMesh().axis[0]->at(getUpperIndex0()); }

        /// \return vert coordinate of the top edge of the element
        inline double getUpper1() const { return rectangularMesh().axis[1]->at(getUpperIndex1()); }

        /// \return size of the element in the tran direction
        inline double getSize0() const { return getUpper0() - getLower0(); }

        /// \return size of the element in the vert direction
        inline double getSize1() const { return getUpper1() - getLower1(); }

        /// \return vector indicating size of the element
        inline Vec<2, double> getSize() const { return getUpUp() - getLoLo(); }

        /// \return position of the middle of the element
        inline Vec<2, double> getMidpoint() const { return filteredMesh.getElementMidpoint(index0, index1); }

        /// @return index of this element
        inline std::size_t getIndex() const {
            if (elementIndex == UNKONOWN_ELEMENT_INDEX)
                elementIndex = filteredMesh.getElementIndexFromLowIndexes(getLowerIndex0(), getLowerIndex1());
            return elementIndex;
        }

        /// \return this element as rectangular box
        inline Box2D toBox() const { return filteredMesh.getElementBox(index0, index1); }

        /// \return total area of this element
        inline double getVolume() const { return getSize0() * getSize1(); }

        /// \return total area of this element
        inline double getArea() const { return getVolume(); }

        /// \return index of the lower left corner of this element
        inline std::size_t getLoLoIndex() const { return filteredMesh.index(getLowerIndex0(), getLowerIndex1()); }

        /// \return index of the upper left corner of this element
        inline std::size_t getLoUpIndex() const { return filteredMesh.index(getLowerIndex0(), getUpperIndex1()); }

        /// \return index of the lower right corner of this element
        inline std::size_t getUpLoIndex() const { return filteredMesh.index(getUpperIndex0(), getLowerIndex1()); }

        /// \return index of the upper right corner of this element
        inline std::size_t getUpUpIndex() const { return filteredMesh.index(getUpperIndex0(), getUpperIndex1()); }

        /// \return position of the lower left corner of this element
        inline Vec<2, double> getLoLo() const { return filteredMesh(getLowerIndex0(), getLowerIndex1()); }

        /// \return position of the upper left corner of this element
        inline Vec<2, double> getLoUp() const { return filteredMesh(getLowerIndex0(), getUpperIndex1()); }

        /// \return position of the lower right corner of this element
        inline Vec<2, double> getUpLo() const { return filteredMesh(getUpperIndex0(), getLowerIndex1()); }

        /// \return position of the upper right corner of this element
        inline Vec<2, double> getUpUp() const { return filteredMesh(getUpperIndex0(), getUpperIndex1()); }

    };  // class Element


    struct Elements: ElementsBase<RectangularFilteredMesh2D> {

        explicit Elements(const RectangularFilteredMesh2D& mesh): ElementsBase(mesh) {}

        Element operator()(std::size_t i0, std::size_t i1) const { return Element(*filteredMesh, Element::UNKONOWN_ELEMENT_INDEX, i0, i1); }

    };  // struct Elements

    /**
     * Construct filtered mesh with elements of rectangularMesh chosen by a @p predicate.
     * @param rectangularMesh input mesh, before filtering
     * @param predicate predicate which returns either @c true for accepting element or @c false for rejecting it
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if @c true) or shared (if @c false; default)
     */
    RectangularFilteredMesh2D(const RectangularMesh<2>& rectangularMesh, const Predicate& predicate, bool clone_axes = false)
        : RectangularFilteredMeshBase(rectangularMesh, clone_axes)
    {
        for (auto el_it = this->rectangularMesh.elements().begin(); el_it != this->rectangularMesh.elements().end(); ++el_it)
            if (predicate(*el_it)) {
                elementsSet.push_back(el_it.index);
                nodesSet.insert(el_it->getLoLoIndex());
                nodesSet.insert(el_it->getLoUpIndex());
                nodesSet.insert(el_it->getUpLoIndex());
                nodesSet.push_back(el_it->getUpUpIndex());  //TODO is this safe for 10 axis?
                if (el_it->getLowerIndex0() < boundaryIndex[0].lo) boundaryIndex[0].lo = el_it->getLowerIndex0();
                if (el_it->getUpperIndex0() > boundaryIndex[0].up) boundaryIndex[0].up = el_it->getUpperIndex0();
                if (el_it->getLowerIndex1() < boundaryIndex[1].lo) boundaryIndex[1].lo = el_it->getLowerIndex1();
                if (el_it->getUpperIndex1() > boundaryIndex[1].up) boundaryIndex[1].up = el_it->getUpperIndex1();
            }
        nodesSet.shrink_to_fit();
        elementsSet.shrink_to_fit();
    }

    /**
     * Construct filtered mesh with all elements of @c rectangularMesh which have required materials in the midpoints.
     * @param rectangularMesh input mesh, before filtering
     * @param geom geometry to get materials from
     * @param materialPredicate predicate which returns either @c true for accepting material or @c false for rejecting it
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if @c true) or shared (if @c false; default)
     */
    RectangularFilteredMesh2D(const RectangularMesh<2>& rectangularMesh, const GeometryObjectD<2>& geom, const std::function<bool(shared_ptr<const Material>)> materialPredicate)
        : RectangularFilteredMesh2D(rectangularMesh, [&](const RectangularMesh2D::Element& el) { return materialPredicate(geom.getMaterial(el.getMidpoint())); })
    {
    }

    /**
     * Construct filtered mesh with all elements of @c rectangularMesh which have required kinds of materials (in the midpoints).
     * @param rectangularMesh input mesh, before filtering
     * @param geom geometry to get materials from
     * @param materialKinds one or more kinds of material encoded with bit @c or operation, e.g. @c DIELECTRIC|METAL
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if @c true) or shared (if @c false; default)
     */
    RectangularFilteredMesh2D(const RectangularMesh<2>& rectangularMesh, const GeometryObjectD<2>& geom, unsigned char materialKinds, bool clone_axes = false)
        : RectangularFilteredMesh2D(rectangularMesh,
                                    [&](const RectangularMesh2D::Element& el) { return (geom.getMaterialOrAir(el.getMidpoint())->kind() & materialKinds) != 0; },
                                    clone_axes)
    {
    }

    Elements elements() const { return Elements(*this); }
    Elements getElements() const { return elements(); }

    Element element(std::size_t i0, std::size_t i1) const { return Element(*this, Element::UNKONOWN_ELEMENT_INDEX, i0, i1); }
    Element getElement(std::size_t i0, std::size_t i1) const { return element(i0, i1); }

    /**
     * Get an element with a given index @p i.
     * @param i index of the element
     * @return the element
     */
    Element element(std::size_t i) const { return Element(*this, i); }

    /**
     * Get an element with a given index @p i.
     * @param i index of the element
     * @return the element
     */
    Element getElement(std::size_t i) const { return element(i); }

    /**
     * Calculate this mesh index using indexes of axis0 and axis1.
     * @param axis0_index index of axis0, from 0 to axis[0]->size()-1
     * @param axis1_index index of axis1, from 0 to axis[1]->size()-1
     * @return this mesh index, from 0 to size()-1, or NOT_INCLUDED
     */
    inline std::size_t index(std::size_t axis0_index, std::size_t axis1_index) const {
        return nodesSet.indexOf(rectangularMesh.index(axis0_index, axis1_index));
    }

    using RectangularFilteredMeshBase<DIM>::index;
    using RectangularFilteredMeshBase<DIM>::at;

    /**
     * Get point with given mesh indices.
     * @param index0 index of point in axis0
     * @param index1 index of point in axis1
     * @return point with given @p index
     */
    inline Vec<2, double> at(std::size_t index0, std::size_t index1) const {
        return rectangularMesh.at(index0, index1);
    }

    /**
     * Get point with given x and y indexes.
     * @param axis0_index index of axis0, from 0 to axis[0]->size()-1
     * @param axis1_index index of axis1, from 0 to axis[1]->size()-1
     * @return point with given axis0 and axis1 indexes
     */
    inline Vec<2,double> operator()(std::size_t axis0_index, std::size_t axis1_index) const {
        return rectangularMesh.operator()(axis0_index, axis1_index);
    }

private:
    bool canBeIncluded(const Vec<2>& point) const {
        return
            rectangularMesh.axis[0]->at(0) <= point[0] && point[0] <= rectangularMesh.axis[0]->at(rectangularMesh.axis[0]->size()-1) &&
            rectangularMesh.axis[1]->at(0) <= point[1] && point[1] <= rectangularMesh.axis[1]->at(rectangularMesh.axis[1]->size()-1);
    }

    bool prepareInterpolation(const Vec<2>& point, Vec<2>& wrapped_point, std::size_t& index0_lo, std::size_t& index0_hi, std::size_t& index1_lo, std::size_t& index1_hi, std::size_t& rectmesh_index_lo, const InterpolationFlags& flags) const;

public:
    /**
     * Calculate (using linear interpolation) value of data in point using data in points described by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(const RandomAccessContainer& data, const Vec<2>& point, const InterpolationFlags& flags) const
        -> typename std::remove_reference<decltype(data[0])>::type
    {
        Vec<2> wrapped_point;
        std::size_t index0_lo, index0_hi, index1_lo, index1_hi, rectmesh_index_lo;

        if (!prepareInterpolation(point, wrapped_point, index0_lo, index0_hi, index1_lo, index1_hi, rectmesh_index_lo, flags))
            return NaNfor<decltype(data[0])>();

        return flags.postprocess(point,
                                 interpolation::bilinear(
                                     rectangularMesh.axis[0]->at(index0_lo), rectangularMesh.axis[0]->at(index0_hi),
                                     rectangularMesh.axis[1]->at(index1_lo), rectangularMesh.axis[1]->at(index1_hi),
                                     data[nodesSet.indexOf(rectmesh_index_lo)],
                                     data[index(index0_hi, index1_lo)],
                                     data[index(index0_hi, index1_hi)],
                                     data[index(index0_lo, index1_hi)],
                                     wrapped_point.c0, wrapped_point.c1));
    }

    /**
     * Calculate (using nearest neighbor interpolation) value of data in point using data in points described by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateNearestNeighbor(const RandomAccessContainer& data, const Vec<2>& point, const InterpolationFlags& flags) const
        -> typename std::remove_reference<decltype(data[0])>::type
    {
        Vec<2> wrapped_point;
        std::size_t index0_lo, index0_hi, index1_lo, index1_hi, rectmesh_index_lo;

        if (!prepareInterpolation(point, wrapped_point, index0_lo, index0_hi, index1_lo, index1_hi, rectmesh_index_lo, flags))
            return NaNfor<decltype(data[0])>();

        return flags.postprocess(point,
                                 data[this->index(
                                     nearest(wrapped_point.c0, *rectangularMesh.axis[0], index0_lo, index0_hi),
                                     nearest(wrapped_point.c1, *rectangularMesh.axis[1], index1_lo, index1_hi)
                                 )]);
    }

    /**
     * Convert mesh indexes of a bottom-left corner of an element to the index of this element.
     * @param axis0_index index of the corner along the axis0 (left), from 0 to axis[0]->size()-1
     * @param axis1_index index of the corner along the axis1 (bottom), from 0 to axis[1]->size()-1
     * @return index of the element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndexes(std::size_t axis0_index, std::size_t axis1_index) const {
        return elementsSet.indexOf(rectangularMesh.getElementIndexFromLowIndexes(axis0_index, axis1_index));
    }

    /**
     * Get an area of a given element.
     * @param index0, index1 axis 0 and axis 1 indexes of the element
     * @return the area of the element with given indexes
     */
    double getElementArea(std::size_t index0, std::size_t index1) const {
        return rectangularMesh.getElementArea(index0, index1);
    }

    /**
     * Get point in center of Elements.
     * @param index0, index1 index of Elements
     * @return point in center of element with given index
     */
    Vec<2, double> getElementMidpoint(std::size_t index0, std::size_t index1) const {
        return rectangularMesh.getElementMidpoint(index0, index1);
    }

    /**
     * Get element as rectangle.
     * @param index0, index1 index of Elements
     * @return box of elements with given index
     */
    Box2D getElementBox(std::size_t index0, std::size_t index1) const {
        return rectangularMesh.getElementBox(index0, index1);
    }

protected:  // boundaries code:

    // Common code for: left, right, bottom, top boundries:
    template <int CHANGE_DIR>
    struct BoundaryIteratorImpl: public plask::BoundaryNodeSetImpl::IteratorImpl {

        const RectangularFilteredMeshBase<DIM> &mesh;

        /// current indexes
        Vec<2, std::size_t> index;

        /// past the last index of change direction
        std::size_t endIndex;

        BoundaryIteratorImpl(const RectangularFilteredMeshBase<DIM>& mesh, Vec<DIM, std::size_t> index, std::size_t endIndex)
            : mesh(mesh), index(index), endIndex(endIndex)
        {
            // go to the first index existed in order to make dereference possible:
            while (this->index[CHANGE_DIR] < this->endIndex && mesh.index(this->index) == NOT_INCLUDED)
                ++this->index[CHANGE_DIR];
        }

        void increment() override {
            do {
                ++index[CHANGE_DIR];
            } while (index[CHANGE_DIR] < endIndex && mesh.index(index) == NOT_INCLUDED);
        }

        bool equal(const plask::BoundaryNodeSetImpl::IteratorImpl& other) const override {
            const BoundaryIteratorImpl& o = static_cast<const BoundaryIteratorImpl&>(other);
            return index == o.index && endIndex == o.endIndex;
        }

        std::size_t dereference() const override {
            return mesh.index(index);
        }

        typename plask::BoundaryNodeSetImpl::IteratorImpl* clone() const override {
            return new BoundaryIteratorImpl<CHANGE_DIR>(*this);
        }

    };

    template <int CHANGE_DIR>
    struct BoundaryNodeSetImpl: public BoundaryNodeSetWithMeshImpl<RectangularFilteredMeshBase<2>> {

        using typename BoundaryNodeSetWithMeshImpl<RectangularFilteredMeshBase<2>>::const_iterator;

        /// first index
        Vec<2, std::size_t> index;

        /// past the last index of change direction
        std::size_t endIndex;

        BoundaryNodeSetImpl(const RectangularFilteredMeshBase<DIM>& mesh, Vec<2, std::size_t> index, std::size_t endIndex)
            : BoundaryNodeSetWithMeshImpl<RectangularFilteredMeshBase<2>>(mesh), index(index), endIndex(endIndex) {}

        BoundaryNodeSetImpl(const RectangularFilteredMeshBase<DIM>& mesh, std::size_t index0, std::size_t index1, std::size_t endIndex)
            : BoundaryNodeSetWithMeshImpl<RectangularFilteredMeshBase<2>>(mesh), index(index0, index1), endIndex(endIndex) {}

        bool contains(std::size_t mesh_index) const override {
            if (mesh_index >= this->mesh.size()) return false;
            Vec<2, std::size_t> mesh_indexes = this->mesh.indexes(mesh_index);
            for (int i = 0; i < 2; ++i)
                if (i == CHANGE_DIR) {
                    if (mesh_indexes[i] < index[i] || mesh_indexes[i] >= endIndex) return false;
                } else
                    if (mesh_indexes[i] != index[i]) return false;
            return true;
        }

        const_iterator begin() const override {
            return Iterator(new BoundaryIteratorImpl<CHANGE_DIR>(this->mesh, index, endIndex));
        }

        const_iterator end() const override {
            Vec<2, std::size_t> index_end = index;
            index_end[CHANGE_DIR] = endIndex;
            return Iterator(new BoundaryIteratorImpl<CHANGE_DIR>(this->mesh, index_end, endIndex));
        }
    };

public:     // boundaries:

    BoundaryNodeSet createVerticalBoundaryAtLine(std::size_t line_nr_axis0) const override {
        return createVerticalBoundaryAtLine(line_nr_axis0, 0, rectangularMesh.axis[1]->size());
    }

    BoundaryNodeSet createVerticalBoundaryAtLine(std::size_t line_nr_axis0, std::size_t indexBegin, std::size_t indexEnd) const override {
        return new BoundaryNodeSetImpl<1>(*this, line_nr_axis0, indexBegin, indexEnd);
    }

    BoundaryNodeSet createVerticalBoundaryNear(double axis0_coord) const override {
        return createVerticalBoundaryAtLine(rectangularMesh.axis[0]->findNearestIndex(axis0_coord));
    }

    BoundaryNodeSet createVerticalBoundaryNear(double axis0_coord, double from, double to) const override {
        std::size_t begInd, endInd;
        if (!details::getIndexesInBoundsExt(begInd, endInd, *rectangularMesh.axis[1], from, to))
            return new EmptyBoundaryImpl();
        return createVerticalBoundaryAtLine(rectangularMesh.axis[0]->findNearestIndex(axis0_coord), begInd, endInd);
    }

    BoundaryNodeSet createLeftBoundary() const override {
        return createVerticalBoundaryAtLine(boundaryIndex[0].lo);
    }

    BoundaryNodeSet createRightBoundary() const override {
        return createVerticalBoundaryAtLine(boundaryIndex[0].up);
    }

    BoundaryNodeSet createLeftOfBoundary(const Box2D& box) const override {
        std::size_t line, begInd, endInd;
        if (details::getLineLo(line, *rectangularMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd, endInd, *rectangularMesh.axis[1], box.lower.c1, box.upper.c1))
            return createVerticalBoundaryAtLine(line, begInd, endInd);
        else
            return new EmptyBoundaryImpl();
    }

   BoundaryNodeSet createRightOfBoundary(const Box2D& box) const override {
        std::size_t line, begInd, endInd;
        if (details::getLineHi(line, *rectangularMesh.axis[0], box.lower.c0, box.upper.c0) &&
            details::getIndexesInBounds(begInd, endInd, *rectangularMesh.axis[1], box.lower.c1, box.upper.c1))
            return createVerticalBoundaryAtLine(line, begInd, endInd);
        else
            return new EmptyBoundaryImpl();
    }

    BoundaryNodeSet createBottomOfBoundary(const Box2D& box) const override {
        std::size_t line, begInd, endInd;
        if (details::getLineLo(line, *rectangularMesh.axis[1], box.lower.c1, box.upper.c1) &&
            details::getIndexesInBounds(begInd, endInd, *rectangularMesh.axis[0], box.lower.c0, box.upper.c0))
            return createHorizontalBoundaryAtLine(line, begInd, endInd);
        else
            return new EmptyBoundaryImpl();
    }

    BoundaryNodeSet createTopOfBoundary(const Box2D& box) const override {
       std::size_t line, begInd, endInd;
       if (details::getLineHi(line, *rectangularMesh.axis[1], box.lower.c1, box.upper.c1) &&
           details::getIndexesInBounds(begInd, endInd, *rectangularMesh.axis[0], box.lower.c0, box.upper.c0))
           return createHorizontalBoundaryAtLine(line, begInd, endInd);
       else
           return new EmptyBoundaryImpl();
    }

    BoundaryNodeSet createHorizontalBoundaryAtLine(std::size_t line_nr_axis1) const override {
        return createHorizontalBoundaryAtLine(line_nr_axis1, 0, rectangularMesh.axis[0]->size());
    }

    BoundaryNodeSet createHorizontalBoundaryAtLine(std::size_t line_nr_axis1, std::size_t indexBegin, std::size_t indexEnd) const override {
        return new BoundaryNodeSetImpl<0>(*this, indexBegin, line_nr_axis1, indexEnd);
    }

    BoundaryNodeSet createHorizontalBoundaryNear(double axis1_coord) const override {
        return createHorizontalBoundaryAtLine(rectangularMesh.axis[1]->findNearestIndex(axis1_coord));
    }

    BoundaryNodeSet createHorizontalBoundaryNear(double axis1_coord, double from, double to) const override {
        std::size_t begInd, endInd;
        if (!details::getIndexesInBoundsExt(begInd, endInd, *rectangularMesh.axis[0], from, to))
            return new EmptyBoundaryImpl();
        return createHorizontalBoundaryAtLine(rectangularMesh.axis[1]->findNearestIndex(axis1_coord), begInd, endInd);
    }

    BoundaryNodeSet createTopBoundary() const override {
        return createHorizontalBoundaryAtLine(boundaryIndex[1].up);
    }

    BoundaryNodeSet createBottomBoundary() const override {
        return createHorizontalBoundaryAtLine(boundaryIndex[1].lo);
    }

};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularFilteredMesh2D, SrcT, DstT, INTERPOLATION_LINEAR> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularFilteredMesh2D>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->empty()) throw BadMesh("interpolate", "Source mesh empty");
        return new LinearInterpolatedLazyDataImpl< DstT, RectangularFilteredMesh2D, SrcT >(src_mesh, src_vec, dst_mesh, flags);
    }
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularFilteredMesh2D, SrcT, DstT, INTERPOLATION_NEAREST> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularFilteredMesh2D>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->empty()) throw BadMesh("interpolate", "Source mesh empty");
        return new NearestNeighborInterpolatedLazyDataImpl< DstT, RectangularFilteredMesh2D, SrcT >(src_mesh, src_vec, dst_mesh, flags);
    }
};

}   // namespace plask

#endif // PLASK__RECTANGULAR_FILTERED2D_H
