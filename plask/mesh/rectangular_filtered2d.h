#ifndef PLASK__RECTANGULAR_FILTERED2D_H
#define PLASK__RECTANGULAR_FILTERED2D_H

#include "rectangular_filtered_common.h"

namespace plask {

/**
 * Rectangular mesh which uses (and indexes) only chosen elements and all nodes in their corners.
 *
 * Objects of this class can be constructed from instences of full rectangular mesh (RectangularFilteredMesh2D)
 * and they can use the same boundary conditions (BoundaryConditions instance for full mesh accepts also objets of this class).
 * Interpolation methods return NaN-s for all elements which have not been chosen.
 */
struct PLASK_API RectangularFilteredMesh2D: public RectangularFilteredMeshBase<2> {

    typedef std::function<bool(const RectangularMesh2D::Element&)> Predicate;

    class PLASK_API Element {

        const RectangularFilteredMesh2D& filteredMesh;

        //std::uint32_t elementNumber;    ///< index of element in oryginal mesh
        std::size_t index0, index1; // probably this form allows to do most operation fastest in average, low indexes of element corner or just element indexes

        /// Index of element. If it equals to UNKNOWN_ELEMENT_INDEX, it will be calculated on-demand from index0 and index1.
        mutable std::size_t elementIndex;

        const RectangularMesh<2>& fullMesh() const { return filteredMesh.fullMesh; }

    public:

        enum: std::size_t { UNKNOWN_ELEMENT_INDEX = std::numeric_limits<std::size_t>::max() };

        Element(const RectangularFilteredMesh2D& filteredMesh, std::size_t elementIndex, std::size_t index0, std::size_t index1)
            : filteredMesh(filteredMesh), index0(index0), index1(index1), elementIndex(elementIndex)
        {
        }

        Element(const RectangularFilteredMesh2D& filteredMesh, std::size_t elementIndex, std::size_t elementIndexOfFullMesh)
            : filteredMesh(filteredMesh), elementIndex(elementIndex)
        {
            const std::size_t v = fullMesh().getElementMeshLowIndex(elementIndexOfFullMesh);
            index0 = fullMesh().index0(v);
            index1 = fullMesh().index1(v);
        }

        Element(const RectangularFilteredMesh2D& filteredMesh, std::size_t elementIndex)
            : Element(filteredMesh, elementIndex, filteredMesh.elementSet.at(elementIndex))
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
        inline double getLower0() const { return fullMesh().axis[0]->at(index0); }

        /// \return vert coordinate of the bottom edge of the element
        inline double getLower1() const { return fullMesh().axis[1]->at(index1); }

        /// \return tran index of the right edge of the element
        inline std::size_t getUpperIndex0() const { return index0+1; }

        /// \return vert index of the top edge of the element
        inline std::size_t getUpperIndex1() const { return index1+1; }

        /// \return tran coordinate of the right edge of the element
        inline double getUpper0() const { return fullMesh().axis[0]->at(getUpperIndex0()); }

        /// \return vert coordinate of the top edge of the element
        inline double getUpper1() const { return fullMesh().axis[1]->at(getUpperIndex1()); }

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
            if (elementIndex == UNKNOWN_ELEMENT_INDEX)
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

    struct PLASK_API Elements: ElementsBase<RectangularFilteredMesh2D> {

        explicit Elements(const RectangularFilteredMesh2D& mesh): ElementsBase(mesh) { mesh.ensureHasElements(); }

        Element operator()(std::size_t i0, std::size_t i1) const { return Element(*filteredMesh, Element::UNKNOWN_ELEMENT_INDEX, i0, i1); }

    };  // struct Elements

    /// Element mesh
    struct PLASK_API ElementMesh: ElementMeshBase<RectangularFilteredMesh2D> {

        explicit ElementMesh(const RectangularFilteredMesh2D* originalMesh): ElementMeshBase<RectangularFilteredMesh2D>(originalMesh) {}

        /**
         * Calculate this mesh index using indexes of axis0 and axis1.
         * \param axis0_index index of axis0, from 0 to axis[0]->size()-1
         * \param axis1_index index of axis1, from 0 to axis[1]->size()-1
         * \return this mesh index, from 0 to size()-1, or NOT_INCLUDED
         */
        inline std::size_t index(std::size_t axis0_index, std::size_t axis1_index) const {
            return originalMesh->elementSet.indexOf(originalMesh->fullMesh.getElement(axis0_index, axis1_index).getIndex());
        }


        bool prepareInterpolation(const Vec<2>& point, Vec<2>& wrapped_point, std::size_t& index0_lo, std::size_t& index0_hi, std::size_t& index1_lo, std::size_t& index1_hi,
                                  const InterpolationFlags& flags) const {
            return originalMesh->prepareInterpolation(point, wrapped_point, index0_lo, index0_hi, index1_lo, index1_hi, flags);
        }

        /**
         * Calculate (using linear interpolation) value of data in point using data in points described by this mesh.
         * \param data values of data in points describe by this mesh
         * \param point point in which value should be calculate
         * \return interpolated value in point \p point
         */
        template <typename RandomAccessContainer>
        auto interpolateLinear(const RandomAccessContainer& data, const Vec<2>& point, const InterpolationFlags& flags) const
            -> typename std::remove_reference<decltype(data[0])>::type {
            typedef typename std::remove_reference<decltype(data[0])>::type DataT;
            Vec<2> p;
            size_t index0, index0_hi, index1, index1_hi;

            if (!originalMesh->prepareInterpolation(point, p, index0, index0_hi, index1, index1_hi, flags))
                return NaNfor<decltype(data[0])>();

            Vec<2> pa = originalMesh->fullMesh.getElement(index0, index1).getMidpoint();

            size_t step0 = (p.c0 < pa.c0)?
                (index0 == 0)? 0 : -1 :
                (index0_hi == originalMesh->fullMesh.axis[0]->size()-1)? 0 : 1;
            size_t step1 = (p.c1 < pa.c1)?
                (index1 == 0)? 0 : -1 :
                (index1_hi == originalMesh->fullMesh.axis[1]->size()-1)? 0 : 1;

            size_t index_aa = index(index0, index1), index_ab, index_ba, index_bb;

            typename std::remove_const<DataT>::type data_aa = data[index_aa], data_ab, data_ba, data_bb;

            if (step0 == 0 && step1 == 0) {
                index_ab = index_ba = index_bb = index_aa;
                data_ab, data_ba, data_bb = data_aa;
            } else {
                index_ab = index(index0, index1+step1);
                index_ba = index(index0+step0, index1);
                index_bb = index(index0+step0, index1+step1);
                data_ab = (index_ab != Element::UNKNOWN_ELEMENT_INDEX)? data[index_ab] : data_aa;
                data_ba = (index_ba != Element::UNKNOWN_ELEMENT_INDEX)? data[index_ba] : data_aa;
                data_bb = (index_bb != Element::UNKNOWN_ELEMENT_INDEX)? data[index_bb] : data_ab + data_ba - data_aa;
            }

            Vec<2> pb = originalMesh->fullMesh.getElement(index0+step0, index1+step1).getMidpoint();
            if (step0 == 0) pb.c0 += 1.; if (step1 == 0) pb.c1 += 1.;

            return flags.postprocess(point,
                interpolation::bilinear(pa.c0, pb.c0, pa.c1, pb.c1,
                                        data_aa, data_ba, data_bb, data_ab, p.c0, p.c1));
        }

        /**
         * Calculate (using nearest neighbor interpolation) value of data in point using data in points described by this mesh.
         * \param data values of data in points describe by this mesh
         * \param point point in which value should be calculate
         * \return interpolated value in point \p point
         */
        template <typename RandomAccessContainer>
        auto interpolateNearestNeighbor(const RandomAccessContainer& data, const Vec<2>& point, const InterpolationFlags& flags) const
            -> typename std::remove_reference<decltype(data[0])>::type {
            Vec<2> wrapped_point;
            std::size_t index0_lo, index0_hi, index1_lo, index1_hi;

            if (!originalMesh->prepareInterpolation(point, wrapped_point, index0_lo, index0_hi, index1_lo, index1_hi, flags))
                return NaNfor<decltype(data[0])>();

            return flags.postprocess(point, data[this->index(index0_lo, index1_lo)]);
        }

    };  // struct ElementMesh

    /**
     * Construct empty/unitialized mesh. One should call reset() method before using this.
     */
    RectangularFilteredMesh2D() = default;

    /**
     * Change a selection of elements used to once pointed by a given @p predicate.
     * @param predicate predicate which returns either @c true for accepting element or @c false for rejecting it
     */
    void reset(const Predicate& predicate);

    /**
     * Construct filtered mesh with elements of @p rectangularMesh chosen by a @p predicate.
     * Preserve order of elements and nodes of @p rectangularMesh.
     * @param rectangularMesh input mesh, before filtering
     * @param predicate predicate which returns either @c true for accepting element or @c false for rejecting it
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if @c true) or shared with @p rectangularMesh (if @c false; default)
     */
    RectangularFilteredMesh2D(const RectangularMesh<2>& fullMesh, const Predicate& predicate, bool clone_axes = false);

    /**
     * Change parameters of this mesh to use elements of @p rectangularMesh chosen by a @p predicate.
     * Preserve order of elements and nodes of @p rectangularMesh.
     * @param rectangularMesh input mesh, before filtering
     * @param predicate predicate which returns either @c true for accepting element or @c false for rejecting it
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if @c true) or shared with @p rectangularMesh (if @c false; default)
     */
    void reset(const RectangularMesh<2>& fullMesh, const Predicate& predicate, bool clone_axes = false);

    /**
     * Construct filtered mesh with all elements of @c rectangularMesh which have required materials in the midpoints.
     * Preserve order of elements and nodes of @p rectangularMesh.
     * @param rectangularMesh input mesh, before filtering
     * @param geom geometry to get materials from
     * @param materialPredicate predicate which returns either @c true for accepting material or @c false for rejecting it
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if @c true) or shared (if @c false; default)
     */
    RectangularFilteredMesh2D(const RectangularMesh<2>& rectangularMesh, const GeometryD<2>& geom, const std::function<bool(shared_ptr<const Material>)> materialPredicate, bool clone_axes = false)
        : RectangularFilteredMesh2D(rectangularMesh,
                                    [&](const RectangularMesh2D::Element& el) { return materialPredicate(geom.getMaterial(el.getMidpoint())); },
                                    clone_axes)
    {
    }

    /**
     * Change parameters of this mesh to use all elements of @c rectangularMesh which have required materials in the midpoints.
     * Preserve order of elements and nodes of @p rectangularMesh.
     * @param rectangularMesh input mesh, before filtering
     * @param geom geometry to get materials from
     * @param materialPredicate predicate which returns either @c true for accepting material or @c false for rejecting it
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if @c true) or shared (if @c false; default)
     */
    void reset(const RectangularMesh<2>& rectangularMesh, const GeometryD<2>& geom, const std::function<bool(shared_ptr<const Material>)> materialPredicate, bool clone_axes = false) {
        reset(rectangularMesh,
              [&](const RectangularMesh2D::Element& el) { return materialPredicate(geom.getMaterial(el.getMidpoint())); },
              clone_axes);
    }

    /**
     * Construct filtered mesh with all elements of @c rectangularMesh which have required kinds of materials (in the midpoints).
     * Preserve order of elements and nodes of @p rectangularMesh.
     * @param rectangularMesh input mesh, before filtering
     * @param geom geometry to get materials from
     * @param materialKinds one or more kinds of material encoded with bit @c or operation,
     *        e.g. @c DIELECTRIC|METAL for selecting all dielectrics and metals,
     *        or @c ~(DIELECTRIC|METAL) for selecting everything else
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if @c true) or shared (if @c false; default)
     */
    RectangularFilteredMesh2D(const RectangularMesh<2>& rectangularMesh, const GeometryD<2>& geom, unsigned int materialKinds, bool clone_axes = false)
        : RectangularFilteredMesh2D(rectangularMesh,
                                    [&](const RectangularMesh2D::Element& el) { return (geom.getMaterial(el.getMidpoint())->kind() & materialKinds) != 0; },
                                    clone_axes)
    {
    }

    /**
     * Change parameters of this mesh to use all elements of @c rectangularMesh which have required kinds of materials (in the midpoints).
     * Preserve order of elements and nodes of @p rectangularMesh.
     * @param rectangularMesh input mesh, before filtering
     * @param geom geometry to get materials from
     * @param materialKinds one or more kinds of material encoded with bit @c or operation,
     *        e.g. @c DIELECTRIC|METAL for selecting all dielectrics and metals,
     *        or @c ~(DIELECTRIC|METAL) for selecting everything else
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if @c true) or shared (if @c false; default)
     */
    void reset(const RectangularMesh<2>& rectangularMesh, const GeometryD<2>& geom, unsigned int materialKinds, bool clone_axes = false) {
        reset(rectangularMesh,
             [&](const RectangularMesh2D::Element& el) { return (geom.getMaterial(el.getMidpoint())->kind() & materialKinds) != 0; },
             clone_axes);
    }

    /**
     * Construct a mesh with given set of nodes.
     *
     * Set of elements are calculated on-demand, just before the first use, according to the rule:
     * An element is selected if and only if all its vertices are included in the @p nodeSet.
     *
     * This constructor is used by getElementMesh.
     */
    RectangularFilteredMesh2D(const RectangularMesh<DIM>& rectangularMesh, Set nodeSet, bool clone_axes = false)
        : RectangularFilteredMeshBase(rectangularMesh, std::move(nodeSet), clone_axes) {}

    Elements elements() const { return Elements(*this); }
    Elements getElements() const { return elements(); }

    Element element(std::size_t i0, std::size_t i1) const { ensureHasElements(); return Element(*this, Element::UNKNOWN_ELEMENT_INDEX, i0, i1); }
    Element getElement(std::size_t i0, std::size_t i1) const { return element(i0, i1); }

    /**
     * Get an element with a given index @p i.
     * @param i index of the element
     * @return the element
     */
    Element element(std::size_t i) const { ensureHasElements(); return Element(*this, i); }

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
        return nodeSet.indexOf(fullMesh.index(axis0_index, axis1_index));
    }

    using RectangularFilteredMeshBase<2>::index;
    using RectangularFilteredMeshBase<2>::at;

    /**
     * Get point with given mesh indices.
     * @param index0 index of point in axis[0]
     * @param index1 index of point in axis[1]
     * @return point with given @p index
     */
    inline Vec<2, double> at(std::size_t index0, std::size_t index1) const {
        return fullMesh.at(index0, index1);
    }

    /**
     * Get point with given x and y indexes.
     * @param axis0_index index of axis[0], from 0 to axis[0]->size()-1
     * @param axis1_index index of axis[1], from 0 to axis[1]->size()-1
     * @return point with given axis0 and axis1 indexes
     */
    inline Vec<2,double> operator()(std::size_t axis0_index, std::size_t axis1_index) const {
        return fullMesh.operator()(axis0_index, axis1_index);
    }

    /**
     * Return a mesh that enables iterating over middle points of the selected rectangles.
     * @return new rectilinear filtered mesh with points in the middles of original, selected rectangles
     */
    shared_ptr<RectangularFilteredMesh2D::ElementMesh> getElementMesh() const {
        return make_shared<RectangularFilteredMesh2D::ElementMesh>(this);
    }

  private:

    void initNodesAndElements(const RectangularFilteredMesh2D::Predicate &predicate);

    bool canBeIncluded(const Vec<2>& point) const {
        return
            fullMesh.axis[0]->at(0) - point[0] < MIN_DISTANCE && point[0] - fullMesh.axis[0]->at(fullMesh.axis[0]->size()-1) < MIN_DISTANCE &&
            fullMesh.axis[1]->at(0) - point[1] < MIN_DISTANCE && point[1] - fullMesh.axis[1]->at(fullMesh.axis[1]->size()-1) < MIN_DISTANCE;
    }

  public:
    /** Prepare point for inteprolation
     * \param point point to check
     * \param[out] wrapped_point point after wrapping with interpolation flags
     * \param[out] index0_lo,index0_hi surrounding indices in the rectantular mesh for axis0
     * \param[out] index1_lo,index1_hi surrounding indices in the rectantular mesh for axis1
     * \param flags interpolation flags
     * \returns \c false if the point falls in the hole or outside of the mesh, \c true if it can be interpolated
     */
    bool prepareInterpolation(const Vec<2>& point, Vec<2>& wrapped_point,
                              std::size_t& index0_lo, std::size_t& index0_hi,
                              std::size_t& index1_lo, std::size_t& index1_hi,
                              const InterpolationFlags& flags) const;

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
        std::size_t index0_lo, index0_hi, index1_lo, index1_hi;

        if (!prepareInterpolation(point, wrapped_point, index0_lo, index0_hi, index1_lo, index1_hi, flags))
            return NaNfor<decltype(data[0])>();

        return flags.postprocess(point,
                                 interpolation::bilinear(
                                     fullMesh.axis[0]->at(index0_lo), fullMesh.axis[0]->at(index0_hi),
                                     fullMesh.axis[1]->at(index1_lo), fullMesh.axis[1]->at(index1_hi),
                                     data[index(index0_lo, index1_lo)],
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
        std::size_t index0_lo, index0_hi, index1_lo, index1_hi;

        if (!prepareInterpolation(point, wrapped_point, index0_lo, index0_hi, index1_lo, index1_hi, flags))
            return NaNfor<decltype(data[0])>();

        return flags.postprocess(point,
                                 data[this->index(
                                     nearest(wrapped_point.c0, *fullMesh.axis[0], index0_lo, index0_hi),
                                     nearest(wrapped_point.c1, *fullMesh.axis[1], index1_lo, index1_hi)
                                 )]);
    }

    /**
     * Convert mesh indexes of a bottom-left corner of an element to the index of this element.
     * @param axis0_index index of the corner along the axis0 (left), from 0 to axis[0]->size()-1
     * @param axis1_index index of the corner along the axis1 (bottom), from 0 to axis[1]->size()-1
     * @return index of the element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndexes(std::size_t axis0_index, std::size_t axis1_index) const {
        return ensureHasElements().indexOf(fullMesh.getElementIndexFromLowIndexes(axis0_index, axis1_index));
    }

    /**
     * Get an area of a given element.
     * @param index0, index1 axis 0 and axis 1 indexes of the element
     * @return the area of the element with given indexes
     */
    double getElementArea(std::size_t index0, std::size_t index1) const {
        return fullMesh.getElementArea(index0, index1);
    }

    /**
     * Get point in center of Elements.
     * @param index0, index1 index of Elements
     * @return point in center of element with given index
     */
    Vec<2, double> getElementMidpoint(std::size_t index0, std::size_t index1) const {
        return fullMesh.getElementMidpoint(index0, index1);
    }

    /**
     * Get element as rectangle.
     * @param index0, index1 index of Elements
     * @return box of elements with given index
     */
    Box2D getElementBox(std::size_t index0, std::size_t index1) const {
        return fullMesh.getElementBox(index0, index1);
    }

  protected:  // boundaries code:

    // Common code for: left, right, bottom, top boundries:
    template <int CHANGE_DIR>
    struct BoundaryIteratorImpl: public BoundaryNodeSetImpl::IteratorImpl {

        const RectangularFilteredMeshBase<2> &mesh;

        /// current indexes
        Vec<2, std::size_t> index;

        /// past the last index of change direction
        std::size_t endIndex;

        BoundaryIteratorImpl(const RectangularFilteredMeshBase<2>& mesh, Vec<2, std::size_t> index, std::size_t endIndex)
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

        bool equal(const BoundaryNodeSetImpl::IteratorImpl& other) const override {
            return index == static_cast<const BoundaryIteratorImpl&>(other).index;
        }

        std::size_t dereference() const override {
            return mesh.index(index);
        }

        typename BoundaryNodeSetImpl::IteratorImpl* clone() const override {
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

        BoundaryNodeSetImpl(const RectangularFilteredMeshBase<2>& mesh, Vec<2, std::size_t> index, std::size_t endIndex)
            : BoundaryNodeSetWithMeshImpl<RectangularFilteredMeshBase<2>>(mesh), index(index), endIndex(endIndex) {}

        BoundaryNodeSetImpl(const RectangularFilteredMeshBase<2>& mesh, std::size_t index0, std::size_t index1, std::size_t endIndex)
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

    BoundaryNodeSet createVerticalBoundaryAtLine(std::size_t line_nr_axis0) const override;

    BoundaryNodeSet createVerticalBoundaryAtLine(std::size_t line_nr_axis0, std::size_t indexBegin, std::size_t indexEnd) const override;

    BoundaryNodeSet createVerticalBoundaryNear(double axis0_coord) const override;

    BoundaryNodeSet createVerticalBoundaryNear(double axis0_coord, double from, double to) const override;

    BoundaryNodeSet createLeftBoundary() const override;

    BoundaryNodeSet createRightBoundary() const override;

    BoundaryNodeSet createLeftOfBoundary(const Box2D& box) const override;

    BoundaryNodeSet createRightOfBoundary(const Box2D& box) const override;

    BoundaryNodeSet createBottomOfBoundary(const Box2D& box) const override;

    BoundaryNodeSet createTopOfBoundary(const Box2D& box) const override;

    BoundaryNodeSet createHorizontalBoundaryAtLine(std::size_t line_nr_axis1) const override;

    BoundaryNodeSet createHorizontalBoundaryAtLine(std::size_t line_nr_axis1, std::size_t indexBegin, std::size_t indexEnd) const override;

    BoundaryNodeSet createHorizontalBoundaryNear(double axis1_coord) const override;

    BoundaryNodeSet createHorizontalBoundaryNear(double axis1_coord, double from, double to) const override;

    BoundaryNodeSet createTopBoundary() const override;

    BoundaryNodeSet createBottomBoundary() const override;

};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularFilteredMesh2D, SrcT, DstT, INTERPOLATION_LINEAR> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularFilteredMesh2D>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->empty()) throw BadMesh("interpolate", "Source mesh empty");
        return new LinearInterpolatedLazyDataImpl<DstT, RectangularFilteredMesh2D, SrcT>(src_mesh, src_vec, dst_mesh, flags);
    }
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularFilteredMesh2D, SrcT, DstT, INTERPOLATION_NEAREST> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularFilteredMesh2D>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->empty()) throw BadMesh("interpolate", "Source mesh empty");
        return new NearestNeighborInterpolatedLazyDataImpl<DstT, RectangularFilteredMesh2D, SrcT>(src_mesh, src_vec, dst_mesh, flags);
    }
};


template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularFilteredMesh2D::ElementMesh, SrcT, DstT, INTERPOLATION_LINEAR> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularFilteredMesh2D::ElementMesh>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->empty()) throw BadMesh("interpolate", "Source mesh empty");
        return new LinearInterpolatedLazyDataImpl<DstT, RectangularFilteredMesh2D::ElementMesh, SrcT>(src_mesh, src_vec, dst_mesh, flags);
    }
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularFilteredMesh2D::ElementMesh, SrcT, DstT, INTERPOLATION_NEAREST> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularFilteredMesh2D::ElementMesh>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->empty()) throw BadMesh("interpolate", "Source mesh empty");
        return new NearestNeighborInterpolatedLazyDataImpl<DstT, RectangularFilteredMesh2D::ElementMesh, SrcT>(src_mesh, src_vec, dst_mesh, flags);
    }
};

}   // namespace plask

#endif // PLASK__RECTANGULAR_FILTERED2D_H
