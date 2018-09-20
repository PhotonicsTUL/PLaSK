#ifndef PLASK__RECTANGULAR_FILTERED3D_H
#define PLASK__RECTANGULAR_FILTERED3D_H

#include "rectangular_filtered_common.h"

namespace plask {

/**
 * Rectangular mesh which uses (and indexes) only chosen elements and all nodes in their corners.
 *
 * Objects of this class can be constructed from instences of full rectangular mesh (RectangularFilteredMesh3D)
 * and they can use the same boundary conditions (BoundaryConditions instance for full mesh accepts also objets of this class).
 * Interpolation methods return NaN-s for all elements which have not been chosen.
 */
struct PLASK_API RectangularFilteredMesh3D: public RectangularFilteredMeshBase<3> {

    /**
     * Calculate index of axis2 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis2, from 0 to axis2->size()-1
     */
    inline std::size_t index2(std::size_t mesh_index) const {   // method missing in the base as it is specific for 3D
        return fullMesh.index2(nodeSet.at(mesh_index));
    }

    /**
     * Calculate index of middle axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of major axis, from 0 to middleIndex.size()-1
     */
    inline std::size_t middleIndex(std::size_t mesh_index) const {   // method missing in the base as it is specific for 3D
        return fullMesh.middleIndex(nodeSet.at(mesh_index));
    }

    /**
     * Get number of elements (for FEM method) in the third direction.
     * @return number of elements in the full rectangular mesh in the third direction (axis2 direction).
     */
    std::size_t getElementsCount2() const {  // method missing in the base as it is specific for 3D
        return fullMesh.getElementsCount2();
    }

    /**
     * Get third coordinate of point in the center of an elements.
     * @param index2 index of the element (axis2 index)
     * @return third coordinate of the point in the center of the element
     */
    double getElementMidpoint2(std::size_t index2) const {   // method missing in the base as it is specific for 3D
        return fullMesh.getElementMidpoint2(index2);
    }

    typedef std::function<bool(const RectangularMesh3D::Element&)> Predicate;

    class PLASK_API Element {

        const RectangularFilteredMesh3D& filteredMesh;

        //std::uint32_t elementNumber;    ///< index of element in oryginal mesh
        std::size_t index0, index1, index2; // probably this form allows to do most operation fastest in average, low indexes of element corner or just element indexes

        /// Index of element. If it equals to UNKNOWN_ELEMENT_INDEX, it will be calculated on-demand from index0 and index1.
        mutable std::size_t elementIndex;

        const RectangularMesh<3>& fullMesh() const { return filteredMesh.fullMesh; }

    public:

        enum: std::size_t { UNKNOWN_ELEMENT_INDEX = std::numeric_limits<std::size_t>::max() };

        Element(const RectangularFilteredMesh3D& filteredMesh, std::size_t elementIndex, std::size_t index0, std::size_t index1, std::size_t index2)
            : filteredMesh(filteredMesh), index0(index0), index1(index1), index2(index2), elementIndex(elementIndex)
        {
        }

        Element(const RectangularFilteredMesh3D& filteredMesh, std::size_t elementIndex, std::size_t elementIndexOfFullMesh)
            : filteredMesh(filteredMesh), elementIndex(elementIndex)
        {
            const std::size_t v = fullMesh().getElementMeshLowIndex(elementIndexOfFullMesh);
            index0 = fullMesh().index0(v);
            index1 = fullMesh().index1(v);
            index2 = fullMesh().index2(v);
        }

        Element(const RectangularFilteredMesh3D& filteredMesh, std::size_t elementIndex)
            : Element(filteredMesh, elementIndex, filteredMesh.elementSet.at(elementIndex))
        {}


        /// \return long index of the element
        inline std::size_t getIndex0() const { return index0; }

        /// \return tran index of the element
        inline std::size_t getIndex1() const { return index1; }

        /// \return vert index of the element
        inline std::size_t getIndex2() const { return index2; }

        /// \return long index of the back edge of the element
        inline std::size_t getLowerIndex0() const { return index0; }

        /// \return tran index of the left edge of the element
        inline std::size_t getLowerIndex1() const { return index1; }

        /// \return vert index of the bottom edge of the element
        inline std::size_t getLowerIndex2() const { return index2; }

        /// \return long coordinate of the back edge of the element
        inline double getLower0() const { return fullMesh().axis[0]->at(index0); }

        /// \return tran coordinate of the left edge of the element
        inline double getLower1() const { return fullMesh().axis[1]->at(index1); }

        /// \return vert coordinate of the bottom edge of the element
        inline double getLower2() const { return fullMesh().axis[2]->at(index2); }

        /// \return long index of the front edge of the element
        inline std::size_t getUpperIndex0() const { return index0+1; }

        /// \return tran index of the right edge of the element
        inline std::size_t getUpperIndex1() const { return index1+1; }

        /// \return vert index of the top edge of the element
        inline std::size_t getUpperIndex2() const { return index2+1; }

        /// \return long coordinate of the front edge of the element
        inline double getUpper0() const { return fullMesh().axis[0]->at(getUpperIndex0()); }

        /// \return tran coordinate of the right edge of the element
        inline double getUpper1() const { return fullMesh().axis[1]->at(getUpperIndex1()); }

        /// \return vert coordinate of the top edge of the element
        inline double getUpper2() const { return fullMesh().axis[2]->at(getUpperIndex2()); }

        /// \return size of the element in the long direction
        inline double getSize0() const { return getUpper0() - getLower0(); }

        /// \return size of the element in the tran direction
        inline double getSize1() const { return getUpper1() - getLower1(); }

        /// \return size of the element in the vert direction
        inline double getSize2() const { return getUpper2() - getLower2(); }

        /// \return vector indicating size of the element
        inline Vec<3, double> getSize() const { return getUpUpUp() - getLoLoLo(); }

        /// \return position of the middle of the element
        inline Vec<3, double> getMidpoint() const { return filteredMesh.getElementMidpoint(index0, index1, index2); }

        /// @return index of this element
        inline std::size_t getIndex() const {
            if (elementIndex == UNKNOWN_ELEMENT_INDEX)
                elementIndex = filteredMesh.getElementIndexFromLowIndexes(getLowerIndex0(), getLowerIndex1(), getLowerIndex2());
            return elementIndex;
        }

        /// \return this element as rectangular box
        inline Box3D toBox() const { return filteredMesh.getElementBox(index0, index1, index2); }

        /// \return total area of this element
        inline double getVolume() const { return getSize0() * getSize1() * getSize2(); }

        /// \return total area of this element
        inline double getArea() const { return getVolume(); }

        /// \return index of the lower left back corner of this element
        inline std::size_t getLoLoLoIndex() const { return filteredMesh.index(getLowerIndex0(), getLowerIndex1(), getLowerIndex2()); }

        /// \return index of the lower left front corner of this element
        inline std::size_t getUpLoLoIndex() const { return filteredMesh.index(getUpperIndex0(), getLowerIndex1(), getLowerIndex2()); }

        /// \return index of the lower right back corner of this element
        inline std::size_t getLoUpLoIndex() const { return filteredMesh.index(getLowerIndex0(), getUpperIndex1(), getLowerIndex2()); }

        /// \return index of the lower right front corner of this element
        inline std::size_t getUpUpLoIndex() const { return filteredMesh.index(getUpperIndex0(), getUpperIndex1(), getLowerIndex2()); }

        /// \return index of the upper left back corner of this element
        inline std::size_t getLoLoUpIndex() const { return filteredMesh.index(getLowerIndex0(), getLowerIndex1(), getUpperIndex2()); }

        /// \return index of the upper left front corner of this element
        inline std::size_t getUpLoUpIndex() const { return filteredMesh.index(getUpperIndex0(), getLowerIndex1(), getUpperIndex2()); }

        /// \return index of the upper right back corner of this element
        inline std::size_t getLoUpUpIndex() const { return filteredMesh.index(getLowerIndex0(), getUpperIndex1(), getUpperIndex2()); }

        /// \return index of the upper right front corner of this element
        inline std::size_t getUpUpUpIndex() const { return filteredMesh.index(getUpperIndex0(), getUpperIndex1(), getUpperIndex2()); }

        /// \return position of the lower left back corner of this element
        inline Vec<3, double> getLoLoLo() const { return filteredMesh(getLowerIndex0(), getLowerIndex1(), getLowerIndex2()); }

        /// \return position of the lower left front corner of this element
        inline Vec<3, double> getUpLoLo() const { return filteredMesh(getUpperIndex0(), getLowerIndex1(), getLowerIndex2()); }

        /// \return position of the lower right back corner of this element
        inline Vec<3, double> getLoUpLo() const { return filteredMesh(getLowerIndex0(), getUpperIndex1(), getLowerIndex2()); }

        /// \return position of the lower right front corner of this element
        inline Vec<3, double> getUpUpLo() const { return filteredMesh(getUpperIndex0(), getUpperIndex1(), getLowerIndex2()); }

        /// \return position of the upper left back corner of this element
        inline Vec<3, double> getLoLoUp() const { return filteredMesh(getLowerIndex0(), getLowerIndex1(), getUpperIndex2()); }

        /// \return position of the upper left front corner of this element
        inline Vec<3, double> getUpLoUp() const { return filteredMesh(getUpperIndex0(), getLowerIndex1(), getUpperIndex2()); }

        /// \return position of the upper right back corner of this element
        inline Vec<3, double> getLoUpUp() const { return filteredMesh(getLowerIndex0(), getUpperIndex1(), getUpperIndex2()); }

        /// \return position of the upper right front corner of this element
        inline Vec<3, double> getUpUpUp() const { return filteredMesh(getUpperIndex0(), getUpperIndex1(), getUpperIndex2()); }

    };  // class Element

    struct PLASK_API Elements: ElementsBase<RectangularFilteredMesh3D> {

        explicit Elements(const RectangularFilteredMesh3D& mesh): ElementsBase(mesh) { mesh.ensureHasElements(); }

        Element operator()(std::size_t i0, std::size_t i1, std::size_t i2) const { return Element(*filteredMesh, Element::UNKNOWN_ELEMENT_INDEX, i0, i1, i2); }

    };  // struct Elements

    /**
     * Construct empty/unitialized mesh. One should call reset() method before using this.
     */
    RectangularFilteredMesh3D() = default;

    /**
     * Change a selection of elements used to once pointed by a given @p predicate.
     * @param predicate predicate which returns either @c true for accepting element or @c false for rejecting it
     */
    void reset(const Predicate& predicate);

    /**
     * Construct filtered mesh with elements of rectangularMesh chosen by a @p predicate.
     * Preserve order of elements and nodes of @p rectangularMesh.
     * @param rectangularMesh input mesh, before filtering
     * @param predicate predicate which returns either @c true for accepting element or @c false for rejecting it
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if @c true) or shared (if @c false; default)
     */
    RectangularFilteredMesh3D(const RectangularMesh<3>& fullMesh, const Predicate& predicate, bool clone_axes = false);

    /**
     * Change parameter of this mesh to use elements of @p rectangularMesh chosen by a @p predicate.
     * Preserve order of elements and nodes of @p rectangularMesh.
     * @param rectangularMesh input mesh, before filtering
     * @param predicate predicate which returns either @c true for accepting element or @c false for rejecting it
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if @c true) or shared with @p rectangularMesh (if @c false; default)
     */
    void reset(const RectangularMesh<3>& fullMesh, const Predicate& predicate, bool clone_axes = false);

    /**
     * Construct filtered mesh with all elements of @c rectangularMesh which have required materials in the midpoints.
     * Preserve order of elements and nodes of @p rectangularMesh.
     * @param rectangularMesh input mesh, before filtering
     * @param geom geometry to get materials from
     * @param materialPredicate predicate which returns either @c true for accepting material or @c false for rejecting it
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if @c true) or shared (if @c false; default)
     */
    RectangularFilteredMesh3D(const RectangularMesh<3>& rectangularMesh, const GeometryD<3>& geom, const std::function<bool(shared_ptr<const Material>)> materialPredicate, bool clone_axes = false)
        : RectangularFilteredMesh3D(rectangularMesh,
                                    [&](const RectangularMesh3D::Element& el) { return materialPredicate(geom.getMaterial(el.getMidpoint())); },
                                    clone_axes)
    {
    }

    /**
     * Change parameter of this mesh to use all elements of @c rectangularMesh which have required materials in the midpoints.
     * Preserve order of elements and nodes of @p rectangularMesh.
     * @param rectangularMesh input mesh, before filtering
     * @param geom geometry to get materials from
     * @param materialPredicate predicate which returns either @c true for accepting material or @c false for rejecting it
     * @param clone_axes whether axes of the @p rectangularMesh should be cloned (if @c true) or shared (if @c false; default)
     */
    void reset(const RectangularMesh<3>& rectangularMesh, const GeometryD<3>& geom, const std::function<bool(shared_ptr<const Material>)> materialPredicate, bool clone_axes = false) {
        reset(rectangularMesh,
              [&](const RectangularMesh3D::Element& el) { return materialPredicate(geom.getMaterial(el.getMidpoint())); },
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
    RectangularFilteredMesh3D(const RectangularMesh<3>& rectangularMesh, const GeometryD<3>& geom, unsigned int materialKinds, bool clone_axes = false)
        : RectangularFilteredMesh3D(rectangularMesh,
                                    [&](const RectangularMesh3D::Element& el) { return (geom.getMaterial(el.getMidpoint())->kind() & materialKinds) != 0; },
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
    void reset(const RectangularMesh<3>& rectangularMesh, const GeometryD<3>& geom, unsigned int materialKinds, bool clone_axes = false) {
        reset(rectangularMesh,
             [&](const RectangularMesh3D::Element& el) { return (geom.getMaterial(el.getMidpoint())->kind() & materialKinds) != 0; },
             clone_axes);
    }

    /**
     * Construct a mesh with given set of nodes.
     *
     * Set of elements are calculated on-demand, just before the first use, according to the rule:
     * An element is selected if and only if all its vertices are included in the @p nodeSet.
     *
     * This constructor is used by getMidpointsMesh.
     */
    RectangularFilteredMesh3D(const RectangularMesh<DIM>& rectangularMesh, Set nodeSet, bool clone_axes = false)
        : RectangularFilteredMeshBase(rectangularMesh, std::move(nodeSet), clone_axes) {}

    Elements elements() const { return Elements(*this); }
    Elements getElements() const { return elements(); }

    Element element(std::size_t i0, std::size_t i1, std::size_t i2) const { ensureHasElements(); return Element(*this, Element::UNKNOWN_ELEMENT_INDEX, i0, i1, i2); }
    Element getElement(std::size_t i0, std::size_t i1, std::size_t i2) const { return element(i0, i1, i2); }

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
     * @param axis2_index index of axis2, from 0 to axis[2]->size()-1
     * @return this mesh index, from 0 to size()-1, or NOT_INCLUDED
     */
    inline std::size_t index(std::size_t axis0_index, std::size_t axis1_index, std::size_t axis2_index) const {
        return nodeSet.indexOf(fullMesh.index(axis0_index, axis1_index, axis2_index));
    }

    using RectangularFilteredMeshBase<3>::index;
    using RectangularFilteredMeshBase<3>::at;

    /**
     * Get point with given mesh indices.
     * @param index0 index of point in axis[0]
     * @param index1 index of point in axis[1]
     * @param index2 index of point in axis[2]
     * @return point with given @p index
     */
    inline Vec<3, double> at(std::size_t index0, std::size_t index1, std::size_t index2) const {
        return fullMesh.at(index0, index1, index2);
    }

    /**
     * Get point with given x and y indexes.
     * @param axis0_index index of axis[0], from 0 to axis[0]->size()-1
     * @param axis1_index index of axis[1], from 0 to axis[1]->size()-1
     * @param axis1_index index of axis[2], from 0 to axis[2]->size()-1
     * @return point with given axis0 and axis1 indexes
     */
    inline Vec<3, double> operator()(std::size_t axis0_index, std::size_t axis1_index, std::size_t axis2_index) const {
        return fullMesh.operator()(axis0_index, axis1_index, axis2_index);
    }

    /**
     * Return a mesh that enables iterating over middle points of the selected rectangles.
     * @param clone_axes whether axes of *this should be cloned (if true) or shared (if false; default) with the mesh returned
     * @return new rectilinear filtered mesh with points in the middles of original, selected rectangles
     */
    shared_ptr<RectangularFilteredMesh3D> getMidpointsMesh(bool clone_axes = false) const {
        return plask::make_shared<RectangularFilteredMesh3D>(*fullMesh.getMidpointsMesh(), ensureHasElements(), clone_axes);
        // elementSet is passed as a second argument since nodes of midpoints mesh coresponds to elements of oryginal mesh
    }

  private:

    void initNodesAndElements(const RectangularFilteredMesh3D::Predicate &predicate);

    bool canBeIncluded(const Vec<3>& point) const {
        return
            fullMesh.axis[0]->at(0) - point[0] < MIN_DISTANCE && point[0] - fullMesh.axis[0]->at(fullMesh.axis[0]->size()-1) < MIN_DISTANCE &&
            fullMesh.axis[1]->at(0) - point[1] < MIN_DISTANCE && point[1] - fullMesh.axis[1]->at(fullMesh.axis[1]->size()-1) < MIN_DISTANCE &&
            fullMesh.axis[2]->at(0) - point[2] < MIN_DISTANCE && point[2] - fullMesh.axis[2]->at(fullMesh.axis[2]->size()-1) < MIN_DISTANCE;
    }

  public:

    /** Prepare point for inteprolation
     * \param point point to check
     * \param[out] wrapped_point point after wrapping with interpolation flags
     * \param[out] index0_lo,index0_hi surrounding indices in the rectantular mesh for axis0
     * \param[out] index1_lo,index1_hi surrounding indices in the rectantular mesh for axis1
     * \param[out] index2_lo,index2_hi surrounding indices in the rectantular mesh for axis2
     * \param[out] rectmesh_index_lo elelemnt index in the rectangular mesh
     * \param flags interpolation flags
     * \returns \c false if the point falls in the hole or outside of the mesh, \c true if it can be interpolated
     */
    bool prepareInterpolation(const Vec<3>& point, Vec<3>& wrapped_point,
                              std::size_t& index0_lo, std::size_t& index0_hi,
                              std::size_t& index1_lo, std::size_t& index1_hi,
                              std::size_t& index2_lo, std::size_t& index2_hi,
                              std::size_t& rectmesh_index_lo, const InterpolationFlags& flags) const;
    /**
     * Calculate (using linear interpolation) value of data in point using data in points described by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(const RandomAccessContainer& data, const Vec<3>& point, const InterpolationFlags& flags) const
        -> typename std::remove_reference<decltype(data[0])>::type
    {
        Vec<3> wrapped_point;
        std::size_t index0_lo, index0_hi, index1_lo, index1_hi, index2_lo, index2_hi, rectmesh_index_lo;

        if (!prepareInterpolation(point, wrapped_point, index0_lo, index0_hi, index1_lo, index1_hi, index2_lo, index2_hi, rectmesh_index_lo, flags))
            return NaNfor<decltype(data[0])>();

        return flags.postprocess(point,
                                 interpolation::trilinear(
                                     fullMesh.axis[0]->at(index0_lo), fullMesh.axis[0]->at(index0_hi),
                                     fullMesh.axis[1]->at(index1_lo), fullMesh.axis[1]->at(index1_hi),
                                     fullMesh.axis[2]->at(index1_lo), fullMesh.axis[2]->at(index1_hi),
                                     data[nodeSet.indexOf(rectmesh_index_lo)],
                                     data[index(index0_lo, index1_lo, index2_lo)],
                                     data[index(index0_hi, index1_lo, index2_lo)],
                                     data[index(index0_hi, index1_hi, index2_lo)],
                                     data[index(index0_lo, index1_hi, index2_lo)],
                                     data[index(index0_lo, index1_lo, index2_hi)],
                                     data[index(index0_hi, index1_lo, index2_hi)],
                                     data[index(index0_hi, index1_hi, index2_hi)],
                                     data[index(index0_lo, index1_hi, index2_hi)],
                                     wrapped_point.c0, wrapped_point.c1, wrapped_point.c2));
    }

    /**
     * Calculate (using nearest neighbor interpolation) value of data in point using data in points described by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateNearestNeighbor(const RandomAccessContainer& data, const Vec<3>& point, const InterpolationFlags& flags) const
        -> typename std::remove_reference<decltype(data[0])>::type
    {
        Vec<3> wrapped_point;
        std::size_t index0_lo, index0_hi, index1_lo, index1_hi, index2_lo, index2_hi, rectmesh_index_lo;

        if (!prepareInterpolation(point, wrapped_point, index0_lo, index0_hi, index1_lo, index1_hi, index2_lo, index2_hi, rectmesh_index_lo, flags))
            return NaNfor<decltype(data[0])>();

        return flags.postprocess(point,
                                 data[this->index(
                                     nearest(wrapped_point.c0, *fullMesh.axis[0], index0_lo, index0_hi),
                                     nearest(wrapped_point.c1, *fullMesh.axis[1], index1_lo, index1_hi),
                                     nearest(wrapped_point.c2, *fullMesh.axis[2], index2_lo, index2_hi)
                                 )]);
    }

    /**
     * Convert mesh indexes of a back-left-bottom corner of an element to the index of this element.
     * @param axis0_index index of the corner along the axis[0] (back), from 0 to axis[0]->size()-1
     * @param axis1_index index of the corner along the axis[1] (left), from 0 to axis[1]->size()-1
     * @param axis2_index index of the corner along the axis[2] (bottom), from 0 to axis[2]->size()-1
     * @return index of the element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndexes(std::size_t axis0_index, std::size_t axis1_index, std::size_t axis2_index) const {
        return ensureHasElements().indexOf(fullMesh.getElementIndexFromLowIndexes(axis0_index, axis1_index, axis2_index));
    }

    /**
     * Get an area of a given element.
     * @param index0, index1, index2 axes 0, 1 and 1 indexes of the element
     * @return the area of the element with given indexes
     */
    double getElementArea(std::size_t index0, std::size_t index1, std::size_t index2) const {
        return fullMesh.getElementArea(index0, index1, index2);
    }

    /**
     * Get point in center of Elements.
     * @param index0, index1, index2 index of Elements
     * @return point in center of element with given index
     */
    Vec<3, double> getElementMidpoint(std::size_t index0, std::size_t index1, std::size_t index2) const {
        return fullMesh.getElementMidpoint(index0, index1, index2);
    }

    /**
     * Get element as rectangle.
     * @param index0, index1, index2 index of Elements
     * @return box of elements with given index
     */
    Box3D getElementBox(std::size_t index0, std::size_t index1, std::size_t index2) const {
        return fullMesh.getElementBox(index0, index1, index2);
    }


  protected:

    // Common code for: left, right, bottom, top boundries:
    template <int CHANGE_DIR_SLOWER, int CHANGE_DIR_FASTER>
    struct BoundaryIteratorImpl: public plask::BoundaryNodeSetImpl::IteratorImpl {

        const RectangularFilteredMeshBase<3> &mesh;

        /// current indexes
        Vec<3, std::size_t> index;

        /// past the last index of change direction
        const std::size_t indexFasterBegin, indexFasterEnd, indexSlowerEnd;

    private:
        /// Increase indexes without filtering.
        void naiveIncrement() {
            ++index[CHANGE_DIR_FASTER];
            if (index[CHANGE_DIR_FASTER] == indexFasterEnd) {
                index[CHANGE_DIR_FASTER] = indexFasterBegin;
                ++index[CHANGE_DIR_SLOWER];
            }
        }
    public:
        BoundaryIteratorImpl(const RectangularFilteredMeshBase<3>& mesh, Vec<3, std::size_t> index, std::size_t indexSlowerEnd, std::size_t indexFasterEnd)
            : mesh(mesh), index(index), indexFasterBegin(index[CHANGE_DIR_FASTER]), indexFasterEnd(indexFasterEnd), indexSlowerEnd(indexSlowerEnd)
        {
            // go to the first index existed in order to make dereference possible:
            while (this->index[CHANGE_DIR_SLOWER] < indexSlowerEnd && mesh.index(this->index) == NOT_INCLUDED)
                naiveIncrement();
        }

        void increment() override {
            do {
                naiveIncrement();
            } while (index[CHANGE_DIR_SLOWER] < indexSlowerEnd && mesh.index(index) == NOT_INCLUDED);
        }

        bool equal(const plask::BoundaryNodeSetImpl::IteratorImpl& other) const override {
            return index == static_cast<const BoundaryIteratorImpl&>(other).index;
        }

        std::size_t dereference() const override {
            return mesh.index(index);
        }

        typename plask::BoundaryNodeSetImpl::IteratorImpl* clone() const override {
            return new BoundaryIteratorImpl<CHANGE_DIR_SLOWER, CHANGE_DIR_FASTER>(*this);
        }

    };

    template <int CHANGE_DIR_SLOWER, int CHANGE_DIR_FASTER>
    struct BoundaryNodeSetImpl: public BoundaryNodeSetWithMeshImpl<RectangularFilteredMeshBase<3>> {

        using typename BoundaryNodeSetWithMeshImpl<RectangularFilteredMeshBase<3>>::const_iterator;

        /// first index
        Vec<3, std::size_t> index;

        /// past the last index of change directions
        std::size_t indexFasterEnd, indexSlowerEnd;

        BoundaryNodeSetImpl(const RectangularFilteredMeshBase<DIM>& mesh, Vec<3, std::size_t> index, std::size_t indexSlowerEnd, std::size_t indexFasterEnd)
            : BoundaryNodeSetWithMeshImpl<RectangularFilteredMeshBase<3>>(mesh), index(index), indexFasterEnd(indexFasterEnd), indexSlowerEnd(indexSlowerEnd) {}

        BoundaryNodeSetImpl(const RectangularFilteredMeshBase<DIM>& mesh, std::size_t index0, std::size_t index1, std::size_t index2, std::size_t indexSlowerEnd, std::size_t indexFasterEnd)
            : BoundaryNodeSetWithMeshImpl<RectangularFilteredMeshBase<3>>(mesh), index(index0, index1, index2), indexFasterEnd(indexFasterEnd), indexSlowerEnd(indexSlowerEnd) {}

        bool contains(std::size_t mesh_index) const override {
            if (mesh_index >= this->mesh.size()) return false;
            Vec<3, std::size_t> mesh_indexes = this->mesh.indexes(mesh_index);
            for (int i = 0; i < 3; ++i)
                if (i == CHANGE_DIR_FASTER) {
                    if (mesh_indexes[i] < index[i] || mesh_indexes[i] >= indexFasterEnd) return false;
                } else if (i == CHANGE_DIR_SLOWER) {
                    if (mesh_indexes[i] < index[i] || mesh_indexes[i] >= indexSlowerEnd) return false;
                } else
                    if (mesh_indexes[i] != index[i]) return false;
            return true;
        }

        const_iterator begin() const override {
            return Iterator(new BoundaryIteratorImpl<CHANGE_DIR_SLOWER, CHANGE_DIR_FASTER>(this->mesh, index, indexSlowerEnd, indexFasterEnd));
        }

        const_iterator end() const override {
            Vec<3, std::size_t> index_end = index;
            index_end[CHANGE_DIR_SLOWER] = indexSlowerEnd;
            return Iterator(new BoundaryIteratorImpl<CHANGE_DIR_SLOWER, CHANGE_DIR_FASTER>(this->mesh, index_end, indexSlowerEnd, indexFasterEnd));
        }
    };

  public:     // boundaries:

    BoundaryNodeSet createIndex0BoundaryAtLine(std::size_t line_nr_axis0,
                                                             std::size_t index1Begin, std::size_t index1End,
                                                             std::size_t index2Begin, std::size_t index2End
                                                             ) const override;

    BoundaryNodeSet createIndex0BoundaryAtLine(std::size_t line_nr_axis0) const override;

    BoundaryNodeSet createIndex1BoundaryAtLine(std::size_t line_nr_axis1,
                                                             std::size_t index0Begin, std::size_t index0End,
                                                             std::size_t index2Begin, std::size_t index2End
                                                             ) const override;

    BoundaryNodeSet createIndex1BoundaryAtLine(std::size_t line_nr_axis1) const override;

    BoundaryNodeSet createIndex2BoundaryAtLine(std::size_t line_nr_axis2,
                                                             std::size_t index0Begin, std::size_t index0End,
                                                             std::size_t index1Begin, std::size_t index1End
                                                             ) const override;

    BoundaryNodeSet createIndex2BoundaryAtLine(std::size_t line_nr_axis2) const override;

    BoundaryNodeSet createBackBoundary() const override;

    BoundaryNodeSet createFrontBoundary() const override;

    BoundaryNodeSet createLeftBoundary() const override;

    BoundaryNodeSet createRightBoundary() const override;

    BoundaryNodeSet createBottomBoundary() const override;

    BoundaryNodeSet createTopBoundary() const override;

    BoundaryNodeSet createBackOfBoundary(const Box3D& box) const override;

    BoundaryNodeSet createFrontOfBoundary(const Box3D& box) const override;

    BoundaryNodeSet createLeftOfBoundary(const Box3D& box) const override;

    BoundaryNodeSet createRightOfBoundary(const Box3D& box) const override;

    BoundaryNodeSet createBottomOfBoundary(const Box3D& box) const override;

    BoundaryNodeSet createTopOfBoundary(const Box3D& box) const override;
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularFilteredMesh3D, SrcT, DstT, INTERPOLATION_LINEAR> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularFilteredMesh3D>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->empty()) throw BadMesh("interpolate", "Source mesh empty");
        return new LinearInterpolatedLazyDataImpl< DstT, RectangularFilteredMesh3D, SrcT >(src_mesh, src_vec, dst_mesh, flags);
    }
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularFilteredMesh3D, SrcT, DstT, INTERPOLATION_NEAREST> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularFilteredMesh3D>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->empty()) throw BadMesh("interpolate", "Source mesh empty");
        return new NearestNeighborInterpolatedLazyDataImpl< DstT, RectangularFilteredMesh3D, SrcT >(src_mesh, src_vec, dst_mesh, flags);
    }
};


}   // namespace plask

#endif // PLASK__RECTANGULAR_FILTERED3D_H
