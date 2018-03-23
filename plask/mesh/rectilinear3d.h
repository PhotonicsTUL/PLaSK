#ifndef PLASK__RECTILINEAR3D_H
#define PLASK__RECTILINEAR3D_H

/** @file
This file contains rectilinear mesh for 3D space.
*/

#include <type_traits>

#include "mesh.h"
#include "axis1d.h"
#include "../utils/interpolation.h"


namespace plask {

/**
 * Rectilinear mesh in 3D space.
 *
 * Includes three 1d rectilinear meshes:
 * - axis0
 * - axis1
 * - axis2
 * Represent all points (c0, c1, c2) such that c0 is in axis0, c1 is in axis1, c2 is in axis2.
 */
class PLASK_API RectilinearMesh3D: public MeshD<3> {

    typedef std::size_t index_ft(const RectilinearMesh3D* mesh, std::size_t c0_index, std::size_t c1_index, std::size_t c2_index);
    typedef std::size_t index012_ft(const RectilinearMesh3D* mesh, std::size_t mesh_index);

    // our own virtual table, changeable in run-time:
    index_ft* index_f;
    index012_ft* index0_f;
    index012_ft* index1_f;
    index012_ft* index2_f;
    const shared_ptr<MeshAxis>* minor_axis;
    const shared_ptr<MeshAxis>* medium_axis;
    const shared_ptr<MeshAxis>* major_axis;

    void onAxisChanged(Event& e);

    void setChangeSignal(const shared_ptr<MeshAxis>& axis) { if (axis) axis->changedConnectMethod(this, &RectilinearMesh3D::onAxisChanged); }
    void unsetChangeSignal(const shared_ptr<MeshAxis>& axis) { if (axis) axis->changedDisconnectMethod(this, &RectilinearMesh3D::onAxisChanged); }

    void setAxis(const shared_ptr<MeshAxis>& axis, shared_ptr<MeshAxis> new_val);

  public:

    /**
     * Represent FEM-like element in Rectilinear.
     */
    class PLASK_API Element {
        const RectilinearMesh3D& mesh;
        std::size_t index0, index1, index2; // probably this form allows to do most operation fastest in average, low indexes of element corner or just element indexes

        public:

        /**
         * Construct element using mesh and element indexes.
         * @param mesh mesh, this element is valid up to time of this mesh life
         * @param index0, index1, index2 axis 0, 1 and 2 indexes of element (equal to low corrner mesh indexes of element)
         */
        Element(const RectilinearMesh3D& mesh, std::size_t index0, std::size_t index1, std::size_t index2): mesh(mesh), index0(index0), index1(index1), index2(index2) {}

        /**
         * Construct element using mesh and element index.
         * @param mesh mesh, this element is valid up to time of this mesh life
         * @param elementIndex index of element
         */
        Element(const RectilinearMesh3D& mesh, std::size_t elementIndex): mesh(mesh) {
            std::size_t v = mesh.getElementMeshLowIndex(elementIndex);
            index0 = mesh.index0(v);
            index1 = mesh.index1(v);
            index2 = mesh.index2(v);
        }

        /// \return long index of the element
        inline std::size_t getIndex0() const { return index0; }

        /// \return tran index of the element
        inline std::size_t getIndex1() const { return index1; }

        /// \return vert index of the element
        inline std::size_t getIndex2() const { return index2; }

        /// \return long index of the left edge of the element
        inline std::size_t getLowerIndex0() const { return index0; }

        /// \return tran index of the left edge of the element
        inline std::size_t getLowerIndex1() const { return index1; }

        /// \return vert index of the bottom edge of the element
        inline std::size_t getLowerIndex2() const { return index2; }

        /// \return long coordinate of the left edge of the element
        inline double getLower0() const { return mesh.axis0->at(index0); }

        /// \return tran coordinate of the left edge of the element
        inline double getLower1() const { return mesh.axis1->at(index1); }

        /// \return vert coordinate of the bottom edge of the element
        inline double getLower2() const { return mesh.axis2->at(index2); }

        /// \return long index of the right edge of the element
        inline std::size_t getUpperIndex0() const { return index0+1; }

        /// \return tran index of the right edge of the element
        inline std::size_t getUpperIndex1() const { return index1+1; }

        /// \return vert index of the top edge of the element
        inline std::size_t getUpperIndex2() const { return index2+1; }

        /// \return long coordinate of the right edge of the element
        inline double getUpper0() const { return mesh.axis0->at(getUpperIndex0()); }

        /// \return tran coordinate of the right edge of the element
        inline double getUpper1() const { return mesh.axis1->at(getUpperIndex1()); }

        /// \return vert coordinate of the top edge of the element
         inline double getUpper2() const { return mesh.axis2->at(getUpperIndex2()); }

        /// \return size of the element in the long direction
        inline double getSize0() const { return getUpper0() - getLower0(); }

        /// \return size of the element in the tran direction
        inline double getSize1() const { return getUpper1() - getLower1(); }

        /// \return size of the element in the vert direction
        inline double getSize2() const { return getUpper2() - getLower2(); }

        /// \return vector indicating size of the element
        inline Vec<3, double> getSize() const { return getUpUpUp() - getLoLoLo(); }

        /// \return position of the middle of the element
        inline Vec<3, double> getMidpoint() const { return mesh.getElementMidpoint(index0, index1, index2); }

        /// @return index of this element
        inline std::size_t getIndex() const { return mesh.getElementIndexFromLowIndex(getLoLoLoIndex()); }

        /// \return index of the lower left back corner of this element
        inline std::size_t getLoLoLoIndex() const { return mesh.index(getLowerIndex0(), getLowerIndex1(), getLowerIndex2()); }

        /// \return index of the lower left front corner of this element
        inline std::size_t getUpLoLoIndex() const { return mesh.index(getUpperIndex0(), getLowerIndex1(), getLowerIndex2()); }

        /// \return index of the lower right back corner of this element
        inline std::size_t getLoUpLoIndex() const { return mesh.index(getLowerIndex0(), getUpperIndex1(), getLowerIndex2()); }

        /// \return index of the lower right front corner of this element
        inline std::size_t getUpUpLoIndex() const { return mesh.index(getUpperIndex0(), getUpperIndex1(), getLowerIndex2()); }

        /// \return index of the upper left back corner of this element
        inline std::size_t getLoLoUpIndex() const { return mesh.index(getLowerIndex0(), getLowerIndex1(), getUpperIndex2()); }

        /// \return index of the upper left front corner of this element
        inline std::size_t getUpLoUpIndex() const { return mesh.index(getUpperIndex0(), getLowerIndex1(), getUpperIndex2()); }

        /// \return index of the upper right back corner of this element
        inline std::size_t getLoUpUpIndex() const { return mesh.index(getLowerIndex0(), getUpperIndex1(), getUpperIndex2()); }

        /// \return index of the upper right front corner of this element
        inline std::size_t getUpUpUpIndex() const { return mesh.index(getUpperIndex0(), getUpperIndex1(), getUpperIndex2()); }

        /// \return position of the lower left back corner of this element
        inline Vec<3, double> getLoLoLo() const { return mesh(getLowerIndex0(), getLowerIndex1(), getLowerIndex2()); }

        /// \return position of the lower left front corner of this element
        inline Vec<3, double> getUpLoLo() const { return mesh(getUpperIndex0(), getLowerIndex1(), getLowerIndex2()); }

        /// \return position of the lower right back corner of this element
        inline Vec<3, double> getLoUpLo() const { return mesh(getLowerIndex0(), getUpperIndex1(), getLowerIndex2()); }

        /// \return position of the lower right front corner of this element
        inline Vec<3, double> getUpUpLo() const { return mesh(getUpperIndex0(), getUpperIndex1(), getLowerIndex2()); }

        /// \return position of the upper left back corner of this element
        inline Vec<3, double> getLoLoUp() const { return mesh(getLowerIndex0(), getLowerIndex1(), getUpperIndex2()); }

        /// \return position of the upper left front corner of this element
        inline Vec<3, double> getUpLoUp() const { return mesh(getUpperIndex0(), getLowerIndex1(), getUpperIndex2()); }

        /// \return position of the upper right back corner of this element
        inline Vec<3, double> getLoUpUp() const { return mesh(getLowerIndex0(), getUpperIndex1(), getUpperIndex2()); }

        /// \return position of the upper right front corner of this element
        inline Vec<3, double> getUpUpUp() const { return mesh(getUpperIndex0(), getUpperIndex1(), getUpperIndex2()); }

    };

    /**
     * Wrapper to Rectilinear which allow to access to FEM-like elements.
     *
     * It works like read-only, random access container of @ref Element objects.
     */
    struct PLASK_API Elements {

        typedef IndexedIterator<const Elements, Element> const_iterator;
        typedef const_iterator iterator;

        /// Mesh which elements will be accessable by this.
        const RectilinearMesh3D* mesh;

        /**
         * Create wrapper which allow to access to FEM-like elements of given @p mesh.
         * @param mesh mesh which elements will be accessable by this
         */
        Elements(const RectilinearMesh3D* mesh): mesh(mesh) {}

        /**
         * Get @p i-th element.
         * @param i element index
         * @return @p i-th element
         */
        Element operator[](std::size_t i) const { return Element(*mesh, i); }

        /**
         * Get element with indices \p i0, \p i1, and \p i2.
         * \param i0, i1, i2 element index
         * \return element with indices \p i0, \p i1, and \p i2
         */
        Element operator()(std::size_t i0, std::size_t i1, std::size_t i2) const { return Element(*mesh, i0, i1, i2); }

        /**
         * Get number of elements.
         * @return number of elements
         */
        std::size_t size() const { return mesh->getElementsCount(); }

        /// @return iterator referring to the first element
        const_iterator begin() const { return const_iterator(this, 0); }

        /// @return iterator referring to the past-the-end element
        const_iterator end() const { return const_iterator(this, size()); }

    };

    /// First coordinate of points in this mesh.
    const shared_ptr<MeshAxis> axis0;

    /// Second coordinate of points in this mesh.
    const shared_ptr<MeshAxis> axis1;

    /// Third coordinate of points in this mesh.
    const shared_ptr<MeshAxis> axis2;

    /// Accessor to FEM-like elements.
    const Elements elements;

    /**
     * Iteration orders:
     * Every other order is proper permutation of indices.
     * They mean ORDER_major,medium,minor, i.e. the last index changes fastest
     * @see setIterationOrder, getIterationOrder, setOptimalIterationOrder
     */
    enum IterationOrder { ORDER_012, ORDER_021, ORDER_102, ORDER_120, ORDER_201, ORDER_210 };

    /**
     * Choose iteration order.
     * @param order iteration order to use
     * @see IterationOrder
     */
    void setIterationOrder(IterationOrder order);

    /**
     * Get iteration order.
     * @return iteration order used by this mesh
     * @see IterationOrder
     */
    IterationOrder getIterationOrder() const;

    /**
     * Set iteration order to the shortest axis changes fastest.
     */
    void setOptimalIterationOrder();

    /**
     * Construct mesh which has all axes of type OrderedAxis and all are empty.
     * @param iterationOrder iteration order
     */
    explicit RectilinearMesh3D(IterationOrder iterationOrder = ORDER_012);

    /**
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     * @param mesh2 mesh for the third coordinate
     * @param iterationOrder iteration order
     */
    RectilinearMesh3D(shared_ptr<MeshAxis> mesh0, shared_ptr<MeshAxis> mesh1, shared_ptr<MeshAxis> mesh2, IterationOrder iterationOrder = ORDER_012);

    /// Copy constructor
    RectilinearMesh3D(const RectilinearMesh3D& src);

    ~RectilinearMesh3D();

    const shared_ptr<MeshAxis> getAxis0() const { return axis0; }

    void setAxis0(shared_ptr<MeshAxis> a0) { setAxis(this->axis0, a0);  }

    const shared_ptr<MeshAxis> getAxis1() const { return axis1; }

    void setAxis1(shared_ptr<MeshAxis> a1) { setAxis(this->axis1, a1); }

    const shared_ptr<MeshAxis> getAxis2() const { return axis2; }

    void setAxis2(shared_ptr<MeshAxis> a2) { setAxis(this->axis2, a2); }

    /**
     * Get numbered axis
     * \param n number of axis
     */
    const shared_ptr<MeshAxis>& axis(size_t n) const {
        if (n == 0) return axis0;
        else if (n == 1) return axis1;
        else if (n != 2) throw Exception("Bad axis number");
        return axis2;
    }

    /// \return major (changing slowest) axis
    inline const shared_ptr<MeshAxis> majorAxis() const {
        return *major_axis;
    }

    /// \return middle (between major and minor) axis
    inline const shared_ptr<MeshAxis> mediumAxis() const {
        return *medium_axis;
    }

    /// \return minor (changing fastes) axis
    inline const shared_ptr<MeshAxis> minorAxis() const {
        return *minor_axis;
    }

    /**
      * Compare meshes
      * @param to_compare mesh to compare
      * @return @c true only if this mesh and @p to_compare represents the same set of points regardless of iteration order
      */
    bool operator==(const RectilinearMesh3D& to_compare) const {
        return *axis0 == *to_compare.axis0 && *axis1 == *to_compare.axis1 && *axis2 == *to_compare.axis2;
    }

    /**
     * Get number of points in the mesh.
     * @return number of points in the mesh
     */
    std::size_t size() const override { return axis0->size() * axis1->size() * axis2->size(); }

    /// @return true only if there are no points in mesh
    bool empty() const override { return axis0->empty() || axis1->empty() || axis2->empty(); }

    /**
     * Get point with given mesh indices.
     * @param index0 index of point in axis0
     * @param index1 index of point in axis1
     * @param index2 index of point in axis2
     * @return point with given @p index
     */
    virtual Vec<3, double> at(std::size_t index0, std::size_t index1, std::size_t index2) const = 0;

    /**
     * Get point with given mesh index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     */
    Vec<3, double> at(std::size_t index) const override {
        return at(index0(index), index1(index), index2(index));
    }

    /**
     * Get point with given mesh index.
     * Points are in order: (c0[0], c1[0], c2[0]), (c0[1], c1[0], c2[0]), ..., (c0[c0.size-1], c1[0], c2[0]), (c0[0], c1[1], c2[0]), ..., (c0[c0.size()-1], c1[c1.size()-1], c2[c2.size()-1])
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     */
    inline Vec<3, double> operator[](std::size_t index) const {
        return at(index0(index), index1(index), index2(index));
    }

    /**
     * Get point with given x and y indexes.
     * @param index0 index of point in axis0
     * @param index1 index of point in axis1
     * @param index2 index of point in axis2
     * @return point with given c0, c1 and c2 indexes
     */
    inline Vec<3, double> operator()(std::size_t index0, std::size_t index1, std::size_t index2) const {
        return at(index0, index1, index2);
    }

    /**
     * Get first coordinate of point in center of Elements.
     * @param index0 index of Elements (axis0 index)
     * @return first coordinate of point point in center of Elements with given index
     */
    double getElementMidpoint0(std::size_t index0) const { return 0.5 * (axis0->at(index0) + axis0->at(index0+1)); }

    /**
     * Get second coordinate of point in center of Elements.
     * @param index1 index of Elements (axis1 index)
     * @return second coordinate of point point in center of Elements with given index
     */
    double getElementMidpoint1(std::size_t index1) const { return 0.5 * (axis1->at(index1) + axis1->at(index1+1)); }

    /**
     * Get second coordinate of point in center of Elements.
     * @param index2 index of Elements (axis2 index)
     * @return second coordinate of point point in center of Elements with given index
     */
    double getElementMidpoint2(std::size_t index2) const { return 0.5 * (axis2->at(index2) + axis2->at(index2+1)); }

    /**
     * Get point in center of Elements.
     * @param index0, index1, index2 index of Elements
     * @return point in center of element with given index
     */
    virtual Vec<3, double> getElementMidpoint(std::size_t index0, std::size_t index1, std::size_t index2) const = 0;

    /**
     * Calculate this mesh index using indexes of c0, c1 and c2.
     * @param c0_index index of c0, from 0 to c0.size()-1
     * @param c1_index index of c1, from 0 to c1.size()-1
     * @param c2_index index of c2, from 0 to c2.size()-1
     * @return this mesh index, from 0 to size()-1
     */
    std::size_t index(std::size_t c0_index, std::size_t c1_index, std::size_t c2_index) const {
        return index_f(this, c0_index, c1_index, c2_index);
    }

    /**
     * Calculate index of c0 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of c0, from 0 to c0.size()-1
     */
    inline std::size_t index0(std::size_t mesh_index) const {
       return index0_f(this, mesh_index);
    }

    /**
     * Calculate index of c1 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of c1, from 0 to c1.size()-1
     */
    inline std::size_t index1(std::size_t mesh_index) const {
        return index1_f(this, mesh_index);
    }

    /**
     * Calculate index of c2 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of c2, from 0 to c2.size()-1
     */
    inline std::size_t index2(std::size_t mesh_index) const {
        return index2_f(this, mesh_index);
    }

    /**
     * Calculate index of major axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of major axis, from 0 to majorAxis.size()-1
     */
    inline std::size_t majorIndex(std::size_t mesh_index) const {
        return mesh_index / (*minor_axis)->size() / (*medium_axis)->size();
    }

    /**
     * Calculate index of middle axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of middle axis, from 0 to mediumAxis.size()-1
     */
    inline std::size_t middleIndex(std::size_t mesh_index) const {
        return (mesh_index / (*minor_axis)->size()) % (*medium_axis)->size();
    }

    /**
     * Calculate index of minor axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of minor axis, from 0 to minorAxis.size()-1
     */
    inline std::size_t minorIndex(std::size_t mesh_index) const {
        return mesh_index % (*minor_axis)->size();
    }

    /**
     * Get number of elements (for FEM method) in the first direction.
     * @return number of elements in this mesh in the first direction (axis0 direction).
     */
    std::size_t getElementsCount0() const {
        return std::max(int(axis0->size())-1, 0);
    }

    /**
     * Get number of elements (for FEM method) in the second direction.
     * @return number of elements in this mesh in the second direction (axis1 direction).
     */
    std::size_t getElementsCount1() const {
        return std::max(int(axis1->size())-1, 0);
    }

    /**
     * Get number of elements (for FEM method) in the third direction.
     * @return number of elements in this mesh in the third direction (axis2 direction).
     */
    size_t getElementsCount2() const {
        return size_t(std::max(int(axis2->size())-1, 0));
    }

    /**
     * Get number of elements (for FEM method).
     * @return number of elements in this mesh
     */
    size_t getElementsCount() const {
        return size_t(std::max(int(axis0->size())-1, 0) * std::max(int(axis1->size())-1, 0) * std::max(int(axis2->size())-1, 0));
    }

    /**
     * Conver element index to mesh index of bottom, left, front element corner.
     * @param element_index index of element, from 0 to getElementsCount()-1
     * @return mesh index
     */
    std::size_t getElementMeshLowIndex(std::size_t element_index) const {
        const std::size_t minor_size_minus_1 = (*minor_axis)->size()-1;
        const std::size_t elements_per_level = minor_size_minus_1 * ((*medium_axis)->size()-1);
        return element_index + (element_index / elements_per_level) * ((*medium_axis)->size() + minor_size_minus_1)
                            + (element_index % elements_per_level) / minor_size_minus_1;
    }

    /**
     * Conver mesh index of bottom, left, front element corner to this element index.
     * @param mesh_index_of_el_bottom_left mesh index
     * @return index of element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndex(std::size_t mesh_index_of_el_bottom_left) const {
        const std::size_t verticles_per_level = (*minor_axis)->size() * (*medium_axis)->size();
        return mesh_index_of_el_bottom_left - (mesh_index_of_el_bottom_left / verticles_per_level) * ((*medium_axis)->size() + (*minor_axis)->size() - 1)
                - (mesh_index_of_el_bottom_left % verticles_per_level) / (*minor_axis)->size();
    }

    /**
     * Convert element index to mesh indexes of bottom left element corner.
     * @param element_index index of element, from 0 to getElementsCount()-1
     * @return axis 0, axis 1 and axis 2 indexes of mesh,
     * you can easy calculate rest indexes of element corner adding 1 to returned coordinates
     */
    Vec<3, std::size_t> getElementMeshLowIndexes(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return Vec<3, std::size_t>(index0(bl_index), index1(bl_index), index2(bl_index));
    }

    /**
     * Calculate (using linear interpolation) value of data in point using data in points describe by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(const RandomAccessContainer& data, const Vec<3>& point, const InterpolationFlags& flags) const
        -> typename std::remove_reference<decltype(data[0])>::type
    {
        auto p = flags.wrap(point);

        size_t index0_lo, index0_hi;
        double back, front;
        bool invert_back, invert_front;
        prepareInterpolationForAxis(*axis0, flags, p.c0, 0, index0_lo, index0_hi, back, front, invert_back, invert_front);

        size_t index1_lo, index1_hi;
        double left, right;
        bool invert_left, invert_right;
        prepareInterpolationForAxis(*axis1, flags, p.c1, 1, index1_lo, index1_hi, left, right, invert_left, invert_right);

        size_t index2_lo, index2_hi;
        double bottom, top;
        bool invert_bottom, invert_top;
        prepareInterpolationForAxis(*axis2, flags, p.c2, 2, index2_lo, index2_hi, bottom, top, invert_bottom, invert_top);

        // all indexes are in bounds
        typename std::remove_const<typename std::remove_reference<decltype(data[0])>::type>::type
            data_lll = data[index(index0_lo, index1_lo, index2_lo)],
            data_hll = data[index(index0_hi, index1_lo, index2_lo)],
            data_hhl = data[index(index0_hi, index1_hi, index2_lo)],
            data_lhl = data[index(index0_lo, index1_hi, index2_lo)],
            data_llh = data[index(index0_lo, index1_lo, index2_hi)],
            data_hlh = data[index(index0_hi, index1_lo, index2_hi)],
            data_hhh = data[index(index0_hi, index1_hi, index2_hi)],
            data_lhh = data[index(index0_lo, index1_hi, index2_hi)];

        if (invert_back)   { data_lll = flags.reflect(0, data_lll); data_llh = flags.reflect(0, data_llh); data_lhl = flags.reflect(0, data_lhl); data_lhh = flags.reflect(0, data_lhh); }
        if (invert_front)  { data_hll = flags.reflect(0, data_hll); data_llh = flags.reflect(0, data_hlh); data_lhl = flags.reflect(0, data_hhl); data_lhh = flags.reflect(0, data_hhh); }
        if (invert_left)   { data_lll = flags.reflect(1, data_lll); data_llh = flags.reflect(1, data_llh); data_hll = flags.reflect(1, data_hll); data_hlh = flags.reflect(1, data_hlh); }
        if (invert_right)  { data_lhl = flags.reflect(1, data_lhl); data_llh = flags.reflect(1, data_lhh); data_hll = flags.reflect(1, data_hhl); data_hlh = flags.reflect(1, data_hhh); }
        if (invert_bottom) { data_lll = flags.reflect(2, data_lll); data_lhl = flags.reflect(2, data_lhl); data_hll = flags.reflect(2, data_hll); data_hhl = flags.reflect(2, data_hhl); }
        if (invert_top)    { data_llh = flags.reflect(2, data_llh); data_lhl = flags.reflect(2, data_lhh); data_hll = flags.reflect(2, data_hlh); data_hhl = flags.reflect(2, data_hhh); }

        return flags.postprocess(point,
            interpolation::trilinear(back, front, left, right, bottom, top,
                                        data_lll, data_hll, data_hhl, data_lhl, data_llh, data_hlh, data_hhh, data_lhh,
                                        p.c0, p.c1, p.c2));
    }

    /**
     * Calculate (using nearest neighbor interpolation) value of data in point using data in points describe by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateNearestNeighbor(const RandomAccessContainer& data, Vec<3> point, const InterpolationFlags& flags) const
        -> typename std::remove_reference<decltype(data[0])>::type {
        auto p = flags.wrap(point);
        /*prepareNearestNeighborInterpolationForAxis(*axis0, flags, p.c0, 0);
        prepareNearestNeighborInterpolationForAxis(*axis1, flags, p.c1, 1);
        prepareNearestNeighborInterpolationForAxis(*axis2, flags, p.c2, 2);*/
        if (flags.periodic(0)) {
            if (p.c0 < axis0->at(0)) {
                if (axis0->at(0) - p.c0 > p.c0 - flags.low(0) + flags.high(0) - axis0->at(axis0->size()-1)) p.c0 = axis0->at(axis0->size()-1);
            } else if (p.c0 > axis0->at(axis0->size()-1)) {
                if (p.c0 - axis0->at(axis0->size()-1) > flags.high(0) - p.c0 + axis0->at(0) - flags.low(0)) p.c0 = axis0->at(0);
            }
        }
        if (flags.periodic(1)) {
            if (p.c1 < axis1->at(0)) {
                if (axis1->at(0) - p.c1 > p.c1 - flags.low(1) + flags.high(1) - axis1->at(axis1->size()-1)) p.c1 = axis1->at(axis1->size()-1);
            } else if (p.c1 > axis1->at(axis1->size()-1)) {
                if (p.c1 - axis1->at(axis1->size()-1) > flags.high(1) - p.c1 + axis1->at(0) - flags.low(1)) p.c1 = axis1->at(0);
            }
        }
        if (flags.periodic(2)) {
            if (p.c2 < axis2->at(0)) {
                if (axis2->at(0) - p.c2 > p.c2 - flags.low(2) + flags.high(2) - axis2->at(axis2->size()-1)) p.c2 = axis2->at(axis2->size()-1);
            } else if (p.c2 > axis2->at(axis2->size()-1)) {
                if (p.c2 - axis2->at(axis2->size()-1) > flags.high(2) - p.c2 + axis2->at(0) - flags.low(2)) p.c2 = axis2->at(0);
            }
        }
        return flags.postprocess(point, data[this->index(axis0->findNearestIndex(p.c0), axis1->findNearestIndex(p.c1), axis2->findNearestIndex(p.c2))]);
    }
};


}   // namespace plask

#endif // PLASK__RECTILINEAR3D_H
