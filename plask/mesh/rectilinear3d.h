#ifndef PLASK__RECTILINEAR3D_H
#define PLASK__RECTILINEAR3D_H

/** @file
This file contains rectilinear mesh for 3D space.
*/

#include <type_traits>

#include "rectangular_common.h"
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
class PLASK_API RectilinearMesh3D: public RectangularMeshBase3D /*MeshD<3>*/ {

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
        inline double getLower0() const { return mesh.axis[0]->at(index0); }

        /// \return tran coordinate of the left edge of the element
        inline double getLower1() const { return mesh.axis[1]->at(index1); }

        /// \return vert coordinate of the bottom edge of the element
        inline double getLower2() const { return mesh.axis[2]->at(index2); }

        /// \return long index of the right edge of the element
        inline std::size_t getUpperIndex0() const { return index0+1; }

        /// \return tran index of the right edge of the element
        inline std::size_t getUpperIndex1() const { return index1+1; }

        /// \return vert index of the top edge of the element
        inline std::size_t getUpperIndex2() const { return index2+1; }

        /// \return long coordinate of the right edge of the element
        inline double getUpper0() const { return mesh.axis[0]->at(getUpperIndex0()); }

        /// \return tran coordinate of the right edge of the element
        inline double getUpper1() const { return mesh.axis[1]->at(getUpperIndex1()); }

        /// \return vert coordinate of the top edge of the element
         inline double getUpper2() const { return mesh.axis[2]->at(getUpperIndex2()); }

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

        static inline Element deref(const RectilinearMesh3D& mesh, std::size_t index) { return mesh.getElement(index); }
    public:
        typedef IndexedIterator<const RectilinearMesh3D, Element, deref> const_iterator;
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
        const_iterator begin() const { return const_iterator(mesh, 0); }

        /// @return iterator referring to the past-the-end element
        const_iterator end() const { return const_iterator(mesh, size()); }

    };

    /// First, second and third coordinates of points in this mesh.
    const shared_ptr<MeshAxis> axis[3];

    /// Accessor to FEM-like elements.
    Elements elements() const { return Elements(this); }
    Elements getElements() const { return elements(); }

    Element element(std::size_t i0, std::size_t i1, std::size_t i2) const { return Element(*this, i0, i1, i2); }
    Element getElement(std::size_t i0, std::size_t i1, std::size_t i2) const { return element(i0, i1, i2); }

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

    /**
     * Copy constructor.
     * @param src mesh to copy
     * @param clone_axes whether axes of the @p src should be cloned (if true) or shared (if false; default)
     */
    RectilinearMesh3D(const RectilinearMesh3D& src, bool clone_axes = false);

    ~RectilinearMesh3D();

    const shared_ptr<MeshAxis> getAxis0() const { return axis[0]; }

    void setAxis0(shared_ptr<MeshAxis> a0) { setAxis(this->axis[0], a0);  }

    const shared_ptr<MeshAxis> getAxis1() const { return axis[1]; }

    void setAxis1(shared_ptr<MeshAxis> a1) { setAxis(this->axis[1], a1); }

    const shared_ptr<MeshAxis> getAxis2() const { return axis[2]; }

    void setAxis2(shared_ptr<MeshAxis> a2) { setAxis(this->axis[2], a2); }

    /**
     * Get numbered axis
     * \param n number of axis
     */
    const shared_ptr<MeshAxis>& getAxis(size_t n) const {
        if (n >= 3) throw Exception("Bad axis number");
        return axis[n];
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
        return *axis[0] == *to_compare.axis[0] && *axis[1] == *to_compare.axis[1] && *axis[2] == *to_compare.axis[2];
    }

    /**
     * Get number of points in the mesh.
     * @return number of points in the mesh
     */
    std::size_t size() const override { return axis[0]->size() * axis[1]->size() * axis[2]->size(); }

    /// @return true only if there are no points in mesh
    bool empty() const override { return axis[0]->empty() || axis[1]->empty() || axis[2]->empty(); }

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
    double getElementMidpoint0(std::size_t index0) const { return 0.5 * (axis[0]->at(index0) + axis[0]->at(index0+1)); }

    /**
     * Get second coordinate of point in center of Elements.
     * @param index1 index of Elements (axis1 index)
     * @return second coordinate of point point in center of Elements with given index
     */
    double getElementMidpoint1(std::size_t index1) const { return 0.5 * (axis[1]->at(index1) + axis[1]->at(index1+1)); }

    /**
     * Get second coordinate of point in center of Elements.
     * @param index2 index of Elements (axis2 index)
     * @return second coordinate of point point in center of Elements with given index
     */
    double getElementMidpoint2(std::size_t index2) const { return 0.5 * (axis[2]->at(index2) + axis[2]->at(index2+1)); }

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
     * Calculate this mesh index using indexes of axis[0], axis[1] and axis[2].
     * @param indexes index of axis[0], axis[1] and axis[2]
     * @return this mesh index, from 0 to size()-1
     */
    inline std::size_t index(const Vec<3, std::size_t>& indexes) const {
        return index(indexes[0], indexes[1], indexes[2]);
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
     * Calculate indexes of axes.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis[0], axis[1], and axis[2]
     */
    inline Vec<3, std::size_t> indexes(std::size_t mesh_index) const {
        return Vec<3, std::size_t>(index0(mesh_index), index1(mesh_index), index2(mesh_index));
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
        const std::size_t s = axis[0]->size();
        return (s != 0)? s-1 : 0;
    }

    /**
     * Get number of elements (for FEM method) in the second direction.
     * @return number of elements in this mesh in the second direction (axis1 direction).
     */
    std::size_t getElementsCount1() const {
        const std::size_t s = axis[1]->size();
        return (s != 0)? s-1 : 0;
    }

    /**
     * Get number of elements (for FEM method) in the third direction.
     * @return number of elements in this mesh in the third direction (axis2 direction).
     */
    size_t getElementsCount2() const {
        const std::size_t s = axis[2]->size();
        return (s != 0)? s-1 : 0;
    }

    /**
     * Get number of elements (for FEM method).
     * @return number of elements in this mesh
     */
    size_t getElementsCount() const {
        return size_t(std::max(int(axis[0]->size())-1, 0) * std::max(int(axis[1]->size())-1, 0) * std::max(int(axis[2]->size())-1, 0));
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
     * Convert indexes of mesh axes of lower (along all axes) element corner to index of this element.
     * @param axis0_index, axis1_index, axis2_index indexes of corner
     * @return index of element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndex(std::size_t axis0_index, std::size_t axis1_index, std::size_t axis2_index) const {
        return getElementIndexFromLowIndex(index(axis0_index, axis1_index, axis2_index));
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
        prepareInterpolationForAxis(*axis[0], flags, p.c0, 0, index0_lo, index0_hi, back, front, invert_back, invert_front);

        size_t index1_lo, index1_hi;
        double left, right;
        bool invert_left, invert_right;
        prepareInterpolationForAxis(*axis[1], flags, p.c1, 1, index1_lo, index1_hi, left, right, invert_left, invert_right);

        size_t index2_lo, index2_hi;
        double bottom, top;
        bool invert_bottom, invert_top;
        prepareInterpolationForAxis(*axis[2], flags, p.c2, 2, index2_lo, index2_hi, bottom, top, invert_bottom, invert_top);

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
        prepareNearestNeighborInterpolationForAxis(*axis[0], flags, p.c0, 0);
        prepareNearestNeighborInterpolationForAxis(*axis[1], flags, p.c1, 1);
        prepareNearestNeighborInterpolationForAxis(*axis[2], flags, p.c2, 2);
        return flags.postprocess(point, data[this->index(axis[0]->findNearestIndex(p.c0), axis[1]->findNearestIndex(p.c1), axis[2]->findNearestIndex(p.c2))]);
    }

private:

  // Common code for: left, right, bottom, top, front, back boundaries:
  struct BoundaryIteratorImpl: public BoundaryNodeSetImpl::IteratorImpl {

      const RectilinearMesh3D &mesh;

      const std::size_t level;

      std::size_t index_f, index_s;

      const std::size_t index_f_begin, index_f_end;

      BoundaryIteratorImpl(const RectilinearMesh3D& mesh, std::size_t level,
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

      FixedIndex0IteratorImpl(const RectilinearMesh3D& mesh, std::size_t level_index0, std::size_t index_1, std::size_t index_1_begin, std::size_t index_1_end, std::size_t index_2)
          : BoundaryIteratorImpl(mesh, level_index0, index_1, index_1_begin, index_1_end, index_2) {}

      virtual std::size_t dereference() const override { return this->mesh.index(this->level, this->index_f, this->index_s); }

      virtual typename BoundaryNodeSetImpl::IteratorImpl* clone() const override {
          return new FixedIndex0IteratorImpl(*this);
      }
  };

  // iterator with fixed second coordinate
  struct FixedIndex1IteratorImpl: public BoundaryIteratorImpl {

      FixedIndex1IteratorImpl(const RectilinearMesh3D& mesh, std::size_t level_index1, std::size_t index_0, std::size_t index_0_begin, std::size_t index_0_end, std::size_t index_2)
          : BoundaryIteratorImpl(mesh, level_index1, index_0, index_0_begin, index_0_end, index_2) {}

      virtual std::size_t dereference() const override { return this->mesh.index(this->index_f, this->level, this->index_s); }

      virtual typename BoundaryNodeSetImpl::IteratorImpl* clone() const override {
          return new FixedIndex1IteratorImpl(*this);
      }
  };

  // iterator with fixed third coordinate
  struct FixedIndex2IteratorImpl: public BoundaryIteratorImpl {

      FixedIndex2IteratorImpl(const RectilinearMesh3D& mesh, std::size_t level_index2, std::size_t index_0, std::size_t index_0_begin, std::size_t index_0_end, std::size_t index_1)
          : BoundaryIteratorImpl(mesh, level_index2, index_0, index_0_begin, index_0_end, index_1) {}

      virtual std::size_t dereference() const override { return this->mesh.index(this->index_f, this->index_s, this->level); }

      virtual typename BoundaryNodeSetImpl::IteratorImpl* clone() const override {
          return new FixedIndex2IteratorImpl(*this);
      }
  };

  struct FixedIndex0Boundary: public BoundaryNodeSetWithMeshImpl<RectilinearMesh3D> {

      typedef typename BoundaryNodeSetImpl::Iterator Iterator;

      std::size_t level_axis0;

      FixedIndex0Boundary(const RectilinearMesh3D& mesh, std::size_t level_axis0): BoundaryNodeSetWithMeshImpl<RectilinearMesh3D>(mesh), level_axis0(level_axis0) {}

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

  struct FixedIndex0BoundaryInRange: public BoundaryNodeSetWithMeshImpl<RectilinearMesh3D> {

      typedef typename BoundaryNodeSetImpl::Iterator Iterator;

      std::size_t level_axis0, beginAxis1, endAxis1, beginAxis2, endAxis2;

      FixedIndex0BoundaryInRange(const RectilinearMesh3D& mesh, std::size_t level_axis0, std::size_t beginAxis1, std::size_t endAxis1, std::size_t beginAxis2, std::size_t endAxis2)
          : BoundaryNodeSetWithMeshImpl<RectilinearMesh3D>(mesh), level_axis0(level_axis0),
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

  struct FixedIndex1Boundary: public BoundaryNodeSetWithMeshImpl<RectilinearMesh3D> {

      typedef typename BoundaryNodeSetImpl::Iterator Iterator;

      std::size_t level_axis1;

      FixedIndex1Boundary(const RectilinearMesh3D& mesh, std::size_t level_axis1): BoundaryNodeSetWithMeshImpl<RectilinearMesh3D>(mesh), level_axis1(level_axis1) {}

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

  struct FixedIndex1BoundaryInRange: public BoundaryNodeSetWithMeshImpl<RectilinearMesh3D> {

      typedef typename BoundaryNodeSetImpl::Iterator Iterator;

      std::size_t level_axis1, beginAxis0, endAxis0, beginAxis2, endAxis2;

      FixedIndex1BoundaryInRange(const RectilinearMesh3D& mesh, std::size_t level_axis1, std::size_t beginAxis0, std::size_t endAxis0, std::size_t beginAxis2, std::size_t endAxis2)
          : BoundaryNodeSetWithMeshImpl<RectilinearMesh3D>(mesh), level_axis1(level_axis1),
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


  struct FixedIndex2Boundary: public BoundaryNodeSetWithMeshImpl<RectilinearMesh3D> {

      typedef typename BoundaryNodeSetImpl::Iterator Iterator;

      std::size_t level_axis2;

      FixedIndex2Boundary(const RectilinearMesh3D& mesh, std::size_t level_axis2): BoundaryNodeSetWithMeshImpl<RectilinearMesh3D>(mesh), level_axis2(level_axis2) {}

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

  struct FixedIndex2BoundaryInRange: public BoundaryNodeSetWithMeshImpl<RectilinearMesh3D> {

      typedef typename BoundaryNodeSetImpl::Iterator Iterator;

      std::size_t level_axis2, beginAxis0, endAxis0, beginAxis1, endAxis1;

      FixedIndex2BoundaryInRange(const RectilinearMesh3D& mesh, std::size_t level_axis2, std::size_t beginAxis0, std::size_t endAxis0, std::size_t beginAxis1, std::size_t endAxis1)
          : BoundaryNodeSetWithMeshImpl<RectilinearMesh3D>(mesh), level_axis2(level_axis2),
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


}   // namespace plask

#endif // PLASK__RECTILINEAR3D_H
