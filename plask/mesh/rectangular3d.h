#ifndef PLASK__RECTANGULAR3D_H
#define PLASK__RECTANGULAR3D_H

/** @file
This file contains rectilinear mesh for 3d space.
*/

#include "rectangular2d.h"

namespace plask {

/**
 * Rectilinear mesh in 3D space.
 *
 * Includes three 1d rectilinear meshes:
 * - axis0 (alternative names: lon(), ee_z(), rad_r())
 * - axis1 (alternative names: tran(), ee_x(), rad_phi())
 * - axis2 (alternative names: vert(), ee_y(), rad_z())
 * Represent all points (x, y, z) such that x is in axis0, y is in axis1, z is in axis2.
 */
template<>
class PLASK_API RectangularMesh<3>: public MeshD<3> {

    typedef std::size_t index_ft(const RectangularMesh<3>* mesh, std::size_t c0_index, std::size_t c1_index, std::size_t c2_index);
    typedef std::size_t index012_ft(const RectangularMesh<3>* mesh, std::size_t mesh_index);

    // our own virtual table, changeable in run-time:
    index_ft* index_f;
    index012_ft* index0_f;
    index012_ft* index1_f;
    index012_ft* index2_f;
    const shared_ptr<RectangularAxis>* minor_axis;
    const shared_ptr<RectangularAxis>* medium_axis;
    const shared_ptr<RectangularAxis>* major_axis;

    void onAxisChanged(Event& e);

    void setChangeSignal(const shared_ptr<RectangularAxis>& axis) { if (axis) axis->changedConnectMethod(this, &RectangularMesh<3>::onAxisChanged); }
    void unsetChangeSignal(const shared_ptr<RectangularAxis>& axis) { if (axis) axis->changedDisconnectMethod(this, &RectangularMesh<3>::onAxisChanged); }

    void setAxis(const shared_ptr<RectangularAxis>& axis, shared_ptr<RectangularAxis> new_val);

  public:

    /**
     * Represent FEM-like element in RectangularMesh.
     */
    class PLASK_API Element {
        const RectangularMesh<3>& mesh;
        std::size_t index0, index1, index2; // probably this form allows to do most operation fastest in average, low indexes of element corner or just element indexes

        public:

        /**
         * Construct element using mesh and element indexes.
         * @param mesh mesh, this element is valid up to time of this mesh life
         * @param index0, index1, index2 axis 0, 1 and 2 indexes of element (equal to low corrner mesh indexes of element)
         */
        Element(const RectangularMesh<3>& mesh, std::size_t index0, std::size_t index1, std::size_t index2): mesh(mesh), index0(index0), index1(index1), index2(index2) {}

        /**
         * Construct element using mesh and element index.
         * @param mesh mesh, this element is valid up to time of this mesh life
         * @param elementIndex index of element
         */
        Element(const RectangularMesh<3>& mesh, std::size_t elementIndex): mesh(mesh) {
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

        /// \return this element as rectangular box
        inline Box3D toBox() const { return mesh.getElementBox(index0, index1, index2); }

        /// \return total volume of this element
        inline double getVolume() const { return getSize0() * getSize1() * getSize2(); }

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
     * Wrapper to RectangularMesh which allow to access to FEM-like elements.
     *
     * It works like read-only, random access container of @ref Element objects.
     */
    struct PLASK_API Elements {

        typedef IndexedIterator<const Elements, Element> const_iterator;
        typedef const_iterator iterator;

        /// Mesh which elements will be accessable by this.
        const RectangularMesh<3>* mesh;

        /**
         * Create wrapper which allow to access to FEM-like elements of given @p mesh.
         * @param mesh mesh which elements will be accessable by this
         */
        Elements(const RectangularMesh<3>* mesh): mesh(mesh) {}

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

    /// Boundary type.
    typedef ::plask::Boundary<RectangularMesh<3>> Boundary;

    /// First coordinate of points in this mesh.
    const shared_ptr<RectangularAxis> axis0;

    /// Second coordinate of points in this mesh.
    const shared_ptr<RectangularAxis> axis1;

    /// Third coordinate of points in this mesh.
    const shared_ptr<RectangularAxis> axis2;

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
    explicit RectangularMesh(IterationOrder iterationOrder = ORDER_012);

    /**
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     * @param mesh2 mesh for the third coordinate
     * @param iterationOrder iteration order
     */
    RectangularMesh(shared_ptr<RectangularAxis> mesh0, shared_ptr<RectangularAxis> mesh1, shared_ptr<RectangularAxis> mesh2, IterationOrder iterationOrder = ORDER_012);

    /// Copy constructor
    RectangularMesh(const RectangularMesh<3>& src);

    ~RectangularMesh();

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<RectangularAxis>& lon() const { return axis0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<RectangularAxis>& tran() const { return axis1; }

    /**
     * Get third coordinate of points in this mesh.
     * @return axis2
     */
    const shared_ptr<RectangularAxis>& vert() const { return axis2; }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<RectangularAxis>& ee_z() const { return axis0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<RectangularAxis>& ee_x() const { return axis1; }

    /**
     * Get third coordinate of points in this mesh.
     * @return axis2
     */
    const shared_ptr<RectangularAxis>& ee_y() const { return axis2; }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<RectangularAxis>& rad_r() const { return axis0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<RectangularAxis>& rad_phi() const { return axis1; }

    /**
     * Get thirs coordinate of points in this mesh.
     * @return axis2
     */
    const shared_ptr<RectangularAxis>& rad_z() const { return axis2; }

    const shared_ptr<RectangularAxis> getAxis0() const { return axis0; }

    void setAxis0(shared_ptr<RectangularAxis> a0) { setAxis(this->axis0, a0);  }

    const shared_ptr<RectangularAxis> getAxis1() const { return axis1; }

    void setAxis1(shared_ptr<RectangularAxis> a1) { setAxis(this->axis1, a1); }

    const shared_ptr<RectangularAxis> getAxis2() const { return axis2; }

    void setAxis2(shared_ptr<RectangularAxis> a2) { setAxis(this->axis2, a2); }

    /**
     * Get numbered axis
     * \param n number of axis
     */
    const shared_ptr<RectangularAxis>& axis(size_t n) const {
        if (n == 0) return axis0;
        else if (n == 1) return axis1;
        else if (n != 2) throw Exception("Bad axis number");
        return axis2;
    }

    /// \return major (changing slowest) axis
    inline const shared_ptr<RectangularAxis> majorAxis() const {
        return *major_axis;
    }

    /// \return middle (between major and minor) axis
    inline const shared_ptr<RectangularAxis> mediumAxis() const {
        return *medium_axis;
    }

    /// \return minor (changing fastes) axis
    inline const shared_ptr<RectangularAxis> minorAxis() const {
        return *minor_axis;
    }

    /**
      * Compare meshes
      * @param to_compare mesh to compare
      * @return @c true only if this mesh and @p to_compare represents the same set of points regardless of iteration order
      */
    bool operator==(const RectangularMesh<3>& to_compare) const {
        return *axis0 == *to_compare.axis0 && *axis1 == *to_compare.axis1 && *axis2 == *to_compare.axis2;
    }

    /**
     * Get number of points in the mesh.
     * @return number of points in the mesh
     */
    virtual std::size_t size() const override { return axis0->size() * axis1->size() * axis2->size(); }

    /**
     * Write mesh to XML
     * \param object XML object to write to
     */
    virtual void writeXML(XMLElement& object) const override;


    /// @return true only if there are no points in mesh
    bool empty() const override { return axis0->empty() || axis1->empty() || axis2->empty(); }

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
     * Get point with given mesh indices.
     * @param index0 index of point in axis0
     * @param index1 index of point in axis1
     * @param index2 index of point in axis2
     * @return point with given @p index
     */
    inline Vec<3, double> at(std::size_t index0, std::size_t index1, std::size_t index2) const {
        return Vec<3, double>(axis0->at(index0), axis1->at(index1), axis2->at(index2));
    }

    /**
     * Get point with given mesh index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     */
    virtual Vec<3, double> at(std::size_t index) const override {
        return operator() (index0(index), index1(index), index2(index));
    }

    /**
     * Get point with given mesh index.
     * Points are in order: (c0[0], c1[0], c2[0]), (c0[1], c1[0], c2[0]), ..., (c0[c0.size-1], c1[0], c2[0]), (c0[0], c1[1], c2[0]), ..., (c0[c0.size()-1], c1[c1.size()-1], c2[c2.size()-1])
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     */
    inline Vec<3, double> operator[](std::size_t index) const {
        return operator() (index0(index), index1(index), index2(index));
    }

    /**
     * Get point with given x and y indexes.
     * @param c0_index index of c0, from 0 to c0.size()-1
     * @param c1_index index of c1, from 0 to c1.size()-1
     * @param c2_index index of c2, from 0 to c2.size()-1
     * @return point with given c0, c1 and c2 indexes
     */
    inline Vec<3, double> operator()(std::size_t c0_index, std::size_t c1_index, std::size_t c2_index) const {
        return Vec<3, double>(axis0->at(c0_index), axis1->at(c1_index), axis2->at(c2_index));
    }

    /**
     * Remove all points from mesh.
     */
    /*void clear() {
        axis0->clear();
        axis1->clear();
        axis2->clear();
    }*/

    /**
     * Return a mesh that enables iterating over middle points of the rectangles
     * \return new rectilinear mesh with points in the middles of original rectangles
     */
    shared_ptr<RectangularMesh> getMidpointsMesh();

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
    std::size_t getElementsCount2() const {
        return std::max(int(axis2->size())-1, 0);
    }

    /**
     * Get number of elements (for FEM method).
     * @return number of elements in this mesh
     */
    std::size_t getElementsCount() const {
        return std::max((int(axis0->size())-1) * (int(axis1->size())-1) * (int(axis2->size())-1), 0);
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
    Vec<3, double> getElementMidpoint(std::size_t index0, std::size_t index1, std::size_t index2) const {
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
     * Get element as rectangle.
     * @param index0, index1, index2 index of Elements
     * @return box of elements with given index
     */
    Box3D getElementBox(std::size_t index0, std::size_t index1, std::size_t index2) const {
        return Box3D(axis0->at(index0), axis1->at(index1), axis2->at(index2), axis0->at(index0+1), axis1->at(index1+1), axis2->at(index2+1));
    }

    /**
     * Get element as rectangle.
     * @param element_index index of element
     * @return box of elements with given index
     */
    Box3D getElementBox(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return getElementBox(index0(bl_index), index1(bl_index), index2(bl_index));
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
        size_t index0 = std::upper_bound(axis0->begin(), axis0->end(), p.c0).index;
        size_t index1 = std::upper_bound(axis1->begin(), axis1->end(), p.c1).index;
        size_t index2 = std::upper_bound(axis2->begin(), axis2->end(), p.c2).index;

        size_t index0_1;
        double back, front;
        bool invert_back = false, invert_front = false;
        if (index0 == 0) {
            if (flags.symmetric(0)) {
                index0_1 = 0;
                back = axis0->at(0);
                if (back > 0.) {
                    back = - back;
                    invert_back = true;
                } else if (flags.periodic(0)) {
                    back = 2. * flags.low(0) - back;
                    invert_back = true;
                } else {
                    back -= 1.;
                }
            } else if (flags.periodic(0)) {
                index0_1 = axis0->size() - 1;
                back = axis0->at(index0_1) - flags.high(0) + flags.low(0);
            } else {
                index0_1 = 0;
                back = axis0->at(0) - 1.;
            }
        } else {
            index0_1 = index0 - 1;
            back = axis0->at(index0_1);
        }
        if (index0 == axis0->size()) {
            if (flags.symmetric(0)) {
                --index0;
                front = axis0->at(index0);
                if (front < 0.) {
                    front = - front;
                    invert_front = true;
                } else if (flags.periodic(0)) {
                    back = 2. * flags.high(0) - front;
                    invert_front = true;
                } else {
                    front += 1.;
                }
            } else if (flags.periodic(0)) {
                index0 = 0;
                front = axis0->at(0) + flags.high(0) - flags.low(0);
                if (front == back) front += 1e-6;
            } else {
                --index0;
                front = axis0->at(index0) + 1.;
            }
        } else {
            front = axis0->at(index0);
        }

        size_t index1_1;
        double left, right;
        bool invert_left = false, invert_right = false;
        if (index1 == 0) {
            if (flags.symmetric(1)) {
                index1_1 = 0;
                left = axis1->at(0);
                if (left > 0.) {
                    left = - left;
                    invert_left = true;
                } else if (flags.periodic(1)) {
                    left = 2. * flags.low(1) - left;
                    invert_left = true;
                } else {
                    left -= 1.;
                }
            } else if (flags.periodic(1)) {
                index1_1 = axis1->size() - 1;
                left = axis1->at(index1_1) - flags.high(1) + flags.low(1);
            } else {
                index1_1 = 0;
                left = axis1->at(0) - 1.;
            }
        } else {
            index1_1 = index1 - 1;
            left = axis1->at(index1_1);
        }
        if (index1 == axis1->size()) {
            if (flags.symmetric(1)) {
                --index1;
                right = axis1->at(index1);
                if (right < 0.) {
                    right = - right;
                    invert_right = true;
                } else if (flags.periodic(1)) {
                    left = 2. * flags.high(1) - right;
                    invert_right = true;
                } else {
                    right += 1.;
                }
            } else if (flags.periodic(1)) {
                index1 = 0;
                right = axis1->at(0) + flags.high(1) - flags.low(1);
                if (right == left) right += 1e-6;
            } else {
                --index1;
                right = axis1->at(index1) + 1.;
            }
        } else {
            right = axis1->at(index1);
        }

        size_t index2_1;
        double bottom, top;
        bool invert_top = false, invert_bottom = false;
        if (index2 == 0) {
            if (flags.symmetric(2)) {
                index2_1 = 0;
                bottom = axis2->at(0);
                if (bottom > 0.) {
                    bottom = - bottom;
                    invert_bottom = true;
                } else if (flags.periodic(2)) {
                    bottom = 2. * flags.low(2) - bottom;
                    invert_bottom = true;
                } else {
                    bottom -= 1.;
                }
            } else if (flags.periodic(2)) {
                index2_1 = axis2->size() - 1;
                bottom = axis2->at(index2_1) - flags.high(2) + flags.low(2);
            } else {
                index2_1 = 0;
                bottom = axis2->at(0) - 1.;
            }
        } else {
            index2_1 = index2 - 1;
            bottom = axis2->at(index2_1);
        }
        if (index2 == axis2->size()) {
            if (flags.symmetric(2)) {
                --index2;
                top = axis2->at(index2);
                if (top < 0.) {
                    top = - top;
                    invert_top = true;
                } else if (flags.periodic(2)) {
                    top = 2. * flags.high(2) - top;
                    invert_top = true;
                } else {
                    top += 1.;
                }
            } else if (flags.periodic(2)) {
                index2 = 0;
                top = axis2->at(0) + flags.high(2) - flags.low(2);
                if (top == bottom) top += 1e-6;
            } else {
                --index2;
                top = axis2->at(index2) + 1.;
            }
        } else {
            top = axis2->at(index2);
        }

        // all indexes are in bounds
        typename std::remove_const<typename std::remove_reference<decltype(data[0])>::type>::type
            data_lll = data[index(index0_1, index1_1, index2_1)],
            data_hll = data[index(index0,   index1_1, index2_1)],
            data_hhl = data[index(index0,   index1  , index2_1)],
            data_lhl = data[index(index0_1, index1  , index2_1)],
            data_llh = data[index(index0_1, index1_1, index2)],
            data_hlh = data[index(index0,   index1_1, index2)],
            data_hhh = data[index(index0,   index1  , index2)],
            data_lhh = data[index(index0_1, index1  , index2)];

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

private:
    // Common code for: left, right, bottom, top, front, back boundaries:
    struct BoundaryIteratorImpl: public BoundaryLogicImpl::IteratorImpl {

        const RectangularMesh &mesh;

        std::size_t level;

        std::size_t index_f, index_s;

        const std::size_t index_f_size;

        BoundaryIteratorImpl(const RectangularMesh& mesh, std::size_t level, std::size_t index_f, std::size_t index_f_size, std::size_t index_s)
            : mesh(mesh), level(level), index_f(index_f), index_s(index_f_size == 0 ? 0 : index_s), index_f_size(index_f_size) {
        }

        virtual void increment() {
            ++index_f;
            if (index_f == index_f_size) {
                index_f = 0;
                ++index_s;
            }
        }

        virtual bool equal(const typename BoundaryLogicImpl::IteratorImpl& other) const {
            return index_f == static_cast<const BoundaryIteratorImpl&>(other).index_f && index_s == static_cast<const BoundaryIteratorImpl&>(other).index_s;
        }

    };

    // iterator with fixed first coordinate
    struct FixedIndex0IteratorImpl: public BoundaryIteratorImpl {

        FixedIndex0IteratorImpl(const RectangularMesh& mesh, std::size_t level_index0, std::size_t index_1, std::size_t index_2)
            : BoundaryIteratorImpl(mesh, level_index0, index_1, mesh.axis1->size(), index_2) {}

        virtual std::size_t dereference() const { return this->mesh.index(this->level, this->index_f, this->index_s); }

        virtual typename BoundaryLogicImpl::IteratorImpl* clone() const {
            return new FixedIndex0IteratorImpl(*this);
        }
    };

    // iterator with fixed second coordinate
    struct FixedIndex1IteratorImpl: public BoundaryIteratorImpl {

        FixedIndex1IteratorImpl(const RectangularMesh& mesh, std::size_t level_index1, std::size_t index_0, std::size_t index_2)
            : BoundaryIteratorImpl(mesh, level_index1, index_0, mesh.axis0->size(), index_2) {}

        virtual std::size_t dereference() const { return this->mesh.index(this->index_f, this->level, this->index_s); }

        virtual typename BoundaryLogicImpl::IteratorImpl* clone() const {
            return new FixedIndex1IteratorImpl(*this);
        }
    };

    // iterator with fixed third coordinate
    struct FixedIndex2IteratorImpl: public BoundaryIteratorImpl {

        FixedIndex2IteratorImpl(const RectangularMesh& mesh, std::size_t level_index2, std::size_t index_0, std::size_t index_1)
            : BoundaryIteratorImpl(mesh, level_index2, index_0, mesh.axis0->size(), index_1) {}

        virtual std::size_t dereference() const { return this->mesh.index(this->index_f, this->index_s, this->level); }

        virtual typename BoundaryLogicImpl::IteratorImpl* clone() const {
            return new FixedIndex2IteratorImpl(*this);
        }
    };

    struct BoundaryInRangeIteratorImpl: public BoundaryLogicImpl::IteratorImpl {

        const RectangularMesh &mesh;

        std::size_t level;

        std::size_t index_f, index_s;

        const std::size_t index_f_begin, index_f_end;

        BoundaryInRangeIteratorImpl(const RectangularMesh& mesh, std::size_t level,
                             std::size_t index_f, std::size_t index_f_begin, std::size_t index_f_end,
                             std::size_t index_s)
            : mesh(mesh), level(level), index_f(index_f), index_s(index_f_begin == index_f_end ? 0 : index_s), index_f_begin(index_f_begin), index_f_end(index_f_end) {
        }

        virtual void increment() {
            ++index_f;
            if (index_f == index_f_end) {
                index_f = index_f_begin;
                ++index_s;
            }
        }

        virtual bool equal(const typename BoundaryLogicImpl::IteratorImpl& other) const {
            return index_f == static_cast<const BoundaryInRangeIteratorImpl&>(other).index_f && index_s == static_cast<const BoundaryInRangeIteratorImpl&>(other).index_s;
        }

    };

    struct FixedIndex0Boundary: public BoundaryWithMeshLogicImpl<RectangularMesh<3>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t level_axis0;

        FixedIndex0Boundary(const RectangularMesh<3>& mesh, std::size_t level_axis0): BoundaryWithMeshLogicImpl<RectangularMesh<3>>(mesh), level_axis0(level_axis0) {}

        bool contains(std::size_t mesh_index) const {
            return this->mesh.index0(mesh_index) == level_axis0;
        }

        Iterator begin() const {
            return Iterator(new FixedIndex0IteratorImpl(this->mesh, level_axis0, 0, 0));
        }

        Iterator end() const {
            return Iterator(new FixedIndex0IteratorImpl(this->mesh, level_axis0, 0, this->mesh.axis2->size()));
        }

        std::size_t size() const {
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

        bool contains(std::size_t mesh_index) const {
            return this->mesh.index0(mesh_index) == level_axis0
                    && in_range(this->mesh.index1(mesh_index), beginAxis1, endAxis1)
                    && in_range(this->mesh.index2(mesh_index), beginAxis2, endAxis2);
        }

        Iterator begin() const {
            return Iterator(new FixedIndex0IteratorImpl(this->mesh, level_axis0, beginAxis1, beginAxis2));
        }

        Iterator end() const {
            return Iterator(new FixedIndex0IteratorImpl(this->mesh, level_axis0, beginAxis1, endAxis2));
        }

        std::size_t size() const {
            return (endAxis1 - beginAxis1) * (endAxis2 - beginAxis2);
        }

        bool empty() const {
            return beginAxis1 == endAxis1 || beginAxis2 == endAxis2;
        }
    };

    struct FixedIndex1Boundary: public BoundaryWithMeshLogicImpl<RectangularMesh<3>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t level_axis1;

        FixedIndex1Boundary(const RectangularMesh<3>& mesh, std::size_t level_axis1): BoundaryWithMeshLogicImpl<RectangularMesh<3>>(mesh), level_axis1(level_axis1) {}

        //virtual LeftBoundary* clone() const { return new LeftBoundary(); }

        bool contains(std::size_t mesh_index) const {
            return this->mesh.index1(mesh_index) == level_axis1;
        }

        Iterator begin() const {
            return Iterator(new FixedIndex1IteratorImpl(this->mesh, level_axis1, 0, 0));
        }

        Iterator end() const {
            return Iterator(new FixedIndex1IteratorImpl(this->mesh, level_axis1, 0, this->mesh.axis2->size()));
        }

        std::size_t size() const {
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

        bool contains(std::size_t mesh_index) const {
            return this->mesh.index1(mesh_index) == level_axis1
                    && in_range(this->mesh.index0(mesh_index), beginAxis0, endAxis0)
                    && in_range(this->mesh.index2(mesh_index), beginAxis2, endAxis2);
        }

        Iterator begin() const {
            return Iterator(new FixedIndex1IteratorImpl(this->mesh, level_axis1, beginAxis0, beginAxis2));
        }

        Iterator end() const {
            return Iterator(new FixedIndex1IteratorImpl(this->mesh, level_axis1, beginAxis0, endAxis2));
        }

        std::size_t size() const {
            return (endAxis0 - beginAxis0) * (endAxis2 - beginAxis2);
        }

        bool empty() const {
            return beginAxis0 == endAxis0 || beginAxis2 == endAxis2;
        }
    };


    struct FixedIndex2Boundary: public BoundaryWithMeshLogicImpl<RectangularMesh<3>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t level_axis2;

        FixedIndex2Boundary(const RectangularMesh<3>& mesh, std::size_t level_axis2): BoundaryWithMeshLogicImpl<RectangularMesh<3>>(mesh), level_axis2(level_axis2) {}

        //virtual LeftBoundary* clone() const { return new LeftBoundary(); }

        bool contains(std::size_t mesh_index) const {
            return this->mesh.index2(mesh_index) == level_axis2;
        }

        Iterator begin() const {
            return Iterator(new FixedIndex2IteratorImpl(this->mesh, level_axis2, 0, 0));
        }

        Iterator end() const {
            return Iterator(new FixedIndex2IteratorImpl(this->mesh, level_axis2, 0, this->mesh.axis1->size()));
        }

        std::size_t size() const {
            return this->mesh.axis0->size() * this->mesh.axis1->size();
        }
    };

    struct FixedIndex2BoundaryInRange: public BoundaryWithMeshLogicImpl<RectangularMesh<3>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t level_axis2, beginAxis0, endAxis0, beginAxis1, endAxis1;

        FixedIndex2BoundaryInRange(const RectangularMesh<3>& mesh, std::size_t level_axis2, std::size_t beginAxis0, std::size_t endAxis0, std::size_t beginAxis1, std::size_t endAxis1)
            : BoundaryWithMeshLogicImpl<RectangularMesh<3>>(mesh), level_axis2(level_axis2),
              beginAxis0(beginAxis0), endAxis0(endAxis0), beginAxis1(beginAxis1), endAxis1(endAxis1)
              {}

        bool contains(std::size_t mesh_index) const {
            return this->mesh.index2(mesh_index) == level_axis2
                    && in_range(this->mesh.index0(mesh_index), beginAxis0, endAxis0)
                    && in_range(this->mesh.index1(mesh_index), beginAxis1, endAxis1);
        }

        Iterator begin() const {
            return Iterator(new FixedIndex2IteratorImpl(this->mesh, level_axis2, beginAxis0, beginAxis1));
        }

        Iterator end() const {
            return Iterator(new FixedIndex2IteratorImpl(this->mesh, level_axis2, beginAxis0, endAxis1));
        }

        std::size_t size() const {
            return (endAxis0 - beginAxis0) * (endAxis1 - beginAxis1);
        }

        bool empty() const {
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
        return new LinearInterpolatedLazyDataImpl< DstT, RectangularMesh<3>, SrcT >(src_mesh, src_vec, dst_mesh, flags);
    }
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh<3>, SrcT, DstT, INTERPOLATION_NEAREST> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh<3>>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->axis0->size() == 0 || src_mesh->axis1->size() == 0 || src_mesh->axis2->size() == 0)
            throw BadMesh("interpolate", "Source mesh empty");
        return new NearestNeighborInterpolatedLazyDataImpl< DstT, RectangularMesh<3>, SrcT >(src_mesh, src_vec, dst_mesh, flags);
    }
};


/**
 * Copy @p to_copy mesh using OrderedAxis to represent each axis in returned mesh.
 * @param to_copy mesh to copy
 * @return mesh with each axis of type OrderedAxis
 */
PLASK_API shared_ptr<RectangularMesh<3> > make_rectilinear_mesh(const RectangularMesh<3> &to_copy);
inline shared_ptr<RectangularMesh<3>> make_rectilinear_mesh(shared_ptr<const RectangularMesh<3>> to_copy) { return make_rectilinear_mesh(*to_copy); }

template <>
inline Boundary<RectangularMesh<3>> parseBoundary<RectangularMesh<3>>(const std::string& boundary_desc, plask::Manager&) { return RectangularMesh<3>::getBoundary(boundary_desc); }

template <>
inline Boundary<RectangularMesh<3>> parseBoundary<RectangularMesh<3>>(XMLReader& boundary_desc, Manager& env) { return RectangularMesh<3>::getBoundary(boundary_desc, env); }

PLASK_API_EXTERN_TEMPLATE_CLASS(RectangularMesh<3>)

}   // namespace plask

#endif // PLASK__RECTANGULAR3D_H
