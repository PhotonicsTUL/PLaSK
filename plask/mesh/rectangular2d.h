#ifndef PLASK__RECTANGULAR2D_H
#define PLASK__RECTANGULAR2D_H

/** @file
This file contains rectangular mesh for 2D space.
*/

#include <iterator>

#include "rectangular_common.h"
#include "../utils/interpolation.h"
#include "../geometry/object.h"
#include "../geometry/space.h"
#include "../math.h"

namespace plask {

/**
 * Rectilinear mesh in 2D space.
 *
 * Includes two 1D rectilinear meshes:
 * - axis0 (alternative names: tran(), ee_x(), rad_r())
 * - axis1 (alternative names: up(), ee_y(), rad_z())
 * Represent all points (x, y) such that x is in axis0 and y is in axis1.
 */
class PLASK_API RectangularMesh2D: public RectangularMeshBase2D {

    typedef std::size_t index_ft(const RectangularMesh2D* mesh, std::size_t axis0_index, std::size_t axis1_index);
    typedef std::size_t index01_ft(const RectangularMesh2D* mesh, std::size_t mesh_index);

    // Our own virtual table, changeable in run-time:
    index_ft* index_f;
    index01_ft* index0_f;
    index01_ft* index1_f;

    const shared_ptr<MeshAxis>* minor_axis; ///< minor (changing fastest) axis
    const shared_ptr<MeshAxis>* major_axis; ///< major (changing slowest) axis

    void onAxisChanged(Event& e);

    void setChangeSignal(const shared_ptr<MeshAxis>& axis) { if (axis) axis->changedConnectMethod(this, &RectangularMesh2D::onAxisChanged); }
    void unsetChangeSignal(const shared_ptr<MeshAxis>& axis) { if (axis) axis->changedDisconnectMethod(this, &RectangularMesh2D::onAxisChanged); }

  public:

    /**
     * Represent FEM-like element in RectangularMesh.
     */
    class PLASK_API Element {
        const RectangularMesh2D& mesh;
        std::size_t index0, index1; // probably this form allows to do most operation fastest in average, low indexes of element corner or just element indexes

        public:

        /**
         * Construct element using mesh and element indexes.
         * @param mesh mesh, this element is valid up to time of this mesh life
         * @param index0, index1 axis 0 and 1 indexes of element (equal to low corrner mesh indexes of element)
         */
        Element(const RectangularMesh2D& mesh, std::size_t index0, std::size_t index1): mesh(mesh), index0(index0), index1(index1) {}

        /**
         * Construct element using mesh and element index.
         * @param mesh mesh, this element is valid up to time of this mesh life
         * @param elementIndex index of element
         */
        Element(const RectangularMesh2D& mesh, std::size_t elementIndex): mesh(mesh) {
            std::size_t v = mesh.getElementMeshLowIndex(elementIndex);
            index0 = mesh.index0(v);
            index1 = mesh.index1(v);
        }

        /// \return tran index of the element
        inline std::size_t getIndex0() const { return index0; }

        /// \return vert index of the element
        inline std::size_t getIndex1() const { return index1; }

        /// \return tran index of the left edge of the element
        inline std::size_t getLowerIndex0() const { return index0; }

        /// \return vert index of the bottom edge of the element
        inline std::size_t getLowerIndex1() const { return index1; }

        /// \return tran coordinate of the left edge of the element
        inline double getLower0() const { return mesh.axis[0]->at(index0); }

        /// \return vert coordinate of the bottom edge of the element
        inline double getLower1() const { return mesh.axis[1]->at(index1); }

        /// \return tran index of the right edge of the element
        inline std::size_t getUpperIndex0() const { return index0+1; }

        /// \return vert index of the top edge of the element
        inline std::size_t getUpperIndex1() const { return index1+1; }

        /// \return tran coordinate of the right edge of the element
        inline double getUpper0() const { return mesh.axis[0]->at(getUpperIndex0()); }

        /// \return vert coordinate of the top edge of the element
        inline double getUpper1() const { return mesh.axis[1]->at(getUpperIndex1()); }

        /// \return size of the element in the tran direction
        inline double getSize0() const { return getUpper0() - getLower0(); }

        /// \return size of the element in the vert direction
        inline double getSize1() const { return getUpper1() - getLower1(); }

        /// \return vector indicating size of the element
        inline Vec<2, double> getSize() const { return getUpUp() - getLoLo(); }

        /// \return position of the middle of the element
        inline Vec<2, double> getMidpoint() const { return mesh.getElementMidpoint(index0, index1); }

        /// @return index of this element
        inline std::size_t getIndex() const { return mesh.getElementIndexFromLowIndexes(getLowerIndex0(), getLowerIndex1()); }

        /// \return this element as rectangular box
        inline Box2D toBox() const { return mesh.getElementBox(index0, index1); }

        /// \return total area of this element
        inline double getVolume() const { return getSize0() * getSize1(); }

        /// \return total area of this element
        inline double getArea() const { return getVolume(); }

        /// \return index of the lower left corner of this element
        inline std::size_t getLoLoIndex() const { return mesh.index(getLowerIndex0(), getLowerIndex1()); }

        /// \return index of the upper left corner of this element
        inline std::size_t getLoUpIndex() const { return mesh.index(getLowerIndex0(), getUpperIndex1()); }

        /// \return index of the lower right corner of this element
        inline std::size_t getUpLoIndex() const { return mesh.index(getUpperIndex0(), getLowerIndex1()); }

        /// \return index of the upper right corner of this element
        inline std::size_t getUpUpIndex() const { return mesh.index(getUpperIndex0(), getUpperIndex1()); }

        /// \return position of the lower left corner of this element
        inline Vec<2, double> getLoLo() const { return mesh(getLowerIndex0(), getLowerIndex1()); }

        /// \return position of the upper left corner of this element
        inline Vec<2, double> getLoUp() const { return mesh(getLowerIndex0(), getUpperIndex1()); }

        /// \return position of the lower right corner of this element
        inline Vec<2, double> getUpLo() const { return mesh(getUpperIndex0(), getLowerIndex1()); }

        /// \return position of the upper right corner of this element
        inline Vec<2, double> getUpUp() const { return mesh(getUpperIndex0(), getUpperIndex1()); }

    };

    /**
     * Wrapper to RectangularMesh which allow to access to FEM-like elements.
     *
     * It works like read-only, random access container of @ref Element objects.
     */
    class PLASK_API Elements {

        static inline Element deref(const RectangularMesh2D& mesh, std::size_t index) { return mesh.getElement(index); }
    public:
        typedef IndexedIterator<const RectangularMesh2D, Element, deref> const_iterator;
        typedef const_iterator iterator;

        const RectangularMesh2D* mesh;

        Elements(const RectangularMesh2D* mesh): mesh(mesh) {}

        /**
         * Get @p i-th element.
         * @param i element index
         * @return @p i-th element
         */
        Element operator[](std::size_t i) const { return Element(*mesh, i); }

        /**
         * Get element with indices \p i0 and \p i1.
         * \param i0, i1 element index
         * \return element with indices \p i0 and \p i1
         */
        Element operator()(std::size_t i0, std::size_t i1) const { return Element(*mesh, i0, i1); }

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

    /// First and second coordinates of points in this mesh.
    const shared_ptr<MeshAxis> axis[2];

    /// Accessor to FEM-like elements.
    Elements elements() const { return Elements(this); }
    Elements getElements() const { return elements(); }

    Element element(std::size_t i0, std::size_t i1) const { return Element(*this, i0, i1); }
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
     * Iteration orders:
     * - 10 iteration order is:
     * (axis0[0], axis1[0]), (axis0[1], axis1[0]), ..., (axis0[axis0->size-1], axis1[0]), (axis0[0], axis1[1]), ..., (axis0[axis0->size()-1], axis1[axis[1]->size()-1])
     * - 01 iteration order is:
     * (axis0[0], axis1[0]), (axis0[0], axis1[1]), ..., (axis0[0], y[axis[1]->size-1]), (axis0[1], axis1[0]), ..., (axis0[axis0->size()-1], axis1[axis[1]->size()-1])
     * @see setIterationOrder, getIterationOrder, setOptimalIterationOrder
     */
    enum IterationOrder { ORDER_10, ORDER_01 };

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
    void setOptimalIterationOrder() {
        setIterationOrder(axis[0]->size() > axis[1]->size() ? ORDER_01 : ORDER_10);
    }

    /**
     * Construct mesh which has all axes of type OrderedAxis and all are empty.
     * @param iterationOrder iteration order
     */
    explicit RectangularMesh2D(IterationOrder iterationOrder = ORDER_01);

    /**
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     * @param iterationOrder iteration order
     */
    RectangularMesh2D(shared_ptr<MeshAxis> axis0, shared_ptr<MeshAxis> axis1, IterationOrder iterationOrder = ORDER_01);

    /**
     * Change axes and iteration order of this mesh.
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     * @param iterationOrder iteration order
     */
    void reset(shared_ptr<MeshAxis> axis0, shared_ptr<MeshAxis> axis1, IterationOrder iterationOrder = ORDER_01);

    /**
     * Copy constructor.
     * @param src mesh to copy
     * @param clone_axes whether axes of the @p src should be cloned (if true) or shared (if false; default)
     */
    RectangularMesh2D(const RectangularMesh2D& src, bool clone_axes = false);

    /**
     * Change axes and iteration order of this mesh to the ones from @p src.
     * @param src mesh to copy
     * @param iterationOrder iteration order
     */
    void reset(const RectangularMesh2D& src, bool clone_axes = false);

    RectangularMesh2D& operator=(const RectangularMesh2D& src) { reset(src, true); return *this; }

    RectangularMesh2D& operator=(RectangularMesh2D&& src) { reset(src, false); return *this; }

    ~RectangularMesh2D();

    /**
     * Change axis.
     * @param axis_nr number of axis to change
     * @param new_val new value for axis
     * @param fireResized whether to call fireResized()
     */
    void setAxis(std::size_t axis_nr, shared_ptr<MeshAxis> new_val, bool fireResized = true);

    const shared_ptr<MeshAxis> getAxis0() const { return axis[0]; }

    void setAxis0(shared_ptr<MeshAxis> a0) { setAxis(0, a0); }

    const shared_ptr<MeshAxis> getAxis1() const { return axis[1]; }

    void setAxis1(shared_ptr<MeshAxis> a1) { setAxis(1, a1); }



    /*
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate, or constructor argument for the first coordinate mesh
     * @param mesh1 mesh for the second coordinate, or constructor argument for the second coordinate mesh
     * @param iterationOrder iteration order
     */
    /*template <typename Mesh0CtorArg, typename Mesh1CtorArg>
    RectangularMesh(Mesh0CtorArg&& mesh0, Mesh1CtorArg&& mesh1, IterationOrder iterationOrder = ORDER_10):
        axis0(std::forward<Mesh0CtorArg>(mesh0)), axis1(std::forward<Mesh1CtorArg>(mesh1)) elements(this) {
        axis0->owner = this; axis[1]->owner = this;
        setIterationOrder(iterationOrder); }*/

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<MeshAxis>& tran() const { return axis[0]; }

    void setTran(shared_ptr<MeshAxis> a0) { setAxis0(a0); }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<MeshAxis>& vert() const { return axis[1]; }

    void setVert(shared_ptr<MeshAxis> a1) { setAxis1(a1); }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<MeshAxis>& ee_x() const { return axis[0]; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<MeshAxis>& ee_y() const { return axis[1]; }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const shared_ptr<MeshAxis>& rad_r() const { return axis[0]; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const shared_ptr<MeshAxis>& rad_z() const { return axis[1]; }

    /**
     * Get numbered axis
     * @param n number of axis
     * @return n-th axis (cn)
     */
    const shared_ptr<MeshAxis>& getAxis(size_t n) const {
        if (n >= 2) throw Exception("Bad axis number");
        return axis[n];
    }

    /// \return major (changing slowest) axis
    inline const shared_ptr<MeshAxis> majorAxis() const {
        return *major_axis;
    }

    /// \return minor (changing fastest) axis
    inline const shared_ptr<MeshAxis> minorAxis() const {
        return *minor_axis;
    }

    /**
      * Compare meshes
      * @param to_compare mesh to compare
      * @return @c true only if this mesh and @p to_compare represents the same set of points regardless of iteration order
      */
    bool operator==(const RectangularMesh2D& to_compare) const {
        return *axis[0] == *to_compare.axis[0] && *axis[1] == *to_compare.axis[1];
    }

    /**
     * Get number of points in mesh.
     * @return number of points in mesh
     */
    std::size_t size() const override { return axis[0]->size() * axis[1]->size(); }

    /**
     * Get maximum of sizes axis0 and axis1
     * @return maximum of sizes axis0 and axis1
     */
    std::size_t getMaxSize() const { return std::max(axis[0]->size(), axis[1]->size()); }

    /**
     * Get minimum of sizes axis0 and axis1
     * @return minimum of sizes axis0 and axis1
     */
    std::size_t getMinSize() const { return std::min(axis[0]->size(), axis[1]->size()); }

    /**
     * Write mesh to XML
     * \param object XML object to write to
     */
    void writeXML(XMLElement& object) const override;

    /// @return true only if there are no points in mesh
    bool empty() const override { return axis[0]->empty() || axis[1]->empty(); }

    /**
     * Calculate this mesh index using indexes of axis[0] and axis[1].
     * @param axis0_index index of axis[0], from 0 to axis[0]->size()-1
     * @param axis1_index index of axis[1], from 0 to axis[1]->size()-1
     * @return this mesh index, from 0 to size()-1
     */
    inline std::size_t index(std::size_t axis0_index, std::size_t axis1_index) const {
        return index_f(this, axis0_index, axis1_index);
    }

    /**
     * Calculate this mesh index using indexes of axis[0] and axis[1].
     * @param indexes index of axis[0] and axis[1]
     * @return this mesh index, from 0 to size()-1
     */
    inline std::size_t index(const Vec<2, std::size_t>& indexes) const {
        return index(indexes[0], indexes[1]);
    }

    /**
     * Calculate index of axis0 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis0, from 0 to axis[0]->size()-1
     */
    inline std::size_t index0(std::size_t mesh_index) const {
        return index0_f(this, mesh_index);
    }

    /**
     * Calculate index of y using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis1, from 0 to axis[1]->size()-1
     */
    inline std::size_t index1(std::size_t mesh_index) const {
        return index1_f(this, mesh_index);
    }

    /**
     * Calculate indexes of axes.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis[0], axis[1], and axis[2]
     */
    inline Vec<2, std::size_t> indexes(std::size_t mesh_index) const {
        return Vec<2, std::size_t>(index0(mesh_index), index1(mesh_index));
    }

    /**
     * Calculate index of major axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of major axis, from 0 to majorAxis.size()-1
     */
    inline std::size_t majorIndex(std::size_t mesh_index) const {
        return mesh_index / (*minor_axis)->size();
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
     * @return point with given @p index
     */
    inline Vec<2, double> at(std::size_t index0, std::size_t index1) const {
        return Vec<2, double>(axis[0]->at(index0), axis[1]->at(index1));
    }

    /**
     * Get point with given mesh index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     */
    virtual Vec<2, double> at(std::size_t index) const override {
        return Vec<2, double>(axis[0]->at(index0(index)), axis[1]->at(index1(index)));
    }

    /**
     * Get point with given mesh index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     * @see IterationOrder
     */
    inline Vec<2,double> operator[](std::size_t index) const {
        return Vec<2, double>(axis[0]->at(index0(index)), axis[1]->at(index1(index)));
    }

    /**
     * Get point with given x and y indexes.
     * @param axis0_index index of axis0, from 0 to axis0->size()-1
     * @param axis1_index index of axis1, from 0 to axis[1]->size()-1
     * @return point with given axis0 and axis1 indexes
     */
    inline Vec<2,double> operator()(std::size_t axis0_index, std::size_t axis1_index) const {
        return Vec<2, double>(axis[0]->at(axis0_index), axis[1]->at(axis1_index));
    }

    /**
     * Remove all points from mesh.
     */
    /*void clear() {
        axis0->clear();
        axis[1]->clear();
    }*/

    /**
     * Return a mesh that enables iterating over middle points of the rectangles
     * \return new rectilinear mesh with points in the middles of original rectangles
     */
    shared_ptr<RectangularMesh2D> getMidpointsMesh() const;

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
        auto p = flags.wrap(point);

        size_t index0_lo, index0_hi;
        double left, right;
        bool invert_left, invert_right;
        prepareInterpolationForAxis(*axis[0], flags, p.c0, 0, index0_lo, index0_hi, left, right, invert_left, invert_right);

        size_t index1_lo, index1_hi;
        double bottom, top;
        bool invert_bottom, invert_top;
        prepareInterpolationForAxis(*axis[1], flags, p.c1, 1, index1_lo, index1_hi, bottom, top, invert_bottom, invert_top);

        typename std::remove_const<typename std::remove_reference<decltype(data[0])>::type>::type
            data_lb = data[index(index0_lo, index1_lo)],
            data_rb = data[index(index0_hi, index1_lo)],
            data_rt = data[index(index0_hi, index1_hi)],
            data_lt = data[index(index0_lo, index1_hi)];
        if (invert_left)   { data_lb = flags.reflect(0, data_lb); data_lt = flags.reflect(0, data_lt); }
        if (invert_right)  { data_rb = flags.reflect(0, data_rb); data_rt = flags.reflect(0, data_rt); }
        if (invert_top)    { data_lt = flags.reflect(1, data_lt); data_rt = flags.reflect(1, data_rt); }
        if (invert_bottom) { data_lb = flags.reflect(1, data_lb); data_rb = flags.reflect(1, data_rb); }

        return flags.postprocess(point, interpolation::bilinear(left, right, bottom, top, data_lb, data_rb, data_rt, data_lt, p.c0, p.c1));
    }

    /**
     * Calculate (using nearest neighbor interpolation) value of data in point using data in points described by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateNearestNeighbor(const RandomAccessContainer& data, const Vec<2>& point, const InterpolationFlags& flags) const
        -> typename std::remove_reference<decltype(data[0])>::type {
        auto p = flags.wrap(point);
        prepareNearestNeighborInterpolationForAxis(*axis[0], flags, p.c0, 0);
        prepareNearestNeighborInterpolationForAxis(*axis[1], flags, p.c1, 1);
        return flags.postprocess(point, data[this->index(axis[0]->findNearestIndex(p.c0), axis[1]->findNearestIndex(p.c1))]);
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
     * Get number of elements (for FEM method).
     * @return number of elements in this mesh
     */
    std::size_t getElementsCount() const {
        return (axis[0]->size() != 0 && axis[1]->size() != 0)?
            (axis[0]->size()-1) * (axis[1]->size()-1) : 0;
    }

    /**
     * Convert mesh index of bottom left element corner to index of this element.
     * @param mesh_index_of_el_bottom_left mesh index
     * @return index of the element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndex(std::size_t mesh_index_of_el_bottom_left) const {
        return mesh_index_of_el_bottom_left - mesh_index_of_el_bottom_left / (*minor_axis)->size();
    }

    /**
     * Convert mesh indexes of a bottom-left corner of an element to the index of this element.
     * @param axis0_index index of the corner along the axis0 (left), from 0 to axis[0]->size()-1
     * @param axis1_index index of the corner along the axis1 (bottom), from 0 to axis[1]->size()-1
     * @return index of the element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndexes(std::size_t axis0_index, std::size_t axis1_index) const {
        return getElementIndexFromLowIndex(index(axis0_index, axis1_index));
    }

    /**
     * Convert element index to mesh index of bottom-left element corner.
     * @param element_index index of element, from 0 to getElementsCount()-1
     * @return mesh index
     */
    std::size_t getElementMeshLowIndex(std::size_t element_index) const {
        return element_index + (element_index / ((*minor_axis)->size()-1));
    }

    /**
     * Convert an element index to mesh indexes of bottom-left corner of the element.
     * @param element_index index of the element, from 0 to getElementsCount()-1
     * @return axis 0 and axis 1 indexes of mesh,
     * you can easy calculate rest indexes of element corner by adding 1 to returned coordinates
     */
    Vec<2, std::size_t> getElementMeshLowIndexes(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return Vec<2, std::size_t>(index0(bl_index), index1(bl_index));
    }

    /**
     * Get an area of a given element.
     * @param index0, index1 axis 0 and axis 1 indexes of the element
     * @return the area of the element with given indexes
     */
    double getElementArea(std::size_t index0, std::size_t index1) const {
        return (axis[0]->at(index0+1) - axis[0]->at(index0)) * (axis[1]->at(index1+1) - axis[1]->at(index1));
    }

    /**
     * Get an area of a given element.
     * @param element_index index of the element
     * @return the area of the element with given index
     */
    double getElementArea(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return getElementArea(index0(bl_index), index1(bl_index));
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
     * Get point in center of Elements.
     * @param index0, index1 index of Elements
     * @return point in center of element with given index
     */
    Vec<2, double> getElementMidpoint(std::size_t index0, std::size_t index1) const {
        return vec(getElementMidpoint0(index0), getElementMidpoint1(index1));
    }

    /**
     * Get point in the center of an element.
     * @param element_index index of the element
     * @return point in the center of the element
     */
    Vec<2, double> getElementMidpoint(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return getElementMidpoint(index0(bl_index), index1(bl_index));
    }

    /**
     * Get element as rectangle.
     * @param index0, index1 index of Elements
     * @return box of elements with given index
     */
    Box2D getElementBox(std::size_t index0, std::size_t index1) const {
        return Box2D(axis[0]->at(index0), axis[1]->at(index1), axis[0]->at(index0+1), axis[1]->at(index1+1));
    }

    /**
     * Get an element as a rectangle.
     * @param element_index index of the element
     * @return the element as a rectangle (box)
     */
    Box2D getElementBox(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return getElementBox(index0(bl_index), index1(bl_index));
    }

  private:

    // Common code for: left, right, bottom, top boundries:
    struct BoundaryIteratorImpl: public BoundaryNodeSetImpl::IteratorImpl {

        const RectangularMesh2D &mesh;

        std::size_t line;

        std::size_t index;

        BoundaryIteratorImpl(const RectangularMesh2D& mesh, std::size_t line, std::size_t index): mesh(mesh), line(line), index(index) {}

        virtual void increment() override { ++index; }

        virtual bool equal(const typename BoundaryNodeSetImpl::IteratorImpl& other) const override {
            return index == static_cast<const BoundaryIteratorImpl&>(other).index;
        }

    };

    // iterator over vertical line (from bottom to top). for left and right boundaries
    struct VerticalIteratorImpl: public BoundaryIteratorImpl {

        VerticalIteratorImpl(const RectangularMesh2D& mesh, std::size_t line, std::size_t index): BoundaryIteratorImpl(mesh, line, index) {}

        virtual std::size_t dereference() const override { return this->mesh.index(this->line, this->index); }

        virtual typename BoundaryNodeSetImpl::IteratorImpl* clone() const override {
            return new VerticalIteratorImpl(*this);
        }
    };

    // iterator over horizonstal line (from left to right), for bottom and top boundaries
    struct HorizontalIteratorImpl: public BoundaryIteratorImpl {

        HorizontalIteratorImpl(const RectangularMesh2D& mesh, std::size_t line, std::size_t index): BoundaryIteratorImpl(mesh, line, index) {}

        virtual std::size_t dereference() const override { return this->mesh.index(this->index, this->line); }

        virtual typename BoundaryNodeSetImpl::IteratorImpl* clone() const override {
            return new HorizontalIteratorImpl(*this);
        }
    };

    struct VerticalBoundary: public BoundaryNodeSetWithMeshImpl<RectangularMesh2D> {

        typedef typename BoundaryNodeSetImpl::Iterator Iterator;

        std::size_t line;

        VerticalBoundary(const RectangularMesh2D& mesh, std::size_t line_axis0): BoundaryNodeSetWithMeshImpl<RectangularMesh2D>(mesh), line(line_axis0) {}

        //virtual LeftBoundary* clone() const { return new LeftBoundary(); }

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index0(mesh_index) == line;
        }

        Iterator begin() const override {
            return Iterator(new VerticalIteratorImpl(this->mesh, line, 0));
        }

        Iterator end() const override {
            return Iterator(new VerticalIteratorImpl(this->mesh, line, this->mesh.axis[1]->size()));
        }

        std::size_t size() const override {
            return this->mesh.axis[1]->size();
        }
    };


    struct VerticalBoundaryInRange: public BoundaryNodeSetWithMeshImpl<RectangularMesh2D> {

        typedef typename BoundaryNodeSetImpl::Iterator Iterator;

        std::size_t line, beginInLineIndex, endInLineIndex;

        VerticalBoundaryInRange(const RectangularMesh2D& mesh, std::size_t line_axis0, std::size_t beginInLineIndex, std::size_t endInLineIndex)
            : BoundaryNodeSetWithMeshImpl<RectangularMesh2D>(mesh), line(line_axis0), beginInLineIndex(beginInLineIndex), endInLineIndex(endInLineIndex) {}

        //virtual LeftBoundary* clone() const { return new LeftBoundary(); }

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index0(mesh_index) == line && in_range(this->mesh.index1(mesh_index), beginInLineIndex, endInLineIndex);
        }

        Iterator begin() const override {
            return Iterator(new VerticalIteratorImpl(this->mesh, line, beginInLineIndex));
        }

        Iterator end() const override {
            return Iterator(new VerticalIteratorImpl(this->mesh, line, endInLineIndex));
        }

        std::size_t size() const override {
            return endInLineIndex - beginInLineIndex;
        }
    };

    struct HorizontalBoundary: public BoundaryNodeSetWithMeshImpl<RectangularMesh2D> {

        typedef typename BoundaryNodeSetImpl::Iterator Iterator;

        std::size_t line;

        HorizontalBoundary(const RectangularMesh2D& mesh, std::size_t line_axis1): BoundaryNodeSetWithMeshImpl<RectangularMesh2D>(mesh), line(line_axis1) {}

        //virtual TopBoundary* clone() const { return new TopBoundary(); }

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index1(mesh_index) == line;
        }

        Iterator begin() const override {
            return Iterator(new HorizontalIteratorImpl(this->mesh, line, 0));
        }

        Iterator end() const override {
            return Iterator(new HorizontalIteratorImpl(this->mesh, line, this->mesh.axis[0]->size()));
        }

        std::size_t size() const override {
            return this->mesh.axis[0]->size();
        }
    };

    struct HorizontalBoundaryInRange: public BoundaryNodeSetWithMeshImpl<RectangularMesh2D> {

        typedef typename BoundaryNodeSetImpl::Iterator Iterator;

        std::size_t line, beginInLineIndex, endInLineIndex;

        HorizontalBoundaryInRange(const RectangularMesh2D& mesh, std::size_t line_axis1, std::size_t beginInLineIndex, std::size_t endInLineIndex)
            : BoundaryNodeSetWithMeshImpl<RectangularMesh2D>(mesh), line(line_axis1), beginInLineIndex(beginInLineIndex), endInLineIndex(endInLineIndex) {}
        //virtual TopBoundary* clone() const { return new TopBoundary(); }

        bool contains(std::size_t mesh_index) const override {
            return this->mesh.index1(mesh_index) == line && in_range(this->mesh.index0(mesh_index), beginInLineIndex, endInLineIndex);
        }

        Iterator begin() const override {
            return Iterator(new HorizontalIteratorImpl(this->mesh, line, beginInLineIndex));
        }

        Iterator end() const override {
            return Iterator(new HorizontalIteratorImpl(this->mesh, line, endInLineIndex));
        }

        std::size_t size() const override {
            return endInLineIndex - beginInLineIndex;
        }
    };

    //TODO
    /*struct HorizontalLineBoundary: public BoundaryLogicImpl<RectangularMesh2D> {

        double height;

        bool contains(const RectangularMesh &mesh, std::size_t mesh_index) const {
            return mesh.index1(mesh_index) == mesh.axis[1]->findNearestIndex(height);
        }
    };*/

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
struct InterpolationAlgorithm<RectangularMesh2D, SrcT, DstT, INTERPOLATION_LINEAR> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh2D>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->axis[0]->size() == 0 || src_mesh->axis[1]->size() == 0)
            throw BadMesh("interpolate", "Source mesh empty");
        return new LinearInterpolatedLazyDataImpl< DstT, RectangularMesh2D, SrcT >(src_mesh, src_vec, dst_mesh, flags);
    }
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh2D, SrcT, DstT, INTERPOLATION_NEAREST> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh2D>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->axis[0]->size() == 0 || src_mesh->axis[1]->size() == 0)
            throw BadMesh("interpolate", "Source mesh empty");
        return new NearestNeighborInterpolatedLazyDataImpl< DstT, RectangularMesh2D, SrcT >(src_mesh, src_vec, dst_mesh, flags);
    }
};

/**
 * Copy @p to_copy mesh using OrderedAxis to represent each axis in returned mesh.
 * @param to_copy mesh to copy
 * @return mesh with each axis of type OrderedAxis
 */
PLASK_API shared_ptr<RectangularMesh2D> make_rectangular_mesh(const RectangularMesh2D& to_copy);
inline shared_ptr<RectangularMesh2D> make_rectangular_mesh(shared_ptr<const RectangularMesh2D> to_copy) { return make_rectangular_mesh(*to_copy); }

} // namespace plask

#include "rectangular_spline.h"

#endif // PLASK__RECTANGULAR2D_H
