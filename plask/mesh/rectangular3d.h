#ifndef PLASK__RECTANGULAR3D_H
#define PLASK__RECTANGULAR3D_H

/** @file
This file includes rectilinear mesh for 3d space.
*/

#include "rectangular2d.h"

namespace plask {

/**
 * Rectilinear mesh in 3D space.
 *
 * Includes three 1d rectilinear meshes:
 * - c0 (alternative names: lon(), ee_z(), r())
 * - c1 (alternative names: tran, ee_x(), phi())
 * - c2 (alternative names: up(), ee_y(), z())
 * Represent all points (x, y, z) such that x is in c0, y is in c1, z is in c2.
 */
//TODO methods which call fireResize() when points are adding, etc.
template <typename Mesh1D>
class RectangularMesh<3,Mesh1D>: public MeshD<3> {

    static_assert(std::is_floating_point< typename std::remove_reference<decltype(std::declval<Mesh1D>().operator[](0))>::type >::value,
                  "Mesh1d must have operator[](std::size_t index) which returns floating-point value");

    typedef std::size_t index_ft(const RectangularMesh* mesh, std::size_t c0_index, std::size_t c1_index, std::size_t c2_index);
    typedef std::size_t index012_ft(const RectangularMesh* mesh, std::size_t mesh_index);

    // our own virtual table, changeable in run-time:
    index_ft* index_f;
    index012_ft* index0_f;
    index012_ft* index1_f;
    index012_ft* index2_f;
    Mesh1D* minor_axis;
    Mesh1D* middle_axis;
    Mesh1D* major_axis;

    shared_ptr<RectangularMesh<3,Mesh1D>> midpoints_cache; ///< cache for midpoints mesh

  protected:

    virtual void onChange(const Event& evt) {
        midpoints_cache.reset();
    }

  public:
    
    /**
     * Represent FEM-like element in RectangularMesh.
     */
    class Element {
        const RectangularMesh<3,Mesh1D>& mesh;
        std::size_t index0, index1, index2; // probably this form allows to do most operation fastest in average, low indexes of element corner or just element indexes

        public:

        /**
         * Construct element using mesh and element indexes.
         * @param mesh mesh, this element is valid up to time of this mesh life
         * @param index0, index1, index2 axis 0, 1 and 2 indexes of element (equal to low corrner mesh indexes of element)
         */
        Element(const RectangularMesh<3,Mesh1D>& mesh, std::size_t index0, std::size_t index1, std::size_t index2): mesh(mesh), index0(index0), index1(index1), index2(index2) {}

        /**
         * Construct element using mesh and element index.
         * @param mesh mesh, this element is valid up to time of this mesh life
         * @param elementIndex index of element
         */
        Element(const RectangularMesh<3,Mesh1D>& mesh, std::size_t elementIndex): mesh(mesh) {
            std::size_t v = mesh.getElementMeshLowIndex(elementIndex);
            index0 = mesh.index0(v);
            index1 = mesh.index1(v);
            index2 = mesh.index2(v);
        }

        typedef typename Mesh1D::PointType PointType;

        inline std::size_t getLowerIndex0() const { return index0; }

        inline std::size_t getLowerIndex1() const { return index1; }
        
        inline std::size_t getLowerIndex2() const { return index2; }

        inline PointType getLower0() const { return mesh.axis0[index0]; }

        inline PointType getLower1() const { return mesh.axis1[index1]; }
        
        inline PointType getLower2() const { return mesh.axis2[index2]; }

        inline std::size_t getUpperIndex0() const { return index0+1; }

        inline std::size_t getUpperIndex1() const { return index1+1; }
        
        inline std::size_t getUpperIndex2() const { return index2+1; }

        inline PointType getUpper0() const { return mesh.axis0[getUpperIndex0()]; }

        inline PointType getUpper1() const { return mesh.axis1[getUpperIndex1()]; }
        
        inline PointType getUpper2() const { return mesh.axis2[getUpperIndex2()]; }

        inline PointType getSize0() const { return getUpper0() - getLower0(); }

        inline PointType getSize1() const { return getUpper1() - getLower1(); }
        
        inline PointType getSize2() const { return getUpper2() - getLower2(); }

        inline Vec<3, PointType> getSize() const { return getUpUpUp() - getLoLoLo(); }

        inline Vec<3, PointType> getMidpoint() const { return mesh.getElementMidpoint(index0, index1, index2); }

        /// @return this element index
        inline std::size_t getIndex() const { return mesh.getElementIndexFromLowIndex(getLoLoLoIndex()); }

        inline Box3D toBox() const { return mesh.getElementBox(index0, index1, index2); }

        inline std::size_t getLoLoLoIndex() const { return mesh.index(getLowerIndex0(), getLowerIndex1(), getLowerIndex2()); }

        //inline std::size_t getLoUpIndex() const { return mesh.index(getLowerIndex0(), getUpperIndex1()); }

        //inline std::size_t getUpLoIndex() const { return mesh.index(getUpperIndex0(), getLowerIndex1()); }

        inline std::size_t getUpUpUpIndex() const { return mesh.index(getUpperIndex0(), getUpperIndex1(), getUpperIndex2()); }

        inline Vec<3, PointType> getLoLoLo() const { return mesh(getLowerIndex0(), getLowerIndex1(), getLowerIndex2()); }

        //inline Vec<2, PointType> getLoUp() const { return mesh(getLowerIndex0(), getUpperIndex1()); }

        //inline Vec<2, PointType> getUpLo() const { return mesh(getUpperIndex0(), getLowerIndex1()); }

        inline Vec<3, PointType> getUpUpUp() const { return mesh(getUpperIndex0(), getUpperIndex1(), getUpperIndex2()); }

    };
    
    /**
     * Wrapper to RectangularMesh which allow to access to FEM-like elements.
     *
     * It works like read-only, random access container of @ref Element objects.
     */
    struct Elements {

        typedef IndexedIterator<const Elements, Element> const_iterator;
        typedef const_iterator iterator;

        const RectangularMesh<3,Mesh1D>* mesh;

        Elements(const RectangularMesh<3,Mesh1D>* mesh): mesh(mesh) {}

        /**
         * Get @p i-th element.
         * @param i element index
         * @return @p i-th element
         */
        Element operator[](std::size_t i) const { return Element(*mesh, i); }

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
    typedef ::plask::Boundary<RectangularMesh<3,Mesh1D>> Boundary;

    /// First coordinate of points in this mesh.
    Mesh1D axis0;

    /// Second coordinate of points in this mesh.
    Mesh1D axis1;

    /// Third coordinate of points in this mesh.
    Mesh1D axis2;
    
    /// Accessor to FEM-like elements.
    const Elements elements;

    /**
     * Iteration orders:
     * - normal iteration order (ORDER_012) is:
     *   (c0[0], c1[0]), (c0[1], c1[0]), ..., (c0[c0.size-1], c1[0]), (c0[0], c1[1]), ..., (c0[c0.size()-1], c1[c1.size()-1])
     * Every other order is proper permutation of indices
     * @see setIterationOrder, getIterationOrder, setOptimalIterationOrder
     */
    enum IterationOrder { ORDER_012, ORDER_021, ORDER_102, ORDER_120, ORDER_201, ORDER_210 };

    /// Construct an empty mesh
    RectangularMesh(IterationOrder iterationOrder=ORDER_210): elements(this) {
        axis0.owner = this; axis1.owner = this; axis2.owner = this;
        setIterationOrder(iterationOrder);
    }

    /// Copy constructor
    RectangularMesh(const RectangularMesh& src): axis0(src.axis0), axis1(src.axis1), axis2(src.axis2), elements(this) {
        axis0.owner = this; axis1.owner = this; axis2.owner = this;
        setIterationOrder(src.getIterationOrder());
    }

    /// Move constructor
    RectangularMesh(RectangularMesh&& src): axis0(std::move(src.axis0)), axis1(std::move(src.axis1)), axis2(std::move(src.axis2)), elements(this) {
        axis0.owner = this; axis1.owner = this; axis2.owner = this;
        setIterationOrder(src.getIterationOrder());
    }

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
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     * @param mesh2 mesh for the third coordinate
     * @param iterationOrder iteration order
     */
    RectangularMesh(Mesh1D mesh0, Mesh1D mesh1, Mesh1D mesh2, IterationOrder iterationOrder = ORDER_210) :
        axis0(std::move(mesh0)),
        axis1(std::move(mesh1)),
        axis2(std::move(mesh2)),
        elements(this)
    {
        axis0.owner = this; axis1.owner = this; axis2.owner = this;
        setIterationOrder(iterationOrder);
    }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    Mesh1D& lon() { return axis0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    const Mesh1D& lon() const { return axis0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    Mesh1D& tran() { return axis1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    const Mesh1D& tran() const { return axis1; }

    /**
     * Get third coordinate of points in this mesh.
     * @return c2
     */
    Mesh1D& up() { return axis2; }

    /**
     * Get third coordinate of points in this mesh.
     * @return c2
     */
    const Mesh1D& up() const { return axis2; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    Mesh1D& ee_z() { return axis0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    const Mesh1D& ee_z() const { return axis0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    Mesh1D& ee_x() { return axis1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    const Mesh1D& ee_x() const { return axis1; }

    /**
     * Get third coordinate of points in this mesh.
     * @return c2
     */
    Mesh1D& ee_y() { return axis1; }

    /**
     * Get third coordinate of points in this mesh.
     * @return c2
     */
    const Mesh1D& ee_y() const { return axis1; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    Mesh1D& rad_r() { return axis0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    const Mesh1D& rad_r() const { return axis0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    Mesh1D& rad_phi() { return axis1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    const Mesh1D& rad_phi() const { return axis1; }

    /**
     * Get thirs coordinate of points in this mesh.
     * @return c1
     */
    Mesh1D& rad_z() { return axis1; }

    /**
     * Get thirs coordinate of points in this mesh.
     * @return c1
     */
    const Mesh1D& rad_z() const { return axis1; }

    /**
     * Get numbered axis
     * \param n number of axis
     */
    Mesh1D& axis(size_t n) {
        if (n == 0) return axis0;
        else if (n == 1) return axis1;
        else if (n != 2) throw Exception("Bad axis number");
        return axis2;
    }

    /**
     * Get numbered axis
     * \param n number of axis
     */
    const Mesh1D& axis(size_t n) const {
        if (n == 0) return axis0;
        else if (n == 1) return axis1;
        else if (n != 2) throw Exception("Bad axis number");
        return axis2;
    }

    /// \return major (changing slowest) axis
    inline Mesh1D& majorAxis() {
        return *major_axis;
    }

    /// \return major (changing slowest) axis
    inline const Mesh1D& majorAxis() const {
        return *major_axis;
    }

    /// \return middle (between major and minor) axis
    inline const Mesh1D& middleAxis() const {
        return *middle_axis;
    }

    /// \return middle (between major and minor) axis
    inline Mesh1D& middleAxis() {
        return *middle_axis;
    }

    /// \return minor (changing fastes) axis
    inline const Mesh1D& minorAxis() const {
        return *minor_axis;
    }

    /// \return minor (changing fastes) axis
    inline Mesh1D& minorAxis() {
        return *minor_axis;
    }

    /**
      * Compare meshes
      * @param to_compare mesh to compare
      * @return @c true only if this mesh and @p to_compare represents the same set of points regardless of iteration order
      */
    bool operator==(const RectangularMesh<3,Mesh1D>& to_compare) {
        return axis0 == to_compare.axis0 && axis1 == to_compare.axis1 && axis2 == to_compare.axis2;
    }

    /**
     * Get number of points in the mesh.
     * @return number of points in the mesh
     */
    virtual std::size_t size() const { return axis0.size() * axis1.size() * axis2.size(); }

    /**
     * Write mesh to XML
     * \param object XML object to write to
     */
    virtual void writeXML(XMLElement& object) const;


    /// @return true only if there are no points in mesh
    bool empty() const { return axis0.empty() || axis1.empty() || axis2.empty(); }

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
        return mesh_index / minorAxis().size() / middleAxis().size();
    }

    /**
     * Calculate index of middle axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of middle axis, from 0 to middleAxis.size()-1
     */
    inline std::size_t middleIndex(std::size_t mesh_index) const {
        return (mesh_index / minorAxis().size()) % middleAxis().size();
    }

    /**
     * Calculate index of minor axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of minor axis, from 0 to minorAxis.size()-1
     */
    inline std::size_t minorIndex(std::size_t mesh_index) const {
        return mesh_index % minorAxis().size();
    }

    /**
     * Get point with given mesh indices.
     * @param index0 index of point in axis0
     * @param index1 index of point in axis1
     * @param index2 index of point in axis2
     * @return point with given @p index
     */
    inline Vec<3, double> at(std::size_t index0, std::size_t index1, std::size_t index2) const {
        return Vec<3, double>(axis0[index0], axis1[index1], axis2[index2]);
    }

    /**
     * Get point with given mesh index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     */
    virtual Vec<3, double> at(std::size_t index) const {
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
        return Vec<3, double>(axis0[c0_index], axis1[c1_index], axis2[c2_index]);
    }

    /**
     * Remove all points from mesh.
     */
    void clear() {
        axis0.clear();
        axis1.clear();
        axis2.clear();
    }

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
        return std::max(int(axis0.size())-1, 0);
    }

    /**
     * Get number of elements (for FEM method) in the second direction.
     * @return number of elements in this mesh in the second direction (axis1 direction).
     */
    std::size_t getElementsCount1() const {
        return std::max(int(axis1.size())-1, 0);
    }
    
    /**
     * Get number of elements (for FEM method) in the third direction.
     * @return number of elements in this mesh in the third direction (axis2 direction).
     */
    std::size_t getElementsCount2() const {
        return std::max(int(axis2.size())-1, 0);
    }
    
    /**
     * Get number of elements (for FEM method).
     * @return number of elements in this mesh
     */
    std::size_t getElementsCount() const {
        return std::max((int(axis0.size())-1) * (int(axis1.size())-1) * (int(axis2.size())-1), 0);
    }
    
    /**
     * Conver element index to mesh index of bottom, left, front element corner.
     * @param element_index index of element, from 0 to getElementsCount()-1
     * @return mesh index
     */
    std::size_t getElementMeshLowIndex(std::size_t element_index) const {
        const std::size_t minor_size_minus_1 = minor_axis->size()-1;
        const std::size_t elements_per_level = minor_size_minus_1 * (middle_axis->size()-1);
        return element_index + (element_index / elements_per_level) * (middle_axis->size() + minor_size_minus_1)
                            + (element_index % elements_per_level) / minor_size_minus_1;
    }
    
    /**
     * Conver mesh index of bottom, left, front element corner to this element index.
     * @param mesh_index_of_el_bottom_left mesh index
     * @return index of element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndex(std::size_t mesh_index_of_el_bottom_left) const {
        const std::size_t verticles_per_level = minor_axis->size() * middle_axis->size();
        return mesh_index_of_el_bottom_left - (mesh_index_of_el_bottom_left / verticles_per_level) * (middle_axis->size() + minor_axis->size() - 1)
                - (mesh_index_of_el_bottom_left % verticles_per_level) / minor_axis->size();
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
        return (axis0[index0+1] - axis0[index0]) * (axis1[index1+1] - axis1[index1]) * (axis2[index2+1] - axis2[index2]);
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
    double getElementMidpoint0(std::size_t index0) const { return 0.5 * (axis0[index0] + axis0[index0+1]); }

    /**
     * Get second coordinate of point in center of Elements.
     * @param index1 index of Elements (axis1 index)
     * @return second coordinate of point point in center of Elements with given index
     */
    double getElementMidpoint1(std::size_t index1) const { return 0.5 * (axis1[index1] + axis1[index1+1]); }
    
    /**
     * Get second coordinate of point in center of Elements.
     * @param index1 index of Elements (axis1 index)
     * @return second coordinate of point point in center of Elements with given index
     */
    double getElementMidpoint2(std::size_t index2) const { return 0.5 * (axis2[index2] + axis2[index2+1]); }
    
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
        return Box3D(axis0[index0], axis1[index1], axis2[index2], axis0[index0+1], axis1[index1+1], axis2[index2+1]);
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
    auto interpolateLinear(const RandomAccessContainer& data, const Vec<3, double>& point) const -> typename std::remove_reference<decltype(data[0])>::type {
        std::size_t index0 = axis0.findIndex(point.c0);
        std::size_t index1 = axis1.findIndex(point.c1);
        std::size_t index2 = axis2.findIndex(point.c2);

        if (index2 == 0)
            return interpolateLinear2D(
                [&] (std::size_t i0, std::size_t i1) { return data[index(i0, i1, 0)]; },
                point.c0, point.c1, axis0, axis1, index0, index1
            );

        if (index2 == axis0.size()) {
            --index2;
            return interpolateLinear2D(
                [&] (std::size_t i0, std::size_t i1) { return data[index(i0, i1, index2)]; },
                point.c0, point.c1, axis0, axis1, index0, index1
            );
        }

        if (index1 == 0)
            return interpolateLinear2D(
                [&] (std::size_t i0, std::size_t i2) { return data[index(i0, 0, i2)]; },
                point.c0, point.c2, axis0, axis2, index0, index2
            );

        if (index1 == axis1.size()) {
            --index1;
            return interpolateLinear2D(
                [&] (std::size_t i0, std::size_t i2) { return data[index(i0, index1, i2)]; },
                point.c0, point.c2, axis0, axis2, index0, index2
            );
        }

        // index1 and index2 are in bounds here:
        if (index0 == 0)
        return interpolation::bilinear(axis1[index1-1], axis1[index1],
                                        axis2[index2-1], axis2[index2],
                                        data[index(0, index1-1, index2-1)],
                                        data[index(0, index1,   index2-1)],
                                        data[index(0, index1,   index2  )],
                                        data[index(0, index1-1, index2  )],
                                        point.c1, point.c2);
        if (index0 == axis0.size()) {
            --index0;
        return interpolation::bilinear(axis1[index1-1], axis1[index1],
                                        axis2[index2-1], axis2[index2],
                                        data[index(index0, index1-1, index2-1)],
                                        data[index(index0, index1,   index2-1)],
                                        data[index(index0, index1,   index2  )],
                                        data[index(index0, index1-1, index2  )],
                                        point.c1, point.c2);
        }

        // all indexes are in bounds
        return interpolation::trilinear(
            axis0[index0-1], axis0[index0],
            axis1[index1-1], axis1[index1],
            axis2[index2-1], axis2[index2],
            data[index(index0-1, index1-1, index2-1)],
            data[index(index0,   index1-1, index2-1)],
            data[index(index0,   index1  , index2-1)],
            data[index(index0-1, index1  , index2-1)],
            data[index(index0-1, index1-1, index2)],
            data[index(index0,   index1-1, index2)],
            data[index(index0,   index1  , index2)],
            data[index(index0-1, index1  , index2)],
            point.c0, point.c1, point.c2
        );

    }
    
private:
    // Common code for: left, right, bottom, top, front, back boundries:
    struct BoundaryIteratorImpl: public BoundaryLogicImpl::IteratorImpl {

        const RectangularMesh &mesh;

        std::size_t level;

        std::size_t index_f, index_s;
        
        const std::size_t index_f_size;

        BoundaryIteratorImpl(const RectangularMesh& mesh, std::size_t level, std::size_t index_f, std::size_t index_f_size, std::size_t index_s)
        : mesh(mesh), level(level), index_f(index_f), index_s(index_s), index_f_size(index_f_size) {}

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
};

template <typename Mesh1D,typename DataT>    //for any data type
struct InterpolationAlgorithm<RectangularMesh<3,Mesh1D>, DataT, INTERPOLATION_LINEAR> {
    static void interpolate(const RectangularMesh<3,Mesh1D>& src_mesh, const DataVector<const DataT>& src_vec, const plask::MeshD<3>& dst_mesh, DataVector<DataT>& dst_vec) {
        #pragma omp parallel for
        for (size_t i = 0; i < dst_mesh.size(); ++i)
            dst_vec[i] = src_mesh.interpolateLinear(src_vec, dst_mesh[i]);
    }
};

}   // namespace plask

#endif // PLASK__RECTANGULAR3D_H
