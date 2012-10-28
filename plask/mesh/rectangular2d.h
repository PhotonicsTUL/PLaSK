#ifndef PLASK__RECTANGULAR2D_H
#define PLASK__RECTANGULAR2D_H

/** @file
This file includes rectilinear mesh for 2d space.
*/

#include <iterator>

#include "mesh.h"
#include "boundary.h"
#include "interpolation.h"
#include "../utils/interpolation.h"
#include "../geometry/object.h"
#include "../geometry/space.h"
#include "../geometry/path.h"
#include "../math.h"
#include "../manager.h"

namespace plask {

/**
 * Rectilinear mesh in 2D space.
 *
 * Includes two 1D rectilinear meshes:
 * - axis0 (alternative names: tran, ee_x(), r())
 * - axis1 (alternative names: up(), ee_y(), z())
 * Represent all points (x, y) such that x is in axis0 and y is in axis1.
 */
//TODO methods which call fireChanged() when points are added, etc.
template <typename Mesh1D>
class RectangularMesh<2,Mesh1D>: public MeshD<2> {

    static_assert(std::is_floating_point< typename std::remove_reference<decltype(std::declval<Mesh1D>().operator[](0))>::type >::value,
                  "Mesh1d must have operator[](std::size_t index) which returns floating-point value");

    typedef std::size_t index_ft(const RectangularMesh<2,Mesh1D>* mesh, std::size_t axis0_index, std::size_t axis1_index);
    typedef std::size_t index01_ft(const RectangularMesh<2,Mesh1D>* mesh, std::size_t mesh_index);

    // Our own virtual table, changeable in run-time:
    index_ft* index_f;
    index01_ft* index0_f;
    index01_ft* index1_f;
    Mesh1D* minor_axis; ///< minor (changing fastest) axis
    Mesh1D* major_axis; ///< major (changing slowest) axis

    shared_ptr<RectangularMesh<2,Mesh1D>> midpoints_cache; ///< cache for midpoints mesh

  protected:

    virtual void onChange(const Event& evt) {
        midpoints_cache.reset();
    }

  public:

    /**
     * Represent FEM-like element in RectangularMesh.
     */
    class Element {
        const RectangularMesh<2,Mesh1D>& mesh;
        std::size_t index0, index1; // probably this form allows to do most operation fastest in average, low indexes of element corner or just element indexes

        public:

        /**
         * Construct element using mesh and element indexes.
         * @param mesh mesh, this element is valid up to time of this mesh life
         * @param index0, index1 axis 0 and 1 indexes of element (equal to low corrner mesh indexes of element)
         */
        Element(const RectangularMesh<2,Mesh1D>& mesh, std::size_t index0, std::size_t index1): mesh(mesh), index0(index0), index1(index1) {}

        /**
         * Construct element using mesh and element index.
         * @param mesh mesh, this element is valid up to time of this mesh life
         * @param elementIndex index of element
         */
        Element(const RectangularMesh<2,Mesh1D>& mesh, std::size_t elementIndex): mesh(mesh) {
            std::size_t v = mesh.getElementMeshLowIndex(elementIndex);
            index0 = mesh.index0(v);
            index1 = mesh.index1(v);
        }

        typedef typename Mesh1D::PointType PointType;

        inline std::size_t getLowerIndex0() const { return index0; }

        inline std::size_t getLowerIndex1() const { return index1; }

        inline PointType getLower0() const { return mesh.axis0[index0]; }

        inline PointType getLower1() const { return mesh.axis1[index1]; }

        inline std::size_t getUpperIndex0() const { return index0+1; }

        inline std::size_t getUpperIndex1() const { return index1+1; }

        inline PointType getUpper0() const { return mesh.axis0[getUpperIndex0()]; }

        inline PointType getUpper1() const { return mesh.axis1[getUpperIndex1()]; }

        inline PointType getSize0() const { return getUpper0() - getLower0(); }

        inline PointType getSize1() const { return getUpper1() - getLower1(); }

        inline Vec<2, PointType> getSize() const { return getUpUp() - getLoLo(); }

        inline Vec<2, PointType> getMidpoint() const { return mesh.getElementMidpoint(index0, index1); }

        /// @return this element index
        inline std::size_t getIndex() const { return mesh.getElementIndexFromLowIndex(getLoLoIndex()); }

        inline Box2D toBox() const { return mesh.getElementBox(index0, index1); }

        inline std::size_t getLoLoIndex() const { return mesh.index(getLowerIndex0(), getLowerIndex1()); }

        inline std::size_t getLoUpIndex() const { return mesh.index(getLowerIndex0(), getUpperIndex1()); }

        inline std::size_t getUpLoIndex() const { return mesh.index(getUpperIndex0(), getLowerIndex1()); }

        inline std::size_t getUpUpIndex() const { return mesh.index(getUpperIndex0(), getUpperIndex1()); }

        inline Vec<2, PointType> getLoLo() const { return mesh(getLowerIndex0(), getLowerIndex1()); }

        inline Vec<2, PointType> getLoUp() const { return mesh(getLowerIndex0(), getUpperIndex1()); }

        inline Vec<2, PointType> getUpLo() const { return mesh(getUpperIndex0(), getLowerIndex1()); }

        inline Vec<2, PointType> getUpUp() const { return mesh(getUpperIndex0(), getUpperIndex1()); }

    };

    /**
     * Wrapper to RectangularMesh which allow to access to FEM-like elements.
     *
     * It works like read-only, random access container of @ref Element objects.
     */
    struct Elements {

        typedef IndexedIterator<const Elements, Element> const_iterator;
        typedef const_iterator iterator;

        const RectangularMesh<2,Mesh1D>* mesh;

        Elements(const RectangularMesh<2,Mesh1D>* mesh): mesh(mesh) {}

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
    typedef ::plask::Boundary<RectangularMesh<2,Mesh1D>> Boundary;

    /// First coordinate of points in this mesh.
    Mesh1D axis0;

    /// Second coordinate of points in this mesh.
    Mesh1D axis1;

    /// Accessor to FEM-like elements.
    const Elements elements;

    /**
     * Iteration orders:
     * - normal iteration order (NORMAL_ORDER) is:
     * (axis0[0], axis1[0]), (axis0[1], axis1[0]), ..., (axis0[axis0.size-1], axis1[0]), (axis0[0], axis1[1]), ..., (axis0[axis0.size()-1], axis1[axis1.size()-1])
     * - transposed iteration order (TRANSPOSED_ORDER) is:
     * (axis0[0], axis1[0]), (axis0[0], axis1[1]), ..., (axis0[0], y[axis1.size-1]), (axis0[1], axis1[0]), ..., (axis0[axis0.size()-1], axis1[axis1.size()-1])
     * @see setIterationOrder, getIterationOrder, setOptimalIterationOrder
     */
    enum IterationOrder { NORMAL_ORDER, TRANSPOSED_ORDER };

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
        setIterationOrder(axis0.size() > axis1.size() ? TRANSPOSED_ORDER : NORMAL_ORDER);
    }

    /// Construct an empty mesh
    RectangularMesh(IterationOrder iterationOrder = NORMAL_ORDER): elements(this) {
        axis0.owner = this; axis1.owner = this;
        setIterationOrder(iterationOrder);
    }

    /// Copy constructor
    RectangularMesh(const RectangularMesh& src): axis0(src.axis0), axis1(src.axis1), elements(this) {
        axis0.owner = this; axis1.owner = this;
        setIterationOrder(src.getIterationOrder());
    }

    /// Move constructor
    RectangularMesh(RectangularMesh&& src): axis0(std::move(src.axis0)), axis1(std::move(src.axis1)), elements(this) {
        axis0.owner = this; axis1.owner = this;
        setIterationOrder(src.getIterationOrder());
    }

    /**
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     * @param iterationOrder iteration order
     */
    RectangularMesh(Mesh1D mesh0, Mesh1D mesh1, IterationOrder iterationOrder = NORMAL_ORDER):
        axis0(std::move(mesh0)), axis1(std::move(mesh1)), elements(this) {
        axis0.owner = this; axis1.owner = this;
        setIterationOrder(iterationOrder);
    }

    /*
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate, or constructor argument for the first coordinate mesh
     * @param mesh1 mesh for the second coordinate, or constructor argument for the second coordinate mesh
     * @param iterationOrder iteration order
     */
    /*template <typename Mesh0CtorArg, typename Mesh1CtorArg>
    RectangularMesh(Mesh0CtorArg&& mesh0, Mesh1CtorArg&& mesh1, IterationOrder iterationOrder = NORMAL_ORDER):
        axis0(std::forward<Mesh0CtorArg>(mesh0)), axis1(std::forward<Mesh1CtorArg>(mesh1)) elements(this) {
        axis0.owner = this; axis1.owner = this;
        setIterationOrder(iterationOrder); }*/

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    Mesh1D& tran() { return axis0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const Mesh1D& tran() const { return axis0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    Mesh1D& vert() { return axis1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const Mesh1D& vert() const { return axis1; }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    Mesh1D& ee_x() { return axis0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const Mesh1D& ee_x() const { return axis0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    Mesh1D& ee_y() { return axis1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const Mesh1D& ee_y() const { return axis1; }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    Mesh1D& rad_r() { return axis0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return axis0
     */
    const Mesh1D& rad_r() const { return axis0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    Mesh1D& rad_z() { return axis1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const Mesh1D& rad_z() const { return axis1; }

    /**
     * Get numbered axis
     * @param n number of axis
     * @return n-th axis (cn)
     */
    Mesh1D& axis(size_t n) {
        if (n == 0) return axis0;
        else if (n != 1) throw Exception("Bad axis number");
        return axis1;
    }

    /**
     * Get numbered axis
     * @param n number of axis
     * @return n-th axis (cn)
     */
    const Mesh1D& axis(size_t n) const {
        if (n == 0) return axis0;
        else if (n != 1) throw Exception("Bad axis number");
        return axis1;
    }

    /// \return major (changing slowest) axis
    inline const Mesh1D& majorAxis() const {
        return *major_axis;
    }

    /// \return major (changing slowest) axis
    inline Mesh1D& majorAxis() {
        return *major_axis;
    }

    /// \return minor (changing fastest) axis
    inline const Mesh1D& minorAxis() const {
        return *minor_axis;
    }

    /// \return minor (changing fastest) axis
    inline Mesh1D& minorAxis() {
        return *minor_axis;
    }

    /**
      * Compare meshes
      * @param to_compare mesh to compare
      * @return @c true only if this mesh and @p to_compare represents the same set of points regardless of iteration order
      */
    bool operator==(const RectangularMesh<2,Mesh1D>& to_compare) {
        return axis0 == to_compare.axis0 && axis1 == to_compare.axis1;
    }

    /**
     * Get number of points in mesh.
     * @return number of points in mesh
     */
    std::size_t size() const { return axis0.size() * axis1.size(); }

    /**
     * Get maximum of sizes axis0 and axis1
     * @return maximum of sizes axis0 and axis1
     */
    std::size_t getMaxSize() const { return std::max(axis0.size(), axis1.size()); }

    /**
     * Get minimum of sizes axis0 and axis1
     * @return minimum of sizes axis0 and axis1
     */
    std::size_t getMinSize() const { return std::min(axis0.size(), axis1.size()); }

    /**
     * Write mesh to XML
     * \param object XML object to write to
     */
    virtual void writeXML(XMLElement& object) const;

    /// @return true only if there are no points in mesh
    bool empty() const { return axis0.empty() || axis1.empty(); }

    /**
     * Calculate this mesh index using indexes of axis0 and axis1.
     * @param axis0_index index of axis0, from 0 to axis0.size()-1
     * @param axis1_index index of axis1, from 0 to axis1.size()-1
     * @return this mesh index, from 0 to size()-1
     */
    inline std::size_t index(std::size_t axis0_index, std::size_t axis1_index) const {
        return index_f(this, axis0_index, axis1_index);
    }

    /**
     * Calculate index of axis0 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis0, from 0 to axis0.size()-1
     */
    inline std::size_t index0(std::size_t mesh_index) const {
        return index0_f(this, mesh_index);
    }

    /**
     * Calculate index of y using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis1, from 0 to axis1.size()-1
     */
    inline std::size_t index1(std::size_t mesh_index) const {
        return index1_f(this, mesh_index);
    }

    /**
     * Calculate index of minor axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of major axis, from 0 to majorAxis.size()-1
     */
    inline std::size_t majorIndex(std::size_t mesh_index) const {
        return mesh_index / minorAxis().size();
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
     * @return point with given @p index
     */
    inline Vec<2, double> at(std::size_t index0, std::size_t index1) const {
        return Vec<2, double>(axis0[index0], axis1[index1]);
    }

    /**
     * Get point with given mesh index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     */
    virtual Vec<2, double> at(std::size_t index) const {
        return Vec<2, double>(axis0[index0(index)], axis1[index1(index)]);
    }

    /**
     * Get point with given mesh index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     * @see IterationOrder
     */
    inline Vec<2,double> operator[](std::size_t index) const {
        return Vec<2, double>(axis0[index0(index)], axis1[index1(index)]);
    }

    /**
     * Get point with given x and y indexes.
     * @param axis0_index index of axis0, from 0 to axis0.size()-1
     * @param axis1_index index of axis1, from 0 to axis1.size()-1
     * @return point with given axis0 and axis1 indexes
     */
    inline Vec<2,double> operator()(std::size_t axis0_index, std::size_t axis1_index) const {
        return Vec<2, double>(axis0[axis0_index], axis1[axis1_index]);
    }

    /**
     * Remove all points from mesh.
     */
    void clear() {
        axis0.clear();
        axis1.clear();
    }

    /**
     * Return a mesh that enables iterating over middle points of the rectangles
     * \return new rectilinear mesh with points in the middles of original rectangles
     */
    shared_ptr<RectangularMesh> getMidpointsMesh();

    /**
     * Calculate (using linear interpolation) value of data in point using data in points described by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(const RandomAccessContainer& data, const Vec<2, double>& point) const -> typename std::remove_reference<decltype(data[0])>::type {
        return interpolateLinear2D(
            [&] (std::size_t i0, std::size_t i1) { return data[this->index(i0, i1)]; },
            point.c0, point.c1, axis0, axis1, axis0.findIndex(point.c0), axis1.findIndex(point.c1)
        );
    }

    /**
     * Get number of elements (for FEM method) in the first direction.
     * @return number of elements in this mesh in the first direction (axis0 direction).
     */
    std::size_t getElementsCount0() const {
        return (axis0.size() != 0)? axis0.size() : 0;
    }

    /**
     * Get number of elements (for FEM method) in the second direction.
     * @return number of elements in this mesh in the second direction (axis1 direction).
     */
    std::size_t getElementsCount1() const {
        return (axis1.size() != 0)? axis1.size() : 0;


    }

    /**
     * Get number of elements (for FEM method).
     * @return number of elements in this mesh
     */
    std::size_t getElementsCount() const {
        return (axis0.size() != 0 && axis1.size() != 0)?
            (axis0.size()-1) * (axis1.size()-1) : 0;
    }

    /**
     * Conver mesh index of bottom left element corner to this element index.
     * @param mesh_index_of_el_bottom_left mesh index
     * @return index of element, from 0 to getElementsCount()-1
     */
    std::size_t getElementIndexFromLowIndex(std::size_t mesh_index_of_el_bottom_left) const {
        return mesh_index_of_el_bottom_left - mesh_index_of_el_bottom_left / minor_axis->size();
    }

    /**
     * Conver element index to mesh index of bottom left element corner.
     * @param element_index index of element, from 0 to getElementsCount()-1
     * @return mesh index
     */
    std::size_t getElementMeshLowIndex(std::size_t element_index) const {
        return element_index + (element_index / (minor_axis->size()-1));
    }

    /**
     * Convert element index to mesh indexes of bottom left element corner.
     * @param element_index index of element, from 0 to getElementsCount()-1
     * @return axis 0 and axis 1 indexes of mesh,
     * you can easy calculate rest indexes of element corner adding 1 to returned coordinates
     */
    Vec<2, std::size_t> getElementMeshLowIndexes(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return Vec<2, std::size_t>(index0(bl_index), index1(bl_index));
    }

    /**
     * Get area of given element.
     * @param index0, index1 axis 0 and axis 1 indexes of element
     * @return area of elements with given index
     */
    double getElementArea(std::size_t index0, std::size_t index1) const {
        return (axis0[index0+1] - axis0[index0]) * (axis1[index1+1] - axis1[index1]);
    }

    /**
     * Get area of given element.
     * @param element_index index of element
     * @return area of elements with given index
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
    double getElementMidpoint0(std::size_t index0) const { return 0.5 * (axis0[index0] + axis0[index0+1]); }

    /**
     * Get second coordinate of point in center of Elements.
     * @param index1 index of Elements (axis1 index)
     * @return second coordinate of point point in center of Elements with given index
     */
    double getElementMidpoint1(std::size_t index1) const { return 0.5 * (axis1[index1] + axis1[index1+1]); }

    /**
     * Get point in center of Elements.
     * @param index0, index1 index of Elements
     * @return point in center of element with given index
     */
    Vec<2, double> getElementMidpoint(std::size_t index0, std::size_t index1) const {
        return vec(getElementMidpoint0(index0), getElementMidpoint1(index1));
    }

    /**
     * Get point in center of Elements.
     * @param element_index index of Elements
     * @return point in center of element with given index
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
        return Box2D(axis0[index0], axis1[index1], axis0[index0+1], axis1[index1+1]);
    }

    /**
     * Get element as rectangle.
     * @param element_index index of element
     * @return box of elements with given index
     */
    Box2D getElementBox(std::size_t element_index) const {
        std::size_t bl_index = getElementMeshLowIndex(element_index);
        return getElementBox(index0(bl_index), index1(bl_index));
    }

  private:

    // Common code for: left, right, bottom, top boundries:
    struct BoundaryIteratorImpl: public BoundaryLogicImpl::IteratorImpl {

        const RectangularMesh &mesh;

        std::size_t line;

        std::size_t index;

        BoundaryIteratorImpl(const RectangularMesh& mesh, std::size_t line, std::size_t index): mesh(mesh), line(line), index(index) {}

        virtual void increment() { ++index; }

        virtual bool equal(const typename BoundaryLogicImpl::IteratorImpl& other) const {
            return index == static_cast<const BoundaryIteratorImpl&>(other).index;
        }

    };

    // iterator over vertical line (from bottom to top). for left and right boundaries
    struct VerticalIteratorImpl: public BoundaryIteratorImpl {

        VerticalIteratorImpl(const RectangularMesh& mesh, std::size_t line, std::size_t index): BoundaryIteratorImpl(mesh, line, index) {}

        virtual std::size_t dereference() const { return this->mesh.index(this->line, this->index); }

        virtual typename BoundaryLogicImpl::IteratorImpl* clone() const {
            return new VerticalIteratorImpl(*this);
        }
    };

    // iterator over horizonstal line (from left to right), for bottom and top boundaries
    struct HorizontalIteratorImpl: public BoundaryIteratorImpl {

        HorizontalIteratorImpl(const RectangularMesh& mesh, std::size_t line, std::size_t index): BoundaryIteratorImpl(mesh, line, index) {}

        virtual std::size_t dereference() const { return this->mesh.index(this->index, this->line); }

        virtual typename BoundaryLogicImpl::IteratorImpl* clone() const {
            return new HorizontalIteratorImpl(*this);
        }
    };

    struct VerticalBoundary: public BoundaryWithMeshLogicImpl<RectangularMesh<2,Mesh1D>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t line;

        VerticalBoundary(const RectangularMesh<2,Mesh1D>& mesh, std::size_t line_axis0): BoundaryWithMeshLogicImpl<RectangularMesh<2,Mesh1D>>(mesh), line(line_axis0) {}

        //virtual LeftBoundary* clone() const { return new LeftBoundary(); }

        bool includes(std::size_t mesh_index) const {
            return this->mesh.index0(mesh_index) == line;
        }

        Iterator begin() const {
            return Iterator(new VerticalIteratorImpl(this->mesh, line, 0));
        }

        Iterator end() const {
            return Iterator(new VerticalIteratorImpl(this->mesh, line, this->mesh.axis1.size()));
        }

        std::size_t size() const {
            return this->mesh.axis1.size();
        }
    };


    struct VerticalBoundaryInRange: public BoundaryWithMeshLogicImpl<RectangularMesh<2,Mesh1D>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t line, beginInLineIndex, endInLineIndex;

        VerticalBoundaryInRange(const RectangularMesh<2,Mesh1D>& mesh, std::size_t line_axis0, std::size_t beginInLineIndex, std::size_t endInLineIndex)
            : BoundaryWithMeshLogicImpl<RectangularMesh<2,Mesh1D>>(mesh), line(line_axis0), beginInLineIndex(beginInLineIndex), endInLineIndex(endInLineIndex) {}

        //virtual LeftBoundary* clone() const { return new LeftBoundary(); }

        bool includes(std::size_t mesh_index) const {
            return this->mesh.index0(mesh_index) == line && in_range(this->mesh.index1(mesh_index), beginInLineIndex, endInLineIndex);
        }

        Iterator begin() const {
            return Iterator(new VerticalIteratorImpl(this->mesh, line, beginInLineIndex));
        }

        Iterator end() const {
            return Iterator(new VerticalIteratorImpl(this->mesh, line, endInLineIndex));
        }

        std::size_t size() const {
            return endInLineIndex - beginInLineIndex;
        }
    };

    struct HorizontalBoundary: public BoundaryWithMeshLogicImpl<RectangularMesh<2,Mesh1D>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t line;

        HorizontalBoundary(const RectangularMesh<2,Mesh1D>& mesh, std::size_t line_axis1): BoundaryWithMeshLogicImpl<RectangularMesh<2,Mesh1D>>(mesh), line(line_axis1) {}

        //virtual TopBoundary* clone() const { return new TopBoundary(); }

        bool includes(std::size_t mesh_index) const {
            return this->mesh.index1(mesh_index) == line;
        }

        Iterator begin() const {
            return Iterator(new HorizontalIteratorImpl(this->mesh, line, 0));
        }

        Iterator end() const {
            return Iterator(new HorizontalIteratorImpl(this->mesh, line, this->mesh.axis0.size()));
        }

        std::size_t size() const {
            return this->mesh.axis0.size();
        }
    };

    struct HorizontalBoundaryInRange: public BoundaryWithMeshLogicImpl<RectangularMesh<2,Mesh1D>> {

        typedef typename BoundaryLogicImpl::Iterator Iterator;

        std::size_t line, beginInLineIndex, endInLineIndex;

        HorizontalBoundaryInRange(const RectangularMesh<2,Mesh1D>& mesh, std::size_t line_axis1, std::size_t beginInLineIndex, std::size_t endInLineIndex)
            : BoundaryWithMeshLogicImpl<RectangularMesh<2,Mesh1D>>(mesh), line(line_axis1), beginInLineIndex(beginInLineIndex), endInLineIndex(endInLineIndex) {}
        //virtual TopBoundary* clone() const { return new TopBoundary(); }

        bool includes(std::size_t mesh_index) const {
            return this->mesh.index1(mesh_index) == line && in_range(this->mesh.index0(mesh_index), beginInLineIndex, endInLineIndex);
        }

        Iterator begin() const {
            return Iterator(new HorizontalIteratorImpl(this->mesh, line, beginInLineIndex));
        }

        Iterator end() const {
            return Iterator(new HorizontalIteratorImpl(this->mesh, line, endInLineIndex));
        }

        std::size_t size() const {
            return endInLineIndex - beginInLineIndex;
        }
    };

    //TODO
    /*struct HorizontalLineBoundary: public BoundaryLogicImpl<RectangularMesh<2,Mesh1D>> {

        double height;

        bool includes(const RectangularMesh &mesh, std::size_t mesh_index) const {
            return mesh.index1(mesh_index) == mesh.axis1.findNearestIndex(height);
        }
    };*/

    /**
     * Helper used by getLeftOfBoundary, etc.
     * @param[out] line index of point in @p axis which lies in bound [@p box_lower, @p box_upper] and is nearest to @p box_lower,
     *  undefined if @c false was returned
     * @param[in] axis axis, 1D mesh
     * @param[in] box_lower, box_upper position of lower and upper box edges
     * @return @c true only if @p axis has point which lies in bounds [@p box_lower, @p box_upper]
     */
    static bool getLineLo(std::size_t& line, const Mesh1D& axis, double box_lower, double box_upper) {
        assert(box_lower <= box_upper);
        line = axis.findIndex(box_lower);
        return line != axis.size() && axis[line] <= box_upper;
    }

    /**
     * Helper used by getRightOfBoundary, etc.
     * @param[out] line index of point in @p axis which lies in bound [@p box_lower, @p box_upper] and is nearest to @p box_upper,
     *  undefined if @c false was returned
     * @param[in] axis axis, 1D mesh
     * @param[in] box_lower, box_upper position of lower and upper box edges
     * @return @c true only if @p axis has point which lies in bounds [@p box_lower, @p box_upper]
     */
    static bool getLineHi(std::size_t& line, const Mesh1D& axis, double box_lower, double box_upper) {
        assert(box_lower <= box_upper);
        line = axis.findIndex(box_upper);
        if (line != axis.size() && axis[line] == box_upper) return true;
        if (line == 0) return false;
        --line;
        return axis[line] >= box_lower;
    }

    /**
     * Helper used by getLeftOfBoundary, etc.
     * @param[out] begInd, endInd range [begInd, endInd) of indices in @p axis which show points which lie in bounds [@p box_lower, @p box_upper],
     *      undefined if @c false was returned
     * @param[in] axis axis, 1D mesh
     * @param[in] box_lower, box_upper position of lower and upper box edges
     * @return @c true only if some of @p axis points lies in bounds [@p box_lower, @p box_upper]
     */
    static bool getIndexesInBounds(std::size_t& begInd, std::size_t& endInd, const Mesh1D& axis, double box_lower, double box_upper) {
        assert(box_lower <= box_upper);
        begInd = axis.findIndex(box_lower);
        endInd = axis.findIndex(box_upper);
        if (endInd != axis.size() && axis[endInd] == box_upper) ++endInd;    //endInd is exluded
        return begInd != endInd;
    }

    /**
     * Decrease @p index if @p real_pos is much closer to axis[index-1] than axis[index].
     * @param[in] axis axis of mesh
     * @param[in, out] index index such that axis[index] <= real_pos < axis[index+1], can be unchanged or decrement by one by this method
     * @param[in] real_pos position
     */
    static void tryMakeLower(const Mesh1D& axis, std::size_t& index, double real_pos) {
        if (index == 0) return;
        if ((real_pos - axis[index-1]) * 100.0 < (axis[index] - axis[index-1])) --index;
    }

    /**
     * Increase @p index if @p real_pos is much closer to axis[index] than axis[index-1].
     * @param[in] axis axis of mesh
     * @param[in, out] index index such that axis[index-1] <= real_pos < axis[index], can be unchanged or increment by one by this method
     * @param[in] real_pos position
     */
    static void tryMakeHigher(const Mesh1D& axis, std::size_t& index, double real_pos) {
        if (index == axis.size() || index == 0) return; //index == 0 means empty mesh
        if ((axis[index] - real_pos) * 100.0 < (axis[index] - axis[index-1])) ++index;
    }

    /**
     * Helper.
     * @param[out] begInd, endInd range [begInd, endInd) of indices in @p axis which show points which lie or almost lie in bounds [@p box_lower, @p box_upper],
     *      undefined if @c false was returned
     * @param[in] axis axis, 1D mesh
     * @param[in] box_lower, box_upper position of lower and upper box edges
     * @return @c true only if some of @p axis points (almost) lies in bounds [@p box_lower, @p box_upper]
     */
    static bool getIndexesInBoundsExt(std::size_t& begInd, std::size_t& endInd, const Mesh1D& axis, double box_lower, double box_upper) {
        getIndexesInBounds(begInd, endInd, axis, box_lower, box_upper);
        tryMakeLower(axis, begInd, box_lower);
        tryMakeHigher(axis, endInd, box_upper);
        return begInd != endInd;
    }

    /**
     * Parse boundary from XML tag in format:
     * \<place dir="i.e. left" [of="object name" [path="path name"] [geometry="name of geometry which is used by solver"]]/>
     * @param boundary_desc XML reader which point to tag to read (after read it will be moved to end of this tag)
     * @param enviroment parser enviroment
     * @param getXBoundary function which creates simple boundary, with edge of mesh, i.e. getLeftBoundary
     * @param getXOfBoundary function which creates simple boundary, with edge of object, i.e. getLeftOfBoundary
     * @return boundary which was read
     */
    static Boundary parseBoundaryFromXML(plask::XMLReader& boundary_desc, plask::BoundaryParserEnviroment enviroment, Boundary (*getXBoundary)(),
                                         Boundary (*getXOfBoundary)(shared_ptr<const GeometryD<2>>, shared_ptr<const GeometryObject>, const PathHints*)) {
        boost::optional<std::string> of = boundary_desc.getAttribute("of");
        if (!of) {
            boundary_desc.requireTagEnd();
            return getXBoundary();
        } else {
            plask::shared_ptr< const plask::GeometryD<2> > geometry;
            boost::optional<std::string> geom_name = boundary_desc.getAttribute("geometry");
            if (geom_name)
                geometry = enviroment.manager.requireGeometry<const plask::GeometryD<2>>(*geom_name);
            else {
                geometry = dynamic_pointer_cast< const plask::GeometryD<2> >(enviroment.geometry);
                if (!geometry)
                    throw Exception("Cannot parse %1% of \"%2%\" object. Geometry is not known (\"geometry\" attribute can be used to provide this information).",
                                    boundary_desc.requireAttribute("dir"), *of);
            }
            boost::optional<std::string> path_name = boundary_desc.getAttribute("path");
            boundary_desc.requireTagEnd();
            return getXOfBoundary(geometry, enviroment.manager.requireGeometryObject(*of),
                                  path_name ? &enviroment.manager.requirePathHints(*path_name) : nullptr);
        }
    }

public:
    // boundaries:

    template <typename Predicate>
    static Boundary getBoundary(Predicate predicate) {
        return Boundary(new PredicateBoundaryImpl<RectangularMesh<2,Mesh1D>, Predicate>(predicate));
    }

    /**
     * Get boundary which show one vertical (from bottom to top) line in mesh.
     * @param line_nr_axis0 number of vertical line, axis 0 index of mesh
     * @return boundary which show one vertical (from bottom to top) line in mesh
     */
    static Boundary getVerticalBoundaryAtLine(std::size_t line_nr_axis0) {
        return Boundary( [line_nr_axis0](const RectangularMesh<2,Mesh1D>& mesh) {return new VerticalBoundary(mesh, line_nr_axis0);} );
    }

    /**
     * Get boundary which show range in vertical (from bottom to top) line in mesh.
     * @param line_nr_axis0 number of vertical line, axis 0 index of mesh
     * @param indexBegin, indexEnd ends of [indexBegin, indexEnd) range in line
     * @return boundary which show range in vertical (from bottom to top) line in mesh.
     */
    static Boundary getVerticalBoundaryAtLine(std::size_t line_nr_axis0, std::size_t indexBegin, std::size_t indexEnd) {
        return Boundary( [=](const RectangularMesh<2,Mesh1D>& mesh) {return new VerticalBoundaryInRange(mesh, line_nr_axis0, indexBegin, indexEnd);} );
    }

    /**
     * Get boundary which show one vertical (from bottom to top) line in mesh which lies nearest given coordinate.
     * @param axis0_coord axis 0 coordinate
     * @return boundary which show one vertical (from bottom to top) line in mesh
     */
    static Boundary getVerticalBoundaryNear(double axis0_coord) {
        return Boundary( [axis0_coord](const RectangularMesh<2,Mesh1D>& mesh) {return new VerticalBoundary(mesh, mesh.axis0.findNearestIndex(axis0_coord));} );
    }

    /**
     * Get boundary which show one vertical (from bottom to top) segment in mesh which lies nearest given coordinate and has ends in given range
     * @param axis0_coord axis 0 coordinate
     * @param from, to ends of line segment, [from, to] range of axis 1 coordinates
     * @return boundary which show one vertical (from bottom to top) segment in mesh
     */
    static Boundary getVerticalBoundaryNear(double axis0_coord, double from, double to) {
        return Boundary( [axis0_coord, from, to](const RectangularMesh<2,Mesh1D>& mesh) -> BoundaryLogicImpl* {
            std::size_t begInd, endInd;
            if (!RectangularMesh<2,Mesh1D>::getIndexesInBoundsExt(begInd, endInd, mesh.axis1, from, to))
                return new EmptyBoundaryImpl();
            return new VerticalBoundaryInRange(mesh, mesh.axis0.findNearestIndex(axis0_coord), begInd, endInd);
        } );
    }

    /**
     * Get boundary which show one vertical, left (from bottom to top) line in mesh.
     * @return boundary which show left line in mesh
     */
    static Boundary getLeftBoundary() {
        return Boundary( [](const RectangularMesh<2,Mesh1D>& mesh) {return new VerticalBoundary(mesh, 0);} );
    }

    /**
     * Get boundary which show one vertical, right (from bottom to top) line in mesh.
     * @return boundary which show right line in mesh
     */
    static Boundary getRightBoundary() {
        return Boundary( [](const RectangularMesh<2,Mesh1D>& mesh) {return new VerticalBoundary(mesh, mesh.axis0.size()-1);} );
    }

    /**
     * Get boundary which lies on left edge of the @p box (at mesh line nearest left edge and inside the box).
     * @param box box in which boundary should lie
     * @return boundary which lies on left edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    static Boundary getLeftOfBoundary(const Box2D& box) {
        return Boundary( [=](const RectangularMesh<2,Mesh1D>& mesh) -> BoundaryLogicImpl* {
            std::size_t line, begInd, endInd;
            if (RectangularMesh<2,Mesh1D>::getLineLo(line, mesh.axis0, box.lower.c0, box.upper.c0) &&
                RectangularMesh<2,Mesh1D>::getIndexesInBounds(begInd, endInd, mesh.axis1, box.lower.c1, box.upper.c1))
                return new VerticalBoundaryInRange(mesh, line, begInd, endInd);
            else
                return new EmptyBoundaryImpl();
        } );
    }

    /**
     * Get boundary which lies on right edge of the @p box (at mesh line nearest right edge and inside the box).
     * @param box box in which boundary should lie
     * @return boundary which lies on right edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    static Boundary getRightOfBoundary(const Box2D& box) {
        return Boundary( [=](const RectangularMesh<2,Mesh1D>& mesh) -> BoundaryLogicImpl* {
            std::size_t line, begInd, endInd;
            if (RectangularMesh<2,Mesh1D>::getLineHi(line, mesh.axis0, box.lower.c0, box.upper.c0) &&
                RectangularMesh<2,Mesh1D>::getIndexesInBounds(begInd, endInd, mesh.axis1, box.lower.c1, box.upper.c1))
                return new VerticalBoundaryInRange(mesh, line, begInd, endInd);
            else
                return new EmptyBoundaryImpl();
        } );
    }

    /**
     * Get boundary which lies on bottom edge of the @p box (at mesh line nearest bottom edge and inside the box).
     * @param box box in which boundary should lie
     * @return boundary which lies on bottom edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    static Boundary getBottomOfBoundary(const Box2D& box) {
        return Boundary( [=](const RectangularMesh<2,Mesh1D>& mesh) -> BoundaryLogicImpl* {
            std::size_t line, begInd, endInd;
            if (RectangularMesh<2,Mesh1D>::getLineLo(line, mesh.axis1, box.lower.c1, box.upper.c1) &&
                RectangularMesh<2,Mesh1D>::getIndexesInBounds(begInd, endInd, mesh.axis0, box.lower.c0, box.upper.c0))
                return new HorizontalBoundaryInRange(mesh, line, begInd, endInd);
            else
                return new EmptyBoundaryImpl();
        } );
    }

    /**
     * Get boundary which lies on top edge of the @p box (at mesh line nearest top edge and inside the box).
     * @param box box in which boundary should lie
     * @return boundary which lies on top edge of the @p box or empty boundary if there are no mesh indexes which lies inside the @p box
     */
    static Boundary getTopOfBoundary(const Box2D& box) {
        return Boundary( [=](const RectangularMesh<2,Mesh1D>& mesh) -> BoundaryLogicImpl* {
            std::size_t line, begInd, endInd;
            if (RectangularMesh<2,Mesh1D>::getLineHi(line, mesh.axis1, box.lower.c1, box.upper.c1) &&
                RectangularMesh<2,Mesh1D>::getIndexesInBounds(begInd, endInd, mesh.axis0, box.lower.c0, box.upper.c0))
                return new HorizontalBoundaryInRange(mesh, line, begInd, endInd);
            else
                return new EmptyBoundaryImpl();
        } );
    }

    /**
     * Get boundary which lies on chosen edge of boxes.
     * @param getBoxes functor which returns 0 or more boxes (vector of boxes)
     * @param getBoundaryForBox functor which returns boundary for box given as parameter, it chooses edge of box (for example this may call getLeftOfBoundary, etc.)
     * @return boundary which represents sum of boundaries returned by getBoundaryForBox for all boxes returned by getBoxes
     */
    template <typename GetBoxes, typename GetBoundaryForBox>
    static Boundary getBoundaryForBoxes(GetBoxes getBoxes, GetBoundaryForBox getBoundaryForBox) {
        return Boundary( [=](const RectangularMesh<2,Mesh1D>& mesh) -> BoundaryWithMesh {
            std::vector<RectangularMesh<2,Mesh1D>::Boundary> boundaries;
            std::vector<typename RectangularMesh<2,Mesh1D>::Boundary::WithMesh> boundaries_with_meshes;
            auto boxes = getBoxes(); // probably std::vector<Box2D>
            for (auto& box: boxes) {
                RectangularMesh<2,Mesh1D>::Boundary boundary = getBoundaryForBox(box);
                typename RectangularMesh<2,Mesh1D>::Boundary::WithMesh boundary_with_mesh = boundary(mesh);
                if (!boundary_with_mesh.empty()) {
                    boundaries.push_back(std::move(boundary));
                    boundaries_with_meshes.push_back(std::move(boundary_with_mesh));
                }
            }
            if (boundaries.empty()) return new EmptyBoundaryImpl();
            if (boundaries.size() == 1) return boundaries_with_meshes[0];
            return new SumBoundaryImpl<RectangularMesh<2,Mesh1D>>(std::move(boundaries_with_meshes));
        } );
    }

    /**
     * Get boundary which lies on left edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geomoetry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geomoetry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of left edges of @p object's bounding-boxes
     */
    static Boundary getLeftOfBoundary(shared_ptr<const GeometryD<2>> geometry, shared_ptr<const GeometryObject> object, const PathHints& path) {
        return getBoundaryForBoxes(
            [=] { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box2D& box) { return RectangularMesh<2,Mesh1D>::getLeftOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on left edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geomoetry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geomoetry
     * @param path (optional) hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of left edges of @p object's bounding-boxes
     */
    static Boundary getLeftOfBoundary(shared_ptr<const GeometryD<2>> geometry, shared_ptr<const GeometryObject> object, const PathHints* path = nullptr) {
        if (path) return getLeftOfBoundary(geometry, object, *path);
        return getBoundaryForBoxes(
            [=] { return geometry->getObjectBoundingBoxes(object); },
            [](const Box2D& box) { return RectangularMesh<2,Mesh1D>::getLeftOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on right edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geomoetry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geomoetry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of right edges of @p object's bounding-boxes
     */
    static Boundary getRightOfBoundary(shared_ptr<const GeometryD<2>> geometry, shared_ptr<const GeometryObject> object, const PathHints& path) {
        return getBoundaryForBoxes(
            [=] { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box2D& box) { return RectangularMesh<2,Mesh1D>::getRightOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on right edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geomoetry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geomoetry
     * @param path (optional) hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of right edges of @p object's bounding-boxes
     */
    static Boundary getRightOfBoundary(shared_ptr<const GeometryD<2>> geometry, shared_ptr<const GeometryObject> object, const PathHints* path = nullptr) {
        if (path) return getRightOfBoundary(geometry, object, *path);
        return getBoundaryForBoxes(
            [=] { return geometry->getObjectBoundingBoxes(object); },
            [](const Box2D& box) { return RectangularMesh<2,Mesh1D>::getRightOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on bottom edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geomoetry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geomoetry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of bottom edges of @p object's bounding-boxes
     */
    static Boundary getBottomOfBoundary(shared_ptr<const GeometryD<2>> geometry, shared_ptr<const GeometryObject> object, const PathHints& path) {
        return getBoundaryForBoxes(
            [=] { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box2D& box) { return RectangularMesh<2,Mesh1D>::getBottomOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on bottom edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geomoetry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geomoetry
     * @param path (optional) hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of bottom edges of @p object's bounding-boxes
     */
    static Boundary getBottomOfBoundary(shared_ptr<const GeometryD<2>> geometry, shared_ptr<const GeometryObject> object, const PathHints* path = nullptr) {
        if (path) return getBottomOfBoundary(geometry, object, *path);
        return getBoundaryForBoxes(
            [=] { return geometry->getObjectBoundingBoxes(object); },
            [](const Box2D& box) { return RectangularMesh<2,Mesh1D>::getBottomOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on top edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geomoetry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geomoetry
     * @param path hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of top edges of @p object's bounding-boxes
     */
    static Boundary getTopOfBoundary(shared_ptr<const GeometryD<2>> geometry, shared_ptr<const GeometryObject> object, const PathHints& path) {
        return getBoundaryForBoxes(
            [=] { return geometry->getObjectBoundingBoxes(object, path); },
            [](const Box2D& box) { return RectangularMesh<2,Mesh1D>::getTopOfBoundary(box); }
        );
    }

    /**
     * Get boundary which lies on top edge of bounding-boxes of @p object (in @p geometry coordinates).
     * @param geometry geomoetry, needs to define coordinates, geometry which is used with using mesh
     * @param object object included in @p geomoetry
     * @param path (optional) hints specifying particular instances of the geometry object
     * @return boundary which represents sum of boundaries of top edges of @p object's bounding-boxes
     */
    static Boundary getTopOfBoundary(shared_ptr<const GeometryD<2>> geometry, shared_ptr<const GeometryObject> object, const PathHints* path = nullptr) {
        if (path) return getTopOfBoundary(geometry, object, *path);
        return getBoundaryForBoxes(
            [=] { return geometry->getObjectBoundingBoxes(object); },
            [](const Box2D& box) { return RectangularMesh<2,Mesh1D>::getTopOfBoundary(box); }
        );
    }

    /**
     * Get boundary which show one horizontal (from left to right) line in mesh.
     * @param line_nr_axis1 number of horizontal line, index of axis1 mesh
     * @return boundary which show one horizontal (from left to right) line in mesh
     */
    static Boundary getHorizontalBoundaryAtLine(std::size_t line_nr_axis1) {
        return Boundary( [line_nr_axis1](const RectangularMesh<2,Mesh1D>& mesh) {return new HorizontalBoundary(mesh, line_nr_axis1);} );
    }

    /**
     * Get boundary which shows range in horizontal (from left to right) line in mesh.
     * @param line_nr_axis1 number of horizontal line, index of axis1 mesh
     * @param indexBegin, indexEnd ends of [indexBegin, indexEnd) range in line
     * @return boundary which show range in horizontal (from left to right) line in mesh.
     */
    static Boundary getHorizontalBoundaryAtLine(std::size_t line_nr_axis1, std::size_t indexBegin, std::size_t indexEnd) {
        return Boundary( [=](const RectangularMesh<2,Mesh1D>& mesh) {return new HorizontalBoundaryInRange(mesh, line_nr_axis1, indexBegin, indexEnd);} );
    }

    /**
     * Get boundary which shows one horizontal (from left to right) line in mesh which lies nearest given coordinate.
     * @param axis1_coord axis 1 coordinate
     * @return boundary which show one horizontal (from left to right) line in mesh
     */
    static Boundary getHorizontalBoundaryNear(double axis1_coord) {
        return Boundary( [axis1_coord](const RectangularMesh<2,Mesh1D>& mesh) {return new HorizontalBoundary(mesh, mesh.axis1.findNearestIndex(axis1_coord));} );
    }

    /**
     * Get boundary which show one horizontal (from left to right) segment in mesh which lies nearest given coordinate and has ends in given range.
     * @param axis1_coord axis 1 coordinate
     * @param from, to ends of line segment, [from, to] range of axis 0 coordinates
     * @return boundary which show one horizontal (from left to right) line in mesh
     */
    static Boundary getHorizontalBoundaryNear(double axis1_coord, double from, double to) {
        return Boundary( [axis1_coord, from, to](const RectangularMesh<2,Mesh1D>& mesh) -> BoundaryLogicImpl* {
            std::size_t begInd, endInd;
            if (!RectangularMesh<2,Mesh1D>::getIndexesInBoundsExt(begInd, endInd, mesh.axis0, from, to))
                return new EmptyBoundaryImpl();
            return new HorizontalBoundaryInRange(mesh, mesh.axis1.findNearestIndex(axis1_coord), begInd, endInd);
        } );
    }

    /**
     * Get boundary which shows one horizontal, top (from left to right) line in mesh.
     * @return boundary which show top line in mesh
     */
    static Boundary getTopBoundary() {
        return Boundary( [](const RectangularMesh<2,Mesh1D>& mesh) {return new HorizontalBoundary(mesh, mesh.axis1.size()-1);} );
    }

    /**
     * Get boundary which shows one horizontal, bottom (from left to right) line in mesh.
     * @return boundary which show bottom line in mesh
     */
    static Boundary getBottomBoundary() {
        return Boundary( [](const RectangularMesh<2,Mesh1D>& mesh) {return new HorizontalBoundary(mesh, 0);} );
    }

    static Boundary getBoundary(const std::string& boundary_desc) {
        if (boundary_desc == "bottom") return getBottomBoundary();
        if (boundary_desc == "left") return getLeftBoundary();
        if (boundary_desc == "right") return getRightBoundary();
        if (boundary_desc == "top") return getTopBoundary();
        return Boundary();
    }

    static Boundary getBoundary(plask::XMLReader& boundary_desc, plask::BoundaryParserEnviroment enviroment) {
        auto side = boundary_desc.getAttribute("side");
        auto line = boundary_desc.getAttribute("line");
        if (side && line) {
            throw XMLConflictingAttributesException(boundary_desc, "size", "line");
        } else if (side) {
            if (*side == "bottom")
                return parseBoundaryFromXML(boundary_desc, enviroment, &getBottomBoundary, &getBottomOfBoundary);
            if (*side == "left")
                return parseBoundaryFromXML(boundary_desc, enviroment, &getLeftBoundary, &getLeftOfBoundary);
            if (*side == "right")
                return parseBoundaryFromXML(boundary_desc, enviroment, &getRightBoundary, &getRightOfBoundary);
            if (*side == "top")
                return parseBoundaryFromXML(boundary_desc, enviroment, &getTopBoundary, &getTopOfBoundary);
            throw XMLBadAttrException(boundary_desc, "side", *side);
        } else if (line) {
            double at = boundary_desc.requireAttribute<double>("at"),
                   start = boundary_desc.requireAttribute<double>("start"),
                   stop = boundary_desc.requireAttribute<double>("stop");
            boundary_desc.requireTagEnd();
            if (*line == "vertical")
                return getVerticalBoundaryNear(at, start, stop);
            if (*line == "horizontal")
                return getHorizontalBoundaryNear(at, start, stop);
            throw XMLBadAttrException(boundary_desc, "line", *line);
        }
        return Boundary();
    }
};

/**
 * Do linear 2d interpolation with checking bounds variants.
 * @param data 2d data source, data(i0, i1) should return data in point (axis0[i0], axis1[i1])
 * @param point_axis0,point_axis1 requested point coordinates
 * @param axis0 first coordinates of points
 * @param axis1 second coordinates of points
 * @param index0 should be equal to axis0.findIndex(point_axis0)
 * @param index1 should be equal to axis1.findIndex(point_axis1)
 * @return value in point point_axis0, point_axis1
 * @tparam DataGetter2D functor
 */
template <typename DataGetter2D, typename Mesh1D>
auto interpolateLinear2D(DataGetter2D data, const double& point_axis0, const double& point_axis1, const Mesh1D& axis0, const Mesh1D& axis1, std::size_t index0, std::size_t index1)
  -> typename std::remove_reference<decltype(data(0, 0))>::type {
    if (index0 == 0) {
        if (index1 == 0) return data(0, 0);
        if (index1 == axis1.size()) return data(0, index1-1);
        return interpolation::linear(axis1[index1-1], data(0, index1-1), axis1[index1], data(0, index1), point_axis1);
    }

    if (index0 == axis0.size()) {
        --index0;
        if (index1 == 0) return data(index0, 0);
        if (index1 == axis1.size()) return data(index0, index1-1);
        return interpolation::linear(axis1[index1-1], data(index0, index1-1), axis1[index1], data(index0, index1), point_axis1);
    }

    if (index1 == 0)
        return interpolation::linear(axis0[index0-1], data(index0-1, 0), axis0[index0], data(index0, 0), point_axis0);

    if (index1 == axis1.size()) {
        --index1;
        return interpolation::linear(axis0[index0-1], data(index0-1, index1), axis0[index0], data(index0, index1), point_axis0);
    }

    return interpolation::bilinear(axis0[index0-1], axis0[index0],
                                   axis1[index1-1], axis1[index1],
                                   data(index0-1, index1-1),
                                   data(index0,   index1-1),
                                   data(index0,   index1  ),
                                   data(index0-1, index1  ),
                                   point_axis0, point_axis1);
}


template <typename Mesh1D, typename DataT>    // for any data type
struct InterpolationAlgorithm<RectangularMesh<2,Mesh1D>, DataT, INTERPOLATION_LINEAR> {
    static void interpolate(const RectangularMesh<2,Mesh1D>& src_mesh, const DataVector<const DataT>& src_vec, const plask::MeshD<2>& dst_mesh, DataVector<DataT>& dst_vec) {
        #pragma omp parallel for
        for (size_t i = 0; i < dst_mesh.size(); ++i)
            dst_vec[i] = src_mesh.interpolateLinear(src_vec, dst_mesh[i]);
    }
};

} // namespace plask

#endif // PLASK__RECTANGULAR2D_H
