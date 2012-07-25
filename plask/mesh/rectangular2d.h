#ifndef PLASK__RECTANGULAR2D_H
#define PLASK__RECTANGULAR2D_H

/** @file
This file includes rectilinear mesh for 2d space.
*/

#include <iterator>

#include "mesh.h"
#include "boundary.h"
#include "interpolation.h"
#include "../geometry/element.h"

namespace plask {

/**
 * Rectilinear mesh in 2D space.
 *
 * Includes two 1D rectilinear meshes:
 * - c0 (alternative names: tran(), ee_x(), r())
 * - c1 (alternative names: up(), ee_y(), z())
 * Represent all points (x, y) such that x is in c0 and y is in c1.
 */
//TODO methods which call fireChanged() when points are added, etc.
template <typename Mesh1D>
class RectangularMesh<2,Mesh1D>: public MeshD<2> {

    static_assert(std::is_floating_point< typename std::remove_reference<decltype(std::declval<Mesh1D>().operator[](0))>::type >::value,
                  "Mesh1d must have operator[](std::size_t index) which returns floating-point value");

    typedef std::size_t index_ft(const RectangularMesh<2,Mesh1D>* mesh, std::size_t c0_index, std::size_t c1_index);
    typedef std::size_t index01_ft(const RectangularMesh<2,Mesh1D>* mesh, std::size_t mesh_index);

    // Our own virtual table, changeable in run-time:
    index_ft* index_f;
    index01_ft* index0_f;
    index01_ft* index1_f;

  public:

    /// Boundary type.
    typedef ::plask::Boundary<RectangularMesh<2,Mesh1D>> Boundary;

    /// First coordinate of points in this mesh.
    Mesh1D c0;

    /// Second coordinate of points in this mesh.
    Mesh1D c1;

    /**
     * Iteration orders:
     * - normal iteration order (NORMAL_ORDER) is:
     * (c0[0], c1[0]), (c0[1], c1[0]), ..., (c0[c0.size-1], c1[0]), (c0[0], c1[1]), ..., (c0[c0.size()-1], c1[c1.size()-1])
     * - transposed iteration order (TRANSPOSED_ORDER) is:
     * (c0[0], c1[0]), (c0[0], c1[1]), ..., (c0[0], y[c1.size-1]), (c0[1], c1[0]), ..., (c0[c0.size()-1], c1[c1.size()-1])
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
        setIterationOrder(c0.size() > c1.size() ? TRANSPOSED_ORDER : NORMAL_ORDER);
    }

    /// Construct an empty mesh
    RectangularMesh(IterationOrder iterationOrder = NORMAL_ORDER) { setIterationOrder(iterationOrder); }

    /**
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     * @param iterationOrder iteration order
     */
    RectangularMesh(Mesh1D mesh0, Mesh1D mesh1, IterationOrder iterationOrder = NORMAL_ORDER) :
        c0(std::move(mesh0)), c1(std::move(mesh1)) { setIterationOrder(iterationOrder); }

    /*
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate, or constructor argument for the first coordinate mesh
     * @param mesh1 mesh for the second coordinate, or constructor argument for the second coordinate mesh
     * @param iterationOrder iteration order
     */
    /*template <typename Mesh0CtorArg, typename Mesh1CtorArg>
    RectangularMesh(Mesh0CtorArg&& mesh0, Mesh1CtorArg&& mesh1, IterationOrder iterationOrder = NORMAL_ORDER) :
        c0(std::forward<Mesh0CtorArg>(mesh0)), c1(std::forward<Mesh1CtorArg>(mesh1)) { setIterationOrder(iterationOrder); }*/

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    Mesh1D& tran() { return c0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    const Mesh1D& tran() const { return c0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    Mesh1D& up() { return c1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    const Mesh1D& up() const { return c1; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    Mesh1D& ee_x() { return c0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    const Mesh1D& ee_x() const { return c0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    Mesh1D& ee_y() { return c1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    const Mesh1D& ee_y() const { return c1; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    Mesh1D& rad_r() { return c0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    const Mesh1D& rad_r() const { return c0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    Mesh1D& rad_z() { return c1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    const Mesh1D& rad_z() const { return c1; }

    /**
     * Get numbered axis
     * \param no
     */
    Mesh1D& axis(size_t n) {
        if (n == 0) return c0;
        else if (n != 1) throw Exception("Bad axis number");
        return c1;
    }

    /**
     * Get numbered axis
     * \param no
     */
    const Mesh1D& axis(size_t n) const {
        if (n == 0) return c0;
        else if (n != 1) throw Exception("Bad axis number");
        return c1;
    }

    /// \return major (changing slowest) axis
    const Mesh1D& majorAxis() const {
        if (getIterationOrder() == NORMAL_ORDER) return c1;
        else return c0;
    }

    /// \return major (changing slowest) axis
    Mesh1D& majorAxis() {
        if (getIterationOrder() == NORMAL_ORDER) return c1;
        else return c0;
    }

    /// \return minor (changing fastes) axis
    const Mesh1D& minorAxis() const {
        if (getIterationOrder() == NORMAL_ORDER) return c0;
        else return c1;
    }

    /// \return minor (changing fastes) axis
    Mesh1D& minorAxis() {
        if (getIterationOrder() == NORMAL_ORDER) return c0;
        else return c1;
    }

    /// Type of points in this mesh.
    typedef Vec<2,double> PointType;

    /**
     * Random access iterator type which allow iterate over all points in this mesh, in order appointed by operator[].
     * This iterator type is indexed, which mean that it have (read-write) index field equal to 0 for begin() and growing up to size() for end().
     */
    typedef IndexedIterator< const RectangularMesh, PointType > const_iterator;

    /// @return iterator referring to the first point in this mesh
    const_iterator begin_fast() const { return const_iterator(this, 0); }

    /// @return iterator referring to the past-the-end point in this mesh
    const_iterator end_fast() const { return const_iterator(this, size()); }

    // implement MeshD<2> polymorphic iterators:
    virtual typename MeshD<2>::Iterator begin() const { return makeMeshIterator(begin_fast()); }
    virtual typename MeshD<2>::Iterator end() const { return makeMeshIterator(end_fast()); }

    /**
      * Compare meshes
      * @param to_compare mesh to compare
      * @return @c true only if this mesh and @p to_compare represents the same set of points regardless of iteration order
      */
    bool operator==(const RectangularMesh<2,Mesh1D>& to_compare) {
        return c0 == to_compare.c0 && c1 == to_compare.c1;
    }

    /**
     * Get number of points in mesh.
     * @return number of points in mesh
     */
    std::size_t size() const { return c0.size() * c1.size(); }

    /**
     * Write mesh to XML
     * \param element XML element to write to
     */
    virtual void writeXML(XMLElement& element) const;

    /// @return true only if there are no points in mesh
    bool empty() const { return c0.empty() || c1.empty(); }

    /**
     * Calculate this mesh index using indexes of c0 and c1.
     * @param c0_index index of c0, from 0 to c0.size()-1
     * @param c1_index index of c1, from 0 to c1.size()-1
     * @return this mesh index, from 0 to size()-1
     */
    inline std::size_t index(std::size_t c0_index, std::size_t c1_index) const {
        return index_f(this, c0_index, c1_index);
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
     * Calculate index of y using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of c1, from 0 to c1.size()-1
     */
    inline std::size_t index1(std::size_t mesh_index) const {
        return index1_f(this, mesh_index);
    }

    /**
     * Get point with given mesh index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     * @see IterationOrder
     */
    inline Vec<2,double> operator[](std::size_t index) const {
        return Vec<2, double>(c0[index0(index)], c1[index1(index)]);
    }

    /**
     * Get point with given x and y indexes.
     * @param c0_index index of c0, from 0 to c0.size()-1
     * @param c1_index index of c1, from 0 to c1.size()-1
     * @return point with given c0 and c1 indexes
     */
    inline Vec<2,double> operator()(std::size_t c0_index, std::size_t c1_index) const {
        return Vec<2, double>(c0[c0_index], c1[c1_index]);
    }

    /**
     * Remove all points from mesh.
     */
    void clear() {
        c0.clear();
        c1.clear();
    }
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
            point.c0, point.c1, c0, c1, c0.findIndex(point.c0), c1.findIndex(point.c1)
        );
    }

    /**
     * Get number of elements (for FEM method) in the first direction.
     * @return number of elements in this mesh in the first direction (c0 direction).
     */
    std::size_t getElementsCount0() const {
        return std::max(int(c0.size())-1, 0);
    }

    /**
     * Get number of elements (for FEM method) in the second direction.
     * @return number of elements in this mesh in the second direction (c1 direction).
     */
    std::size_t getElementsCount1() const {
        return std::max(int(c1.size())-1, 0);
    }

    /**
     * Get number of elements (for FEM method).
     * @return number of elements in this mesh
     */
    std::size_t getElementsCount() const {
        return std::max((int(c0.size())-1) * (int(c1.size())-1), 0);
    }

    /**
     * Get area of given element.
     * @param c0index, c1index index of element
     * @return area of element with given index
     */
    double getElementArea(std::size_t c0index, std::size_t c1index) const {
        return (c0[c0index+1] - c0[c0index])*(c1[c1index+1] - c1[c1index]);
    }

    /**
     * Get first coordinate of point in center of element.
     * @param c0index index of element (c0 index)
     * @return first coordinate of point point in center of element with given index
     */
    double getElementCenter0(std::size_t c0index) const { return (c0[c0index+1] + c0[c0index]) / 2.0; }

    /**
     * Get second coordinate of point in center of element.
     * @param c1index index of element (c1 index)
     * @return second coordinate of point point in center of element with given index
     */
    double getElementCenter1(std::size_t c1index) const { return (c1[c1index+1] + c1[c1index]) / 2.0; }

    /**
     * Get point in center of element.
     * @param c0index, c1index index of element
     * @return point in center of element with given index
     */
    Vec<2, double> getElementCenter(std::size_t c0index, std::size_t c1index) const {
        return vec(getElementCenter0(c0index), getElementCenter1(c1index));
    }

    /**
     * Get element (as rectangle).
     * @param c0index, c1index index of element
     * @return element with given index
     */
    Box2D getElement(std::size_t c0index, std::size_t c1index) const {
        return Box2D(c0[c0index], c1[c1index], c0[c0index+1], c1[c1index+1]);
    }

    /**
     * Return a mesh that enables iterating over middle points of the rectangles
     * \return new rectilinear mesh with points in the middles of original rectangles
     */
    RectangularMesh getMidpointsMesh() const;

private:

    // Common code for: left, right, bottom, top boundries:
    struct BoundaryIteratorImpl: public BoundaryImpl<RectangularMesh>::IteratorImpl {

        const RectangularMesh &mesh;

        std::size_t index;

        BoundaryIteratorImpl(const RectangularMesh& mesh, std::size_t index): mesh(mesh), index(index) {}

        virtual void increment() { ++index; }

        virtual bool equal(const typename BoundaryImpl<RectangularMesh>::IteratorImpl& other) const {
            return index == static_cast<const BoundaryIteratorImpl&>(other).index;
        }

    };

    struct LeftBoundary: public BoundaryImpl<RectangularMesh<2,Mesh1D>> {

        struct IteratorImpl: public BoundaryIteratorImpl {

            IteratorImpl(const RectangularMesh<2,Mesh1D>& mesh, std::size_t index): BoundaryIteratorImpl(mesh, index) {}

            virtual std::size_t dereference() const { return this->mesh.index(0, index); }

            virtual typename BoundaryImpl<RectangularMesh>::IteratorImpl* clone() const {
                return new IteratorImpl(*this);
            }

        };

        typedef typename BoundaryImpl<RectangularMesh<2,Mesh1D>>::Iterator Iterator;

        //virtual LeftBoundary* clone() const { return new LeftBoundary(); }

        bool includes(const RectangularMesh<2,Mesh1D> &mesh, std::size_t mesh_index) const {
            return mesh.index0(mesh_index) == 0;
        }

        Iterator begin(const RectangularMesh<2,Mesh1D> &mesh) const {
            return Iterator(new IteratorImpl(mesh, 0));
        }

        Iterator end(const RectangularMesh<2,Mesh1D> &mesh) const {
            return Iterator(new IteratorImpl(mesh, mesh.c1.size()));
        }

    };

    struct RightBoundary: public BoundaryImpl<RectangularMesh<2,Mesh1D>> {

        struct IteratorImpl: public BoundaryIteratorImpl {

            IteratorImpl(const RectangularMesh<2,Mesh1D>& mesh, std::size_t index): BoundaryIteratorImpl(mesh, index) {}

            virtual std::size_t dereference() const { return this->mesh.index(this->mesh.c0.size()-1, index); }

            virtual typename BoundaryImpl<RectangularMesh<2,Mesh1D>>::IteratorImpl* clone() const {
                return new IteratorImpl(*this);
            }

        };

        typedef typename BoundaryImpl<RectangularMesh<2,Mesh1D>>::Iterator Iterator;

        //virtual RightBoundary* clone() const { return new RightBoundary(); }

        bool includes(const RectangularMesh<2,Mesh1D> &mesh, std::size_t mesh_index) const {
            return mesh.index0(mesh_index) == mesh.c0.size()-1;
        }

        Iterator begin(const RectangularMesh<2,Mesh1D> &mesh) const {
            return Iterator(new IteratorImpl(mesh, 0));
        }

        Iterator end(const RectangularMesh<2,Mesh1D> &mesh) const {
            return Iterator(new IteratorImpl(mesh, mesh.c1.size()));
        }

    };

    struct TopBoundary: public BoundaryImpl<RectangularMesh<2,Mesh1D>> {

        struct IteratorImpl: public BoundaryIteratorImpl {

            IteratorImpl(const RectangularMesh<2,Mesh1D>& mesh, std::size_t index): BoundaryIteratorImpl(mesh, index) {}

            virtual std::size_t dereference() const { return this->mesh.index(index, 0); }

            virtual typename BoundaryImpl<RectangularMesh<2,Mesh1D>>::IteratorImpl* clone() const {
                return new IteratorImpl(*this);
            }

        };

        typedef typename BoundaryImpl<RectangularMesh<2,Mesh1D>>::Iterator Iterator;

        //virtual TopBoundary* clone() const { return new TopBoundary(); }

        bool includes(const RectangularMesh<2,Mesh1D> &mesh, std::size_t mesh_index) const {
            return mesh.index1(mesh_index) == 0;
        }

        Iterator begin(const RectangularMesh<2,Mesh1D> &mesh) const {
            return Iterator(new IteratorImpl(mesh, 0));
        }

        Iterator end(const RectangularMesh<2,Mesh1D> &mesh) const {
            return Iterator(new IteratorImpl(mesh, mesh.c0.size()));
        }
    };

    struct BottomBoundary: public BoundaryImpl<RectangularMesh<2,Mesh1D>> {

        struct IteratorImpl: public BoundaryIteratorImpl {

            IteratorImpl(const RectangularMesh<2,Mesh1D>& mesh, std::size_t index): BoundaryIteratorImpl(mesh, index) {}

            virtual std::size_t dereference() const { return this->mesh.index(index, this->mesh.c1.size()-1); }

            virtual typename BoundaryImpl<RectangularMesh<2,Mesh1D>>::IteratorImpl* clone() const {
                return new IteratorImpl(*this);
            }

        };

        typedef typename BoundaryImpl<RectangularMesh<2,Mesh1D>>::Iterator Iterator;

        //virtual BottomBoundary* clone() const { return new BottomBoundary(); }

        bool includes(const RectangularMesh &mesh, std::size_t mesh_index) const {
            return mesh.index1(mesh_index) == mesh.c1.size()-1;
        }

        Iterator begin(const RectangularMesh<2,Mesh1D> &mesh) const {
            return Iterator(new IteratorImpl(mesh, 0));
        }

        Iterator end(const RectangularMesh<2,Mesh1D> &mesh) const {
            return Iterator(new IteratorImpl(mesh, mesh.c0.size()));
        }

    };

public:
    // boundaries:

    template <typename Predicate>
    static Boundary getBoundary(Predicate predicate) {
        return Boundary(new PredicateBoundary<RectangularMesh<2,Mesh1D>, Predicate>(predicate));
    }

    static Boundary getLeftBoundary() {
        return Boundary(new LeftBoundary());
    }

    static Boundary getRightBoundary() {
        return Boundary(new RightBoundary());
    }

    static Boundary getTopBoundary() {
        return Boundary(new TopBoundary());
    }

    static Boundary getBottomBoundary() {
        return Boundary(new BottomBoundary());
    }
};

/**
 * Do linear 2d interpolation with checking bounds variants.
 * @param data 2d data source, data(i0, i1) should return data in point (c0[i0], c1[i1])
 * @param point_c0,point_c1 requested point coordinates
 * @param c0 first coordinates of points
 * @param c1 second coordinates of points
 * @param index0 should be equal to c0.findIndex(point_c0)
 * @param index1 should be equal to c1.findIndex(point_c1)
 * @return value in point point_c0, point_c1
 * @tparam DataGetter2D functor
 */
template <typename DataGetter2D, typename Mesh1D>
auto interpolateLinear2D(DataGetter2D data, const double& point_c0, const double& point_c1, const Mesh1D& c0, const Mesh1D& c1, std::size_t index0, std::size_t index1)
  -> typename std::remove_reference<decltype(data(0, 0))>::type {
    if (index0 == 0) {
        if (index1 == 0) return data(0, 0);
        if (index1 == c1.size()) return data(0, index1-1);
        return interpolation::linear(c1[index1-1], data(0, index1-1), c1[index1], data(0, index1), point_c1);
    }

    if (index0 == c0.size()) {
        --index0;
        if (index1 == 0) return data(index0, 0);
        if (index1 == c1.size()) return data(index0, index1-1);
        return interpolation::linear(c1[index1-1], data(index0, index1-1), c1[index1], data(index0, index1), point_c1);
    }

    if (index1 == 0)
        return interpolation::linear(c0[index0-1], data(index0-1, 0), c0[index0], data(index0, 0), point_c0);

    if (index1 == c1.size()) {
        --index1;
        return interpolation::linear(c0[index0-1], data(index0-1, index1), c0[index0], data(index0, index1), point_c0);
    }

    return interpolation::bilinear(c0[index0-1], c0[index0],
                                   c1[index1-1], c1[index1],
                                   data(index0-1, index1-1),
                                   data(index0,   index1-1),
                                   data(index0,   index1  ),
                                   data(index0-1, index1  ),
                                   point_c0, point_c1);
}


template <typename Mesh1D, typename DataT>    // for any data type
struct InterpolationAlgorithm<RectangularMesh<2,Mesh1D>, DataT, INTERPOLATION_LINEAR> {
    static void interpolate(const RectangularMesh<2,Mesh1D>& src_mesh, const DataVector<DataT>& src_vec, const plask::MeshD<2>& dst_mesh, DataVector<DataT>& dst_vec) {
        auto dst = dst_vec.begin();
        for (auto p: dst_mesh)
            *dst++ = src_mesh.interpolateLinear(src_vec, p);
    }
};

} // namespace plask

namespace std { // use fast iterator if we know mesh type at compile time:

    template <typename Mesh1D>
    inline auto begin(const plask::RectangularMesh<2,Mesh1D>& m) -> decltype(m.begin_fast()) {
        return m.begin_fast();
    }

    template <typename Mesh1D>
    inline auto end(const plask::RectangularMesh<2,Mesh1D>& m) -> decltype(m.end_fast()) {
        return m.end_fast();
    }

} // namespace std

#endif // PLASK__RECTANGULAR2D_H
