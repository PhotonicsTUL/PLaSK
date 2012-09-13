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
    Mesh1D* minor_axis;
    Mesh1D* major_axis;

  public:

    /// Boundary type.
    typedef ::plask::Boundary<RectangularMesh<2,Mesh1D>> Boundary;

    /// First coordinate of points in this mesh.
    Mesh1D axis0;

    /// Second coordinate of points in this mesh.
    Mesh1D axis1;

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
    RectangularMesh(IterationOrder iterationOrder = NORMAL_ORDER) { setIterationOrder(iterationOrder); }

    /**
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     * @param iterationOrder iteration order
     */
    RectangularMesh(Mesh1D mesh0, Mesh1D mesh1, IterationOrder iterationOrder = NORMAL_ORDER) :
        axis0(std::move(mesh0)), axis1(std::move(mesh1)) { setIterationOrder(iterationOrder); }

    /*
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate, or constructor argument for the first coordinate mesh
     * @param mesh1 mesh for the second coordinate, or constructor argument for the second coordinate mesh
     * @param iterationOrder iteration order
     */
    /*template <typename Mesh0CtorArg, typename Mesh1CtorArg>
    RectangularMesh(Mesh0CtorArg&& mesh0, Mesh1CtorArg&& mesh1, IterationOrder iterationOrder = NORMAL_ORDER) :
        axis0(std::forward<Mesh0CtorArg>(mesh0)), axis1(std::forward<Mesh1CtorArg>(mesh1)) { setIterationOrder(iterationOrder); }*/

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
    Mesh1D& up() { return axis1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return axis1
     */
    const Mesh1D& up() const { return axis1; }

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
     * \param no
     */
    Mesh1D& axis(size_t n) {
        if (n == 0) return axis0;
        else if (n != 1) throw Exception("Bad axis number");
        return axis1;
    }

    /**
     * Get numbered axis
     * \param no
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
     * Get number of objects (for FEM method) in the first direction.
     * @return number of objects in this mesh in the first direction (axis0 direction).
     */
    std::size_t getObjectsCount0() const {
        return std::max(int(axis0.size())-1, 0);
    }

    /**
     * Get number of objects (for FEM method) in the second direction.
     * @return number of objects in this mesh in the second direction (axis1 direction).
     */
    std::size_t getObjectsCount1() const {
        return std::max(int(axis1.size())-1, 0);
    }

    /**
     * Get number of objects (for FEM method).
     * @return number of objects in this mesh
     */
    std::size_t getObjectsCount() const {
        return std::max((int(axis0.size())-1) * (int(axis1.size())-1), 0);
    }

    /**
     * Get area of given object.
     * @param index0, index1 index of object
     * @return area of object with given index
     */
    double getObjectArea(std::size_t index0, std::size_t index1) const {
        return (axis0[index0+1] - axis0[index0])*(axis1[index1+1] - axis1[index1]);
    }

    /**
     * Get first coordinate of point in center of object.
     * @param index0 index of object (axis0 index)
     * @return first coordinate of point point in center of object with given index
     */
    double getObjectCenter0(std::size_t index0) const { return (axis0[index0+1] + axis0[index0]) / 2.0; }

    /**
     * Get second coordinate of point in center of object.
     * @param index1 index of object (axis1 index)
     * @return second coordinate of point point in center of object with given index
     */
    double getObjectCenter1(std::size_t index1) const { return (axis1[index1+1] + axis1[index1]) / 2.0; }

    /**
     * Get point in center of object.
     * @param index0, index1 index of object
     * @return point in center of object with given index
     */
    Vec<2, double> getObjectCenter(std::size_t index0, std::size_t index1) const {
        return vec(getObjectCenter0(index0), getObjectCenter1(index1));
    }

    /**
     * Get object (as rectangle).
     * @param index0, index1 index of object
     * @return object with given index
     */
    Box2D getObject(std::size_t index0, std::size_t index1) const {
        return Box2D(axis0[index0], axis1[index1], axis0[index0+1], axis1[index1+1]);
    }

    /**
     * Return a mesh that enables iterating over middle points of the rectangles
     * \return new rectilinear mesh with points in the middles of original rectangles
     */
    RectangularMesh getMidpointsMesh() const;

private:

    // Common code for: left, right, bottom, top boundries:
    struct BoundaryIteratorImpl: public BoundaryLogicImpl<RectangularMesh>::IteratorImpl {

        const RectangularMesh &mesh;

        std::size_t line;

        std::size_t index;

        BoundaryIteratorImpl(const RectangularMesh& mesh, std::size_t line, std::size_t index): mesh(mesh), line(line), index(index) {}

        virtual void increment() { ++index; }

        virtual bool equal(const typename BoundaryLogicImpl<RectangularMesh>::IteratorImpl& other) const {
            return index == static_cast<const BoundaryIteratorImpl&>(other).index;
        }

    };

    // iterator over vertical line (from bottom to top). for left and right boundaries
    struct VerticalIteratorImpl: public BoundaryIteratorImpl {

        VerticalIteratorImpl(const RectangularMesh& mesh, std::size_t line, std::size_t index): BoundaryIteratorImpl(mesh, line, index) {}

        virtual std::size_t dereference() const { return this->mesh.index(this->line, this->index); }

        virtual typename BoundaryLogicImpl<RectangularMesh>::IteratorImpl* clone() const {
            return new VerticalIteratorImpl(*this);
        }
    };

    // iterator over horizonstal line (from left to right), for bottom and top boundaries
    struct HorizontalIteratorImpl: public BoundaryIteratorImpl {

        HorizontalIteratorImpl(const RectangularMesh& mesh, std::size_t line, std::size_t index): BoundaryIteratorImpl(mesh, line, index) {}

        virtual std::size_t dereference() const { return this->mesh.index(this->index, this->line); }

        virtual typename BoundaryLogicImpl<RectangularMesh>::IteratorImpl* clone() const {
            return new HorizontalIteratorImpl(*this);
        }
    };

    struct VerticalBoundary: public BoundaryWithMeshLogicImpl<RectangularMesh<2,Mesh1D>> {

        typedef typename BoundaryLogicImpl<RectangularMesh<2,Mesh1D>>::Iterator Iterator;

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

    };

    struct HorizontalBoundary: public BoundaryWithMeshLogicImpl<RectangularMesh<2,Mesh1D>> {

        typedef typename BoundaryLogicImpl<RectangularMesh<2,Mesh1D>>::Iterator Iterator;

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
    };

    //TODO
    struct HorizontalLineBoundary: public BoundaryLogicImpl<RectangularMesh<2,Mesh1D>> {

        double height;

        bool includes(const RectangularMesh &mesh, std::size_t mesh_index) const {
            return mesh.index1(mesh_index) == mesh.axis1.findNearestIndex(height);
        }
    };

public:
    // boundaries:

    template <typename Predicate>
    static Boundary getBoundary(Predicate predicate) {
        return Boundary(new PredicateBoundary<RectangularMesh<2,Mesh1D>, Predicate>(predicate));
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
	 * Get boundary which show one vertical (from bottom to top) line in mesh which lies nearest given coordinate.
	 * @param axis0_coord axis 0 coordinate
	 * @return boundary which show one vertical (from bottom to top) line in mesh
	 */
	static Boundary getVerticalBoundaryNear(double axis0_coord) {
		return Boundary( [axis0_coord](const RectangularMesh<2,Mesh1D>& mesh) {return new VerticalBoundary(mesh, mesh.axis0.findNearestIndex(axis0_coord));} );
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
	 * Get boundary which show one horizontal (from left to right) line in mesh.
	 * @param line_nr_axis1 number of horizontal line, axis 1 index of mesh
	 * @return boundary which show one horizontal (from left to right) line in mesh
	 */
	static Boundary getHorizontalBoundaryAtLine(std::size_t line_nr_axis1) {
		return Boundary( [line_nr_axis1](const RectangularMesh<2,Mesh1D>& mesh) {return new HorizontalBoundary(mesh, line_nr_axis1);} );
	}

	/**
	 * Get boundary which show one horizontal (from left to right) line in mesh which lies nearest given coordinate.
	 * @param axis1_coord axis 1 coordinate
	 * @return boundary which show one horizontal (from left to right) line in mesh
	 */
	static Boundary getHorizontalBoundaryNear(double axis1_coord) {
		return Boundary( [axis1_coord](const RectangularMesh<2,Mesh1D>& mesh) {return new HorizontalBoundary(mesh, mesh.axis1.findNearestIndex(axis1_coord));} );
	}

	/**
	 * Get boundary which show one horizontal, top (from left to right) line in mesh.
	 * @return boundary which show top line in mesh
	 */
    static Boundary getTopBoundary() {
        return Boundary( [](const RectangularMesh<2,Mesh1D>& mesh) {return new HorizontalBoundary(mesh, mesh.axis1.size()-1);} );
    }

	/**
	 * Get boundary which show one horizontal, bottom (from left to right) line in mesh.
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
