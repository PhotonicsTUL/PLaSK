#ifndef PLASK__RECTILINEAR_H
#define PLASK__RECTILINEAR_H

#include <vector>

#include "../vector/2d.h"
#include "../utils/iterators.h"

namespace plask {

/**
 * Rectilinear mesh in 1d space.
 */
class RectilinearMesh1d {

    ///Points coordinates in ascending order.
	std::vector<double> points;

public:
    
    ///Type of points in this mesh.
    typedef double PointType;
    
    ///Random access iterator type which alow iterate over all points in this mesh, in ascending order.
    typedef std::vector<double>::const_iterator const_iterator;
    
    ///@return iterator referring to the first point in this mesh
	const_iterator begin() const { return points.begin(); }
	
	///@return iterator referring to the past-the-end point in this mesh
    const_iterator end() const { return points.end(); }
    
    //should we allow for non-const iterators?
    /*typedef std::vector<double>::iterator iterator;
    iterator begin() { return points.begin(); }
    iterator end() { return points.end(); }*/
    
    //Construct empty mesh.
    //RectilinearMesh1d() {}
    
    ///@return number of points in mesh
    std::size_t getSize() const { return points.size(); }
    
    /**
     * Add (1d) point to this mesh.
     * Point is add to mesh only if it is not already included in mesh.
     * @param new_node_cord coordinate of point to add
     */
    void addPoint(double new_node_cord);
    
    /**
     * Get point by index.
     * @param index index of point, from 0 to getSize()-1
     * @return point with given @a index
     */
    const double& operator[](std::size_t index) const { return points[index]; }
    
};

/**
 * Rectilinear mesh in 2d space.
 */
struct RectilinearMesh2d {
	
	///First coordinate of points in this mesh.
	RectilinearMesh1d x;
	
	///Second coordinate of points in this mesh.
	RectilinearMesh1d y;
	
	///Type of points in this mesh.
	typedef Vector2d<double> PointType;
	
	/**
	 * Random access iterator type which alow iterate over all points in this mesh, in order appointed by operator[].
	 * This iterator type is indexed, which mean that it have (read-write) index field equal to 0 for begin() and growing up to getSize() for end().
	 */
	typedef IndexedIterator< const RectilinearMesh2d, PointType > const_iterator;
	
	///@return iterator referring to the first point in this mesh
	const_iterator begin() const { return const_iterator(this, 0); }
	
	///@return iterator referring to the past-the-end point in this mesh
	const_iterator end() const { return const_iterator(this, getSize()); }
	
	///@return number of points in mesh
	std::size_t getSize() const { return x.getSize() * y.getSize(); }
	
	/**
     * Add (2d) point to this mesh.
     * @param to_add point to add
     */
    void addPoint(const Vector2d<double>& to_add) {
        x.addPoint(to_add.x);
        y.addPoint(to_add.y);
    }
	
	/**
	 * Get point by index.
	 * Points are in order: (x[0], y[0]), (x[1], y[0]), ..., (x[x.getSize-1], y[0]), (x[0], y[1]), ..., (x[x.getSize()-1], y[y.getSize()-1])
     * @param index index of point, from 0 to getSize()-1
     * @return point with given @a index
     */
    Vector2d<double> operator[](std::size_t index) const {
        const std::size_t x_size = x.getSize();
        return Vector2d<double>(x[index % x_size], y[index / x_size]);
    }
};

}	//namespace plask

#endif // PLASK__RECTILINEAR_H
