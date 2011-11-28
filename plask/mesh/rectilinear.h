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
    
    /**
     * Find position where @a to_find point could be insert.
     * @param to_find point to find
     * @return First position where to_find could be insert.
     *         Reffer to value equal to @a to_find only if @a to_find is already in mesh.
     *         Can be equal to end() if to_find is higher than all points in mesh
     *         (in such case returned iterator can't be dereferenced).
     */
    const_iterator find(double to_find) const;
    
    /**
     * Find index where @a to_find point could be insert.
     * @param to_find point to find
     * @return First index where to_find could be insert.
     *         Reffer to value equal to @a to_find only if @a to_find is already in mesh.
     *         Can be equal to getSize() if to_find is higher than all points in mesh.
     */
    std::size_t findIndex(double to_find) const { return find(to_find) - begin(); }
    
    template <typename RandomAccessContainer>
    typename RandomAccessContainer::value_type linearInterpolation(RandomAccessContainer& data, double point);
    
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

//RectilinearMesh1d method templates implementation
template <typename RandomAccessContainer>
typename RandomAccessContainer::value_type RectilinearMesh1d::linearInterpolation(RandomAccessContainer& data, double point) {
    //TODO what should do if mesh is empty?
    std::size_t index = findIndex(point);
    if (index == getSize()) return data[index - 1];
    if (index == 0 || operator [](index) == point) return data[index]; //hit exactly
    //here: d0=data[index-1] < point < data[index]=d1
    auto d0 = data[index-1];
    return d0 + (data[index] - d0) * (point - operator [](index-1)) / (operator [](index) - operator [](index-1));
}

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
	 * Calculate this mesh index using indexes of x and y.
	 * @param x_index index of x, from 0 to x.getSize()-1
	 * @param y_index index of y, from 0 to y.getSize()-1
	 * @return this mesh index, from 0 to getSize()-1
	 */
	std::size_t index(std::size_t x_index, std::size_t y_index) const {
        return x_index + x.getSize() * y_index;
	}
	
	/**
	 * Calculate index of x using this mesh index.
	 * @param mesh_index this mesh index, from 0 to getSize()-1
	 * @return index of x, from 0 to x.getSize()-1
	 */
	std::size_t xIndex(std::size_t mesh_index) const {
        return mesh_index % x.getSize();
	}
	
	/**
	 * Calculate index of y using this mesh index.
	 * @param mesh_index this mesh index, from 0 to getSize()-1
	 * @return index of y, from 0 to y.getSize()-1
	 */
	std::size_t yIndex(std::size_t mesh_index) const {
        return mesh_index / x.getSize();
	}
	
	/**
     * Add (2d) point to this mesh.
     * @param to_add point to add
     */
    void addPoint(const Vector2d<double>& to_add) {
        x.addPoint(to_add.x);
        y.addPoint(to_add.y);
    }
	
	/**
	 * Get point with given mesh index.
	 * Points are in order: (x[0], y[0]), (x[1], y[0]), ..., (x[x.getSize-1], y[0]), (x[0], y[1]), ..., (x[x.getSize()-1], y[y.getSize()-1])
     * @param index index of point, from 0 to getSize()-1
     * @return point with given @a index
     */
    Vector2d<double> operator[](std::size_t index) const {
        const std::size_t x_size = x.getSize();
        return Vector2d<double>(x[index % x_size], y[index / x_size]);
    }
    
    /**
	 * Get point with given x and y indexes.
	 * @param x_index index of x, from 0 to x.getSize()-1
	 * @param y_index index of y, from 0 to y.getSize()-1
     * @return point with given x and y indexes
     */
    Vector2d<double> operator()(std::size_t x_index, std::size_t y_index) const {
        return Vector2d<double>(x[x_index], y[y_index]);
    }
};

}	//namespace plask

#endif // PLASK__RECTILINEAR_H
