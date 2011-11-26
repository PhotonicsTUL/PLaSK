#ifndef PLASK__RECTILINEAR_H
#define PLASK__RECTILINEAR_H

#include <vector>
#include <boost/iterator/iterator_facade.hpp>

#include "../vector/2d.h"

namespace plask {

class RectilinearMesh1d {

	std::vector<double> points;

public:
    
    typedef double PointType;
    
    typedef std::vector<double>::const_iterator const_iterator;
	const_iterator begin() const { return points.begin(); }
    const_iterator end() const { return points.end(); }
    
    //should we allow for non-const iterators?
    /*typedef std::vector<double>::iterator iterator;
    iterator begin() { return points.begin(); }
    iterator end() { return points.end(); }*/
    

    
    RectilinearMesh1d() {}
    
    std::size_t getSize() const { return points.size(); }
    
    /**
     * Add (1d) point to this mesh.
     * @param new_node_cord coordinate of point to add
     */
    void addPoint(double new_node_cord);
    
    /**
     * Get point by index.
     * @param index index of point, from 0 to getSize()-1
     * @return point with given index
     */
    double operator[](std::size_t index) const { return points[index]; }
    
};

class RectilinearMesh2d {
	
	public:
	
	RectilinearMesh1d x;
	RectilinearMesh1d y;
	
	/*class Iterator: public boost::iterator_facade< Iterator, Vector2d<double>, boost::random_access_traversal_tag > {
	
		const RectilinearMesh1d* x_nodes;
		RectilinearMesh1d::const_iterator x;	//TODO może 1 indeks... wszystkie operacje prościutkie poza dereferencją wymagającą dzielenia
		RectilinearMesh1d::const_iterator y;
		
		public:
		
		//default constructor to satisfy the iterator requirements
		Iterator() {}
		
		Iterator(const RectilinearMesh1d* x_nodes, RectilinearMesh1d::const_iterator x, RectilinearMesh1d::const_iterator y)
		: x_̣̣̣̣(x_nodes), x(x), y(y) {}
		
		bool equal(const Iterator& other) const {
			return x == other.x && y == other.y;
		}

		void increment() {
			++x;
			if (x == x_nodes->end()) {
				x = x_nodes->begin();
				++y;
			}
		}
		
		void decrement() {
			if (x == x_nodes->begin()) {
				--y;
				x = x_nodes->end();
			}
			--x;
		}
		
	};*/
	
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
     * @param index index of point, from 0 to getSize()-1
     * @return point with given index
     */
    Vector2d<double> operator[](std::size_t index) const {
        const std::size_t x_size = x.getSize();
        return Vector2d<double>(x[index % x_size], y[index / x_size]);
    }
};

}	//namespace plask

#endif // PLASK__RECTILINEAR_H
