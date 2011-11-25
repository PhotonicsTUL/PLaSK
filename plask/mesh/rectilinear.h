#ifndef PLASK__RECTILINEAR_H
#define PLASK__RECTILINEAR_H

#include <vector>
#include <boost/iterator/iterator_facade.hpp>

#include "../vector/2d.h"

namespace plask {

class RectilinearMesh1d {

	std::vector<double> nodes;

public:
    
    typedef double PointType;
    
    typedef std::vector<double>::const_iterator const_iterator;
	const_iterator begin() const { return nodes.begin(); }
    const_iterator end() const { return nodes.end(); }
    
    //should we allow for non-const iterators?
    /*typedef std::vector<double>::iterator iterator;
    iterator begin() { return nodes.begin(); }
    iterator end() { return nodes.end(); }*/
    

    
    RectilinearMesh1d() {}
    
    std::size_t getSize() const { return nodes.size(); }
    
    /**
     * Add node to this mesh.
     * @param new_node_cord coordinate of node to add
     */
    void addNode(double new_node_cord);
    
};

class RectilinearMesh2d {
	
	RectilinearMesh1d x;
	RectilinearMesh1d y;
	
	public:
	
	class Iterator: public boost::iterator_facade< Iterator, Vector2d<double>, boost::random_access_traversal_tag > {
	
		const RectilinearMesh1d* x_nodes;
		RectilinearMesh1d::const_iterator x;	//TODO może 1 indeks... wszystkie operacje prościutkie poza dereferencją wymagającą dzielenia
		RectilinearMesh1d::const_iterator y;
		
		public:
		
		//default constructor to satisfy the iterator requirements
		Iterator() {}
		
		Iterator(const RectilinearMesh1d* x_nodes, RectilinearMesh1d::const_iterator x, RectilinearMesh1d::const_iterator y)
		: x_nodes(x_nodes), x(x), y(y) {}
		
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
		
	};
	
	std::size_t getSize() const { return x.getSize() * y.getSize(); }
};

}	//namespace plask

#endif // PLASK__RECTILINEAR_H
