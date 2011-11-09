#include "../vec.h"
#include <memory>

namespace plast {

/**
Base class for all grids in given space.
Grid represent set of points in space.
@tparam dim number of space dimentions
*/
template <int dim>
struct Grid {

    typedef Vec<dim> Vec_t;

    /**
    Base class for grid iterator implementation. Iterate over points in grid.
    */
    struct IteratorImpl {
	///@return current point
	virtual Vec_t get() const = 0;
	
	///Iterate to next point.
	virtual void next() = 0;
	
	///@return @c true only if there are more points to iterate over
	virtual void hasNext() const = 0;
	
	///@return clone of @c *this
	virtual IteratorImpl* clone() const = 0;
	
	//Do nothing.
	virtual ~IteratorImpl() {}
    };

    /**
    Iterator over points in grid.
    
    Wrapper over IteratorImpl.
    */
    class Iterator {
	
	IteratorImpl* impl;
	
	public:
	
	Iterator(IteratorImpl* impl): impl(impl) {}
	
	~Iterator() { delete impl; }
	
	//Copy constructor
	Iterator(const Iterator& src) { impl = src.impl->clone(); }
	
	//Move constructor
	Iterator(Iterator &&src): impl(src.impl) { src.impl = 0; }
	
	Vec_t operator*() const { return impl->get(); }
    };
    
    virtual Iterator begin();
    
    //TODO end()
};


}	//namespace plast
