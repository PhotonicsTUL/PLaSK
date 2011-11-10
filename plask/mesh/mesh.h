#include "../vec.h"
#include <memory>

namespace plask {

/**
 * Base class for all meshs.
 * Mesh represent set of points in 3d space nad has/know/allow for:
 * - number of points
 * - iterator over this points
 * - can calculate interpolated value for given points (in 3d), source values, and interpolation method
 */
struct Mesh {
};

//TODO nieaktualne, ale coś może się przydać:
/**
Base class for all meshs in given space.
Mesh represent set of points in space.
@tparam dim number of space dimentions
*/
template <int dim>
struct Mesh {

    // Point type used by this mesh.
    typedef Vec<dim> Vec_t;
    
    // Base class for all meshs inharited from this class (1d, 2d, 3d mesh base).
    typedef Mesh<dim> BaseClass;

    /**
    Base class for mesh iterator implementation. Iterate over points in mesh.
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
    Iterator over points in mesh.
    
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


} // namespace plask
