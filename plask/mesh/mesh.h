#ifndef PLASK__MESH_H
#define PLASK__MESH_H

/** @page meshes Meshes
 * @section meshes_about About
 * Mesh represent (ordered) set of points in 3d space. All meshes in plask inherits from and implements plask::Mesh interface.
 * 
 * Typically, there are some data connected with points in mesh.
 * In plask, all this data are not stored in mesh class, and so they must be store separately.
 * Because points in mesh are ordered, and each one have unique index from <code>0</code> to <code>getSize()-1</code>
 * you can sore data in any indexed structure, like array or std::vector (which is recommended),
 * storing data for i-th point in mesh under i-th index.
 * 
 * @section meshes_interpolation Data interpolation
 * 
 * @section meshes_write How to implement new mesh and interpolation methods
 */

#include "../space.h"
#include <memory>

#include "interpolation.h"

namespace plask {

/**
 * Base class for all meshes.
 * Mesh represent set of points in 3d space and:
 * - know number of points,
 * - allow for iterate over this points,
 * - can calculate interpolated value for given points (in 3d), source values, and interpolation method.
 * 
 * @see @ref meshes
 */
struct Mesh {

    ///@return number of points in mesh
    virtual std::size_t getSize() const;
    
    /**
     * Interpolate values (@a src_vec) from one mesh (@a src_mesh) to this one using given interpolation method.
     * @param src_mesh, src_vec source
     * @param method interpolation method to use
     * @return vector with interpolated values
     * @throw NotImplemented if given interpolation method is not implemented for used source mesh type
     * @throw CriticalException if given interpolation method is not valid
     */
    template <typename SrcMeshT, typename DataT>
    inline std::shared_ptr<const std::vector<DataT>>
    fill(SrcMeshT& src_mesh, std::shared_ptr<const std::vector<DataT>>& src_vec, InterpolationMethod method = DEFAULT)
    throw (NotImplemented, CriticalException) {
        return interpolate(src_mesh, src_vec, *this, method);
    }

};

#ifdef disable_fragment

//TODO nieaktualne, ale coś może się przydać:
//TODO zrobić Meshe o dużej wydajności w poszczególnych przestrzeniach (specjalizowane przestrzenią)
//TODO i zaimplementować interfejs Mesh uzywając Mesh optymalizowany + funkcja konwertująca pkt. do 3d
//TODO zamiast dim i Vec<dim> zrobić i specjalizować typami przestrzeni udostępniającymi typ punktu, itp.
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

#endif


} // namespace plask

#endif  //PLASK__MESH_H
