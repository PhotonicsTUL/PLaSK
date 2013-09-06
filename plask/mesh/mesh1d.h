#ifndef PLASK__MESH1D_H
#define PLASK__MESH1D_H

#include "mesh.h"
#include "rectilinear1d.h"
#include "regular1d.h"

namespace plask {
 
/**
 * Wrapper class for 1D mesh
 * \tparam AxisT wrapped 1D mesh
 * \tparam PtrT intelligent pointer holding the axis
 */
template <typename AxisT>
class RectangularMesh<1,AxisT>: public MeshD<1> {

  public:
    
    /// Original mesh type for further access
    typedef AxisT AxisType;
    
    /// Pointer to original mesh
    AxisT axis;
    
    /**
     * Create the mesh and pin an axis to it
     * \param args... arguments forwarded to the axis
     */
    template <typename... Args>
    RectangularMesh(const Args&... args): axis(args...) {
        axis.owner = this;
    }
    
    /**
     * Copy constructor
     * \param source object to copy from
     */
    RectangularMesh(const RectangularMesh<1,AxisT>& source): axis(source.axis) {
        axis.owner = this;
    }
    
    /// Implicit conversion to the axis
    operator const AxisT&() const { return axis; }
    
    virtual size_t size() const {
        return axis.size();
    }

    virtual Vec<1> at(size_t index) const {
        return axis[index];
    }

    virtual void writeXML(XMLElement& object) const;

    /**
     * Get point at given index
     * \param index index in the meshes
     * \return axis coordinate
     */
    double operator[](size_t index) const {
        return axis[index];
    }
    
    /**
     * Compare meshes
     * \param to_compare mesh to compare
     * \return \c true only if this mesh and \p to_compare represents the same set of points
     */
    bool operator==(const RectangularMesh<1,AxisT>& to_compare) {
        return axis == to_compare.axis;
    }
};

typedef RectangularMesh<1,RectilinearAxis> RectilinearMesh1D;
typedef RectangularMesh<1,RegularAxis> RegularMesh1D;


} // namespace plask

#endif // PLASK__MESH1D_H