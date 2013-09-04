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
struct Mesh1D: public MeshD<1> {
    
    /// Original mesh type for further access
    typedef AxisT AxisType;
    
    /// Pointer to original mesh
    AxisT axis;
    
    /**
     * Create the mesh and pin an axis to it
     * \param args... arguments forwarded to the axis
     */
    template <typename... Args>
    Mesh1D(const Args&... args): axis(args...) {
        axis.owner = this;
    }
    
    /**
     * Copy constructor
     * \param source object to copy from
     */
    Mesh1D(const Mesh1D<AxisT>& source): axis(source.axis) {
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
    bool operator==(const Mesh1D<AxisT>& to_compare) {
        return axis == to_compare.axis;
    }
};

typedef Mesh1D<RectilinearAxis> RectilinearMesh1D;
typedef Mesh1D<RegularAxis> RegularMesh1D;


} // namespace plask

#endif // PLASK__MESH1D_H