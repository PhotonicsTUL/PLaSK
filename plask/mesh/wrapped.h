#ifndef PLASK__MESH_MIRROR_H
#define PLASK__MESH_MIRROR_H

#include "mesh.h"
#include "../geometry/space.h"

namespace plask {

/**
 * Mesh adapter for meshes extending outside of the computational regions and wrapping coordinates for
 * mirror symmetry and periodicity.
 *
 * Right now it only handles mirror symmetry in specified directions. It takes other mesh in the constructor
 * and simple delegates the calls to it replacing negative position on specified axis with
 * its absolute value.
 *
 * This is a dumb version designed for use only as temporary local variables. Do not use it for storing any
 * mesh permanently!
 *
 * TODO handle periodicity as well.
 */
template <int dim>
struct WrappedMesh: public MeshD<dim> {

  protected:

      const MeshD<dim>& original;   ///< Original mesh
      bool symmetric[dim];          ///< Indicates whether there is a symmetry along specified axes

  public:

    /**
     * Construct mirror adapter
     * \param original original mesh
     * \param geometry geometry in which the mesh is defined
     */
    WrappedMesh(const MeshD<dim>& original, const shared_ptr<GeometryD<dim>>& geometry);

    virtual ~WrappedMesh() {}

    virtual std::size_t size() const {
        return original.size();
    }

    virtual Vec<dim> at(std::size_t index) const {
        Vec<dim> pos = original.at(index);
        for (int dir = 0; dir < dim; ++dir)
            if (symmetric[dir]) pos[dir] = abs(pos[dir]);
        return pos;
    }

    virtual void writeXML(XMLElement& object) const {
        original.writeXML(object);
    }
};

} // namespace plask

#endif // PLASK__MESH_MIRROR_H