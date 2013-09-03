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
      const shared_ptr<GeometryD<dim>>& geometry; ///< Geometry of the mesh

  public:

    /**
     * Construct mirror adapter
     * \param original original mesh
     * \param geometry geometry in which the mesh is defined
     */
    WrappedMesh(const MeshD<dim>& original, const shared_ptr<GeometryD<dim>>& geometry): original(original), geometry(geometry) {}

    virtual ~WrappedMesh() {}

    virtual std::size_t size() const {
        return original.size();
    }

    virtual Vec<dim> at(std::size_t index) const {
        Vec<dim> pos = original.at(index);
        auto box = geometry->getChild()->getBoundingBox();
        for (int i = 0; i < dim; ++i) {
            auto dir = Geometry::Direction(i+3-dim);
            if (geometry->isPeriodic(dir)) {
                double l = box.lower[i], h = box.upper[i];
                double d = h - l;
                if (geometry->isSymmetric(dir)) {
                    pos[i] = std::fmod(abs(pos[i]), 2*d);
                    if (pos[i] > d) pos[i] = 2*d - pos[i];
                } else {
                    pos[i] = std::fmod(pos[i]-l, d);
                    pos[i] += (pos[i] >= 0)? l : h;
                }
            } else
                if (geometry->isSymmetric(dir)) pos[i] = abs(pos[i]);
        }
        return pos;
    }

    virtual void writeXML(XMLElement& object) const {
        original.writeXML(object);
    }
};

} // namespace plask

#endif // PLASK__MESH_MIRROR_H
