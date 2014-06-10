#include "wrapped.h"

namespace plask {

template <int dim>
WrappedMesh<dim>::WrappedMesh(shared_ptr<const MeshD<dim> > original, const shared_ptr<const GeometryD<dim> > &geometry, const bool ignore_symmetry[dim]):
    original(original), geometry(geometry) {
    for (size_t i = 0; i != dim; ++i) this->ignore_symmetry[i] = ignore_symmetry[i];
}

template <int dim>
std::size_t WrappedMesh<dim>::size() const {
    return original->size();
}

template <int dim>
Vec<dim> WrappedMesh<dim>::at(std::size_t index) const {
    Vec<dim> pos = original->at(index);
    for (int i = 0; i < dim; ++i) {
        auto dir = Geometry::Direction(i+3-dim);
        if (geometry->isPeriodic(dir)) {
           auto box = geometry->getChildBoundingBox();
           double l = box.lower[i], h = box.upper[i];
           double d = h - l;
           if (geometry->isSymmetric(dir)) {
                if (ignore_symmetry[i]) {
                    pos[i] = std::fmod(pos[i], 2*d);
                    if (pos[i] > d) pos[i] = -2*d + pos[i];
                    else if (pos[i] < -d) pos[i] = 2*d + pos[i];
                } else {
                    pos[i] = std::fmod(abs(pos[i]), 2*d);
                    if (pos[i] > d) pos[i] = 2*d - pos[i];
                }
           } else {
               pos[i] = std::fmod(pos[i]-l, d);
               pos[i] += (pos[i] >= 0)? l : h;
           }
       } else {
           if (geometry->isSymmetric(dir) && !ignore_symmetry[i]) pos[i] = abs(pos[i]);
        }
    }
    return pos;
}

template <int dim>
void WrappedMesh<dim>::writeXML(XMLElement& object) const {
    original->writeXML(object);
}

template struct WrappedMesh<2>;
template struct WrappedMesh<3>;

} // namespace plask
