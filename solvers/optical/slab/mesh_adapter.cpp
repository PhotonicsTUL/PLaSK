#include "mesh_adapter.h"

namespace plask { namespace solvers { namespace slab {

template<> shared_ptr<LevelsAdapter<2>::Mesh> LevelsAdapterRectangular<2>::yield()  {
    if (idx == src->axis1->size()) return shared_ptr<LevelsAdapter<2>::Mesh>();
    return make_shared<LevelsAdapterRectangular<2>::Level>(src, idx++);
}

template<> shared_ptr<LevelsAdapter<3>::Mesh> LevelsAdapterRectangular<3>::yield()  {
    if (idx == src->axis2->size()) return shared_ptr<LevelsAdapter<3>::Mesh>();
    return make_shared<LevelsAdapterRectangular<3>::Level>(src, idx++);
}

template<> std::size_t LevelsAdapterRectangular<2>::Level::size() const {
    return src->axis0->size();
}

template<> plask::Vec<2> LevelsAdapterRectangular<2>::Level::at(std::size_t i) const {
    return src->at(i, vert);
}

template<> size_t LevelsAdapterRectangular<2>::Level::index(size_t i) const {
    return src->index(i, vert);
}

template<> double LevelsAdapterRectangular<2>::Level::vpos() const {
    return src->axis1->at(vert);
}

template<> std::size_t LevelsAdapterRectangular<3>::Level::size() const {
    return src->axis0->size() * src->axis1->size();
}

template<> plask::Vec<3> LevelsAdapterRectangular<3>::Level::at(std::size_t i) const {
    return src->at(i % src->axis0->size(), i / src->axis0->size(), vert);
}

template<> size_t LevelsAdapterRectangular<3>::Level::index(size_t i) const {
    return src->index(i % src->axis0->size(), i / src->axis0->size(), vert);
}

template<> double LevelsAdapterRectangular<3>::Level::vpos() const {
    return src->axis2->at(vert);
}

template struct LevelsAdapterRectangular<2>;
template struct LevelsAdapterRectangular<3>;

}}} // namespace plask::solvers::slab
