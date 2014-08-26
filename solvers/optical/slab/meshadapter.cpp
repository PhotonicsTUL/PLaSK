#include "meshadapter.h"

namespace plask { namespace solvers { namespace slab {

template<int dim>
size_t LevelsAdapterGeneric<dim>::GenericLevel::index(size_t i) const { return matching[i]; }

template<int dim>
double LevelsAdapterGeneric<dim>::GenericLevel::vpos() const { return vert; }


template<> shared_ptr<LevelsAdapter::Level> LevelsAdapterRectangular<2>::yield()  {
    if (idx == src->axis1->size()) return shared_ptr<typename LevelsAdapter::Level>();
    return make_shared<LevelsAdapterRectangular<2>::RectangularLevel>(src, idx++);
}

template<> shared_ptr<LevelsAdapter::Level> LevelsAdapterRectangular<3>::yield()  {
    if (idx == src->axis2->size()) return shared_ptr<typename LevelsAdapter::Level>();
    return make_shared<LevelsAdapterRectangular<3>::RectangularLevel>(src, idx++);
}

template<> std::size_t LevelsAdapterRectangular<2>::RectangularLevel::size() const {
    return src->axis0->size();
}

template<> std::size_t LevelsAdapterRectangular<2>::Mesh::size() const {
    return level->src->axis0->size();
}

template<> plask::Vec<2> LevelsAdapterRectangular<2>::Mesh::at(std::size_t i) const {
    return level->src->at(i, level->vert);
}

template<> size_t LevelsAdapterRectangular<2>::RectangularLevel::index(size_t i) const {
    return src->index(i, vert);
}

template<> double LevelsAdapterRectangular<2>::RectangularLevel::vpos() const {
    return src->axis1->at(vert);
}

template<> std::size_t LevelsAdapterRectangular<3>::RectangularLevel::size() const {
    return src->axis0->size() * src->axis1->size();
}

template<> std::size_t LevelsAdapterRectangular<3>::Mesh::size() const {
    return level->src->axis0->size() * level->src->axis1->size();
}

template<> plask::Vec<3> LevelsAdapterRectangular<3>::Mesh::at(std::size_t i) const {
    return level->src->at(i % level->src->axis0->size(), i / level->src->axis0->size(), level->vert);
}

template<> size_t LevelsAdapterRectangular<3>::RectangularLevel::index(size_t i) const {
    return src->index(i % src->axis0->size(), i / src->axis0->size(), vert);
}

template<> double LevelsAdapterRectangular<3>::RectangularLevel::vpos() const {
    return src->axis2->at(vert);
}


std::unique_ptr<LevelsAdapter> makeLevelsAdapter(const shared_ptr<const Mesh>& src)
{
    typedef std::unique_ptr<LevelsAdapter> ReturnT;

    if (auto mesh = dynamic_pointer_cast<const RectangularMesh<2>>(src))
        return ReturnT(new LevelsAdapterRectangular<2>(mesh));
    else if (auto mesh = dynamic_pointer_cast<const RectangularMesh<3>>(src))
        return ReturnT(new LevelsAdapterRectangular<3>(mesh));
    else if (auto mesh = dynamic_pointer_cast<const MeshD<2>>(src))
        return ReturnT(new LevelsAdapterGeneric<2>(mesh));
    else if (auto mesh = dynamic_pointer_cast<const MeshD<3>>(src))
        return ReturnT(new LevelsAdapterGeneric<3>(mesh));
    return ReturnT();
}

template struct LevelsAdapterGeneric<2>;
template struct LevelsAdapterGeneric<3>;
template struct LevelsAdapterRectangular<2>;
template struct LevelsAdapterRectangular<3>;

}}} // namespace plask::solvers::slab
