#ifndef PLASK__SOLVER_SLAB_MESHADAPTER_H
#define PLASK__SOLVER_SLAB_MESHADAPTER_H

#include <boost/iterator/filter_iterator.hpp>

#include <plask/mesh/mesh.h>

#include "matrices.h"

namespace plask { namespace solvers { namespace slab {


/// Simple adapter that allows to process single level in the mesh
template <int dim> struct LevelMeshAdapter: public MeshD<dim>
{
    /// Base of adapter generators
    struct GeneratorBase {
        /**
         * Copy generator
         */
        virtual std::unique_ptr<typename LevelMeshAdapter<dim>::GeneratorBase> copy() const = 0;
        /** Yield pointers to the consecutive levels
         * \return pointer to the new level object or \c nullptr at the end
         */
        virtual shared_ptr<LevelMeshAdapter<dim>> yield() = 0;
        /**
         * Return level at given vertical position
         * \param level at given vertical posiotion
         */
        virtual shared_ptr<LevelMeshAdapter<dim>> at(double pos) const = 0;
    };

    /**
     * Unscramble indices
     * \param i index in the adapter
     * \return index in the original mesh
     */
    virtual size_t index(size_t i) const = 0;

    /**
     * Get level vertical position
     */
    virtual double vpos() const = 0;
};


/// Generic implementation of the level adapter
template <int dim>
struct LevelMeshAdapterGeneric: public LevelMeshAdapter<dim>
{
  protected:
    /// Indices of matching points
    std::vector<size_t> matching;

    /// Original mesh
    shared_ptr<const MeshD<dim>> src;

    /// Interesting level
    double vert;

  public:
    /// Generator for the generic levels
    struct Generator: public LevelMeshAdapter<dim>::GeneratorBase {
        shared_ptr<const MeshD<dim>> src;
        shared_ptr<std::set<double>> levels;
        std::set<double>::iterator iter;
        Generator(shared_ptr<const MeshD<dim>> src): src(src), levels(new std::set<double>) {
            for (auto point: *src) {
                levels->insert(point[dim-1]);
            }
            iter = levels->begin();
        }
        Generator(const Generator& orig): src(orig.src), levels(orig.levels) {}
        std::unique_ptr<typename LevelMeshAdapter<dim>::GeneratorBase> copy() const override {
            return std::unique_ptr<typename LevelMeshAdapter<dim>::GeneratorBase>(new Generator(*this));
        }
        shared_ptr<LevelMeshAdapter<dim>> yield() override {
            if (iter == levels->end()) return shared_ptr<LevelMeshAdapter<dim>>();
            return shared_ptr<LevelMeshAdapter<dim>>(new LevelMeshAdapterGeneric<dim>(src, *(iter++)));
        }
        shared_ptr<LevelMeshAdapter<dim>> at(double pos) const override {
            assert(levels->find(pos) != levels->end());
            return shared_ptr<LevelMeshAdapter<dim>>(new LevelMeshAdapterGeneric<dim>(src, pos));
        }
    };

    /// Create mesh adapter
    LevelMeshAdapterGeneric(shared_ptr<const MeshD<dim>> src, double level): src(src), vert(level) {
        for (auto it = src->begin(); it != src->end(); ++it) {
            if ((*it)[dim-1] == level) matching.push_back(it.index);
        }
    }

    virtual std::size_t size() const override { return matching.size(); }

    virtual plask::Vec<dim> at(std::size_t i) const override { return (*src)[matching[i]]; }

    virtual size_t index(size_t i) const override { return matching[i]; }

    virtual double vpos() const override { return vert; }
};


template <int dim> struct LevelMeshAdapterRectangular;

/// More efficient Rectangular2D implementation of the level adapter
template<>
struct LevelMeshAdapterRectangular<2>: public LevelMeshAdapter<2>
{
    typedef RectangularMesh<2> RectangularT;

  protected:
    /// Original mesh
    shared_ptr<const RectangularT> src;

    /// Interesting level
    size_t vert;

  public:
    /// Generator for the generic levels
    struct Generator: public LevelMeshAdapter<2>::GeneratorBase {
        shared_ptr<const RectangularT> src;
        size_t idx;
        Generator(shared_ptr<const RectangularT> src): src(src), idx(0) {}
        Generator(const Generator& orig): src(orig.src), idx(0) {}
        std::unique_ptr<typename LevelMeshAdapter<2>::GeneratorBase> copy() const override {
            return std::unique_ptr<typename LevelMeshAdapter<2>::GeneratorBase>(new Generator(*this));
        }
        shared_ptr<LevelMeshAdapter<2>> yield() override {
            if (idx == src->axis1->size()) return shared_ptr<LevelMeshAdapter<2>>();
            return shared_ptr<LevelMeshAdapter<2>>(new LevelMeshAdapterRectangular<2>(src, idx++));
        }
        shared_ptr<LevelMeshAdapter<2>> at(double pos) const override {
            size_t i = src->axis1->findIndex(pos);
            assert (i < src->axis1->size() && src->axis1->at(i) == pos);
            return shared_ptr<LevelMeshAdapter<2>>(new LevelMeshAdapterRectangular<2>(src, i));
        }
    };

    /// Create mesh adapter
    LevelMeshAdapterRectangular(shared_ptr<const RectangularT> src, size_t vert): src(src), vert(vert) {}

    virtual std::size_t size() const override { return src->axis0->size(); }

    virtual plask::Vec<2> at(std::size_t i) const override { return src->at(i, vert); }

    virtual size_t index(size_t i) const override { return src->index(i, vert); }

    virtual double vpos() const override { return src->axis1->at(vert); }
};

/// More efficient Rectangular3D implementation of the level adapter
template<>
struct LevelMeshAdapterRectangular<3>: public LevelMeshAdapter<3>
{
    typedef RectangularMesh<3> RectangularT;

  protected:
    /// Original mesh
    shared_ptr<const RectangularT> src;

    /// Interesting level
    size_t vert;

  public:
    /// Generator for the generic levels
    struct Generator: public LevelMeshAdapter<3>::GeneratorBase {
        shared_ptr<const RectangularT> src;
        size_t idx;
        Generator(shared_ptr<const RectangularT> src): src(src), idx(0) {}
        Generator(const Generator& orig): src(orig.src), idx(0) {}
        std::unique_ptr<typename LevelMeshAdapter<3>::GeneratorBase> copy() const override {
            return std::unique_ptr<typename LevelMeshAdapter<3>::GeneratorBase>(new Generator(*this));
        }
        shared_ptr<LevelMeshAdapter<3>> yield() override {
            if (idx == src->axis2->size()) return shared_ptr<LevelMeshAdapter<3>>();
            return shared_ptr<LevelMeshAdapter<3>>(new LevelMeshAdapterRectangular<3>(src, idx++));
        }
        shared_ptr<LevelMeshAdapter<3>> at(double pos) const override {
            size_t i = src->axis2->findIndex(pos);
            assert (i < src->axis2->size() && src->axis2->at(i) == pos);
            return shared_ptr<LevelMeshAdapter<3>>(new LevelMeshAdapterRectangular<3>(src, i));
        }
    };

    /// Create mesh adapter
    LevelMeshAdapterRectangular(shared_ptr<const RectangularT> src, size_t vert): src(src), vert(vert) {}

    virtual std::size_t size() const override { return src->axis0->size() * src->axis1->size(); }

    virtual plask::Vec<3> at(std::size_t i) const override { return src->at(i % src->axis0->size(), i / src->axis0->size(), vert); }

    virtual size_t index(size_t i) const override { return src->index(i % src->axis0->size(), i / src->axis0->size(), vert); }

    virtual double vpos() const override { return src->axis2->at(vert); }
};


template <int dim>
std::unique_ptr<typename LevelMeshAdapter<dim>::GeneratorBase> LevelsGenerator(const shared_ptr<const MeshD<dim>>& src)
{
    typedef std::unique_ptr<typename LevelMeshAdapter<dim>::GeneratorBase> ReturnT;

    if (auto mesh = dynamic_pointer_cast<const RectangularMesh<dim>>(src))
        return ReturnT(new typename LevelMeshAdapterRectangular<dim>::Generator(mesh));
    return ReturnT(new typename LevelMeshAdapterGeneric<dim>::Generator(src));
}

/**
 * Lazy data implementation for leveled data
 */
template <int dim, typename T>
struct LevelLazyDataImpl: public LazyDataImpl<T> {

    shared_ptr<const MeshD<dim>> mesh;
    std::unique_ptr<typename LevelMeshAdapter<dim>::GeneratorBase> levels;

    std::function<LazyData<T>(const shared_ptr<LevelMeshAdapter<dim>>&)> func;

    LevelLazyDataImpl(const std::function<LazyData<T>(const shared_ptr<LevelMeshAdapter<dim>>&)>& func, const shared_ptr<const MeshD<dim>>& mesh):
        func(func), mesh(mesh), levels(LevelsGenerator<dim>(mesh))
    {
    }

    size_t size() const override {
        return mesh->size();
    }

    T at(std::size_t idx) const override {
        // This method is rather inefficient
        auto level = levels->at(mesh[idx][dim-1]);
        for (size_t i = 0, end = level->size(); i != end; ++i)
            if (level->index(i) == idx) return func(level)[i];
        assert(false);
    }

    DataVector<const T> getAll() const override {
        DataVector<T> result(mesh->size());
        auto local_levels = levels.copy();
        while (auto level = local_levels->yield()) {
            auto data = func(level);
            for (size_t i = 0; i != level->size(); ++i) result[level->index(i)] = data[i];
        }
        return result;
    }
};






}}} // namespace plask::solvers::slab

#endif // PLASK__SOLVER_SLAB_MESHADAPTER_H
