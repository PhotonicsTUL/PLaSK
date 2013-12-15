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
        /** Yield pointers to the consecutive levels
         * \return pointer to the new level object or \c nullptr at the end
         */
        virtual std::unique_ptr<LevelMeshAdapter<dim>> yield() = 0;
    };

    /**
     * Unscramble indices
     * \param i index in the adapter
     * \return index in the original mesh
     */
    virtual size_t index(size_t i) const = 0;

    /// Get level vertical position
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
    const MeshD<dim>* src;

    /// Interesting level
    double vert;

  public:
    /// Generator for the generic levels
    struct Generator: public LevelMeshAdapter<dim>::GeneratorBase {
        const MeshD<dim>* src;
        std::set<double> levels;
        std::set<double>::iterator iter;
        Generator(const MeshD<dim>* src): src(src) {
            for (auto point: *src) {
                levels.insert(point[dim-1]);
            }
            iter = levels.begin();
        }
        virtual std::unique_ptr<LevelMeshAdapter<dim>> yield() override {
            if (iter == levels.end()) return std::unique_ptr<LevelMeshAdapter<dim>>();
            return std::unique_ptr<LevelMeshAdapter<dim>>(new LevelMeshAdapterGeneric<dim>(src, *(iter++)));
        }
    };

    /// Create mesh adapter
    LevelMeshAdapterGeneric(const MeshD<dim>* src, double level): src(src), vert(level) {
        for (auto it = src->begin(); it != src->end(); ++it) {
            if ((*it)[dim-1] == level) matching.push_back(it.index);
        }
    }

    virtual std::size_t size() const override { return matching.size(); }

    virtual plask::Vec<dim> at(std::size_t i) const override { return (*src)[matching[i]]; }

    virtual size_t index(size_t i) const override { return matching[i]; }

    virtual double vpos() const override { return vert; }
};


template <int dim, typename AxisT> struct LevelMeshAdapterRectangular;

/// More efficient Rectangular2D implementation of the level adapter
template <typename AxisT>
struct LevelMeshAdapterRectangular<2,AxisT>: public LevelMeshAdapter<2>
{
    typedef RectangularMesh<2,AxisT> RectangularT;

  protected:
    /// Original mesh
    const RectangularT* src;

    /// Interesting level
    size_t vert;

  public:
    /// Generator for the generic levels
    struct Generator: public LevelMeshAdapter<2>::GeneratorBase {
        const RectangularT* src;
        size_t idx;
        Generator(const RectangularT* src): src(src), idx(0) {}
        virtual std::unique_ptr<LevelMeshAdapter<2>> yield() override {
            if (idx == src->axis1.size()) return std::unique_ptr<LevelMeshAdapter<2>>();
            return std::unique_ptr<LevelMeshAdapter<2>>(new LevelMeshAdapterRectangular<2,AxisT>(src, idx++));
        }
    };

    /// Create mesh adapter
    LevelMeshAdapterRectangular(const RectangularT* src, size_t vert): src(src), vert(vert) {}

    virtual std::size_t size() const override { return src->axis0.size(); }

    virtual plask::Vec<2> at(std::size_t i) const override { return src->at(i, vert); }

    virtual size_t index(size_t i) const override { return src->index(i, vert); }

    virtual double vpos() const override { return src->axis1[vert]; }
};

/// More efficient Rectangular3D implementation of the level adapter
template <typename AxisT>
struct LevelMeshAdapterRectangular<3,AxisT>: public LevelMeshAdapter<3>
{
    typedef RectangularMesh<3,AxisT> RectangularT;

  protected:
    /// Original mesh
    const RectangularT* src;

    /// Interesting level
    size_t vert;

  public:
    /// Generator for the generic levels
    struct Generator: public LevelMeshAdapter<3>::GeneratorBase {
        const RectangularT* src;
        size_t idx;
        Generator(const RectangularT* src): src(src), idx(0) {}
        virtual std::unique_ptr<LevelMeshAdapter<3>> yield() override {
            if (idx == src->axis2.size()) return std::unique_ptr<LevelMeshAdapter<3>>();
            return std::unique_ptr<LevelMeshAdapter<3>>(new LevelMeshAdapterRectangular<3,AxisT>(src, idx++));
        }
    };

    /// Create mesh adapter
    LevelMeshAdapterRectangular(const RectangularT* src, size_t vert): src(src), vert(vert) {}

    virtual std::size_t size() const override { return src->axis0.size() * src->axis1.size(); }

    virtual plask::Vec<3> at(std::size_t i) const override { return src->at(i % src->axis0.size(), i / src->axis0.size(), vert); }

    virtual size_t index(size_t i) const override { return src->index(i % src->axis0.size(), i / src->axis0.size(), vert); }

    virtual double vpos() const override { return src->axis2[vert]; }
};


template <int dim>
std::unique_ptr<typename LevelMeshAdapter<dim>::GeneratorBase> LevelsGenerator(const MeshD<dim>& src)
{
    typedef std::unique_ptr<typename LevelMeshAdapter<dim>::GeneratorBase> ReturnT;
    typedef RectangularMesh<dim,RectilinearAxis> RectilinearMesh;
    typedef RectangularMesh<dim,RegularAxis> RegularMesh;

    if (auto mesh = dynamic_cast<const RegularMesh*>(&src))
        return ReturnT(new typename LevelMeshAdapterRectangular<dim,RegularAxis>::Generator(mesh));
    if (auto mesh = dynamic_cast<const RectilinearMesh*>(&src))
        return ReturnT(new typename LevelMeshAdapterRectangular<dim,RectilinearAxis>::Generator(mesh));
    return ReturnT(new typename LevelMeshAdapterGeneric<dim>::Generator(&src));
}


}}} // namespace plask::solvers::slab

#endif // PLASK__SOLVER_SLAB_MESHADAPTER_H
