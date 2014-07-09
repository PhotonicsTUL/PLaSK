#ifndef PLASK__SOLVER_SLAB_MESHADAPTER_H
#define PLASK__SOLVER_SLAB_MESHADAPTER_H

#include <boost/iterator/filter_iterator.hpp>

#include <plask/mesh/mesh.h>

#include "matrices.h"

namespace plask { namespace solvers { namespace slab {


/// Simple adapter that allows to process single level in the mesh
template <int dim>
struct LevelsAdapter
{
    struct Mesh: public MeshD<dim> {
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

    virtual shared_ptr<Mesh> yield() = 0;
};


/// Generic implementation of the level adapter
template <int dim>
struct LevelsAdapterGeneric: public LevelsAdapter<dim>
{
    struct Level: public LevelsAdapter<dim>::Mesh {
      protected:
        std::vector<size_t> matching;       ///< Indices of matching points
        shared_ptr<const MeshD<dim>> src;   ///< Original mesh
        double vert;                        ///< Interesting level
      public:
        /// Create mesh adapter
        Level(shared_ptr<const MeshD<dim>> src, double level): src(src), vert(level) {
            for (auto it = src->begin(); it != src->end(); ++it) {
                if ((*it)[dim-1] == level) matching.push_back(it.index);
            }
        }
        // Overrides
        std::size_t size() const override { return matching.size(); }
        plask::Vec<dim> at(std::size_t i) const override { return (*src)[matching[i]]; }
        size_t index(size_t i) const override { return matching[i]; }
        double vpos() const override { return vert; }
    };

    /// Original mesh
    shared_ptr<const MeshD<dim>> src;

    /// Set of detected levels
    std::set<double> levels;

    /// Iterator over levels
    std::set<double>::iterator iter;

    LevelsAdapterGeneric(shared_ptr<const MeshD<dim>> src): src(src) {
        for (auto point: *src) {
            levels.insert(point[dim-1]);
        }
        iter = levels.begin();
    }

    shared_ptr<typename LevelsAdapter<dim>::Mesh> yield() override {
        if (iter == levels.end()) return shared_ptr<typename LevelsAdapter<dim>::Mesh>();
        return make_shared<Level>(src, *(iter++));
    }
};

/// More efficient Rectangular implementation of the level adapter
template <int dim>
struct LevelsAdapterRectangular: public LevelsAdapter<dim>
{
    struct Level: public LevelsAdapter<dim>::Mesh {
      protected:
        shared_ptr<const RectangularMesh<dim>> src; ///< Original mesh
        size_t vert;                        ///< Interesting level
      public:
        /// Create mesh adapter
        Level(shared_ptr<const RectangularMesh<dim>> src, size_t vert): src(src), vert(vert) {}
        // Overrides
        std::size_t size() const override;
        plask::Vec<dim> at(std::size_t i) const override;
        size_t index(size_t i) const override;
        double vpos() const override;
    };

    /// Original mesh
    shared_ptr<const RectangularMesh<dim>> src;

    /// Index of a current level
    size_t idx;

    LevelsAdapterRectangular(shared_ptr<const RectangularMesh<dim>> src): src(src), idx(0) {}

    shared_ptr<typename LevelsAdapter<dim>::Mesh> yield() override;
};


/**
 * Adapter factory. Choose the best class based on the mesh type
 * \param src source mesh
 */
template <int dim>
std::unique_ptr<LevelsAdapter<dim>> makeLevelsAdapter(const shared_ptr<const MeshD<dim>>& src)
{
    typedef std::unique_ptr<LevelsAdapter<dim>> ReturnT;

    if (auto mesh = dynamic_pointer_cast<const RectangularMesh<dim>>(src))
        return ReturnT(new LevelsAdapterRectangular<dim>(mesh));
    return ReturnT(new LevelsAdapterGeneric<dim>(src));
}


}}} // namespace plask::solvers::slab

#endif // PLASK__SOLVER_SLAB_MESHADAPTER_H
