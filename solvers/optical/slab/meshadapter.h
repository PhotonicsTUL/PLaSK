#ifndef PLASK__SOLVER_SLAB_MESHADAPTER_H
#define PLASK__SOLVER_SLAB_MESHADAPTER_H

#include <boost/iterator/filter_iterator.hpp>

#include <plask/mesh/mesh.h>

#include "matrices.h"

namespace plask { namespace optical { namespace slab {


/// Simple adapter that allows to process single level in the mesh
struct PLASK_SOLVER_API LevelsAdapter
{
    struct Level {
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

        /**
         * Get mesh size
         */
        virtual size_t size() const = 0;

        /**
         * Get the mesh for this level
         */
        virtual shared_ptr<const Mesh> mesh() const = 0;
    };

    virtual shared_ptr<Level> yield() = 0;
};


/// Generic implementation of the level adapter
template <int dim>
struct PLASK_SOLVER_API LevelsAdapterGeneric: public LevelsAdapter
{
    struct GenericLevel;

    struct Mesh: public MeshD<dim> {
      protected:
        const GenericLevel* level;
      public:
        Mesh(const GenericLevel* level): level(level) {}
        // Overrides
        std::size_t size() const override { return level->matching.size(); }
        plask::Vec<dim> at(std::size_t i) const override { return (*level->src)[level->matching[i]]; }
    };

    struct GenericLevel: public LevelsAdapter::Level {
      protected:
        std::vector<size_t> matching;       ///< Indices of matching points
        shared_ptr<const MeshD<dim>> src;   ///< Original mesh
        double vert;                        ///< Interesting level
        friend struct LevelsAdapterGeneric<dim>::Mesh;
      public:
        GenericLevel(shared_ptr<const MeshD<dim>> src, double level): src(src), vert(level) {
            for (auto it = src->begin(); it != src->end(); ++it) {
                if ((*it)[dim-1] == level) matching.push_back(it.index);
            }
        }
        // Overrides
        size_t index(size_t i) const override;
        double vpos() const override;
        std::size_t size() const override { return matching.size(); }
        shared_ptr<const plask::Mesh> mesh() const override { return plask::make_shared<const Mesh>(this); }
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

    shared_ptr<typename LevelsAdapter::Level> yield() override {
        if (iter == levels.end()) return shared_ptr<typename LevelsAdapter::Level>();
        return plask::make_shared<GenericLevel>(src, *(iter++));
    }
};

/// More efficient Rectangular implementation of the level adapter
template <int dim>
struct PLASK_SOLVER_API LevelsAdapterRectangular: public LevelsAdapter
{
    struct RectangularLevel;

    struct Mesh: public MeshD<dim> {
      protected:
        const RectangularLevel* level;
      public:
        /// Create mesh adapter
        Mesh(const RectangularLevel* level): level(level) {}
        // Overrides
        std::size_t size() const override;
        plask::Vec<dim> at(std::size_t i) const override;
    };

    struct RectangularLevel: public LevelsAdapter::Level {
      protected:
        shared_ptr<const RectangularMesh<dim>> src; ///< Original mesh
        size_t vert;                        ///< Interesting level
        friend struct LevelsAdapterRectangular<dim>::Mesh;
      public:
        /// Create mesh adapter
        RectangularLevel(shared_ptr<const RectangularMesh<dim>> src, size_t vert): src(src), vert(vert) {}
        // Overrides
        size_t index(size_t i) const override;
        double vpos() const override;
        std::size_t size() const override;
        shared_ptr<const plask::Mesh> mesh() const override { return plask::make_shared<const Mesh>(this); }
    };

    /// Original mesh
    shared_ptr<const RectangularMesh<dim>> src;

    /// Index of a current level
    size_t idx;

    LevelsAdapterRectangular(shared_ptr<const RectangularMesh<dim>> src): src(src), idx(0) {}

    shared_ptr<typename LevelsAdapter::Level> yield() override;
};


/**
 * Adapter factory. Choose the best class based on the mesh type
 * \param src source mesh
 */
std::unique_ptr<LevelsAdapter> PLASK_SOLVER_API makeLevelsAdapter(const shared_ptr<const Mesh>& src);


}}} // namespace plask::optical::slab

#endif // PLASK__SOLVER_SLAB_MESHADAPTER_H
