#ifndef PLASK__SOLVER_SLAB_MESHADAPTER_H
#define PLASK__SOLVER_SLAB_MESHADAPTER_H

#include <boost/iterator/filter_iterator.hpp>

#include <plask/mesh/mesh.h>

#include "matrices.h"

namespace plask { namespace solvers { namespace slab {

/// Simple adapter that allows to process single level in the mesh

template <int dim> struct LevelMeshAdapter: public MeshD<dim>
{
    /**
     * Unscramble indices
     * \param i index in the adapter
     * \return index in the original mesh
     */
    virtual size_t index(size_t i) const = 0;
    
    /// Get level value
    virtual double level() const = 0;
};
    
/// Generic implementation of the level adapter
template <int dim>
struct LevelMeshAdapterGeneric: public LevelMeshAdapter<dim>
{
    /// Original mesh
    const MeshD<dim>* src;
    
    /// Interesting level
    double vert;

    /// Indices of matching points
    std::vector<size_t> matching;
    
    /// Create mesh adapter
    LevelMeshAdapterGeneric(const MeshD<dim>& src, double level): src(&src), vert(level) {
        for (auto it = src.begin(); it != src.end(); ++it) {
            if ((*it)[dim-1] == level) matching.push_back(it.index);
        }
    }
    
    virtual std::size_t size() const override { return matching.size(); }
    
    virtual plask::Vec<dim> at(std::size_t i) const override { return src[matching[i]]; }

    virtual size_t index(size_t i) const override { return matching[i]; }
    
    virtual double level() const override { return vert; }
};

template <int dim, typename AxisT> struct LevelMeshAdapterRectangular;

/// More efficient Rectangular2D implementation of the level adapter
template <typename AxisT>
struct LevelMeshAdapterRectangular<2,AxisT>: public LevelMeshAdapter<2>
{
    typedef RectangularMesh<2,AxisT> RectangularT;
    
    /// Original mesh
    const RectangularT* src;
    
    /// Interesting level
    size_t vert;

    /// Create mesh adapter
    LevelMeshAdapterRectangular(const RectangularT& src, size_t vert): src(&src), vert(vert) {}
    
    virtual std::size_t size() const override { return src->axis0.size(); }
    
    virtual plask::Vec<2> at(std::size_t i) const override { return src->at(i, vert); }

    virtual size_t index(size_t i) const override { return src->index(i, vert); }
};

/// More efficient Rectangular3D implementation of the level adapter
template <typename AxisT>
struct LevelMeshAdapterRectangular<3,AxisT>: public LevelMeshAdapter<3>
{
    typedef RectangularMesh<3,AxisT> RectangularT;
    
    /// Original mesh
    const RectangularT* src;
    
    /// Interesting level
    size_t vert;

    /// Create mesh adapter
    LevelMeshAdapterRectangular(const RectangularT& src, size_t vert): src(&src), vert(vert) {}
    
    virtual std::size_t size() const override { return src->axis0.size() * src->axis1.size(); }
    
    virtual plask::Vec<3> at(std::size_t i) const override { return src->at(i % src->axis0.size(), i / src->axis0.size(), vert); }

    virtual size_t index(size_t i) const override { return src->index(i % src->axis0.size(), i / src->axis0.size(), vert); }
};

// Generator of level adapters
template <int dim>
struct LevelGenerator {
    
    typedef RectangularMesh<dim,RectilinearAxis> RectilinearMesh;
    typedef RectangularMesh<dim,RegularAxis> RegularMesh;
    
    MeshD<dim>* src;
    
    LevelGenerator(const MeshD<dim>& src): src(&src) {
//         if (!rectilinear = dynamic_cast<RectilinearMesh*>(src))
//              regular = dynamic_cast<RegularMesh*>(src);
    }

    
};

}}} // namespace plask::solvers::slab

#endif // PLASK__SOLVER_SLAB_MESHADAPTER_H