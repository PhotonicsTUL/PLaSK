#ifndef PLASK__RECTANGULAR1D_H
#define PLASK__RECTANGULAR1D_H

#include "mesh.h"

#include "memory"

namespace plask {

template<>
struct RectangularMesh<1>: public MeshD<1> {

    /**
     * Create new mesh wich has copy of data included in @c this.
     *
     * By default RectangularMesh is used for result, but subclasses can use another types for less memory usage or better performence.
     * @return new mesh wich has copy of data included in @c this
     */
    virtual shared_ptr<RectangularMesh<1>> clone() const;

    //virtual void clear() = 0;

    virtual std::size_t findIndex(double to_find) const;

    /**
     * Find index nearest to @p to_find.
     * @param to_find
     * @return index i for which abs((*this)[i]-to_find) is minimal
     */
    virtual std::size_t findNearestIndex(double to_find) const;

    /**
     * Return a mesh that enables iterating over middle points of the ranges
     * \return new rectilinear mesh with points in the middles of original ranges
     */
    virtual shared_ptr<RectangularMesh<1>> getMidpointsMesh() const;

    /**
     * @return @c true only if points are in increasing order, @c false if points are in decreasing order
     */
    virtual bool isIncreasing() const = 0;
};

typedef RectangularMesh<1> RectangularAxis;

}   // namespace plask

#endif // PLASK__RECTANGULAR1D_H
