#ifndef PLASK__RECTANGULAR1D_H
#define PLASK__RECTANGULAR1D_H

#include "mesh.h"

#include "memory"

namespace plask {

template<>
class PLASK_API RectangularMesh<1>: public MeshD<1> {

public:

    /**
     * Create new mesh wich has copy of data included in @c this.
     *
     * By default RectangularMesh is used for result, but subclasses can use another types for less memory usage or better performence.
     * @return new mesh wich has copy of data included in @c this
     */
    virtual shared_ptr<RectangularMesh<1>> clone() const;

    /**
     * Find index where @p to_find point could be inserted.
     * @param to_find point to find
     * @return First index where to_find could be inserted.
     *         Refer to value equal to @p to_find only if @p to_find is already in mesh, in other case it refer to value bigger than to_find.
     *         Can be equal to size() if to_find is higher than all points in mesh.
     */
    virtual std::size_t findIndex(double to_find) const;

    /**
     * Find index nearest to @p to_find.
     * @param to_find
     * @return index i for which abs((*this)[i]-to_find) is minimal
     */
    virtual std::size_t findNearestIndex(double to_find) const;

    /**
     * Return a mesh that enables iterating over middle points of the ranges.
     *
     * Throw exception if this mesh has less than two points.
     * \return new rectilinear mesh with points in the middles of original ranges
     */
    virtual shared_ptr<RectangularMesh<1>> getMidpointsMesh() const;

    /**
     * @return @c true only if points are in increasing order, @c false if points are in decreasing order
     */
    virtual bool isIncreasing() const = 0;

protected:
    /// Throw exception if this mesh has less than two point.
    void beforeCalcMidpointMesh() const;
};

typedef RectangularMesh<1> RectangularAxis;

PLASK_API_EXTERN_TEMPLATE_CLASS(RectangularMesh<1>)

}   // namespace plask

#endif // PLASK__RECTANGULAR1D_H
