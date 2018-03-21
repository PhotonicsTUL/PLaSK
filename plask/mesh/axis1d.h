#ifndef PLASK__AXIS1D_H
#define PLASK__AXIS1D_H

#include "mesh.h"
#include "interpolation.h"

#include <memory>

namespace plask {

/**
 * Abstract class to derive all mesh axes from.
 */
class PLASK_API MeshAxis: public MeshD<1> {

public:

    /**
     * Create new mesh wich has copy of data included in @c this.
     *
     * By default RectangularMesh is used for result, but subclasses can use another types for less memory usage or better performence.
     * @return new mesh wich has copy of data included in @c this
     */
    virtual shared_ptr<MeshAxis> clone() const;

    /**
     * Find index where @p to_find point could be inserted.
     * @param to_find point to find
     * @return First index where to_find could be inserted.
     *         Refer to value equal to @p to_find only if @p to_find is already in mesh, in other case it refer to value larger than to_find.
     *         Can be equal to size() if to_find is larger than all points in mesh.
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
    virtual shared_ptr<MeshAxis> getMidpointsMesh() const;

    /**
     * @return @c true only if points are in increasing order, @c false if points are in decreasing order
     */
    virtual bool isIncreasing() const = 0;

protected:
    /// Throw exception if this mesh has less than two point.
    void beforeCalcMidpointMesh() const;
};

/**
 * TODO doc - this is code by M. Dems copied from interpolation methods
 * @param axis
 * @param flags
 * @param wrapped_point_coord
 * @param axis_nr
 * @param index
 * @param index_1
 * @param lo
 * @param hi
 * @param invert_lo
 * @param invert_hi
 */
void prepareLinearInterpolationForAxis(MeshAxis& axis, const InterpolationFlags& flags, double wrapped_point_coord, int axis_nr, std::size_t& index, std::size_t& index_1, double& lo, double& hi, bool& invert_lo, bool& invert_hi);



/**
 * A trivial axis that contains only one point.
 */
struct OnePointAxis: MeshAxis {

    double value;

    /**
     * Create the trivial axis.
     * \param val value of the single point in the axis
     */
    OnePointAxis(double val): value(val) {}

    std::size_t size() const override { return 1; }

    double at(std::size_t) const override { return value; }

    bool isIncreasing() const override { return true; }
};


}   // namespace plask

#endif // PLASK__AXIS1D_H
