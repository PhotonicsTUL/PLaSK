#ifndef PLASK__RECTANGULAR1D_H
#define PLASK__RECTANGULAR1D_H

#include "mesh.h"

#include "memory"

namespace plask {

struct RectangularAxis: public MeshD<1> {

    virtual std::unique_ptr<RectangularAxis> clone() const = 0;

    virtual void clear() = 0;

    virtual std::size_t findIndex(double to_find) const = 0;

    /**
     * Find index nearest to @p to_find.
     * @param to_find
     * @return index i for which abs((*this)[i]-to_find) is minimal
     */
    virtual std::size_t findNearestIndex(double to_find) const = 0;
};



}   // namespace plask

#endif // PLASK__RECTANGULAR1D_H
