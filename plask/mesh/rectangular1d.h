#ifndef PLASK__RECTANGULAR1D_H
#define PLASK__RECTANGULAR1D_H

#include "mesh.h"

#include "memory"

namespace plask {

template<>
struct RectangularMesh<1>: public MeshD<1> {

    virtual shared_ptr<RectangularMesh<1>> clone() const = 0;

    virtual void clear() = 0;

    virtual std::size_t findIndex(double to_find) const = 0;

    /**
     * Find index nearest to @p to_find.
     * @param to_find
     * @return index i for which abs((*this)[i]-to_find) is minimal
     */
    virtual std::size_t findNearestIndex(double to_find) const = 0;
};

typedef RectangularMesh<1> RectangularAxis;



}   // namespace plask

#endif // PLASK__RECTANGULAR1D_H
