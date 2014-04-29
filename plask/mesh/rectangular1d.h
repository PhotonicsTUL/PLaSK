#ifndef PLASK__RECTANGULAR1D_H
#define PLASK__RECTANGULAR1D_H

#include "mesh.h"

#include "memory"

namespace plask {

template<>
struct RectangularMesh<1>: public MeshD<1> {

    virtual shared_ptr<RectangularMesh<1>> clone() const = 0;

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


//TODO enable_shared_from_this for Mesh (for getMidpointsMesh impl. and change to shared_ptr)
class MidpointsMesh: public RectangularMesh<1> {

    //shared_ptr<RectangularMesh<1>> wrapped;
    const RectangularMesh<1>& wrapped;

public:

    //MidpointsMesh(shared_ptr<const RectangularMesh<1>> wrapped = nullptr): wrapped(nullptr) { setWrapped(wrapped); }
    MidpointsMesh(const RectangularMesh<1>& wrapped): wrapped(wrapped) { }

    //shared_ptr<const RectangularMesh<1> > getWrapped() const;

    //void setWrapped(shared_ptr<const RectangularMesh<1> > wrapped);

    virtual shared_ptr<RectangularMesh<1>> clone() const override;

    //virtual void clear() override { setWrapped(nullptr); }

    virtual std::size_t size() const override;

    double at(std::size_t index) const override;

    bool isIncreasing() const override;
};



}   // namespace plask

#endif // PLASK__RECTANGULAR1D_H
