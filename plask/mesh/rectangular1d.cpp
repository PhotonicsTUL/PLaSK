#include "rectangular1d.h"
#include "ordered1d.h"

#include "../utils/stl.h"

namespace plask {


//enable_shared_from_this for Mesh (for getMidpointsMesh impl. and change to shared_ptr)    ???
class MidpointsMesh: public RectangularMesh<1> {

    //shared_ptr<RectangularMesh<1>> wrapped;
    const RectangularMesh<1>& wrapped;

public:

    //MidpointsMesh(shared_ptr<const RectangularMesh<1>> wrapped = nullptr): wrapped(nullptr) { setWrapped(wrapped); }
    MidpointsMesh(const RectangularMesh<1>& wrapped): wrapped(wrapped) { }

    //shared_ptr<const RectangularMesh<1> > getWrapped() const;

    //void setWrapped(shared_ptr<const RectangularMesh<1> > wrapped);

    //virtual void clear() override { setWrapped(nullptr); }

    virtual std::size_t size() const override;

    double at(std::size_t index) const override;

    bool isIncreasing() const override;
};

/*shared_ptr<RectangularMesh<1> > MidpointsMesh::getWrapped() const {
    return wrapped;
}

void MidpointsMesh::setWrapped(shared_ptr<RectangularMesh<1> > value) {
    wrapped = value;
}

shared_ptr<RectangularMesh<1> > MidpointsMesh::clone() const {
    return make_shared<MidpointMesh>(wrapped->clone());
}

std::size_t MidpointsMesh::size() const {
    if (!wrapped) return 0;
    std::size_t wrapped_size = wrapped->size();
    return wrapped_size ? wrapped_size - 1 : 0;
}

double MidpointsMesh::at(std::size_t index) const {
    return (wrapped->at(index) + wrapped->at(index+1)) * 0.5;
}*/

std::size_t MidpointsMesh::size() const {
    //if (!wrapped) return 0;
    std::size_t wrapped_size = wrapped.size();
    return wrapped_size ? wrapped_size - 1 : 0;
}

double MidpointsMesh::at(std::size_t index) const {
    return (wrapped.at(index) + wrapped.at(index+1)) * 0.5;
}

bool MidpointsMesh::isIncreasing() const {
    return wrapped.isIncreasing();
}


// -------------- RectangularMesh<1> ---------------------------------------------

shared_ptr<RectangularMesh<1> > RectangularMesh<1>::clone() const {
    //return make_shared<MidpointsMesh>(wrapped);
    return make_shared<OrderedAxis>(*this);
}

std::size_t RectangularMesh<1>::findIndex(double to_find) const {
    return std::lower_bound(begin(), end(), to_find).index;
}

std::size_t RectangularMesh<1>::findNearestIndex(double to_find) const {
    return find_nearest_binary(begin(), end(), to_find).index;
}

shared_ptr<RectangularMesh<1> > RectangularMesh<1>::getMidpointsMesh() const {
    beforeCalcMidpointMesh();
    /*const std::size_t s = this->size();
    if (s == 0) return this->clone();
    auto result = make_shared<OrderedAxis>();*/
    return make_shared<MidpointsMesh>(*this)->clone();
}

void RectangularMesh<1>::beforeCalcMidpointMesh() const {
    if (this->size() < 2)
        throw BadMesh("getMidpointsMesh", "at least two points are required");
}

template struct PLASK_API RectangularMesh<1>;

}   // namespace plask
