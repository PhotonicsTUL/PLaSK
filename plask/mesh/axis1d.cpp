#include "axis1d.h"
#include "ordered1d.h"

#include "../utils/stl.h"

namespace plask {


//enable_shared_from_this for Mesh (for getMidpointsMesh impl. and change to shared_ptr)    ???
class MidpointsMesh: public MeshAxis {

    //shared_ptr<MeshAxis> wrapped;
    const MeshAxis& wrapped;

public:

    //MidpointsMesh(shared_ptr<const MeshAxis> wrapped = nullptr): wrapped(nullptr) { setWrapped(wrapped); }
    MidpointsMesh(const MeshAxis& wrapped): wrapped(wrapped) { }

    //shared_ptr<const MeshAxis > getWrapped() const;

    //void setWrapped(shared_ptr<const MeshAxis > wrapped);

    //virtual void clear() override { setWrapped(nullptr); }

    virtual std::size_t size() const override;

    double at(std::size_t index) const override;

    bool isIncreasing() const override;
};

/*shared_ptr<MeshAxis> MidpointsMesh::getWrapped() const {
    return wrapped;
}

void MidpointsMesh::setWrapped(shared_ptr<MeshAxis> value) {
    wrapped = value;
}

shared_ptr<MeshAxis> MidpointsMesh::clone() const {
    return plask::make_shared<MidpointMesh>(wrapped->clone());
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


// -------------- MeshAxis ---------------------------------------------

shared_ptr<MeshAxis> MeshAxis::clone() const {
    //return plask::make_shared<MidpointsMesh>(wrapped);
    return plask::make_shared<OrderedAxis>(*this);
}

std::size_t MeshAxis::findIndex(double to_find) const {
    return std::lower_bound(begin(), end(), to_find).index;
}

std::size_t MeshAxis::findNearestIndex(double to_find) const {
    return find_nearest_binary(begin(), end(), to_find).index;
}

shared_ptr<MeshAxis> MeshAxis::getMidpointsMesh() const {
    beforeCalcMidpointMesh();
    /*const std::size_t s = this->size();
    if (s == 0) return this->clone();
    auto result = plask::make_shared<OrderedAxis>();*/
    return plask::make_shared<MidpointsMesh>(*this)->clone();
}

void MeshAxis::beforeCalcMidpointMesh() const {
    if (this->size() < 2)
        throw BadMesh("getMidpointsMesh", "at least two points are required");
}

}   // namespace plask
