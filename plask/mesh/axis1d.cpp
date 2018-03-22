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

std::size_t MeshAxis::findUpIndex(double to_find) const {
    return std::upper_bound(begin(), end(), to_find).index;
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

void prepareNearestNeighborInterpolationForAxis(const MeshAxis& axis, const InterpolationFlags& flags, double& wrapped_point_coord, int axis_nr) {
    if (flags.periodic(axis_nr) && !flags.symmetric(axis_nr)) {
        if (wrapped_point_coord < axis.at(0)) {
            if (axis.at(0) - wrapped_point_coord > wrapped_point_coord - flags.low(axis_nr) + flags.high(axis_nr) - axis.at(axis.size()-1)) wrapped_point_coord = axis.at(axis.size()-1);
        } else if (wrapped_point_coord > axis.at(axis.size()-1)) {
            if (wrapped_point_coord - axis.at(axis.size()-1) > flags.high(axis_nr) - wrapped_point_coord + axis.at(0) - flags.low(axis_nr)) wrapped_point_coord = axis.at(0);
        }
    }
}

void prepareLinearInterpolationForAxis(const MeshAxis& axis, const InterpolationFlags& flags, double wrapped_point_coord, int axis_nr, std::size_t& index, std::size_t& index_1, double& lo, double& hi, bool& invert_lo, bool& invert_hi) {
    index = axis.findUpIndex(wrapped_point_coord);
    invert_lo = false; invert_hi = false;
    if (index == 0) {
        if (flags.symmetric(axis_nr)) {
            index_1 = 0;
            lo = axis.at(0);
            if (lo > 0.) {
                lo = - lo;
                invert_lo = true;
            } else if (flags.periodic(axis_nr)) {
                lo = 2. * flags.low(axis_nr) - lo;
                invert_lo = true;
            } else {
                lo -= 1.;
            }
        } else if (flags.periodic(axis_nr)) {
            index_1 = axis.size() - 1;
            lo = axis.at(index_1) - flags.high(axis_nr) + flags.low(axis_nr);
        } else {
            index_1 = 0;
            lo = axis.at(0) - 1.;
        }
    } else {
        index_1 = index - 1;
        lo = axis.at(index_1);
    }
    if (index == axis.size()) {
        if (flags.symmetric(axis_nr)) {
            --index;
            hi = axis.at(index);
            if (hi < 0.) {
                hi = - hi;
                invert_hi = true;
            } else if (flags.periodic(axis_nr)) {
                lo = 2. * flags.high(axis_nr) - hi;
                invert_hi = true;
            } else {
                hi += 1.;
            }
        } else if (flags.periodic(axis_nr)) {
            index = 0;
            hi = axis.at(0) + flags.high(axis_nr) - flags.low(axis_nr);
            if (hi == lo) hi += 1e-6;
        } else {
            --index;
            hi = axis.at(index) + 1.;
        }
    } else {
        hi = axis.at(index);
    }
}

}   // namespace plask
