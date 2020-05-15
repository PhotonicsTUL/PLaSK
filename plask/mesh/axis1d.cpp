#include "axis1d.h"
#include "ordered1d.h"

#include "../utils/stl.h"

namespace plask {


//enable_shared_from_this for Mesh (for getMidpointAxis impl. and change to shared_ptr)    ???
class MidpointAxis: public MeshAxis {

    //shared_ptr<MeshAxis> wrapped;
    const MeshAxis& wrapped;

public:

    //MidpointAxis(shared_ptr<const MeshAxis> wrapped = nullptr): wrapped(nullptr) { setWrapped(wrapped); }
    MidpointAxis(const MeshAxis& wrapped): wrapped(wrapped) { }

    //shared_ptr<const MeshAxis > getWrapped() const;

    //void setWrapped(shared_ptr<const MeshAxis > wrapped);

    //void clear() override { setWrapped(nullptr); }

    std::size_t size() const override;

    double at(std::size_t index) const override;

    bool isIncreasing() const override;
};

std::size_t MidpointAxis::size() const {
    std::size_t wrapped_size = wrapped.size();
    return wrapped_size ? wrapped_size - 1 : 0;
}

double MidpointAxis::at(std::size_t index) const {
    return (wrapped.at(index) + wrapped.at(index+1)) * 0.5;
}

bool MidpointAxis::isIncreasing() const {
    return wrapped.isIncreasing();
}


// -------------- MeshAxis ---------------------------------------------

shared_ptr<MeshAxis> MeshAxis::clone() const {
    //return plask::make_shared<MidpointAxis>(wrapped);
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

shared_ptr<MeshAxis> MeshAxis::getMidpointAxis() const {
    beforeCalcMidpointMesh();
    return plask::make_shared<MidpointAxis>(*this)->clone();
}

void MeshAxis::beforeCalcMidpointMesh() const {
    if (this->size() < 2)
        throw BadMesh("getMidpointAxis", "at least two points are required");
}

PLASK_API void prepareNearestNeighborInterpolationForAxis(const MeshAxis& axis, const InterpolationFlags& flags, double& wrapped_point_coord, int axis_nr) {
    if (flags.periodic(axis_nr) && !flags.symmetric(axis_nr)) {
        if (wrapped_point_coord < axis.at(0)) {
            if (axis.at(0) - wrapped_point_coord > wrapped_point_coord - flags.low(axis_nr) + flags.high(axis_nr) - axis.at(axis.size()-1)) wrapped_point_coord = axis.at(axis.size()-1);
        } else if (wrapped_point_coord > axis.at(axis.size()-1)) {
            if (wrapped_point_coord - axis.at(axis.size()-1) > flags.high(axis_nr) - wrapped_point_coord + axis.at(0) - flags.low(axis_nr)) wrapped_point_coord = axis.at(0);
        }
    }
}

PLASK_API void prepareInterpolationForAxis(const MeshAxis& axis, const InterpolationFlags& flags, double wrapped_point_coord, int axis_nr, std::size_t& index_lo, std::size_t& index_hi, double& lo, double& hi, bool& invert_lo, bool& invert_hi) {
    index_hi = axis.findUpIndex(wrapped_point_coord);
    invert_lo = false; invert_hi = false;
    if (index_hi == 0) {
        if (flags.symmetric(axis_nr)) {
            index_lo = 0;
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
            index_lo = axis.size() - 1;
            lo = axis.at(index_lo) - flags.high(axis_nr) + flags.low(axis_nr);
        } else {
            index_lo = 0;
            lo = axis.at(0) - 1.;
        }
    } else {
        index_lo = index_hi - 1;
        lo = axis.at(index_lo);
    }
    if (index_hi == axis.size()) {
        if (flags.symmetric(axis_nr)) {
            --index_hi;
            hi = axis.at(index_hi);
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
            index_hi = 0;
            hi = axis.at(0) + flags.high(axis_nr) - flags.low(axis_nr);
            if (hi == lo) hi += 1e-6;
        } else {
            --index_hi;
            hi = axis.at(index_hi) + 1.;
        }
    } else {
        hi = axis.at(index_hi);
    }
}

}   // namespace plask
