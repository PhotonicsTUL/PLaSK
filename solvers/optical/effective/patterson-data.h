#ifndef PLASK__SOLVER__EFFECTIVE_PATTERSONDATA_H
#define PLASK__SOLVER__EFFECTIVE_PATTERSONDATA_H

namespace plask { namespace solvers { namespace effective {

extern const double patterson_points[];

extern const double patterson_weights_data[];

constexpr const double* patterson_weights[] = {
    patterson_weights_data,
    patterson_weights_data + 1,
    patterson_weights_data + 3,
    patterson_weights_data + 7,
    patterson_weights_data + 15,
    patterson_weights_data + 31,
    patterson_weights_data + 63,
    patterson_weights_data + 127,
    patterson_weights_data + 255
};

}}} // namespace plask::solvers::effective

#endif // PLASK__SOLVER__EFFECTIVE_PATTERSONDATA_H