#ifndef PLASK__OPTIONAL_H
#define PLASK__OPTIONAL_H

#include <plask/config.h>

#ifdef PLASK_OPTIONAL_STD

#include <optional>
namespace plask {
    using std::optional;
}

#else // PLASK_OPTIONAL_STD
// Use boost::optional

#include <boost/optional.hpp>
namespace plask {
    using boost::optional;
}

#endif // PLASK_SHARED_PTR_STD

#endif // PLASK__OPTIONAL_H
