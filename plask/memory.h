#ifndef PLASK__MEMORY_H
#define PLASK__MEMORY_H

// Declare shared pointer to use
// (when boost::python gets compatibile with std::shared_ptr, check boost version and use std if possible)
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

namespace plask {
    using boost::shared_ptr;
    using boost::make_shared;
    using boost::dynamic_pointer_cast;
    using boost::static_pointer_cast;
    using boost::const_pointer_cast;
    using boost::weak_ptr;
    using boost::enable_shared_from_this;
}

#endif // PLASK__MEMORY_H
