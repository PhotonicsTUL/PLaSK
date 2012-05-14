#ifndef PLASK__MEMORY_H
#define PLASK__MEMORY_H

#include <plask/config.h>

#ifdef PLASK_SHARED_PTR_STD

#include <memory>
namespace plask {
    using std::shared_ptr;
    using std::make_shared;
    using std::dynamic_pointer_cast;
    using std::static_pointer_cast;
    using std::const_pointer_cast;
    using std::weak_ptr;
    using std::enable_shared_from_this;
}

#else // PLASK_SHARED_PTR_STD
// Use boost::shared_ptr

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

#endif // PLASK_SHARED_PTR_STD




#endif // PLASK__MEMORY_H
