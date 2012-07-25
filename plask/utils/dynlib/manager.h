#ifndef PLASK__UTILS_DYNLIB_MANAGER_H
#define PLASK__UTILS_DYNLIB_MANAGER_H

#include "loader.h"
#include <set>

namespace plask {

class DynamicLibraries {

    typedef std::set<DynamicLibrary> DynamicLibrarySet;

    DynamicLibrarySet loaded;

public:

    /**
     * Load dynamic library and add it to this or throw excpetion if it's not possible.
     *
     * Loaded library will be closed by destructor of this or can also be explicted close by close(const DynamicLibrary& to_close) method.
     * @param name of file with library to load
     * @return loaded library
     */
    const DynamicLibrary& load(const std::string& file_name);

    /**
     * Close given library if it is in this set.
     * @param to_close library to close
     */
    void close(const DynamicLibrary& to_close);

    /**
     * Return default set of dynamic libraries (this set is deleted, so all libraries in it are closed, on program exit).
     * @param default set of dynamic libraries
     */
    static DynamicLibraries& defaultSet();

    /**
     * Load dynamic library and add it to default set or throw excpetion if it's not possible.
     *
     * Loaded library will be closed on program exit or can also be explicted close by defaultClose(const DynamicLibrary& to_close) method.
     * @param name of file with library to load
     * @return loaded library
     */
    const DynamicLibrary& defaultLoad(const std::string& file_name) { return defaultSet().load(file_name); }

    /**
     * Close given library if it is in default set.
     * @param to_close library to close
     */
    const void defaultClose(const DynamicLibrary& to_close) { defaultSet().close(to_close); }

};


}   // namespace plask

#endif // PLASK__UTILS_DYNLIB_MANAGER_H
