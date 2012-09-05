#ifndef PLASK__UTILS_DYNLIB_MANAGER_H
#define PLASK__UTILS_DYNLIB_MANAGER_H

#include "loader.h"
#include <set>

namespace plask {

/**
 * Represent set of dynamically loaded library.
 *
 * Close held libraries in destructor.
 *
 * It also includes one static, singleton instance available by defaultSet().
 * This set is especially useful to load libraries which should be closed on program exit (see defaultLoad method).
 */
class DynamicLibraries {

    /// Set of library, low level container type.
    typedef std::set<DynamicLibrary> DynamicLibrarySet;

    /// Set of library, low level container.
    DynamicLibrarySet loaded;

public:

    /// Type of iterator over loaded libraries.
    typedef DynamicLibrarySet::const_iterator iterator;

    /// Type of iterator over loaded libraries.
    typedef DynamicLibrarySet::const_iterator const_iterator;

    /**
     * Allow to iterate over opened library included in this set.
     * @return begin iterator, which point to first library
     */
    const_iterator begin() const { return loaded.begin(); }

    /**
     * Allow to iterate over opened library included in this set.
     * @return end iterator,  which point just over last library
     */
    const_iterator end() const { return loaded.end(); }

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

    /// Close all holded libraries.
    void closeAll();

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
    static const DynamicLibrary& defaultLoad(const std::string& file_name) { return defaultSet().load(file_name); }

    /**
     * Close given library if it is in default set.
     * @param to_close library to close
     */
    static const void defaultClose(const DynamicLibrary& to_close) { defaultSet().close(to_close); }

    /// Close all libraries holded in default set.
    static const void defaultCloseAll() { defaultSet().closeAll(); }

};


}   // namespace plask

#endif // PLASK__UTILS_DYNLIB_MANAGER_H
