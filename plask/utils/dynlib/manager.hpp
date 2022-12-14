/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#ifndef PLASK__UTILS_DYNLIB_MANAGER_H
#define PLASK__UTILS_DYNLIB_MANAGER_H

#include "loader.hpp"
#include <set>

namespace plask {

/**
 * Represent set of dynamically loaded library.
 *
 * Close held libraries in destructor.
 *
 * It also contains one static, singleton instance available by defaultSet().
 * This set is especially useful to load libraries which should be closed on program exit (see defaultLoad method).
 */
class PLASK_API DynamicLibraries {

    /// Set of library, low level container type.
    typedef std::set<DynamicLibrary> DynamicLibrarySet;

    /// Set of library, low level container.
    DynamicLibrarySet loaded;

public:

    /// Type of iterator over loaded libraries.
    typedef DynamicLibrarySet::const_iterator iterator;

    /// Type of iterator over loaded libraries.
    typedef DynamicLibrarySet::const_iterator const_iterator;

    DynamicLibraries() = default;
    DynamicLibraries(const DynamicLibraries&) = delete; // MSVC require this since DynamicLibrarySet is not-copyable
    DynamicLibraries& operator=(const DynamicLibraries&) = delete;  // MSVC require this since DynamicLibrarySet is not-copyable
    DynamicLibraries(DynamicLibraries&&) = default;
    DynamicLibraries& operator=(DynamicLibraries&&) = default;

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
     * Load dynamic library and add it to this or throw exception if it's not possible.
     *
     * Loaded library will be closed by destructor of this or can also be explicitly closed by close(const DynamicLibrary& to_close) method.
     * @param file_name name of file with library to load
     * @param flags flags which describe configuration of open/close process, one or more (or-ed) flags from DynamicLibrary::Flags set
     * @return loaded library
     */
    const DynamicLibrary& load(const std::string& file_name, unsigned flags=0);

    /**
     * Close given library if it is in this set.
     * @param to_close library to close
     */
    void close(const DynamicLibrary& to_close);

    /// Close all holded libraries.
    void closeAll();

    /**
     * Return default set of dynamic libraries (this set is deleted, so all libraries in it are closed, on program exit).
     * @return default set of dynamic libraries
     */
    static DynamicLibraries& defaultSet();

    /**
     * Load dynamic library and add it to default set or throw exception if it's not possible.
     *
     * Loaded library will be closed on program exit or can also be explicitly closed by defaultClose(const DynamicLibrary& to_close) method.
     * @param file_name name of file with library to load
     * @param flags flags which describe configuration of open/close process, one or more (or-ed) flags from DynamicLibrary::Flags set
     * @return loaded library
     */
    static const DynamicLibrary& defaultLoad(const std::string& file_name, unsigned flags=0) { return defaultSet().load(file_name, flags); }

    /**
     * Close given library if it is in default set.
     * @param to_close library to close
     */
    static void defaultClose(const DynamicLibrary& to_close) { defaultSet().close(to_close); }

    /// Close all libraries holded in default set.
    static void defaultCloseAll() { defaultSet().closeAll(); }

};


}   // namespace plask

#endif // PLASK__UTILS_DYNLIB_MANAGER_H
