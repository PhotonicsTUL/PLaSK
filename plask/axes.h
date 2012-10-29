#ifndef PLASK__AXES_H
#define PLASK__AXES_H

#include <string>
#include <map>

/** @file
This file includes utils connected with names of axes.
*/

#include "vector/3d.h"

namespace plask {

/**
 * Held names of axises.
 *
 * Can change: axis number (from 0 to 2) <-> axis name (string)
 */
struct AxisNames {

    /**
     * Register of axis names.
     */
    struct Register {
        /// Name of system of axis names -> AxisNames
        std::map<std::string, AxisNames> axisNames;

        /// Construct empty register.
        Register() {}

        template<typename... Params>
        Register(const std::string& c0_name, const std::string& c1_name, const std::string& c2_name, const Params&... names) {
            this->operator()(c0_name, c1_name, c2_name, names...);
        }

        /**
         * Add axis names to register.
         * @param c0_name, c1_name, c2_name axis names
         * @param name name of axis names, register key
         */
        void addname(const std::string& c0_name, const std::string& c1_name, const std::string& c2_name, const std::string& name) {
            axisNames[name] = AxisNames(c0_name, c1_name, c2_name);
        }

        /**
         * Add axis names using as key: c0_name + c1_name + c2_name
         * @param c0_name, c1_name, c2_name axis names
         */
        Register& operator()(const std::string& c0_name, const std::string& c1_name, const std::string& c2_name) {
            addname(c0_name, c1_name, c2_name, c0_name + c1_name + c2_name);
            return *this;
        }

        /**
         * Add axis names to register using as keys given @p name and c0_name + c1_name + c2_name.
         * @param c0_name, c1_name, c2_name axis names
         * @param name name of axis names, register key
         * @tparam Param1 std::string or const char*
         */
        template<typename Param1>
        Register& operator()(const std::string& c0_name, const std::string& c1_name, const std::string& c2_name, const Param1& name) {
            addname(c0_name, c1_name, c2_name, name);
            return this->operator()(c0_name, c1_name, c2_name);
        }

        /**
         * Add axis names to register using as keys given names and c0_name + c1_name + c2_name.
         * @param c0_name, c1_name, c2_name axis names
         * @param firstName, names names of axis names, register keys
         * @tparam Param1, Params each of type std::string or const char*
         */
        template<typename Param1, typename... Params>
        Register& operator()(const std::string& c0_name, const std::string& c1_name, const std::string& c2_name, const Param1& firstName, const Params&... names) {
            addname(c0_name, c1_name, c2_name, firstName);
            return this->operator()(c0_name, c1_name, c2_name, names...);
        }

        /**
         * Get axis names with given name (key).
         * @param name register keys
         * @return axis names
         * @throw NoSuchAxisNames if axis names with given name not exists in register
         */
        const AxisNames& get(const std::string& name) const;
    };

    /// Register of standard axis names.
    static Register axisNamesRegister;

    /// Get absolute names.
    static const AxisNames& getAbsoluteNames();

    /// Name of axises (by index).
    std::string byIndex[3];

    /// Construct uninitialized object, with empty names of axes.
    AxisNames() {}

    AxisNames(const std::string& c0_name, const std::string& c1_name, const std::string& c2_name);

    /**
     * Get axis name by index.
     * @param i index of axis name, from 0 to 2
     * @return name of i-th axis
     */
    const std::string& operator[](const std::size_t i) const { return byIndex[i]; }

    /**
     * Get axis index by name.
     * @param name axis name
     * @return index (from 0 to 2) of axis with given @p name or 3 if no axis with given name
     */
    std::size_t operator[](const std::string& name) const;

    /// \return string representation of the axes for the register
    std::string str() const;

    std::string getNameForLong() const { return operator [](axis::lon_index); }

    std::string getNameForTran() const { return operator [](axis::tran_index); }

    std::string getNameForVert() const { return operator [](axis::up_index); }

    /**
     * Check if this and @p to_compare have equals names of all axes.
     * @param to_compare object to compare to @c this
     * @return @c true only if @c this and @p to_compar are equals
     */
    bool operator==(const AxisNames& to_compare) const;

    /**
     * Check if this and @p to_compare have not equals names of all axes.
     * @param to_compare object to compare to @c this
     * @return @c true only if @c this and @p to_compar are not equals
     */
    bool operator!=(const AxisNames& to_compare) const { return !(*this == to_compare); }
};



} // namespace plask

#endif // PLASK__AXES_H
