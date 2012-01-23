#ifndef PLASK__UTILS_STL_H
#define PLASK__UTILS_STL_H

/** @file
This file includes tools which provide compability with STL containers, etc.
*/

namespace plask {

/**
 * Try find value in map by key and return @a if_not_found value if element was not found.
 * @param map map to find in it
 * @param to_find key to find
 * @param if_not_found value to return when there is no element with @a to_find key in @a map
 * @return founded element or @a if_not_found value
 */
template <typename map_t>
inline typename map_t::mapped_type map_find(map_t& map, const typename map_t::key_type& to_find, typename map_t::mapped_type if_not_found = nullptr) {
    typename map_t::iterator f = map.find(to_find);
    return f == map.end() ? if_not_found : f->second;
}

/**
 * Try find value in (const) map by key and return @a if_not_found value if element was not found.
 * @param map map to find in it
 * @param to_find key to find
 * @param if_not_found value to return when there is no element with @a to_find key in @a map
 * @return founded element or @a if_not_found value
 */
template <typename map_t>
inline const typename map_t::mapped_type map_find(const map_t& map, const typename map_t::key_type& to_find, const typename map_t::mapped_type if_not_found = nullptr) {
    typename map_t::const_iterator f = map.find(to_find);
    return f == map.end() ? if_not_found : f->second;
}

} // namespace plask

#endif // PLASK__STL_H
