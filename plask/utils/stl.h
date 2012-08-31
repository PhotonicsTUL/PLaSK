#ifndef PLASK__UTILS_STL_H
#define PLASK__UTILS_STL_H

/** @file
This file includes tools which provide compability with STL containers, etc.
*/

#include <algorithm>

namespace plask {

/**
 * Try find value in map by key and return @a if_not_found value if element was not found.
 * @param map map to find in it
 * @param to_find key to find
 * @param if_not_found value to return when there is no element with @a to_find key in @a map
 * @return founded element or @a if_not_found value
 */
template <typename map_t>
inline typename map_t::mapped_type map_find(map_t& map, const typename map_t::key_type& to_find, typename map_t::mapped_type&& if_not_found = nullptr) {
    auto f = map.find(to_find);
    return f == map.end() ? std::forward<typename map_t::mapped_type>(if_not_found) : f->second;
}

/*
 * Try find value in (const) map by key and return @a if_not_found value if element was not found.
 * @param map map to find in it
 * @param to_find key to find
 * @param if_not_found value to return when there is no element with @a to_find key in @a map
 * @return founded element or @a if_not_found value
 */
/*template <typename map_t>
inline const typename map_t::mapped_type map_find(const map_t& map, const typename map_t::key_type& to_find, const typename map_t::mapped_type&& if_not_found = nullptr) {
    auto f = map.find(to_find);
    return f == map.end() ? std::forward<const typename map_t::mapped_type>(if_not_found) : f->second;
}*/

/**
 * Find position in ascending ordered, radnom access, seqence [begin, end) of floats or doubles nearest to @p to_find.
 * @param begin, end ordered, radnom access, seqence [begin, end), can't be empty
 * @param to_find value to which nearest one should be found
 * @param lower_bound must be equal to std::lower_bound(begin, end, to_find)
 * @return first position pos for which abs(*pos-to_find) is minimal
 */
template <typename Iter, typename Val>
inline Iter find_nearest_using_lower_bound(Iter begin, Iter end, const Val& to_find, Iter lower_bound) {
    if (lower_bound == begin) return lower_bound; //before first
    if (lower_bound == end) return lower_bound-1; //after last
    Iter lo_candidate = lower_bound - 1;
    //now: *lo_candidate <= to_find < *lower_bound
    if (to_find - *lo_candidate <= *lower_bound - to_find) //nearest to *lo_candidate?
        return lo_candidate;
    else
        return lower_bound;
}

/**
 * Find position in ascending ordered, radnom access, seqence [begin, end) of floats or doubles nearest to @p to_find.
 * @param begin, end ordered, radnom access, seqence [begin, end), can't be empty
 * @param to_find value to which nearest one should be found
 * @return first position pos for which abs(*pos-to_find) is minimal
 */
template <typename Iter, typename Val>
inline Iter find_nearest_binary(Iter begin, Iter end, const Val& to_find) {
    return find_nearest_using_lower_bound(begin, end, to_find, std::lower_bound(begin, end, to_find));
}

} // namespace plask

#endif // PLASK__STL_H
