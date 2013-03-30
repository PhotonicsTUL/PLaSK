#ifndef PLASK__UTILS_STL_H
#define PLASK__UTILS_STL_H

/** @file
This file includes tools which provide compability with STL containers, etc.
*/

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace plask {

/**
 * Try find value in map by key and return @a if_not_found value if object was not found.
 * @param map map to find in it
 * @param to_find key to find
 * @param if_not_found value to return when there is no object with @a to_find key in @a map
 * @return founded object or @a if_not_found value
 */
template <typename map_t>
inline typename map_t::mapped_type map_find(map_t& map, const typename map_t::key_type& to_find, typename map_t::mapped_type&& if_not_found = nullptr) {
    auto f = map.find(to_find);
    return f == map.end() ? std::forward<typename map_t::mapped_type>(if_not_found) : f->second;
}

/*
 * Try find value in (const) map by key and return @a if_not_found value if object was not found.
 * @param map map to find in it
 * @param to_find key to find
 * @param if_not_found value to return when there is no object with @a to_find key in @a map
 * @return founded object or @a if_not_found value
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

template <typename... Types>
struct VariadicTemplateTypesHolder {};

/// Don't use this directly, use applyTuple instead.
template<size_t N>
struct ApplyTuple {
    template<typename F, typename T, typename... A>
    static inline auto apply(F&& f, T && t, A &&... a)
        -> decltype(ApplyTuple<N-1>::apply(::std::forward<F>(f), ::std::forward<T>(t),
            ::std::get<N-1>(::std::forward<T>(t)), ::std::forward<A>(a)...
        ))
    {
        return ApplyTuple<N-1>::apply(::std::forward<F>(f), ::std::forward<T>(t),
            ::std::get<N-1>(::std::forward<T>(t)), ::std::forward<A>(a)...
        );
    }
};

/// Don't use this directly, use applyTuple instead.
template<>
struct ApplyTuple<0> {
    template<typename F, typename T, typename... A>
    static inline auto apply(F && f, T &&, A &&... a)
        -> decltype(::std::forward<F>(f)(::std::forward<A>(a)...))
    {
        return ::std::forward<F>(f)(::std::forward<A>(a)...);
    }
};

/**
 * Call @p f using arguments from tuple.
 * @param f functor to call
 * @param t tuple which includes all @p f arguments
 * @return result returned by @p f
 */
template<typename F, typename T>
inline auto applyTuple(F && f, T && t)
     -> decltype(ApplyTuple< ::std::tuple_size<
         typename ::std::decay<T>::type
     >::value>::apply(::std::forward<F>(f), ::std::forward<T>(t)))
{
    return ApplyTuple< ::std::tuple_size<
        typename ::std::decay<T>::type
    >::value>::apply(::std::forward<T>(f), ::std::forward<T>(t));
}   //note: if this will not work, try http://preney.ca/paul/archives/486

} // namespace plask

#endif // PLASK__STL_H
