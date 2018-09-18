#ifndef PLASK__UTILS_STL_H
#define PLASK__UTILS_STL_H

/** @file
This file contains tools which provide compability with STL containers, etc.
*/

#include <algorithm>
#include <cstddef> // for std::size_t

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

/**
 * Print range separating elements by @p separator (which is not printed after last element).
 * @param out stream to print to
 * @param begin, end [begin, end) range to print
 * @param separator string placed between each pair of adjacent elements
 * @return out
 */
template <typename ForwadIterator>
std::ostream& printRange(std::ostream& out, ForwadIterator begin, ForwadIterator end, const char* separator = ", ") {
    if (begin != end) {
        out << *begin;
        for (++begin ; begin != end; ++begin)
            out << separator << *begin;
    }
    return out;
}

/**
 * Assign new value to varible and back the old value in destructor.
 *
 * It:
 * 1. constructs new value using given constructor parameters
 * 2. swaps values of given variable and constructed one
 * 3. in destructor: swaps the values back
 */
template <typename T>
struct AssignWithBackup {
    T& original;
    T store;

    template <typename... ConstructorArgsT>
    AssignWithBackup(T& varible_to_back, ConstructorArgsT&&... newValueCtrArgs)
    : original(varible_to_back), store(std::forward<ConstructorArgsT>(newValueCtrArgs)...) {
        std::swap(varible_to_back, store);
    }

    ~AssignWithBackup() { std::swap(original, store); }
};

/// Don't use this directly, use applyTuple instead.


#include <cstddef>
#include <tuple>
#include <type_traits>

//===================================================================

template <std::size_t...>
struct indices { };

//===================================================================

template <
  std::size_t Begin,
  std::size_t End,
  typename Indices
>
struct make_seq_indices_impl;

template <
  std::size_t Begin,
  std::size_t End,
  std::size_t... Indices
>
struct make_seq_indices_impl<Begin, End, indices<Indices...>>
{
  using type =
    typename
      make_seq_indices_impl<Begin+1, End,indices<Indices..., Begin>
    >::type
  ;
};

template <std::size_t End, std::size_t... Indices>
struct make_seq_indices_impl<End, End, indices<Indices...>>
{
  using type = indices<Indices...>;
};

template <std::size_t Begin, std::size_t End>
using make_seq_indices =
  typename make_seq_indices_impl<Begin, End, indices<>>::type;

//===================================================================

template <typename F>
using return_type = typename std::result_of<F>::type;

template <typename Tuple>
constexpr std::size_t tuple_size()
{
  return std::tuple_size<typename std::decay<Tuple>::type>::value;
}

// No definition exists for the next prototype...
template <
  typename Op,
  typename T,
  template <std::size_t...> class I, std::size_t... Indices
>
constexpr auto apply_tuple_return_type_impl(
  Op&& op,
  T&& t,
  I<Indices...>
) ->
  return_type<Op(
    decltype(std::get<Indices>(std::forward<T>(t)))...
  )>
;

// No definition exists for the next prototype...
template <typename Op, typename T>
constexpr auto apply_tuple_return_type(Op&& op, T&& t) ->
  decltype(apply_tuple_return_type_impl(
    op, t, make_seq_indices<0,tuple_size<T>()>{}
  ));

template <
  typename Ret, typename Op, typename T,
  template <std::size_t...> class I, std::size_t... Indices
>
inline Ret apply_tuple(Op&& op, T&& t, I<Indices...>)
{
  return op(std::get<Indices>(std::forward<T>(t))...);
}

/**
 * Call @p f using arguments from tuple.
 * @param op functor to call
 * @param t tuple which contains all @p f arguments
 * @return result returned by @p f
 */
template <typename Op, typename Tuple>
inline auto apply_tuple(Op&& op, Tuple&& t)
  -> decltype(apply_tuple_return_type(
    std::forward<Op>(op), std::forward<Tuple>(t)
  ))
{
  return
    apply_tuple<
      decltype(apply_tuple_return_type(
        std::forward<Op>(op), std::forward<Tuple>(t)
      ))
    >(
      std::forward<Op>(op), std::forward<Tuple>(t),
      make_seq_indices<0,tuple_size<Tuple>()>{}
    );
  ;
}   //http://preney.ca/paul/archives/934


/**
 * This template can be used to show fileds shich should be just ignore by copy constructors and assign operators of the class.
 */
template <typename T>
struct DontCopyThisField: public T {

    /*template <typename... Args>
    DontCopyThisField(Args&&... args) noexcept(noexcept(T(::std::forward<Args>(args)...)))
        : T(::std::forward<Args>(args)...)
    {}*/

    DontCopyThisField() = default;

    DontCopyThisField(const DontCopyThisField<T>&) noexcept {}

    DontCopyThisField(const T&) noexcept {}

    DontCopyThisField(DontCopyThisField&& from) = default;

    DontCopyThisField(T&& from) noexcept(noexcept(T(::std::move(from))))
        : T(::std::move(from))
    {}

    DontCopyThisField& operator=(const T&) noexcept {
        return *this;
    }

    DontCopyThisField& operator=(const DontCopyThisField&) noexcept {
        return *this;
    }

    DontCopyThisField& operator=(T&& from) noexcept(noexcept(T::operator=(::std::move(from)))) {
        T::operator=(::std::move(from));
        return *this;
    }

    DontCopyThisField& operator=(DontCopyThisField&& from) = default;
};


} // namespace plask

#endif // PLASK__STL_H
