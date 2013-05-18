#ifndef PLASK__UTILS_METAPROG_H
#define PLASK__UTILS_METAPROG_H

/** @file
This file contains meta-programing tools.
*/

#include <tuple>
#include <type_traits>

namespace plask {

/**
Choose nr-th type from types list.

For example:
@code
  chooseType<2, A, B, C, D>::type c;    //is equal to: C c;
  chooseType<false, A, B>::type a;      //is equal to: A a;
  chooseType<true, A, B>::type b;       //is equal to: B b;
@endcode
*/
template <int nr, typename... types>
struct chooseType {
    typedef typename std::tuple_element<nr, std::tuple<types...>>::type type;
};

//TODO better impl. but not compilable with GCC 4.6 (unimplemented)
/*template <int nr, typename firstType, typename... restTypes> struct chooseType {
    ///Choosed type.
    typedef typename chooseType<nr-1, restTypes...>::type type;
};

template <typename firstType, typename... restTypes> struct chooseType<0, firstType, restTypes...> {
    typedef firstType type;
};*/

/**
 * Check if PotentiallyCallable is callable with given Args types (answare is in bool value static field).
 * @tparam PotentiallyCallable somthing which can be a functor
 * @tparam Args types of arguments
 */
// from http://stackoverflow.com/questions/5100015/c-metafunction-to-determine-whether-a-type-is-callable
template < typename PotentiallyCallable, typename... Args>
struct is_callable
{
  typedef char (&no)  [1];
  typedef char (&yes) [2];

  template < typename T > struct dummy;

  template < typename CheckType>
  static yes check(dummy<decltype(std::declval<CheckType>()(std::declval<Args>()...))> *);
  template < typename CheckType>
  static no check(...);

  /// @c true only if PotentiallyCallable is callable with given Args.
  enum { value = sizeof(check<PotentiallyCallable>(0)) == sizeof(yes) };
};


}   // namespace plask


#endif // PLASK__UTILS_METAPROG_H
