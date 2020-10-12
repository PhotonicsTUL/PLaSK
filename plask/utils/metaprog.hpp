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


//http://talesofcpp.fusionfenix.com/post-11/true-story-call-me-maybe
namespace hyman {
  template <typename T>
  using always_void = void;

  template <typename Expr, std::size_t Step = 0, typename Enable = void>
  struct is_callable_impl
    : is_callable_impl<Expr, Step + 1>
  {};

  // (t1.*f)(t2, ..., tN) when f is a pointer to a member function of a class T
  // and t1 is an object of type T or a reference to an object of type T or a
  // reference to an object of a type derived from T;
  template <typename F, typename T, typename ...Args>
  struct is_callable_impl<F(T, Args...), 0,
    always_void<decltype(
      (std::declval<T>().*std::declval<F>())(std::declval<Args>()...)
    )>
  > : std::true_type
  {};

  // ((*t1).*f)(t2, ..., tN) when f is a pointer to a member function of a class T
  // and t1 is not one of the types described in the previous item;
  template <typename F, typename T, typename ...Args>
  struct is_callable_impl<F(T, Args...), 1,
    always_void<decltype(
      ((*std::declval<T>()).*std::declval<F>())(std::declval<Args>()...)
    )>
  > : std::true_type
  {};

  // t1.*f when N == 1 and f is a pointer to member data of a class T and t1 is an
  // object of type T or a reference to an object of type T or a reference to an
  // object of a type derived from T;
  template <typename F, typename T>
  struct is_callable_impl<F(T), 2,
    always_void<decltype(
      std::declval<T>().*std::declval<F>()
    )>
  > : std::true_type
  {};

  // (*t1).*f when N == 1 and f is a pointer to member data of a class T and t1 is
  // not one of the types described in the previous item;
  template <typename F, typename T>
  struct is_callable_impl<F(T), 3,
    always_void<decltype(
      (*std::declval<T>()).*std::declval<F>()
    )>
  > : std::true_type
  {};

  // f(t1, t2, ..., tN) in all other cases.
  template <typename F, typename ...Args>
  struct is_callable_impl<F(Args...), 4,
    always_void<decltype(
      std::declval<F>()(std::declval<Args>()...)
    )>
  > : std::true_type
  {};

  template <typename Expr>
  struct is_callable_impl<Expr, 5>
    : std::false_type
  {};
}

/**
 * Check if Expr is callable.
 */
template <typename Expr>
struct is_callable
  : hyman::is_callable_impl<Expr>
{};


}   // namespace plask


#endif // PLASK__UTILS_METAPROG_H
