#ifndef PLASK__METAPROG_H
#define PLASK__METAPROG_H

/** @file
This file includes meta-programing tools.
*/

#include <tuple>

namespace plask {

/**
Choost nr-th type from types list.

For example:
@code
  chooseType<2, A, B, C, D>::type c;    //is equal to: C c;
  chooseType<false, A, B>::type a;      //is equal to: A a;
  chooseType<true, A, B>::type b;       //is equal to: B b;
@endcode
*/
template <int nr, typename... types>
struct chooseType {
    typedef typename std::tuple_element<nr, std::tuple<types...>> type;
};

//TODO better impl. but not compilable in GCC 4.6 (unimplement)
/*template <int nr, typename firstType, typename... restTypes> struct chooseType {
    ///Choosed type.
    typedef typename chooseType<nr-1, restTypes...>::type type;
};

template <typename firstType, typename... restTypes> struct chooseType<0, firstType, restTypes...> {
    typedef firstType type;
};*/

}   // namespace plask


#endif // PLASK__METAPROG_H
