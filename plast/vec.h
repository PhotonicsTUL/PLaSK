
namespace plast {

/**
Provide type for vector in given space: double for 1d, etc.

Use example: <code>Vec<2>::type my_2dvec_obj;</code>
@tparam dim number of space dimentions
*/
template <int dim>
struct Vec {};

template <>
struct Vec<1> {
    typedef double type;
    //typedef Vec<2>::type 
};

//TODO Vec<2>, Vec<3>

}	//namespace plast
