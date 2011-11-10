
namespace plask {

//TODO Vec -> typy przestrzeni udostępniające typ dla pkt.

/**
Provide type for vector in given space: double for 1d, etc.

Use example: <code>Vec<2>::type my_2dvec_obj;</code>
@tparam dim number of space dimentions
*/
template <int dim>
struct Cartesian {};

template <>
struct Cartesian<1> {
    typedef double PointType;
    //typedef Cartesian<2>::type upspace_type;
};

//TODO Cartesian<2>, Cartesian<3>

} // namespace plask
