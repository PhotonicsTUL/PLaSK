#ifndef PLASK__SOLVER_SIMPLE_OPTICAL
#define PLASK__SOLVER_SIMPLE_OPTICAL

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace simple_optical {

/**
 * This is Doxygen documentation of your solver.
 * Write a brief description of it.
 */
struct PLASK_SOLVER_API SimpleOptical: public SolverOver<Geometry2DCylindrical> {
  
     struct Field {
          dcomplex F, B;
 	  Field() = default;
          Field(dcomplex f, dcomplex b): F(f), B(b) {}
          Field operator*(dcomplex a) const { return Field(F*a, B*a); }
          Field operator/(dcomplex a) const { return Field(F/a, B/a); }
          Field operator*=(dcomplex a) { F *= a; B *= a; return *this; }
          Field operator/=(dcomplex a) { F /= a; B /= a; return *this; }
      };
     
     struct Matrix {
          dcomplex ff, fb, bf, bb;
 	 Matrix() = default;
 	 Matrix(dcomplex t1, dcomplex t2, dcomplex t3, dcomplex t4): ff(t1), fb(t2), bf(t3), bb(t4) {}
 	 static Matrix eye() { return Matrix(1., 0., 0., 1.); }
 	 static Matrix diag(dcomplex f, dcomplex b) {return Matrix(f,0.,0.,b); }
 	 Matrix operator*(const Matrix& T) {
 	   return Matrix( ff*T.ff + fb*T.bf, ff*T.fb + fb*T.bb,
 			  bf*T.ff + bb*T.bf, bf*T.fb + bb*T.bb);
 	}
      };
	
	
	
//          Matrix(dcomplex t1, dcomplex t2, dcomplex t3, dcomplex t4): ff(t1), fb(t2), bf(t3), bb(t4) {}
//          static Matrix eye() { return Matrix(1.,0.,0.,1.); }
//          static Matrix diag(dcomplex f, dcomplex b) { return Matrix(f,0.,0.,b); }
//          Matrix operator*(const Matrix& T) {
//              return Matrix( ff*T.ff + fb*T.bf,   ff*T.fb + fb*T.bb,
//                             bf*T.ff + bb*T.bf,   bf*T.fb + bb*T.bb );
//       };

   SimpleOptical(const std::string& name="SimpleOptical");
   
   void loadConfiguration(XMLReader& reader, Manager& manager);

   virtual std::string getClassName() const { return "SimpleOptical"; }
   
   virtual void onInitialize() {
      if (!geometry) throw NoGeometryException(getId());
   }
   
   
   void simpleVerticalSolver();

   void say_hello();
   
   
private:
   plask::DataVector<double> boundary_layer;
   std::string axis_name;

};
  
}}} // namespace

#endif