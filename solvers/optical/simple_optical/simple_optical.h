#ifndef PLASK__SOLVER_SIMPLE_OPTICAL
#define PLASK__SOLVER_SIMPLE_OPTICAL

#include <plask/plask.hpp>
#include "rootdigger.h"

namespace plask { namespace optical { namespace simple_optical {

/**
 * This is Doxygen documentation of your solver.
 * Write a brief description of it.
 */
struct PLASK_SOLVER_API SimpleOptical: public SolverOver<Geometry2DCylindrical> {
     
     SimpleOptical(const std::string& name="SimpleOptical");
     
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

     enum Polarization {
        TE,
        TM,
	};
	
     
	
     plask::optical::simple_optical::RootDigger::Params stripe_root; 
     
     void loadConfiguration(XMLReader& reader, Manager& manager);

     virtual std::string getClassName() const { return "SimpleOptical"; }
   
     virtual void onInitialize() {
	  if (!geometry) throw NoGeometryException(getId());
     }
     
     /// \return current wavelength
    dcomplex getWavelength() const { return 2e3*M_PI / k0; }

    /**
     * Set new wavelength
     * \param wavelength new wavelength
     */
    void setWavelength(dcomplex wavelength) {
        k0 = 2e3*M_PI / wavelength;
        invalidate();
    }
   
     void simpleVerticalSolver();

     void say_hello();
     
     dcomplex get_T_bb(const dcomplex& x, const std::vector< dcomplex >& NR);
   
protected:
  friend struct RootDigger;
  
  double stripex;             ///< Position of the main stripe

  size_t xbegin,  ///< First element of horizontal mesh to consider
         xend,    ///< Last element of horizontal mesh to consider
         ybegin,  ///< First element of vertical mesh to consider
         yend;    ///< Last element of vertical mesh to consider
         
  Polarization polarization;  ///< Chosen light polarization
  
  shared_ptr<RectangularMesh<2>> mesh;   /// Mesh over which the calculations are performed
  
  std::vector<std::vector<dcomplex>>nrCache; /// Cached refractive indices
  
  dcomplex detS1(const dcomplex& x, const std::vector<dcomplex>& NR, bool save=false);
  
  std::vector<Field,aligned_allocator<Field>> yfields; /// Computed horizontal and vertical fields
  
  dcomplex k0;
  
  std::vector<dcomplex> refractive_index;
   
};
  
}}} // namespace

#endif