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
	
     
     void loadConfiguration(XMLReader& reader, Manager& manager);

     virtual std::string getClassName() const { return "SimpleOptical"; }
   
     void onInitialize() override;
     
     shared_ptr<MeshAxis> axis_vertical;   
     
     shared_ptr<MeshAxis> axis_horizontal;
     
     shared_ptr<MeshAxis> axis_midpoints_vertical;
     
     shared_ptr<MeshAxis> axis_midpoints_horizontal;
     
     /// \return current wavelength
     double getWavelength() const { return 2e3*M_PI / k0; }

    /**
     * Set new wavelength
     * \param wavelength new wavelength
     */
     void setWavelength(double wavelength) {
        k0 = 2e3*M_PI / wavelength;
        invalidate();
    }
   
     void simpleVerticalSolver();

     void say_hello();
    
     dcomplex comput_T_bb(const dcomplex& x, const std::vector< dcomplex >& NR);
     
   
protected:
  friend struct RootDigger;
  
   /// Computed horizontal and vertical fields
  std::vector<Field,aligned_allocator<Field>> yfields;
    
  double stripex;             ///< Position of the main stripe

  size_t x,	   ///< poitn, when program computed vertical fields
         ybegin,  ///< First element of vertical mesh to consider
         yend;    ///< Last element of vertical mesh to consider
         
  Polarization polarization;  ///< Chosen light polarization
  
  shared_ptr<RectangularMesh<2>> mesh;   /// Mesh over which the calculations are performed
  
  double k0;
  
  dcomplex t_bb;
  
  std::vector<dcomplex> refractive_index_vec;
  
  void initialize_refractive_index_vec();
};
  
}}} // namespace

#endif