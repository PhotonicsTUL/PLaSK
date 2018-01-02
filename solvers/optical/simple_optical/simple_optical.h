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
     
     struct FieldZ {
          dcomplex F, B;
	  FieldZ() = default;

          FieldZ(dcomplex f, dcomplex b): F(f), B(b) {}

          FieldZ operator*(dcomplex a) const { return FieldZ(F*a, B*a); }
          FieldZ operator*(const Matrix m) {return FieldZ(m.ff*F+m.fb*B, m.bf*F+m.bb*B);}
          FieldZ operator/(dcomplex a) const { return FieldZ(F/a, B/a); }
          FieldZ operator*=(dcomplex a) { F *= a; B *= a; return *this; }
          FieldZ operator/=(dcomplex a) { F /= a; B /= a; return *this; }
     };
      

     enum Polarization {
        TE,
        TM,
      };
      
    struct Mode {
      SimpleOptical* solver; ///< Solver this mode belongs to Simple Optical
      int m;		     ///< Number of mode
      dcomplex lam;         ///< Stored wavelength 
      double power;          ///< Mode power [mW];
      
      Mode(SimpleOptical* solver, int m=0):
	solver(solver), m(m), power(1.) {}
     
     
     bool operator==(const Mode& other) const {
            return m == other.m && is_zero(lam - other.lam);
     }
     
     };
     
     size_t nmodes() const {
	return modes.size();
     }
     
     /// Insert mode to the list or return the index of the exiting one
     size_t insertMode(const Mode& mode) {
        for (size_t i = 0; i != modes.size(); ++i)
        if (modes[i] == mode) return i;
        modes.push_back(mode);
        return modes.size()-1;
     }

     void loadConfiguration(XMLReader& reader, Manager& manager);

     virtual std::string getClassName() const { return "SimpleOptical"; }
 
     void onInitialize() override;
     void onInvalidate() override;

     shared_ptr<MeshAxis> axis_vertical;   
     shared_ptr<MeshAxis> axis_horizontal;
     shared_ptr<MeshAxis> axis_midpoints_vertical;
     shared_ptr<MeshAxis> axis_midpoints_horizontal;

     /// \return current wavelength
     dcomplex getWavelength() const { return 2e3*M_PI / k0; }

    /**
     * Set new wavelength
     * \param wavelength new wavelength
     */
     void setWavelength(double wavelength) {
        k0 = 2e3*M_PI / wavelength;
        invalidate();
     }

     void simpleVerticalSolver(double wave_length);
     
     dcomplex get_T_bb();
     dcomplex compute_transfer_matrix(const dcomplex& k, const std::vector<dcomplex> & NR);

     /// Parameters for main rootdigger
     RootDigger::Params root;

     /// Parameters for sripe rootdigger
     RootDigger::Params stripe_root;
      
     void computeField(double wavelength, double s, double e, int n);
     std::vector<dcomplex> computeEz(const dcomplex& x, const std::vector<double> & dst_mesh);
     std::vector<double> getZ();
     std::vector<dcomplex> getEz();
     
     typename ProviderFor<LightMagnitude, Geometry2DCylindrical>::Delegate outLightMagnitude;
     const DataVector<double> getLightMagnitude(int num, const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod);
     std::vector<Mode> modes;
     void stageOne();
     
     size_t findMode(double lambda, int m=0);
     
     std::vector<dcomplex> getNrCache();
     

protected:

  friend struct RootDigger;

  size_t x,           ///< poitn, when program computed vertical fields
         ybegin,  ///< First element of vertical mesh to consider
         yend;    ///< Last element of vertical mesh to consider

  Polarization polarization;  ///< Chosen light polarization

  shared_ptr<RectangularMesh<2>> mesh;   /// Mesh over which the calculations are performed
  
  dcomplex k0;

  dcomplex t_bb;

  Matrix transfer_matrix;

  std::vector<dcomplex> refractive_index_vec;

  dcomplex mode; 

  std::vector<double> edge_vert_layer_point;

  void initialize_refractive_index_vec();
  
  std::vector<dcomplex> nrCache;
  
  std::vector<double> z;

  /// Computed vertical fields
  std::vector<dcomplex> zfields; 

  
  std::vector<FieldZ> vecE;

  void print_vector(std::vector<double> vec);

};

  

}}} // namespace

#endif
