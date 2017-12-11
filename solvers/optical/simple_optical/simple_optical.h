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
     
     /// Mode symmetry in horizontal axis
     enum Symmetry {
	  SYMMETRY_DEFAULT,
	  SYMMETRY_POSITIVE,
	  SYMMETRY_NEGATIVE,
	  SYMMETRY_NONE
      };
    
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
    
     /// Details of the computed mode
    struct Mode {
        SimpleOptical* solver;       ///< Solver this mode belongs to
        Symmetry symmetry;              ///< Horizontal symmetry of the modes
        dcomplex neff;                  ///< Stored mode effective index
        bool have_fields;               ///< Did we compute fields for current state?
        std::vector<Field,aligned_allocator<Field>> xfields; ///< Computed horizontal fields
        std::vector<double,aligned_allocator<double>> xweights; ///< Computed horizontal weights
        double power;                   ///< Mode power [mW]

        Mode(SimpleOptical* solver, Symmetry sym):
            solver(solver), have_fields(false), xfields(solver->xend), xweights(solver->xend), power(1.) {
            setSymmetry(sym);
        }

        void setSymmetry(Symmetry sym) {
            if (solver->geometry->isSymmetric(Geometry::DIRECTION_TRAN)) {
                if (sym == SYMMETRY_DEFAULT)
                    sym = SYMMETRY_POSITIVE;
                else if (sym == SYMMETRY_NONE)
                    throw BadInput(solver->getId(), "For symmetric geometry specify positive or negative symmetry");
            } else {
                if (sym == SYMMETRY_DEFAULT)
                    sym = SYMMETRY_NONE;
                else if (sym != SYMMETRY_NONE)
                    throw BadInput(solver->getId(), "For non-symmetric geometry no symmetry may be specified");
            }
            symmetry = sym;
        }

        bool operator==(const Mode& other) const {
            return symmetry == other.symmetry && is_zero( neff - other.neff );
        }

        /// Return mode loss
        double loss() const {
            return - 2e7 * imag(neff * solver->k0);
        }
    };
     
     void loadConfiguration(XMLReader& reader, Manager& manager);

     virtual std::string getClassName() const { return "SimpleOptical"; }
   
     void onInitialize() override;
     
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
        onInvalidate();
    }
   
     void simpleVerticalSolver(double wave_length);
     
     dcomplex get_T_bb();
         
     dcomplex compute_transfer_matrix(const dcomplex& k, const std::vector<dcomplex> & NR);
     
     /// Parameters for main rootdigger
     RootDigger::Params root;

     /// Parameters for sripe rootdigger
     RootDigger::Params stripe_root;
     
     void compute_electric_field_distribution(double wave_length);
        
     std::vector<dcomplex> compute_eField(const dcomplex& x, const std::vector<dcomplex> & NR);
     
     std::vector<dcomplex> get_eField();
     
     std::vector<dcomplex> get_bField();
     
     std::vector<double> get_z();
     
     virtual void onInvalidate();
     
     /// Computed modes
     std::vector<Mode> modes;
     
     /// Provider of optical field
     typename ProviderFor<LightMagnitude, Geometry2DCylindrical>::Delegate outLightMagnitude;
     
     /// Method computing the distribution of light intensity
     const LazyData<double> getLightMagnitude(int num, shared_ptr<const plask::MeshD<2>> dst_mesh, plask::InterpolationMethod=INTERPOLATION_DEFAULT);
     
     /// Return number of found modes
     size_t nmodes() const {
        return modes.size();
     }
     
     
protected:
  friend struct RootDigger;
  
   /// Computed horizontal and vertical fields
  std::vector<Field,aligned_allocator<Field>> yfields;
    
  double stripex;             ///< Position of the main stripe

  size_t x,	   ///< poitn, when program computed vertical fields
         xend,
	 ybegin,  ///< First element of vertical mesh to consider
         yend;    ///< Last element of vertical mesh to consider
         
  Polarization polarization;  ///< Chosen light polarization
  
  shared_ptr<RectangularMesh<2>> mesh;   /// Mesh over which the calculations are performed
  
  dcomplex k0;
  
  dcomplex t_bb;
  
  Matrix transfer_matrix;
  
  std::vector<dcomplex> refractive_index_vec;
  
  dcomplex vneff; 
  
  std::vector<double> edge_vert_layer_point;
  
  std::vector<double> z;
  
  std::vector<dcomplex> eField;
  
  std::vector<dcomplex> bField;
  
  void initialize_refractive_index_vec();
  
};
  
}}} // namespace

#endif