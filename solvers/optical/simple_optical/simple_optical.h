#ifndef PLASK__SOLVER_SIMPLE_OPTICAL

#define PLASK__SOLVER_SIMPLE_OPTICAL

#include <plask/plask.hpp>

#include "rootdigger.h"

namespace plask { namespace optical { namespace simple_optical {

/**
 * Solver performing calculations in 2D Cylindrical geometry by solve Helmholtz equation
 */

struct PLASK_SOLVER_API SimpleOptical: public SolverOver<Geometry2DCylindrical> {
//MD: to powinno być szablonem — wtedy łatwo można zrobić solver dla każdej geometrii

    SimpleOptical(const std::string& name="");

    struct Matrix {
         dcomplex ff, fb, bf, bb;
         Matrix() = default;
         Matrix(dcomplex t1, dcomplex t2, dcomplex t3, dcomplex t4): ff(t1), fb(t2), bf(t3), bb(t4) {}
         static Matrix eye() { return Matrix(1., 0., 0., 1.); }
         static Matrix diag(dcomplex f, dcomplex b) {return Matrix(f,0.,0.,b); }
         Matrix operator*(const Matrix& T) {
            return Matrix( ff*T.ff + fb*T.bf, ff*T.fb + fb*T.bb,
                           bf*T.ff + bb*T.bf, bf*T.fb + bb*T.bb);}
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
      

      
    struct Mode {
      SimpleOptical* solver; ///< Solver this mode belongs to Simple Optical
      dcomplex lam;          ///< Stored wavelength

      Mode(SimpleOptical* solver):
	solver(solver) {}
    };
    
    size_t nmodes() const {
	return modes.size();
    }
    
     /// Insert mode to the list or return the index of the exiting one
     size_t insertMode(const Mode& mode) {
        modes.push_back(mode);
        return modes.size()-1;
    }

     void loadConfiguration(XMLReader& reader, Manager& manager) override;

     virtual std::string getClassName() const override  {return "SimpleOpticalCyl";} 
 
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
     void setWavelength(dcomplex wavelength) {
        k0 = 2e3*M_PI / wavelength;
        nrCache.clear();
     }
     
     dcomplex getVertDeterminant(dcomplex wavelength);
     
     dcomplex computeTransferMatrix(const dcomplex& k, const std::vector<dcomplex> & NR);

     /// Parameters for rootdigger
     RootDigger::Params root;
     
     typename ProviderFor<LightMagnitude, Geometry2DCylindrical>::Delegate outLightMagnitude;
     
     /// Provider of refractive index
     typename ProviderFor<RefractiveIndex, Geometry2DCylindrical>::Delegate outRefractiveIndex;
          
     std::vector<Mode> modes;
     
     size_t findMode(double lambda);
     
     /// \return position of the main stripe
    double getStripeX() const { return stripex; }

    /**
     * Set position of the main stripe
     * \param x horizontal position of the main stripe
     */
    void setStripeX(double x) {
        stripex = x;
        invalidate();
    }


protected:

    friend struct RootDigger;

    size_t ybegin,  ///< First element of vertical mesh to consider
           yend;    ///< Last element of vertical mesh to consider

    shared_ptr<RectangularMesh<2>> mesh;   /// Mesh over which the calculations are performed
  
    dcomplex k0;
  
    double lam0; /// wavelength to start rootdigger 

    std::vector<double> edgeVertLayerPoint;

    void initializeRefractiveIndexVec();
  
    std::vector<dcomplex> nrCache; // Vector to hold refractive index
    
    std::vector<FieldZ> vecE;
  
    const DataVector<double> getLightMagnitude(int num, const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod);
     
    const LazyData<Tensor3<dcomplex>> getRefractiveIndex(const shared_ptr<const MeshD<2>> &dst_mesh, InterpolationMethod);
  
    double stripex;             ///< Position of the main stripe
  
    /// Invalidate the data
    virtual void onInvalidate() override;
  
    void updateCache();
};

}}} // namespace

#endif
