#ifndef PLASK__SOLVER_SEMIVECTORIAL

#define PLASK__SOLVER_SEMIVECTORIAL

#include <plask/plask.hpp>



namespace plask { namespace optical { namespace semivectorial {
/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space by use semivectorial method based on effective index
 */
template<typename Geometry2DType>
struct PLASK_SOLVER_API SemiVectorial: public SolverOver<Geometry2DType>
{
        
    SemiVectorial(const std::string& name="");
    
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
    
    virtual void loadConfiguration(XMLReader&, Manager&) override;
    
    virtual void onInitialize() override;
    
    virtual std::string getClassName() const override;
    
    shared_ptr<MeshAxis> axis_vertical;   
    shared_ptr<MeshAxis> axis_horizontal;
    shared_ptr<MeshAxis> axis_midpoints_vertical;
    shared_ptr<MeshAxis> axis_midpoints_horizontal;
    
    void refractive_index(double x);
    
protected:

    size_t ybegin,  ///< First element of vertical mesh to consider
           yend;    ///< Last element of vertical mesh to consider

    shared_ptr<RectangularMesh<2>> mesh;   /// Mesh over which the calculations are performed
  
    double stripex;  
    
    dcomplex k0;
    
    void initializeRefractiveIndexVec();
    
    dcomplex computeTransferMatrix(const dcomplex& x, const std::vector<dcomplex>& NR);
  
    std::vector<dcomplex> nrCache; // Vector to hold refractive index
    
    std::vector<FieldZ> vecE;
  
    double lam0; /// wavelength to start rootdigger 

    std::vector<double> edgeVertLayerPoint;

    
};

#endif

}}}