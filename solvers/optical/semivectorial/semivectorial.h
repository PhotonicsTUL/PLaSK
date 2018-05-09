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
  
    double lam0; /// wavelength to start rootdigger 

    std::vector<double> edgeVertLayerPoint;

    
};

#endif

}}}