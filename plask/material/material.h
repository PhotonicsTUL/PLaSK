#ifndef PLASK__MATERIAL_H
#define PLASK__MATERIAL_H

/** @file
This file includes base classes for materials and matrial database class.
*/

#include <string>
#include <memory>	//shared_ptr
#include <map>
#include <vector>
#include "../exceptions.h"

namespace plask {

//TODO lepiej użyć inteligętnych wskaźników czy obciążyć użytkowników potrzebą kasowania niektórych materiałów
//co jeśli klasa bazy materiałowej zostanie skasowana?

/**
 * Represent material, its physical properties.
 */
struct Material {

    ///Do nothing.
    virtual ~Material() {}

    // @return @c true only for dyanimicaly created materials which are not managed by MaterialDB and must be manualy deleted in some situations
    // virtual bool isDynamic() { return false; }

    ///@return material name
    virtual std::string getName() const = 0;

};

/**
 * Material which consist with few real materials.
 * It calculate avarages for all properties.
 */
struct MixedMaterial: public Material {
    
    //std::map<std::shared_ptr<Material>, double> materials;

    //virtual bool isDynamic() { return true; }

};

/**
 * Material which wrap one material and rotate its tensors properties.
 */
struct RotatedMaterial: public Material {
    
    std::shared_ptr<Material> wrapped;
    
};

/**
 * Materials database.
 *
 * Create materials (each on first get), and cache it.
 */
struct MaterialsDB {
    
    enum DOPANT_AMOUNT_TYPE { NO_DOPANT, DOPING_CONCENTRATION, CARRIER_CONCENTRATION };
    
    typedef Material* construct_material_f(const std::vector<double>& components_amounts, DOPANT_AMOUNT_TYPE dopant_amount_type, double dopant_amount);

private:
    std::map<std::string, construct_material_f*> constructors;
    
public:
    
    std::shared_ptr<Material> get(const std::string& parsed_name_with_donor, const std::vector<double>& components_amounts, DOPANT_AMOUNT_TYPE dopant_amount_type = NO_DOPANT, double dopant_amount = 0.0) const throw (NoSuchMaterial);
    
    std::shared_ptr<Material> get(const std::string& name_with_components, const std::string& dopant_descr) const throw (NoSuchMaterial, MaterialParseException);
    
    /**
     * @param full_name material name (with encoded parameters)
     * @return material with given name
     * @throw NoSuchMaterial if material with given name not exists
     */
    std::shared_ptr<Material> get(const std::string& full_name) const throw (NoSuchMaterial, MaterialParseException);
   
    /**
     * Add material to DB. Replace existing material if there is one already in DB.
     * @param name material name (with donor after ':')
     * @param constructor function which can create material instance
     */
    void add(const std::string& name, construct_material_f* constructor);

    /**
     * Fill database with default materials creators.
     */
    //TODO materials will be cr
    //void init();
};

} // namespace plask

#endif	//PLASK__MATERIAL_H
