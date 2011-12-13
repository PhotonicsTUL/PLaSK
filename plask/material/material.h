#ifndef PLASK__MATERIAL_H
#define PLASK__MATERIAL_H

/** @file
This file includes base classes for materials and matrial database class.
*/

#include <string>
#include <memory>	//shared_ptr
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
Materials database.
*/
struct MaterialsDB {

    /**
    @return material with given name
    @throw NoSuchMaterial if material with given name not exists
    */
    std::shared_ptr<Material> get(const std::string& name);

    /**
    Fill database with default materials.
    */
    void init();

};

} // namespace plask

#endif	//PLASK__MATERIAL_H
