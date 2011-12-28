from modplask import *

## ## plask vectors ## ##

def vector(*args):
    classes = {(2,float): vector2d_float, (2,complex): vector2d_complex, (3,float): vector3d_float, (3,complex): vector3d_complex}
    t = args[-1]
    if type(t) == type:
        typ = t
        c = args[:-1]
    else:
        typ = float
        c = args
    try:
        return classes[len(c),typ](*c)
    except KeyError:
        raise TypeError("unrecognized vector type")

for V in [vector3d_float, vector3d_complex]:
    V.phi = V.y
    V.a, V.b, V.c = V.x, V.y, V.z

## ## plask.material ## ##

def registerMaterial(Material):
    '''Function to register a new material'''
    if not 'name' in Material.__dict__:
        Material.name = Material.__name__
    material._registerMaterial(Material.name, Material) # register to C++
    return Material

material.new = material.registerMaterial = registerMaterial
#del registerMaterial

## ## plask.geometry ## ##



## ##  ## ##
