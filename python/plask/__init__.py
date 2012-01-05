from modplask import *

## ## plask vectors ## ##

def vector(*args):
    '''Create a new vector'''
    classes = {(2,float): vector2d_float, (2,complex): vector2d_complex, (3,float): vector3d_float, (3,complex): vector3d_complex}
    t = args[-1]
    if type(t) == type or t is None:
        typ = t
        comps = args[:-1]
    else:
        typ = None
        comps = args
    if typ is None:
        try:
            comps = [ float(c) for c in comps ]
        except TypeError:
            c = [ complex(c) for c in comps ]
            typ = complex
        else:
            typ = float
    try:
        cls = classes[len(comps),typ]
    except KeyError:
        raise TypeError("unrecognized vector type")
    if config.vaxis == "y":
        if len(comps) == 3: comps = comps[2],comps[0],comps[1]


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
