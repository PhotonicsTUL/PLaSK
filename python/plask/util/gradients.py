from .. import geometry as _geometry


def simplify(item, lam, T=300., linear='nr'):
    """Return stack of two layers providing the same optical parameters as the gradient
    layer.

    Args:
        item (~plask.geometry.Rectangle or ~plask.geometry.Cuboid): object to simplify
        lam (float): reference wavelength
        T (float): temperature for the refractive indices
        linear ('nr' or 'eps'): which parameter should be linear

    Returns:
        ~plask.geometry.GeometryObject: an object repacing the original leaf
    """
    if isinstance(item, _geometry.Rectangle):
        Block = _geometry.Rectangle
        Stack = _geometry.SingleStack2D
    elif isinstance(item, _geometry.Cuboid):
        Block = _geometry.Cuboid
        Stack = _geometry.SingleStack3D
    else:
        raise TypeError("'item' must be Rectangle or Cuboid")

    try:
        m0, m1 = item.material
    except TypeError:   # the object is uniform
        return item

    material = str(item.representative_material)

    from ._gradients import simplify_gradient_nr

    stack = Stack()
    dims = list(item.dims)
    for Nr, d in simplify_gradient_nr(item.height, m0.Nr(lam, T), m1.Nr(lam, T), lam, linear):
        dims[-1] = d
        nr = Nr.real
        absp = - Nr.imag / (7.95774715459e-09 * lam)
        stack.append(Block(dims, '{} [nr={} absp={}]'.format(material, nr, absp)))

    return stack


def simplify_all(geometry, lam, T=300., linear='nr'):
    """Replace all rectangular blocks with gradients with two-layer simplification
    in geometry tree.

    Args:
        geometry (: object to simplify
        lam (float): reference wavelength
        T (float): temperature for the refractive indices
        linear ('nr' or 'eps'): which parameter should be linear

    Returns:
        ~plask.geometry.GeometryObject: an object repacing the original leaf
    """

    def filter(item):
        if isinstance(item, (_geometry.Rectangle, _geometry.Cuboid)):
            new_item = simplify(item, lam, T, linear)
            if new_item is not item:
                return new_item

    return geometry.modify_objects(filter)
