def geometry_object_names(constructor, *allowed_types):
    """:return: list of names"""
    if len(allowed_types) == 0:
        from .types import geometry_types_2d, geometry_types_3d, geometry_types_other
        return geometry_object_names(constructor, geometry_types_2d, geometry_types_3d, geometry_types_other)
    res = []
    for t in allowed_types:
        for n, c in t.items():
            if c == constructor: res.append(n)
    return res


def construct_geometry_object(element, conf, *allowed_types):
    if element is None: return None
    if len(allowed_types) == 0:
        from .types import geometry_types_2d, geometry_types_3d, geometry_types_other
        d = conf.dim
        if d == 2: return construct_geometry_object(element, conf, geometry_types_2d, geometry_types_other)
        elif d == 3: return construct_geometry_object(element, conf, geometry_types_3d, geometry_types_other)
        else: return construct_geometry_object(element, conf, geometry_types_2d, geometry_types_3d, geometry_types_other)
    s = conf.suffix
    for m in allowed_types:
        c = m.get(element.tag)
        if c is None and s is not None: c = m.get(element.tag + s)
        if c is not None: return c(element, conf)
    raise ValueError('Unexpected tag: <{}>'.format(element.tag))