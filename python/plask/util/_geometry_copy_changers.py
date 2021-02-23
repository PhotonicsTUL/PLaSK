from .. import Manager, geometry

@Manager._geometry_changer('simplify-gradients')
class GradientChanger:

    def __init__(self, xpl, manager):
        self.lam = xpl['lam']
        self.T = xpl.get('temp', 300.)
        self.linear = xpl.get('linear', 'nr')
        if self.linear != 'eps' and self.linear.lower() != 'nr':
            raise ValueError("'linear' argument must be either 'eps' or 'nr'")
        self.dT = xpl.get('dtemp', 100.)
        self.only_role = xpl.get('only-role')

    def __call__(self, item):
        from .gradients import simplify
        if isinstance(item, (geometry.Rectangle, geometry.Cuboid)) and (self.only_role is None or self.only_role in item.roles):
            new_item = simplify(item, self.lam, self.T, self.linear, self.dT)
            if new_item is not item:
                return new_item
