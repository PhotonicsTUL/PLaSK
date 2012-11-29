#!/usr/bin/env plask

import sys
import os

def pad(val, l=7):
    s = str(val)
    return s + ''.join([' ']*max(l-len(s), 0))

def parse_material(mat):
    mat = str(mat).split(':')
    if len(mat) == 1:
        return mat[0], 'ST', 0.
    dp, dc = mat[i].split('=')
    if dp[-2] == ' ': dp = dp[:-2] # there is no common way to get dopant concentration from carriers concentration
    return mat[0], dp, float(dc)
    
    
def write_dan(name, allm=True):
    '''Write dan files for given prefix
    
       'allm' indicates whether consider all materials as ones with constant values
    '''

    print("Writing %s_temp.dan" % name)
    ofile = open(name+'_temp.dan', 'w')
    def out(text):
        ofile.write(text + ' ')
    def outl(text):
        ofile.write(text + '\n')

    geo = GEO.values()[0]
        
    outl(name)
    outl("All             material_lasera")
    if type(geo) == geometry.Cylindrical2D:
        outl("1       1       symetria_osiowa_i_dlugosc_lasera")
    elif type(geo) == geometry.Cartesian2D:
        outl("0       %g      symetria_osiowa_i_dlugosc_lasera" % geometry.extrusion.length)
    else:
        raise TypeError("3D geometry not supported")
   
    leafs = geo.get_leafs()
    boxes = geo.get_leafs_bboxes()
    
    outl("%s 1e-6    ilosc_obszarow_i_skala_wymiarowania" % pad(len(leafs)))
    
    for i,(obj,box) in enumerate(zip(leafs, boxes)):
        # first line
        out("\n" + pad(i+1))
        out(pad(box.lower[0])); out(pad(box.lower[1])); out(pad(box.upper[0])); out(pad(box.upper[1]));
        point = vec(0.5*(box.lower[0]+box.upper[0]), 0.5*(box.lower[1]+box.upper[1]))
        mat = geo.get_material(point)
        mn, dp, dc = parse_material(mat)
        outl(mn)
        # second line
        if geo.child.has_role('active', point):
            mt = 'j'
            cx, cy = 1e-6, 0.2
            #TODO get it from the solver
        elif allm or mat.cond(300.) == mat.cond(400.):
            mt = 'm'
            try: cx, cy = mat.cond(300.)
            except NotImplementedError: cx, cy = mat.air.cond(300.)
        else:
            raise NotImplementedError("Allowing not constant maters is not implemented")
        outl( "%s %s %s       przewodnosc_wlasciwa" % (pad(cx,15), pad(cy,15), mt) )
        # third line
        outl( "%s %s koncentracja_domieszka" % (pad(dc,31), pad(dp)) )
        # fourth line
        if allm or mat.thermk(300.) == material.thermk(400.):
            mt = 'm'
            try: kx, ky = mat.thermk(300.)
            except NotImplementedError: kx, ky = material.air.thermk(300.)
        else:
            raise NotImplementedError("Allowing not constant materials is not implemented")
        outl( "%s %s %s       przewodnosc_wlasciwa" % (pad(kx,15), pad(ky,15), mt) )
        # firth line
        if geo.child.has_role('noheat', point):
            outl("0               0.0                     wydajnosc_zrodel_ciepla")
        elif geo.child.has_role('active', point):
            outl("-200            0.0                     wydajnosc_zrodel_ciepla")
        # TODO constant heats
        else:
            outl("-100            0.0                     wydajnosc_zrodel_ciepla")

    outl("")
    
    outl("0                                       warunki_brzegowe_potencjal")      #TODO
    outl("0                                       warunki_brzegowe_temperatura")    #TODO
    outl("0                                       warunki_brzegowe_konwekcja")      #TODO
    outl("0                                       warunki_brzegowe_strumien")       #TODO
    outl("0                                       warunki_brzegowe_radiacja")       #TODO
    
    outl("KONIEC")
             

if __name__ == "__main__":

    code = 0

    try:
        iname = sys.argv[1]
    except IndexError:
        sys.stderr.write("Usage: %s input_file_temp.dan\n" % sys.argv[0])
        code = 2
    else:
        dest_dir = os.path.dirname(iname)
        name = os.path.join(dest_dir, iname[:-4])

        try:
            load(iname)
            if len(GEO) != 1:
                raise ValueError("More than one geometry defined in %s" % iname)
            write_dan(name)
        except Exception as err:
            import traceback as tb
            tb.print_exc()
            #sys.stderr.write("\n%s: %s\n" % (err.__class__.__name__, err))
            code = 1
        else:
            print("\nDone!")

    sys.exit(code)
