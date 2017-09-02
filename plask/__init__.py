#coding: utf8
"""

PLaSK (Photonic Laser Simulation Kit) is a comprehensive toolkit for simulation
of various micro-scale photonic devices. It is particularly well suited for
analysis of semiconductor lasers, as it allows to perform simulations of various
physical phenomena with different models: thermal, electrical, quantum and optical.
PLaSK takes care of considering mutual interactions between these models and
allows to easily perform complex self-consistent analysis of complete devices.
"""

algorithm = module()
flow = module()
geometry = module()
h5py = module()
hdf5 = module()
license = dict()

class config(object):
    axes = None
    log = None


ARRAYID = None


class ComputationError(ArithmeticError):
    """
    Computational error in some PLaSK solver.
    """
    args = None
    
    message = None
    

def Data(array, mesh):
    """
    Data(array, mesh)
    
        Data returned by field providers.
        
        This class is returned by field providers and receivers and cointains the values
        of the computed field at specified mesh points. It can be passed to the field
        plotting and saving functions or even feeded to some receivers. Also, if the
        mesh is a rectangular one, the data can be converted into an multi-dimensional
        numpy array.
        
        You may access the data by indexing the :class:`~plask.Data` object, where the
        index always corresponds to the index of the mesh point where the particular
        value is specified. Hence, you may also iterate :class:`~plask.Data` objects as
        normal Python sequences.
        
        You may construct the data object manually from a numpy array and a mesh.
        The constructor always take two argumentsa as specified below:
        
        Args:
            array: The array with a custom data.
                It must be either a one dimensional array with sequential data of the
                desired type corresponding to the sequential mesh points or (for the
                rectangular meshes) an array with the same shape as returned by the
                :attr:`array` attribute.
            mesh: The mesh specifying where the data points are located.
                The size of the mesh must be equal to the size of the provided array.
                Furthermore, when constructing the data from the structured array, the
                mesh ordering must match the data stride, so it is possible to avoid
                data copying (defaults for both are fine).
        Returns:
            plask._Data: Data based on the specified mesh and array.
        
        Examples:
            To create the data from the flat sequential array:
        
            >>> msh = plask.mesh.Rectangular2D(plask.mesh.Rectilinear([1, 2, 3]),
            ... plask.mesh.Rectilinear([10, 20]))
            >>> Data(array([1., 2., 3., 4., 5., 6.]), msh)
            <plask.Data at 0x4698938>
        
            As the ``msh`` is a rectangular mesh, the data can be created from the
            structured array with the shape (3, 2), as the first and second mesh
            dimensions are 3 and 2, respectively:
        
            >>> dat = Data(array([[1., 2.], [3., 4.], [5., 6.]]), msh)
            >>> dat[0]
            1.0
        
            By adding one more dimension, you can create an array of vectors:
        
            >>> d = Data(array([[[1.,0.], [2.,0.]], [[3.,0.], [4.,1.]],
            ...                 [[5.,1.], [6.,1.]]]), msh)
            >>> d.dtype
            plask.vec
            >>> d[1]
            plask.vec(2, 0)
            >>> d.array[:,:,0]    # retrieve first components of all the vectors
            array([[1., 2.], [3., 4.], [5., 6.]])
        
        Construction of the data objects is efficient i.e. no data is copied in the
        memory from the provided array.
    """


class DataLog2(object):
    """
    Class used to log relations between two variables (argument and value)
    
    DataLog2(prefix, arg_name, val_name)
        Create log with specified prefix, name, and argument and value names
    """
    def __call__(self, arg, val):
        """
        __call__(self, arg, val)
        
            Log value pair
        """

    def count(self, arg, val):
        """
        count(self, arg, val)
        
            Log value pair and count successive logs
        """

    def reset(self):
        """
        reset(self)
        
            Reset logs counter
        """


LOG_CRITICAL = 'CRITICAL'
LOG_CRITICAL_ERROR = 'CRITICAL_ERROR'
LOG_DATA = 'DATA'
LOG_DEBUG = 'DEBUG'
LOG_DETAIL = 'DETAIL'
LOG_ERROR = 'ERROR'
LOG_ERROR_DETAIL = 'ERROR_DETAIL'
LOG_INFO = 'INFO'
LOG_RESULT = 'RESULT'
LOG_WARNING = 'WARNING'

class LoggingConfig(object):
    """
    Settings of the logging system
    """
    colors = None
    """
    Output color type ('ansi' or 'none').
    """

    level = None
    """
    Maximum log level.
    """

    output = None
    """
    Output destination ('stderr' or 'stdout').
    """

    def use_python(*args, **kwargs):
        """
        use_python() -> None :
            Use Python for log output.
            
            By default PLaSK uses system calls for printing. This is more efficient,
            but can make log not visible if PLaSK is used interactively. Call this method
            to use Python sys.stderr or sys.stdout for log printing.
        """



class Manager(_plask.Manager):
    """
    Main input manager.
    
    Object of this class provides methods to read the XML file and fetch geometry
    objects, pathes, meshes, and generators by name. It also allows to access
    solvers defined in the XPL file.
    
    Some global PLaSK function like :func:`~plask.loadxpl` or :func:`~plask.runxpl`
    create a default manager and use it to load the data from XPL into ther global
    namespace.
    
    Manager(materials=None, draft=False)
    
    Args:
        materials: Material database to use.
                   If *None*, the default material database is used.
        draft (bool): If *True* then partially incomplete XML is accepted
                      (e.g. non-existent materials are allowed).
    """
    class _Roots(object):
        pass

    _roots = None
    """
    Root geometries.
    """

    _scriptline = None
    """
    First line of the script.
    """

    defs = None
    """
    Local defines.
    
    This is a combination of the values specified in the :xml:tag:`<defines>`
    section of the XPL file and the ones specified by the user in the
    :meth:`~plask.Manager.load` method.
    """

    draft_material = None
    """
    Flag indicating if unknown materials are allowed. If True then dummy material
    is created if the proper one cannot be found in the database.
    Otherwise an exception is raised.
    """

    def export(self, target):
        """
        export(self, target)
        
            Export loaded objects into a target dictionary.
            
            All the loaded solvers are exported with keys equal to their names and the other objects
            under the following keys:
            
            * geometries and geometry objects (:attr:`~plask.Manager.geo`): ``GEO``,
            
            * paths to geometry objects (:attr:`~plask.Manager.pth`): ``PTH``,
            
            * meshes (:attr:`~plask.Manager.msh`): ``MSH``,
            
            * mesh generators (:attr:`~plask.Manager.msg`): ``MSG``,
            
            * custom defines (:attr:`~plask.Manager.defs`): ``DEF``.
        """

    geo = None
    """
    Dictionary of all named geometries and geometry objects.
    """

    def load(self, source, vars=None, sections=None):
        """
        load(self, source, vars, sections=None)
        
            Load data from source.
            
            Args:
                source (string or file): File to read.
                    The value of this argument can be either a file name, an open file
                    object, or an XML string to read.
                vars (dict): Dictionary of user-defined variables (which string keys).
                    The values of this dictionary overrides the ones given in the
                    :xml:tag:`<defines>` section of the XPL file.
                sections (list): List of section to read.
                    If this parameter is given, only the listed sections of the XPL file are
                    read and the other ones are skipped.
        """

    msg = None
    """
    Dictionary of all named mesh generators.
    """

    msh = None
    """
    Dictionary of all named meshes.
    """

    pth = None
    """
    Dictionary of all named paths.
    """

    script = None
    """
    Script read from XML file.
    """

    solvers = None
    """
    Dictionary of all named solvers.
    """


PROCID = None

class ScaledLightMagnitude(plask.flow.LightMagnitudeProvider2D):
    """
    Scaled provider for optical field magnitude
    """
    def __call__(*args, **kwargs):
        """
        __call__(self, mesh, interpolation='DEFAULT')
        __call__(self, n, mesh, interpolation='DEFAULT')
        
            Get value from the provider.
            
            :param int n: Value number.
            :param mesh mesh: Target mesh to get the field at.
            :param str interpolation: Requested interpolation method.
        """

    scale = None



class Solver(object):
    """
    Base class for all solvers.
    """
    id = None
    """
    Id of the solver object. (read only)
    
    Example:
        >>> mysolver.id
        mysolver:category.type
    """

    def __init__(self, name=None):
        pass

    def initialize(self):
        """
        initialize(self)
        
            Initialize solver.
            
            This method manually initialized the solver and sets :attr:`initialized` to
            *True*. Normally calling it is not necessary, as each solver automatically
            initializes itself when needed.
            
            Returns:
                bool: solver :attr:`initialized` state prior to this method call.
        """

    initialized = None
    """
    True if the solver has been initialized. (read only)
    
    Solvers usually get initialized at the beginning of the computations.
    You can clean the initialization state and free the memory by calling
    the :meth:`invalidate` method.
    """

    def invalidate(self):
        """
        invalidate(self)
        
            Set the solver back to uninitialized state.
            
            This method frees the memory allocated by the solver and sets
            :attr:`initialized` to *False*.
        """

    def load_xml(self, xml, manager):
        """
        load_xml(self, xml, manager)
        
            Load configuration from XML reader.
            
            This method should be overriden in custom Python solvers.
            
            Example:
                >>> def load_xml(self, xml, manager):
                ...     for tag in xml:
                ...         if tag.name == 'something':
                ...             for sub in tag:
                ...                 if sub.name == 'withtext':
                ...                     self.text = sub.text
                ...         elif tag.name == 'config':
                ...             self.a = tag['a']
                ...             self.b = tag.get('b', 0)
                ...         elif tag.name == 'geometry':
                ...             self.geometry = manager.geo[tag['ref']]
        """



class StepProfile(object):
    """
        Step profile for use in custom providers.
    
        Create a step profile class that can set a constant value of any scalar field
        in an arbitrary geometry object. Typical use of this class is setting an
        arbitrary heat source or step-profile material gain located in a chosen geometry
        object.
    
        Args:
            geometry: Geometry in which the step-profile is defined.
                It must be known in order to properly map the absolute mesh coordinates
                to the step-profile items.
            default: Default value of the provided field, returned in all non-referenced
                geometry objects.
            dtype: Type of the returned value. Defaults to `None`, in which case it is
                determined by the type of `default`.
    
        After creation, set the desired values at chosen geometry objects using item
        access [] notation:
    
        >>> profile[geometry_object] = value
    
        Then, you may retrieve the provider of a desired type using the normal outXXX
        name:
    
        >>> solver.inProperty = profile.outProperty
    
        This way you create a provider of the proper type and  associate it with the
        profile, so each time, the profile is in any way changed, all the receivers
        connected to the provider get notified.
    
        Example:
            To create a heat source profile that sets some heat at the object named
            `hot`:
    
            >>> hot = geometry.Rectangle(20,2, 'GaAs')
            >>> cold = geometry.Rectangle(20,10, 'GaAs')
            >>> stack = geometry.Stack2D()
            >>> stack.prepend(hot)
            <plask.geometry.PathHint at 0x47466b0>
            >>> stack.prepend(cold)
            <plask.geometry.PathHint at 0x47469e0>
            >>> geom = geometry.Cylindrical2D(stack)
            >>> profile = StepProfile(geom)
            >>> profile[hot] = 1e7
            >>> receiver = flow.HeatReceiverCyl()
            >>> receiver.connect(profile.outHeat)
            >>> list(receiver(mesh.Rectangular2D([10], [5, 11])))
            [0.0, 10000000.0]
            >>> receiver.changed
            False
            >>> profile[hot] = 2e7
            >>> receiver.changed
            True
            >>> list(receiver(mesh.Rectangular2D([10], [5, 11])))
            [0.0, 20000000.0]
    """
    def __call__(self, mesh, *args):
        pass


    def _fix_key(self, key):
        pass


    default = None
    """
    Default value of the profile.
    
               This value is returned for all mesh points that are located outside any
               of the geometry objects with a specified value.
    """

    geometry = None
    """
    Profile geometry. (read only)
    """



class XMLError(exceptions.Exception):
    """
    Error in XML file.
    """
    args = getset_descriptor()
    
    message = getset_descriptor()
    


class XmlReader(object):
    class _Iterator(object):
        def next(self):
            """
            next(self)
            """


    attrs = None
    """
    List of all the tag attributes.
    """

    def get(self, key, default=None):
        """
        get(self, key, default=None)
        
            Return tag attribute value or default if the attribute does not exist.
        """

    name = None
    """
    Current tag name.
    """

    text = None
    """
    Text in the current tag.
    """



class XmlWriter(object):
    """
    XML writer that can save existing geometries and meshes to the XML.
    
    Objects of this class contain three dictionaries:
    :attr:`~plask.XmlWriter.geometry` and :attr:`~plask.XmlWriter.mesh`
    that should contain the geometries or meshes, which should be saved and
    :attr:`~plask.XmlWriter.names` with other geometry objects that should be
    explicitly named in the resulting XML. All these dictionaries must have strings
    as their keys and corresponding objects as values.
    
    Args:
        geo (dict): Dictionary with geometries that should be saved to the file.
        mesh (dict): Dictionary with meshes that should be saved to the file.
        names (dict): Dictionary with names of the geometry objects that should be
                      explicitly named in the file.
    
    The final xml can be simply retrieved as a string (``str(writer)``) or saved to
    an XPL file with the :meth:`~plask.XmlWriter.saveto` method.
    
    Example:
        Create an XML file with a simple geometry:
    
        >>> rect = plask.geometry.Rectangle(2, 1, 'GaAs')
        >>> geo = plask.geometry.Cartesian2D(rect)
        >>> xml = plask.XmlWriter({'geo': geo}, {}, {'rect': rect})
        >>> print(xml)
        <plask>
          <geometry>
            <cartesian2d name="geo" axes="zxy">
              <extrusion length="inf">
                <block2d name="rect" material="GaAs" dx="2" dy="1"/>
              </extrusion>
            </cartesian2d>
          </geometry>
          <grids/>
        </plask>
    """
    geometry = None
    """
    Dictionary with geometries that should be saved to the file.
    """

    mesh = None
    """
    Dictionary with meshes that should be saved to the file.
    """

    names = None
    """
    Dictionary with names of the geometry objects that should be explicitly named
    in the file.
    """

    def saveto(self, target):
        """
        saveto(self, target)
        
            Save the resulting XML to the file.
            
            Args:
                target (string or file): A file name or an open file object to save to.
        """


_any = any # function alias

_os = module()

_plask = module()

def _showwarning(message, category, filename, lineno, file=None, line=None):
    """
        Implementation of showwarnings which redirects to PLaSK logs
    """

def axeslist_by_name(self):
    """
    axeslist_by_name(self)
    """


def load_field(file, path=''):
    """
        Load field from HDF5 file.
    
        Args:
            file (str or file): File to load from.
                It should be eiher a filename or a h5py.File object opened for
                reading.
            path (str): HDF5 path (group and dataset name), under which the
               data is located in the HDF5 file.
        Returns:
            Read plask.Data object.
    
        If ``file`` is a string, a new HDF5 file is opened for reading. Then both
        the data and its mesh are read from this file from the path specified by
        the ``path`` argument. ``path`` can contain slashes ('/'), in which case
        the corresponding hierarchy in the HDF5 file is used.
    
        Example:
           You may save data retrieved from a provider to file as follows:
    
           >>> data = my_solver.outMyData(my_mesh)
           >>> save_field('myfile.h5', data, 'mygroup/mydata', 'a')
    
           In another PLaSK session, you may retrieve this data and plot it
           or provide to some receiver:
    
           >>> data = load_field('myfile.h5', 'mygroup/mydata')
           >>> plot_field(data)
           >>> other_solver.inMyData = data
    """

def load_rectangular1d(src_group, name):
    pass


def loadxpl(source, vars={}, sections=None, destination=None, update=False):
    """
        Load the XPL file. All sections contents is read into the `destination` scope.
    
        Args:
            source (str): Name of the XPL file or open file object.
            vars (dict): Optional dictionary with substitution variables. Values
                         specified in the <defines> section of the XPL file are
                         overridden with the one specified in this parameter.
            sections (list): List of section names to read.
            destination (dict): Destination scope. If None, ``globals()`` is used.
            update (bool): If the flag is ``False``, all data got from the previous
                           call to :func:`loadxpl` are discarded. Set it to ``True``
                           if you want to append some data from another file.
    """

material = module()

mesh = module()

numpy = module()

phys = module()

plask = module()

prefix = '/usr/local'
def print_exc():
    """
    Print last exception to PLaSK log.
    """

def print_log(*args, **kwargs):
    """
    object print_log(tuple args, dict kwds) :
        print_log(level, *args)
        
        Print log message into a specified log level.
        
        Args:
            level (str): Log level to print the message to.
            args: Items to print. They are concatenated togeter and separated by space,
                  similarly to the ``print`` function.
    """

def runxpl(source, vars={}):
    """
        Load and run the code from the XPL file. Unlike :func:`loadxpl` this function
        does not modify the current global scope.
    
        Args:
            source (str): Name of the XPL file or open file object.
            vars (dict): Optional dictionary with substitution variables. Values
                         specified in the <defines> section of the XPL file are
                         overridden with the one specified in this parameter.
    """

def save_field(field, file, path='', mode='a'):
    """
        Save field to HDF5 file.
    
        Args:
            file (str or file): File to save to.
                It should be eiher a filename or a h5py.File object opened for
                writing.
            field (plask.Data): Field to save.
               It should be an object returned by PLaSK provider that contains
               calculated field.
            path (str): HDF5 path (group and dataset name), under which the
               data is saved in the HDF5 file.
            mode (str): Mode used for opening new files.
    
        If ``file`` is a string, a new HDF5 file is opened with the mode
        specified by ``mode``. Then both the data and its mesh are written to
        this file under the path specified by the ``path`` argument. It can
        contain slashes ('/'), in which case a corresponding hierarchy is created
        in the HDF5 file.
    
        Saved data can later be restored by any HDF5-aware application, or by
        ``load_field`` function of PLaSK.
    
        Example:
           You may save data retrieved from a provider to file as follows:
    
           >>> data = my_solver.outMyData(my_mesh)
           >>> save_field('myfile.h5', data, 'mygroup/mydata', 'a')
    
           In another PLaSK session, you may retrieve this data and plot it
           or provide to some receiver:
    
           >>> data = load_field('myfile.h5', 'mygroup/mydata')
           >>> plot_field(data)
           >>> other_solver.inMyData = data
    """

def save_rectangular1d(dest_group, name, mesh):
    pass


def vec(*args, **kwargs):
    """
    vec(x,y,z, dtype=None)
    vec(z,x,y, dtype=None)
    vec(r,p,z, dtype=None)
    vec(x,y, dtype=None)
    vec(z,x, dtype=None)
    vec(r,z, dtype=None)
    
    PLaSK vector.
    
    The constructor arguments depend on the current value of
    :attr:`plask.config.axes`. However, you must either specify all the components
    either as the unnamed sequence or as the named keywords.
    
    Args:
        _letter_ (dtype): Vector components.
            Their choice depends on the current value of :attr:`plask.config.axes`.
        dtype (type): type of the vector components.
            If this argument is omitted or `None`, the type is determined
            automatically.
    
    The order of vector components is always [`longitudinal`, `transverse`,
    `vertical`] for 3D vectors or [`transverse`, `vertical`] for 2D vectors.
    However, the component names depend on the :attr:`~plask.config.axes`
    configuration option. Changing this option will change the order of component
    names (even for existing vectors) accordingly:
    
    ============================== ======================== ========================
    plask.config.axes value        2D vector components     3D vector components
    ============================== ======================== ========================
    `xyz`, `yz`, `z_up`            [`y`, `z`]               [`x`, `y`, `z`]
    `zxy`, `xy`, `y_up`            [`x`, `y`]               [`z`, `x`, `y`]
    `prz`, `rz`, `rad`             [`r`, `z`]               [`p`, `r`, `z`]
    `ltv`, `abs`                   [`t`, `v`]               [`l`, `t`, `v`]
    `long,tran,vert`, `absolute`   [`tran`, `vert`]         [`long`, `tran`, `vert`]
    ============================== ======================== ========================
    
    Examples:
        Create two-dimensional vector:
    
        >>> vector(1, 2)
        vector(1, 2)
    
        Create 3D vector specifying components in rotated coordinate system:
    
        >>> config.axes = 'xy'
        >>> vec(x=1, y=2, z=3)
        plask.vec(3, 1, 2)
    
        Create 3D vector specifying components:
    
        >>> config.axes = 'xyz'
        >>> vec(x=1, z=2, y=3)
        plask.vec(1, 3, 2)
    
        Create 2D vector in cylindrical coordinates, specifying dtype:
    
        >>> config.axes = 'rz'
        >>> vec(r=2, z=0, dtype=complex)
        plask.vec((2+0j), (0+0j))
    
    To access vector components you may either use attribute names or numerical
    indexing. The ordering and naming rules are the same as for the construction.
    
    Examples:
    
        >>> config.axes = 'xyz'
        >>> v = vec(1, 2, 3)
        >>> v.z
        3
        >>> v[0]
        1
    
    You may perform all the proper algebraic operations on PLaSK vectors like
    addition, subtraction, multiplication by scalar, multiplication by another
    vector (which results in a dot product).
    
    Example:
    
        >>> v1 = vec(1, 2, 3)
        >>> v2 = vec(10, 20, 30)
        >>> v1 + v2
        plask.vec(11, 22, 33)
        >>> 2 * v1
        plask.vec(2, 4, 6)
        >>> v1 * v2
        140.0
        >>> abs(v1)
        >>> v3 = vec(0, 1+2j)
        >>> v3.conj()
        plask.vec(0, 1-2j)
        >>> v3.abs2()
        5.0
    """

version = None
version_major = None
version_minor = None
