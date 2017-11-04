#coding: utf8
import exceptions
import plask.flow
import _plask
import __builtin__


class ComputationError(exceptions.ArithmeticError):
    """
    Computational error in some PLaSK solver.
    """
    args = __builtin__.getset_descriptor()
    
    message = __builtin__.getset_descriptor()
    

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
LOG_IMPORTANT = 'IMPORTANT'
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

    draft = None
    """
    Flag indicating draft mode. If True then dummy material is created if the proper
    one cannot be found in the database.
     Also some objects do not need to have all
    the atttributes set, which are then filled with some reasonable defaults.Otherwise an exception is raised.
    """

    def export(self, target):
        """
        export(self, target)
        
            Export loaded objects into a target dictionary.
            
            All the loaded solvers are exported with keys equal to their names and the other objects
            under the following keys:
            
            * geometries and geometry objects (:attr:`~plask.Manager.geo`): ``GEO``,
            
            * paths to geometry objects (:attr:`~plask.Manager.pth`): ``PTH``,
            
            * meshes and generators (:attr:`~plask.Manager.msh`): ``MSH``,
            
            * custom defines (:attr:`~plask.Manager.defs`): ``DEF``.
        """

    geo = None
    """
    Dictionary of all named geometries and geometry objects.
    """

    def load(self, source, vars, sections=None):
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
    Dictionary of all named meshes and generators.
    """

    msh = None
    """
    Dictionary of all named meshes and generators.
    """

    overrites = None
    """
    Overriden local defines.
    
    This is a list of local defines that have been overriden in a ``plask`` command
    line or specified as a ``vars`` argument to the :meth:`~plask.Manager.load`
    method.
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
    
    Solver(name='')
    
    Args:
        name: Solver name for its identification in logs.
    You should inherit this class if you are creating custom Python solvers
    in Python, which can read its configuration from the XPL file. Then you need to
    override the :meth:`load_xml` method, which reads the configuration. If you
    override :meth:`on_initialize` of :meth:`on_invalidate` methods, they will be
    called once on the solver initialization/invalidation.
    
    Example:
      .. code-block:: python
    
         class MySolver(Solver):
    
             def __init__(self, name=''):
                 super(MySolver, self).__init__(name)
                 self.param = 0.
                 self.geometry = None
                 self.mesh = None
                 self.workspace = None
                 self.bc = plask.mesh.Rectangular2D.BoundaryConditions()
    
             def load_xpl(self, xpl, manager):
                 for tag in xpl:
                     if tag == 'config':
                         self.param = tag.get('param', self.param)
                     elif tag == 'geometry':
                         self.geometry = tag.getitem(manager.geo, 'ref')
                     elif tag == 'mesh':
                         self.mesh = tag.getitem(manager.msh, 'ref')
                     elif tag == 'boundary':
                         self.bc.read_from_xpl(tag, manager)
    
             def on_initialize(self):
                 self.workspace = zeros(1000.)
    
             def on_invalidate(self):
                 self.workspace = None
    
             def run_computations(self):
                 pass
    
    To make your solver visible in GUI, you must write the ``solvers.yml`` file
    and put it in the same directory as your data file.
    
    Example:
      .. code-block:: yaml
    
         - solver: MySolver
           lib: mymodule
           category: local
           geometry: Cartesian2D
           mesh: Rectangular2D
           tags:
           - tag: config
             label: Solver Configuration
             help: Configuration of the effective model of p-n junction.
             attrs:
             - attr: param
               label: Parameter
               type: float
               unit: V
               help: Some voltage parameter.
           - bcond: boundary
             label: Something
    """
    id = None
    """
    Id of the solver object. (read only)
    
    Example:
        >>> mysolver.id
        mysolver:category.type
    """

    def __init__(self, name=''):
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

    def load_xpl(self, xpl, manager):
        """
        load_xpl(self, xpl, manager)
        
            Load configuration from XPL reader.
            
            This method should be overriden in custom Python solvers.
            
            Example:
              .. code-block:: python
            
                 def load_xpl(self, xpl, manager):
                     for tag in xpl:
                         if tag == 'config':
                             self.a = tag['a']
                             self.b = tag.get('b', 0)
                             if 'c' in tag:
                                 self.c = tag['c']
                         if tag == 'combined':
                             for subtag in tag:
                                 if subtag == 'withtext':
                                     self.data = subtag.attrs
                                     # Text must be read last
                                     self.text = subtag.text
                         elif tag == 'geometry':
                             self.geometry = tag.getitem(manager.geo, 'ref')
        """



class XMLError(exceptions.Exception):
    """
    Error in XML file.
    """
    args = __builtin__.getset_descriptor()
    
    message = __builtin__.getset_descriptor()
    

def XmlWriter(geo=None, msh=None, names=None):
    """
    XmlWriter(geo=None, msh=None, names=None)
    """


class XplReader(object):
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

    def getitem(self, key, default=''):
        """
        getitem(self, key, default='')
        
            Return tag attribute value as raw string or default if the attribute does not exist.
        """

    name = None
    """
    Current tag name.
    """

    text = None
    """
    Text in the current tag.
    """



class XplWriter(object):
    """
    XPL writer that can save existing geometries and meshes to the XPL.
    
    Objects of this class contain three dictionaries:
    :attr:`~plask.XplWriter.geometry` and :attr:`~plask.XplWriter.mesh`
    that should contain the geometries or meshes, which should be saved and
    :attr:`~plask.XplWriter.names` with other geometry objects that should be
    explicitly named in the resulting XPL. All these dictionaries must have strings
    as their keys and corresponding objects as values.
    
    Args:
        geo (dict): Dictionary with geometries that should be saved to the file.
        mesh (dict): Dictionary with meshes that should be saved to the file.
        names (dict): Dictionary with names of the geometry objects that should be
                      explicitly named in the file.
    
    The final XPL can be simply retrieved as a string (``str(writer)``) or saved to
    a file with the :meth:`~plask.XplWriter.saveto` method.
    
    Example:
        Create an XML file with a simple geometry:
    
        >>> rect = plask.geometry.Rectangle(2, 1, 'GaAs')
        >>> geo = plask.geometry.Cartesian2D(rect)
        >>> xml = plask.XplWriter({'geo': geo}, {}, {'rect': rect})
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
        
            Save the resulting XPL to the file.
            
            Args:
                target (string or file): A file name or an open file object to save to.
        """



_Data = Data # class alias

__xml__globals = __builtin__.dict()

def _print_exception(exc_type, exc_value, exc_traceback, scriptname='', second_is_script=False):
    """
    _print_exception(exc_type, exc_value, exc_traceback, scriptname='', second_is_script=False)
    
        Print exception information to PLaSK logging system
    """

def axeslist_by_name(self):
    """
    axeslist_by_name(self)
    """

config = _plask.config()

flow = __builtin__.module()

geometry = __builtin__.module()

lib_path = '/home/maciek/Dokumenty/PLaSK/plask/build-release/lib/plask/'
material = __builtin__.module()

mesh = __builtin__.module()

prefix = '/home/maciek/Dokumenty/PLaSK/plask/build-release'
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

version = 'plask_version'
