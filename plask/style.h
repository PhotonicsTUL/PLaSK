/** \file
This is an empty file containing only style guide for Doxygen.

\page style Coding style guide

Below you have the style rules that should be obeyed when writing your code for PLaSK.
Its aim is to ensure consistent style for all solvers, which makes them easier and
more natural to use by both other developers and the the end user. It is split into
two parts: first the naming convention for classes, methods, functions, variables, etc.
in C++ is discussed, and later there are other important considerations ensuring that
the user experience is unique.

Please obey the guidelines stated in both sections, as more often than not, the naming scheme
in C++ is reflected in Python interface! You are strongly advised to keep to this guidelines
everywhere in your code, however in its private parts it is allowed (although \a really \a strongly
discouraged) to use your own style. For public interface consider the rules stated here as more
obligatory than not.

The reasoning of this style guide is consistency. But you should also know when to be inconsistent
-- sometimes the style guide just doesn't apply. When in doubt, use your best judgment. Look at other
examples and decide what looks best. And don't hesitate to ask!

Two good reasons to break a particular rule:

- When applying the rule would make the code less readable, even for someone who is used
  to reading code that follows the rules.
  .
- To be consistent with surrounding code that also breaks it (maybe for historic reasons) -- although
  this is also an opportunity to clean up someone's else mess.


\section style_cxx Style guide for writing in C++

The formatting of the source is left to your taste, however the general rule of tjum is to make
it legible! The more specific rules are below:

- Always indent logical blocks: class content, function/method body, loops, conditional clauses, etc.
  Generally indent everything around which you put, or could put the `{...}` braces. The only exception
  is the global namespace \p plask and \p plask::solvers::your_solver, which contents should start from
  the first column.
  .
- The official PLaSK language ins English. Every name, string, or comment should be written in English.
  The used character encoding must be UTF-8 (make sure you set-up your editor properly for this).
  .
- Put spaces at both sides of binary operators and at outer sides of brackets `(...)` and `[...]`. Also
  for inline comments you should put single space after double slash: <tt>// comment</tt>.


\subsection naming Naming convention

- First rule of thumb: keep your names short and easy, but at the same time make sure that they are
  meaningful.
  .
- Use \c AllCapitalizedNames for class names.
  .
- For method names use \c mixedCase style, with the first letter being lowercase.
  .
  As a sporadic exception you can chose \c lower_case_with_underscores names for very technical methods
  written for internal use (\c DataVector::remove_const being an example). However, never do this for
  methods exported to the user interface!
  .
- For class fields and variables it is best to use a single word: just a name or a typical physical symbol.
  For short two-word terms it is best to glue them together (e.g. \c bbox for bounding box) unless it obscures
  the legibility. If this rule is impossible to hold, you are allowed to sometimes use \c lower_case_with_underscores.
  .
- Do not use prefixed Hungarian notation for class members and variable names.
  .
- The only prefixed class fields in PLaSK are providers and receivers:
  - receivers are always prefixed with \a in, as \c inTemperature,
  - providers have \a out for their prefix, as \c outVoltage.
  .
- In all your names respect case convention regardless if you use abbreviations or not. So the correct name would be
  \c FemSolver and not \c FEMSolver.
  .
- If your solver provides only a single calculation method, name it \c compute. Otherwise name the methods such a way
  that it is clear what they do.


\section style_interfce Style guide for user interface (Python and XML)

Generally Python interface must follow the \link naming naming convention \endlink for C++. The macros prepared
for creating the interface the easy way assume that the same name is used for methods and class fields in both
languages (with the exception of Python properties, where you specify its name yourself). However in this case,
the rules MUST be even more strictly obeyed, as this interface is visible to the end user and the consistency
at this level is crucial!

- Solver library names should be all lowercase.
  .
- Python names of your classes should rather reflect the method used, not the analyzed phenomena (in the user inerface
  they are prefixed with the category). Furthermore they must be suffixed with \c 2D, \c 3D, or \c Cyl, which states
  the type of the geometry of the solver. So the correct name for e.g. a thermal solver in a 2D Cartesian geometry
  would be \c Fem2D. User would see it as <code>thermal.Fem2D</code>.
  .
- XML configuration tag names must match the Python class properties for accessing the same configuration parameter
  (the usual rules for class fields names apply). The possible exceptions are stated below
  .
- When reading the configuration from XML put the parameters as attributes either of a single tag <code><options></code>, or
  better grouped under some other tags. Python properties need not to be similarly grouped (although they may).
  .
- Boundary conditions in XML should be named by stating what is fixed (in lowercase). In Python they are suffixed with
  \p _boundary. So for example constant temperature should be named as <code><temperature></code> in XML and \c temperature_boundary
  in Python.

*/