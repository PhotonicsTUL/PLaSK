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

- To be consistent with surrounding code that also breaks it (maybe for historic reasons) -- although
  this is also an opportunity to clean up someone's else mess.


\section style_cxx Style guide for writing in C++

The formatting of the source is left to your taste, however the general rule of tjum is to make
it legible! The more specific rules are below:

- Always indent logical blocks: class content, function/method body, loops, conditional clauses, etc.
  Generally indent everything around which you put, or could put the `{...}` braces. The only exception
  is the global namespace \p plask and \p plask::solvers::your_solver, which contents should start from
  the first column.

- The official PLaSK language ins English. Every name, string, or comment should be written in English.
  The used character encoding must be UTF-8 (make sure you set-up your editor properly for this).

- Put spaces at both sides of binary operators and at outer sides of brackets `(...)` and `[...]`. Also
  for inline comments you should put single space after double slash: <tt>// comment</tt>.


\subsection naming Naming convention

- First rule of thumb: keep your names short and easy, but at the same time make sure that they are
  meaningful.

- Use \c AllCapitalizedNames for class names.

- For method names use \c mixedCase style, with the first letter being lowercase.

  As a sporadic exception you can chose \c lower_case_with_underscores names for very technical methods
  written for internal use (\c DataVector::remove_const being an example). However, never do this for
  methods exported to the user interface!

- For class fields and variables it is best to use a single word: just a name or a typical physical symbol.
  For short two-word terms it is best to glue them together (e.g. \c bbox for bounding box) unless it obscures
  the legibility. If this rule is impossible to hold, you are allowed to sometimes use \c lower_case_with_underscores.

- Do not use prefixed Hungarian notation for class members and variable names. It is really more legible to have
  just \c index as loop index than \c tIndex.

- The only prefixed class fields in PLaSK are providers and receivers:
   - receivers are always prefixed with \a in, as \c inTemperature,
   - providers have \a out for their prefix, as \c outVoltage.


\section style_interfce Style guide for user interface (Python and XML)

Generally Python interface must follow the \link naming naming convention \endlink for C++. The macros prepared
for creating the interface the easy way assume that the same name is used for methods and class fields in both
languages (with the exception of Python properties, where you specify its name yourself). However in this case,
the rules MUST be even more strictly obeyed, as this interface is visible to the end user and the consistency
at this level is crucial!

*/