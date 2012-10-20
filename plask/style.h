/** \file
This is an empty file cointaining only style guide for Doxygen.

\page style Coding style guide

Below you have the style rules that should be obeyed when writing your code for PLaSK.
Its aim is to ensure consistent style for all solvers, which makes them easier and
more natural to use by both other developers and the the end user. It is split into
two parts: first the naming convention for classes, methods, functions, variables, etc.
in C++ is discussed, and later there are other important considerations ensuring that
the user experience is unique.

Plase obey the guidelines stated in both sections, as more often than not, the naming scheme
in C++ is reflected in Python interface!

The reasonig of this style guide is consistency. But you should also know when to be inconsistent
-- sometimes the style guide just doesn't apply. When in doubt, use your best judgment. Look at other
examples and decide what looks best. And don't hesitate to ask!

Two good reasons to break a particular rule:

- When applying the rule would make the code less readable, even for someone who is used
  to reading code that follows the rules.

- To be consistent with surrounding code that also breaks it (maybe for historic reasons) -- although
  this is also an opportunity to clean up someone else's mess.


\section style_c++ Style guide for writing in C++

The formating of the source is left to your taste, however the general rule of tjum is to make
it legible! The more specific rules are below:

-# Always indent logical blocks: class content, function/method body, loops, conditional clauses, etc.
   Generaly indent everything around which you put, or could put the \p{ \p} braces. The only exception
   is the global namespace \p plask and \p plask::solvers::your_solver, which contents should start from
   the first column.

-# The official PLaSK language ins English. Every name, string, or comment should be written in English.
   The used character encoding must be UTF-8 (make sure you set-up your editor properly for this).

-# Put spaces at both sides of binary operators and at outer sides of brackets \p(\p) and \p[\p]. Also
   for inline comments you should put single space after double slash: <pre>// comment</pre>.

\subsection naming Naming convention



\section style_interfce Style guide for user interface (Python and XML)

*/