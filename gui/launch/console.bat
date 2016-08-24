@ECHO OFF
WHILE
:shift
IF "%~1" NEQ "--" (
  SHIFT
  GOTO :shift
)
TITLE %~2 (PLaSK)
%*
PAUSE
