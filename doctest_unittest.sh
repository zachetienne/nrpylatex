#!/bin/bash

export PYTHONPATH=`pwd`

if [ -z "$1" ]
then
  echo "ERROR: missing Python executable"
  exit
fi

PYTHONEXEC=$1

echo -e Python PATH:"\t" $PYTHONPATH
echo -e Python Version:"\t" `$PYTHONEXEC --version` "\n"

test_failure=0

add_doctest () {
  file=$1
  echo Running doctest on $file
  $PYTHONEXEC -m doctest $file
  if [ $? == 1 ]
  then
    test_failure=1
  fi
  echo -e Done. "\n"
}

add_doctest nrpylatex/core/functional.py
add_doctest nrpylatex/core/symtree.py
add_doctest nrpylatex/core/indexed_symbol.py
add_doctest nrpylatex/core/assert_equal.py

add_unittest () {
  file=$1
  echo Running unittest on $file
  $PYTHONEXEC -m unittest $file
  if [ $? == 1 ]
  then
    test_failure=1
  fi
}

add_unittest nrpylatex.test.test_parse_latex

if [ $test_failure == 0 ]
then
  printf "\nTest(s) Passed!\n\n"
else
  printf "\nTest(s) Failed!\n\n"
fi
exit $test_failure
