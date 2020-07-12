#!/bin/sh
# Finds the LaTeX representation of a formula with an ID.
ssh mir "
  cd /mnt/storage/ARQMath_CLEF2020/Formulas
  grep '^$1\\s' latex_representation/*.tsv
"
