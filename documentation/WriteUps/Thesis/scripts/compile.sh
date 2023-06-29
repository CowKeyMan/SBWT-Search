#!/bin/bash

# Compiles the thesis

pdflatex -synctex=1 -interaction=nonstopmode "main.tex"
pdflatex -synctex=1 -interaction=nonstopmode "main.tex"
bibtex "main.aux"
pdflatex -synctex=1 -interaction=nonstopmode "main.tex"
pdflatex -synctex=1 -interaction=nonstopmode "main.tex"
echo "done"
