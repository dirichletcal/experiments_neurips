#!/bin/bash

echo "Unittest dirichlet"
python -m unittest discover dirichlet
echo "Unittest betacal"
python -m unittest discover betacal
echo "Unittest calib"
python -m unittest discover calib
