#!/bin/bash

echo "Unittest dirichlet"
python -m unittest discover dirichlet
echo "Unittest betacal"
python -m unittest discover betacal
