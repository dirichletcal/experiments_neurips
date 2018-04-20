# Development

Please follow this instructions to be sure that we all have the same library
versions (it may take 30 minutes or more to install all packages).

```
# Clone the repository
git clone git@bitbucket.org:dirichlet_cal/experiments.git dirichlet_cal_experiments
# Go into the folder
cd dirichlet_cal_experiments
# Clone the submodules
git submodule update --init --recursive
# Create a new virtual environment with Python3
python3 -m venv venv
# Load the generated virtual environment
source venv/bin/activate
# Install all the dependencies
pip install -r requirements.txt
```

From now, every time that you want to run the main or other code first load the
environment with

```
source venv/bin/activate
```

# Notebooks

In order to run the notebooks it is necessary to install IPython and
Jupyter-notebooks. But the setuptools and pip need to be upgraded.
Follow the next instructions (with the loaded virtual environment):

```
pip install --upgrade setuptools pip
pip install ipython
pip install jupyter
```

Then start the Jupyter notebook with

```
jupyter notebook
```

And go to the notebook folder.
