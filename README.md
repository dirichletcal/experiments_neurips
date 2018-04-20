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
