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

# Run experiments

Experiments can be run calling __python main.py__ and the arguments, or in
parallel using scoop.

Here is an example with Scoop.

```
python -m scoop main.py --classifier nbayes --seed 42 --iterations 2 \
                       --folds 3 --datasets iris,autos \
                       --output-path results_test
```

To see all the available options pass the argument __--help__

```
$ python main.py --help
usage: main.py [-h] [-c CLASSIFIER_NAME] [-s SEED_NUM] [-i MC_ITERATIONS]
               [-f N_FOLDS] [-o RESULTS_PATH]

Runs all the experiments with the given arguments

optional arguments:
  -h, --help            show this help message and exit
  -c CLASSIFIER_NAME, --classifier CLASSIFIER_NAME
                        Classifier to use for evaluation (default: nbayes)
  -s SEED_NUM, --seed SEED_NUM
                        Seed for the random number generator (default: 42)
  -i MC_ITERATIONS, --iterations MC_ITERATIONS
                        Number of Markov Chain iterations (default: 10)
  -f N_FOLDS, --folds N_FOLDS
                        Folds to create for cross-validation (default: 5)
  -o RESULTS_PATH, --output_path RESULTS_PATH
                        Path to store all the results (default: results_test)
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
