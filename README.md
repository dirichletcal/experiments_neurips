# Development

Please follow this instructions to be sure that we all have the same library
versions (it may take 30 minutes or more to install all packages).

```
# Clone the repository
git clone https://bitbucket.org/dirichlet_cal/experiments.git dirichlet_cal_experiments
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
                       --folds 3 --datasets iris,spambase \
                       --output-path results_test
```

If you run several classifiers and store the results in the same folder. For
example, here a random forest.

```
python -m scoop main.py --classifier forest --seed 42 --iterations 2 \
                       --folds 3 --datasets iris,spambase \
                       --output-path results_test
```

Then, it is possible to unify and compute meta-summaries by indicating the
folder containing all the results

```
python generate_summaries.py results_test/
```

# Help

To see all the available options pass the argument __--help__

```
$ python main.py --help
usage: main.py [-h] [-c CLASSIFIER_NAME] [-s SEED_NUM] [-i MC_ITERATIONS]
               [-f N_FOLDS] [--inner-folds INNER_FOLDS] [-o RESULTS_PATH] [-v]
               [-d DATASETS]

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
  --inner-folds INNER_FOLDS
                        Folds to perform in any given training fold to train
                        the different calibration methods (default: 3)
  -o RESULTS_PATH, --output-path RESULTS_PATH
                        Path to store all the results (default: results_test)
  -v, --verbose         Show additional messages (default: False)
  -d DATASETS, --datasets DATASETS
                        Comma separated dataset names or one of the defined
                        groups in the datasets package (default: iris,autos)
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
