# Experiments for NeurIPS

In this repository you can find all the code to run the non-neural experiments.
For the experiments with Deep Neural Networks check the
[experiments_dnn](https://github.com/dirichletcal/experiments_dnn) repository.

The code works as expected for multiclass classification datasets. It is
possible to use binary datasets but it has not been extensively tested and may
have some problems. In fact, some of the reliability calibration plots may show
inverted scores at the moment.

Some of the visualisation issues for binary datasets may be fixed by using our
own Python library
[PyCalib](https://github.com/classifier-calibration/PyCalib).

# Development

The code has been tested with Ubuntu 20.04.5 LTS, Python 3.8.10, and it may require some of the
following libraries in order to properly install the necessary Python packages:
blas, lapack, cython3.

Those packages can be installed in Ubuntu with

```
sudo aptitude install blas lapack cython3
```

Please follow this instructions to be sure that we all have the same library
versions (it may take 30 minutes or more to install all packages).

```
# Clone the repository
git clone https://github.com/dirichletcal/experiments_neurips.git
# Go into the folder
cd experiments_neurips
# Clone the submodules
git submodule update --init --recursive
# Create a new virtual environment with Python3
python3 -m venv venv
# Load the generated virtual environment
source venv/bin/activate
# Install all the dependencies
pip install -r requirements.txt
```

For following pulls that include the submodule updates you can run

```
sh git_pull.sh
```

From now, every time that you want to run the main or other code first load the
environment with

```
source venv/bin/activate
```

# Run experiments

Experiments can be run calling __python main.py__ and the optional arguments. The optional argument __-w | --n-workers__ indicates how many parallel processes to run. By default it has a value of __-1__ which runs one parallel process per available cpu.

```
python main.py --classifier forest,nbayes --seed 42 --iterations 2 \
                       --folds 3 --datasets iris,spambase \
                       --output-path results_test
```

Once multiple classifiers, datasets and calibrators have been run, it is possible to unify and compute meta-summaries by indicating the folder containing all the results

```
python generate_summaries.py results_test/
```

# Debugging

This code uses multiprocessing capabilities to accelerate the
execution. This can make difficult debugging certain exceptions as those are
captured inside the pool of processes. If you find an error and want to debug
the code you may want to run as a single process by indicating the _main.py_
argument __--workers 1__. Which avoids using the multiprocessing module.

```
python main.py --classifier forest,nbayes --seed 42 --iterations 2 \
                       --folds 3 --datasets iris,spambase \
                       --output-path results_test \
                       --workers 1
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



pyvenv-3.4 --without-pip venv
source venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python


# Unittest

Currently there is only tests for Dirichlet

    python -m unittest discover dirichlet
    python -m unittest discover betacal

That can be run together with the **run_unittests.sh** script

## Blue Crystal 3

First need to load the following modules

```
module load languages/python-anaconda3-5.2.0
module load tools/git-2.22.0
```

Then download the repository

```
git clone https://github.com/dirichletcal/experiments_neurips.git
cd experiments_neurips
```

Pull the dependencies

```
./git_pull
```

And then create a virtual environment

```
python -m venv venv
```

Load the environment

```
source venv/bin/activate
```

Install all dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

test that the script works with

```
python main.py -m uncalibrated,vector_scaling,temperature_scaling -d iris -i 2 -f 2 -c nbayes
```

If it runs then submit to the queue (HS stands for Hao Song)

```
qsub HS_BC3_add_queue_dirichlet.sh
```

## Blue Crystal 4


First need to load the following modules

```
module load languages/anaconda3/2019.07-3.7.3-biopython
module load tools/git/2.18.0
```

Then download the repository

```
git clone https://github.com/dirichletcal/experiments_neurips.git
cd experiments_neurips
```

Pull the dependencies

```
./git_pull
```

And then create a virtual environment

```
python -m venv venv
```

Load the environment

```
source venv/bin/activate
```

Install all dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

test that the script works with

```
python main.py -m uncalibrated,vector_scaling,temperature_scaling -d iris -i 2 -f 2 -c nbayes
```

If it runs then submit to the queue (HS stands for Hao Song)

```
qsub HS_BC4_add_queue_dirichlet.sh
```


