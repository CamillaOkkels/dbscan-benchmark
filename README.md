# DBSCAN-Bench

Benchmarking DBSCAN implementations

# HOWTO

## Installation

Algorithms are carried out in Docker containers, which means that you will need a running docker installation. See for example [this website](https://www.digitalocean.com/community/tutorial-collections/how-to-install-and-use-docker) to get started.

Assuming you have Python version >= 3.8 installed, run

```bash
python3 -m pip install -r requirements.txt 
```

to install all necessary packages. (Starting in a fresh environment might be a good idea.)

All implementations can be installed using
```bash
python3 install.py
```

## Datasets

Currently the following datasets are supported:

- ADD INFO

- mnist, household, pamap2, gist, glove, TODO

## Algorithms

Currently the following algorithms are supported: 

# Running an experiment

The standard way to run an experiment is

```
python3 run.py --dataset <DATASET> --algorithm <ALGORITHM> 
```

This will run all configurations known for algorithm on the dataset with default choices of epsilon and minPts. To specify different dbscan parameters, use --eps and --minPts. An example call would be 

```
python3 run.py --dataset mnist --algorithm srrdbscan --eps 1700 --minPts 100
```

After running the experiments, make sure to fix the file permissions by running something like 

```
sudo chmod -R 777 results/
```

## Algorithm configuration

Algorithm configurations are stored in YAML files. The are available in `benchmark/algorithms/<ALGORITHM>/config.yml.`

An example looks like this:

```yaml
docker-image: dbscan-benchmarks-srrdbscan
module: srrdbscan
name: srrdbscan
constructor: SRRDBSCAN
args:
  - [0.1, 0.3, 0.5, 0.7, 0.9] # failure probability
  - [0.1, 1] # memory usage
  - [56] # number of threads
  - [1] # shrinkage parameter (fraction of repetitions inspected)
```

The `args` part specifies all the experiments that are going to be run. 
The cartesion product of all the given lists is making up the list of individual runs that are tried out in the experiment. (In the example, 5 * 2 * 1 * 1 runs are conducted).
`args` have to match the number of arguments expected by the constructor. 


## Evaluation

All results are stored in the following scheme `results/<dataset>/<eps>/<minPts>/<algorithm>/`, with one hdf5 file per run. The easiest way to handle them is to post-process them using a Jupyer notebook. 
An example is given in `evaluation/cluster_analysis.ipynb`.

To get an idea about the clustering structure, try running 
```bash
python3 evaluation/summarize.py <path-to-hdf5-file>
```






