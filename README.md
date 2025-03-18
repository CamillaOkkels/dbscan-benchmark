# DBSCAN-Bench

This repository contains infrastructure to benchmark DBSCAN implementations.
It takes care of dataset generation according and contains a few implementations of different DBSCAN variants. 
Consider adding your own implementations through pull requests.

## Supported implementations

We currently support the following implementations:

Exact implementations: 
- [sklearn's DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html): the popular DBSCAN implementation in sklearn.
- [tpeDBSCAN](https://github.com/wangyiqiu/dbscan-python): A grid-based dbscan that works really well in low-dimensional data, see [Wang et al.](https://arxiv.org/abs/1912.06255)
- [FAISS-based DBSCAN](https://github.com/facebookresearch/faiss), a naïve baseline used for high-dimensional data, see [our implementation](benchmark/algorithms/faiss/module.py) for details.

Approximate implementations (these do not guarantee exact clustering results):
- [sngDBSCAN](https://github.com/jenniferjang/subsampled_neighborhood_graph_dbscan): A sampling-based approach to DBSCAN by [Jiang et al.](https://arxiv.org/abs/2006.06743)
- [srrDBSCAN](https://github.com/CamillaOkkels/srrdbscan): An LSH-based dbscan for high-dimensional Euclidean data, see [Okkels et al.](https://openproceedings.org/2025/conf/edbt/paper-208.pdf)

## Supported datasets

We currently support the following datasets under Euclidean distance. 
We primarily focus on high-dimensional datasets as the pose a significant challenge to scalable clustering algorithms.
As reference, we include two popular low-dimensional datasets, Household and Pamap2. 
Each dataset comes with a suggested number of neighbors (minPts) and suggested epsilon values for DBSCAN. 

| Dataset     | Size       | Dimensions | minPts | Epsilon | Method |
|------------|-----------|------------|-----------|----------------|--------|
| MNIST      | 60,000    | 784        | 100       | 1000, 1600, 4000 | FAISS  |
| GIST       | 1,000,000 | 960        | 20        | 0.3, 1.3, 6.0 | ---    |
| GloVe      | 1,183,514 | 100        | 100       | 2.0, 5.0, 20.0 | FAISS  |
| Aloi       | 49,534    | 27         | 100       | 0.0008, 0.01, 0.1 | FAISS  |
| Census     | 299,285   | 500        | 650       | 0.03, 1.0, 8.0 | FAISS  |
| Celeba     | 202,599   | 39         | 200       | 1.0, 1.6, 4.0 | FAISS  |
| Pamap2     | 2,872,533 | 4          | 100       | 0.5, 3.0, 30.0 | TPEDBSCAN    |
| Household  | 2,049,280 | 7          | 100       | 0.5, 1.0, 500 | TPEDBSCAN    |

The final column contains our suggestion to compute baseline results.


# HOWTO 

## Installation

Algorithms are carried out in Docker containers, which means that you will need a running docker installation. See for example [this website](https://www.digitalocean.com/community/tutorial-collections/how-to-install-and-use-docker) to get started.

Assuming you have Python version >= 3.8 installed, run

```bash
python3 -m pip install -r requirements.txt 
```

to install all necessary packages. Starting in a fresh python environment is suggested. 

All implementations can be installed using
```bash
python3 install.py
```

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

In our experiments, we measured the following clustering times (in seconds).

| Dataset  |  ϵ  | #clusters | SRR | SNG | TPE | FAISS | SKL |
|----------|----:|----:|----:|----:|----:|----:|----:|
| **Aloi** | 0.0008 | 2 | **0.37** | 1.95 | ---₍HD₎ | 21.75 | 7.69 |
|          | 0.01   | 243 | 0.47 | **0.46** | ---₍HD₎ | 300.02 | 12.84 |
|          | 0.1    | 2 | 1.44 | **1.4** | ---₍HD₎ | 173.44 | 49.88 |
| **Celeba** | 1.0  | 2291 | **4.35** | 15.91 | ---₍HD₎ | 323.04 | 4369.7 |
|          | 1.6    | 51 | **7.71** | 18.37 | ---₍HD₎ | 1078.9 | 4727.9 |
|          | 4.0    | 1 | **5.85** | 43.83 | ---₍HD₎ | 1616.0 | ---₍M₎ |
| **Census** | 0.03  | 37 | **21.12** | 23.4 | ---₍HD₎ | 3768.1 | ---₍T₎ |
|          | 1.0    | 1816 | **30.96** | 93.78 | ---₍HD₎ | 6357.3 | ---₍T₎ |
|          | 8.0    | 2 | **62.36** | 772.88 | ---₍HD₎ | 11823 | ---₍M₎ |
| **Gist**  | 0.3*   | 794 | **486.6** | 1205.0 | ---₍HD₎ | ---₍T₎ | ---₍T₎ |
|          | 1.3*   | 27 | **3367** | 8224.89 | ---₍HD₎ | ---₍T₎ | ---₍T₎ |
|          | 6.0*   | 1 | **625.7** | 15627 | ---₍HD₎ | ---₍T₎ | ---₍M₎ |
| **Glove** | 2.0    | 8 | **176.14** | 1064.84 | ---₍HD₎ | 7433.47 | ---₍T₎ |
|          | 5.0*   | 11 | **165.9** | *2036.14* | ---₍HD₎ | ---₍T₎ | ---₍T₎ |
|          | 20.0   | 1 | **99.22** | 4282.37 | ---₍HD₎ | 18487 | ---₍M₎ |
| **Mnist** | 1000   | 2 | 14.41 | **13.64** | ---₍HD₎ | 197.51 | 3201.3 |
|          | 1600   | 7 | 30.49 | **20.05** | ---₍HD₎ | 683.1 | 3225.7 |
|          | 4000   | 1 | **25.2** | 42.69 | ---₍HD₎ | 492.7 | 2946.9 |
| **Household** | 0.5 | 69 | 56.19 | ---₍M₎ | **1.13** | ---₍T₎ | ---₍M₎ |
|          | 1.0    | 63 | 49.43 | ---₍M₎ | **1.43** | ---₍T₎ | ---₍M₎ |
|          | 500    | 1 | 24.24 | ---₍M₎ | **0.64** | 15609 | ---₍M₎ |
| **Pamap2** | 0.5   | 276 | 15.37 | ---₍M₎ | **1.6** | 13095 | 183.86 |
|          | 3.0    | 275 | 80.16 | 1657.3 | **1.49** | ---₍T₎ | 319.85 |
|          | 30.0   | 3 | 29.96 | 3572.3 | **1.11** | ---₍T₎ | ---₍M₎ |

# Reference

This benchmarking environment was used to run the experiments for [the paper](https://openproceedings.org/2025/conf/edbt/paper-208.pdf): 

> Camilla Birch Okkels, Martin Aumüller, Viktor Bello Thomsen, Arthur Zimek:
High-dimensional density-based clustering using locality-sensitive hashing. EDBT 2025: 694-706

If you find our work useful, please reference this paper.


# Credits

The basic infrastructure is largely inspired by the ann-benchmarks project <https://github.com/erikbern/ann-benchmarks/>. 






