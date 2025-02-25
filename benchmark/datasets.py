"""
Functionality to create datasets used in the evaluation.
"""
from glob import glob
import h5py
import os
import time
import numpy as np
import zipfile, sklearn


from typing import Dict, Tuple

from urllib.request import urlopen


def download(src: str, dst: str):
    """ download an URL """
    if os.path.exists(dst):
        #print("Already exists")
        return
    print('downloading %s -> %s...' % (src, dst))

    t0 = time.time()
    outf = open(dst, "wb")
    inf = urlopen(src)
    info = dict(inf.info())
    content_size = int(info.get('Content-Length', -1))
    bs = 1 << 20
    totsz = 0
    while True:
        block = inf.read(bs)
        elapsed = time.time() - t0
        print(
            "  [%.2f s] downloaded %.2f MiB / %.2f MiB at %.2f MiB/s   " % (
                elapsed,
                totsz / 2**20, content_size / 2**20 if content_size != -1 else -1,
                totsz / 2**20 / elapsed),
            flush=True, end="\r"
        )
        if not block:
            break
        outf.write(block)
        totsz += len(block)
    print()
    print("download finished in %.2f s, total size %d bytes" % (
        time.time() - t0, totsz
    ))

def get_dataset_fn(dataset_name: str) -> str:
    """
    Returns the full file path for a given dataset name in the data directory.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        str: The full file path of the dataset.
    """
    if not os.path.exists("data"):
        os.mkdir("data")
    return os.path.join("data", f"{dataset_name}.hdf5")


def get_dataset(dataset_name: str, path: str = ".") -> h5py.File:
    """
    Fetches a dataset by downloading it from a known URL or creating it locally
    if it's not already present. The dataset file is then opened for reading, 
    and the file handle and the dimension of the dataset are returned.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        Tuple[h5py.File, int]: A tuple containing the opened HDF5 file object and
            the dimension of the dataset.
    """
    hdf5_filename = get_dataset_fn(dataset_name)
    try:
        dataset_url = f"https://ann-benchmarks.com/{dataset_name}.hdf5"
        download(dataset_url, hdf5_filename)
    except:
        print(f"Cannot download {dataset_url}")
        if dataset_name in DATASETS:
            print("Creating dataset locally")
            DATASETS[dataset_name]['prepare']()#(hdf5_filename)

    hdf5_file = h5py.File(f"{path}/{hdf5_filename}", "r")

    return hdf5_file

def compute_groundtruth(X: np.ndarray, eps: float, minPts: int) -> Tuple[np.ndarray, np.ndarray]:
    from benchmark.algorithms.sklearn.module import SKLearnDBSCAN
    from benchmark.algorithms.tpedbscan.module import TPEDBSCAN
    print("Computing groundtruth...")
    start = time.time()
    d = X.shape[1]
    if d > 10:
        dbscan = SKLearnDBSCAN()
    else:
        dbscan = TPEDBSCAN()
    dbscan.cluster(X, eps, minPts)
    end = time.time()

    print(f"Computing groundtruth took {(end - start):.2f}s.")
    return dbscan.retrieve_labels()

def write_output(X: np.ndarray, name: str, compute_gt=False):
    eps = DATASETS[name]['eps']
    minPts = DATASETS[name]['minPts']
    f = h5py.File(get_dataset_fn(name), "w")
    f.create_dataset("data", data=X)
    group = f.create_group(f"eps={eps}, minPts={minPts}")
    group.attrs["eps"] = eps
    group.attrs["minPts"] = minPts
    if compute_gt:
        labels, core_indices = compute_groundtruth(X, eps, minPts)
        group.create_dataset("clustering_labels", data=labels)
        group.create_dataset("core_indices", data=core_indices)
    f.close()

def mnist():
    if os.path.exists(get_dataset_fn("mnist")):
        return
    src = "http://ann-benchmarks.com/mnist-784-euclidean.hdf5"
    dst = "mnist.hdf5" #get_dataset_fn("mnist")
    download(src, dst)

    f = h5py.File(dst)
    X = np.array(f['train'])
    f.close()
    write_output(X, "mnist")

def gist():
    if os.path.exists(get_dataset_fn("gist")):
        return
    src = "http://ann-benchmarks.com/gist-960-euclidean.hdf5"
    dst = "gist.hdf5" #get_dataset_fn("mnist")
    download(src, dst)

    f = h5py.File(dst)
    X = np.array(f['train'])
    f.close()
    write_output(X, "gist") 
    
def glove():
    if os.path.exists(get_dataset_fn("glove")):
        return
    src = "http://ann-benchmarks.com/glove-100-angular.hdf5"
    dst = "glove.hdf5" #get_dataset_fn("glove")

    download(src, dst)
    f = h5py.File(dst)
    X = np.array(f['train'])
    f.close()
    write_output(X, "glove")

def bremen():
    if os.path.exists(get_dataset_fn("bremen-small")):
        return
    src = "https://b2share.eudat.eu/api/files/189c8eaf-d596-462b-8a07-93b5922c4a9f/bremenSmall.h5.h5"
    dst = "bremen-small.hdf5" #get_dataset_fn("bremen-small")
    download(src, dst)

    f = h5py.File(dst)
    X = np.array(f['DBSCAN'])
    f.close()
    write_output(X, "bremen-small")


def twitter():
    src = "https://b2share.eudat.eu/api/files/189c8eaf-d596-462b-8a07-93b5922c4a9f/twitterSmall.h5.h5"
    dst = "twitter-small.hdf5" #get_dataset_fn("twitter-small")

    download(src, dst)
    f = h5py.File(dst)
    X = np.array(f['DBSCAN'])
    f.close()
    write_output(X, "twitter_small")

def pamap2(apply_pca=False):
    from sklearn.decomposition import PCA
    fn = "pamap2" if apply_pca else "pamap2-full"
    if os.path.exists(get_dataset_fn(fn)):
        return
    
    src = "http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
    download(src, "PAMAP2.zip")

    with zipfile.ZipFile("PAMAP2.zip") as zn:
        arr = []
        for i in range(1, 10):
            zfn = f"PAMAP2_Dataset/Protocol/subject10{i}.dat"
            zf = zn.open(zfn)
            for line in zf:
                line = line.decode()
                l = list(map(float, line.strip().split()))
                # remove timestamp
                arr.append(l[1:])
        X = np.nan_to_num(np.array(arr)) # many NaNs in data, replace them with 0.
        if apply_pca:
            X = PCA(n_components=4).fit_transform(X) # PCA of first four components
        write_output(X, fn)

    # PAMAP2_Dataset/Protocol/subject101.dat 

def household():
    # https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
    if os.path.exists(get_dataset_fn("household")):
        return

    src = 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'
    fn = "household.zip"
    download(src, "household.zip")

    with zipfile.ZipFile(fn) as z:
        zn = z.open('household_power_consumption.txt')
        zn.readline()
        cnt = []
        for line in zn:
            line = line.decode()
            if "?" not in line:
                cnt.append(list(map(float, line.strip().split(";")[2:])))
        X = np.array(cnt,dtype=np.float32)
        write_output(X, "household")

def aloi():
    if os.path.exists(get_dataset_fn("aloi")):
        return
    src = "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/1_ALOI.npz"
    download(src, "aloi.npz")
    X = np.load("aloi.npz")['X']
    write_output(X, "aloi")

def census():
    if os.path.exists(get_dataset_fn("census")):
        return
    src = "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/9_census.npz"
    download(src, "census.npz")
    X = np.load("census.npz")['X']
    write_output(X, "census")

def celeba():
    if os.path.exists(get_dataset_fn("celeba")):
        return
    src = "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/8_celeba.npz"
    download(src, "celeba.npz")
    X = np.load("celeba.npz")['X']
    write_output(X, "celeba")


def synthetic_small():
    # generated using simdengen from https://github.com/ParAlg/ParGeo
    if os.path.exists(get_dataset_fn("synthetic-small")):
        return
    src = 'https://itu.dk/people/maau/seedspreader.zip'
    fn = "synthetic_small.zip"
    n = 100_000
    d = 9
    download(src, fn)

    with zipfile.ZipFile(fn) as z:
        zn = z.open('seedspreader.txt')
        X = np.array(zn.read().split(), dtype=np.float32)
        X = X.reshape((n, d))
        write_output(X, "synthetic-small")

def synthetic():
    # generated using simdengen from https://github.com/ParAlg/ParGeo
    if os.path.exists(get_dataset_fn("synthetic")):
        return
    src = 'https://itu.dk/people/maau/synthetic-10M.zip'
    fn = "synthetic.zip"
    n = 10_000_000
    d = 9
    download(src, fn)

    with zipfile.ZipFile(fn) as z:
        zn = z.open('synthetic-10M.txt')
        X = np.array(zn.read().split(), dtype=np.float32)
        X = X.reshape((n, d))
        write_output(X, "synthetic")

def geolife():
    if os.path.exists(get_dataset_fn("geolife")):
        return
    src = "https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip"
    fn = "geolife.zip"
    download(src, fn)
    #os.system("unzip geolife.zip")
    fns = list(glob("Geolife Trajectories 1.3/Data/*/Trajectory/*.plt"))
    print(len(fns))
    X = []
    for fn in fns:
        with open(fn) as f:
            [f.readline() for _ in range(6)]
            for line in f:
                line = line.split(",")
                lon, lat, alt = float(line[0]), float(line[1]), float(line[3])
                X.append([lon, lat, alt])
    X = np.array(X)
    print(X.shape)
    write_output(X, "geolife")
    os.system("rm -rf 'Geolife Trajectories 1.3'")

def openai(n=1_000_000):
    assert n in [100_000, 1_000_000]
    fn = "wikipedia-full" if n == 1_000_000 else "wikipedia-small"
    if os.path.exists(get_dataset_fn(fn)):
        return    

    from datasets import load_dataset

    data = load_dataset("KShivendu/dbpedia-entities-openai-1M", split="train")
    if n is not None and n >= 100_000:
        data = data.select(range(n))

    embeddings = data.to_pandas()['openai'].to_numpy()
    embeddings = np.vstack(embeddings).reshape((-1, 1536))

    write_output(embeddings, fn)


def teraclicklog():
    pass

def test():
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler

    if os.path.exists(get_dataset_fn("test")):
        return

    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=750, centers=centers, cluster_std=0.4, random_state=0
    )

    X = StandardScaler().fit_transform(X)
    write_output(np.array(X, dtype=np.float32), "test")   

DATASETS = {
    'mnist': {
        'prepare': mnist,
        'data': lambda f: f['data'],
        'eps': 1300,
        'minPts': 100,
    }, 
    'gist': {
        'prepare': gist,
        'data': lambda f: f['data'],
        'eps': 1,
        'minPts': 100,
    }, 
    'glove': {
        'prepare': glove,
        'data': lambda f: f['data'],
        'eps': 4,
        'minPts': 100,
    },
    'bremen-small': {
        'prepare': bremen,
        'data': lambda f: f['data'],
        'eps': 312,
        'minPts': 100,
    },
    'twitter-small': {
        'prepare': twitter,
        'data': lambda f: f['data'],
        'eps': 0.01,
        'minPts': 40,
    },
    'pamap2': {
        'prepare': lambda: pamap2(True),
        'data': lambda f: f['data'],
        'eps': 500,
        'minPts': 100,
    },
    'pamap2-full': {
        'prepare': lambda: pamap2(False),
        'data': lambda f: f['data'],
        'eps': 500,
        'minPts': 100,
    },
    'household': {
        'prepare': household,
        'data': lambda f: f['data'],
        'eps': 2000,
        'minPts': 100, 
    },
    "geolife": {
        'prepare': geolife,
        'data': lambda f: f['data'],
        'eps': 50,
        'minPts': 100,
    },
    'synthetic-small': {
        'prepare': synthetic_small,
        'data': lambda f: f['data'],
        'eps': 2500,
        'minPts': 100,
    },
    'synthetic': {
        'prepare': synthetic,
        'data': lambda f: f['data'],
        'eps': 2500,
        'minPts': 100,
    },
    'aloi': {
        'prepare': aloi,
        'data': lambda f: f['data'],
        'eps': 100,
        'minPts': 100,
    },
    'census': {
        'prepare': census,
        'data': lambda f: f['data'],
        'eps': 100,
        'minPts': 650,
    },
    'celeba': {
        'prepare': celeba,
        'data': lambda f: f['data'],
        'eps': 100,
        'minPts': 200,
    },
    'test': {
        'prepare': test,
        'data': lambda f: f['data'],
        'eps': 0.3,
        'minPts': 10
    },
    'wikipedia-small': {
        'prepare': lambda: openai(100_000),
        'data': lambda f: f['data'],
        'eps': 1,
        'minPts': 10,
    },
    'wikipedia-full': {
        'prepare': lambda: openai(1_000_000),
        'data': lambda f: f['data'],
        'eps': 1,
        'minPts': 10,
    },
}

