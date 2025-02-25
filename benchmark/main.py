import argparse
import numpy as np
import time
import docker
import multiprocessing.pool
import os
import threading
import json


from typing import List, Set

from benchmark.datasets import DATASETS, get_dataset
from benchmark.definitions import instantiate_algorithm, get_definitions, list_algorithms, Definition
from benchmark.algorithms.base.module import BaseDBSCAN
from benchmark.results import store_results

def run_experiment(X: np.array, algo: BaseDBSCAN , eps: float, minPts: int):
    start = time.time()
    algo.cluster(X, eps, minPts)
    end = time.time()
    return end - start, algo.retrieve_labels()

def run_docker(dataset: str, eps: float, minPts: int, definition: Definition) -> None:
    cmd = [
        "--dataset",
        dataset,
        "--algorithm",
       definition.algorithm,
        "--eps",
        eps,
        "--minPts",
        minPts,
        "--nodocker",
        "--arguments",
        "\"" + json.dumps(definition.arguments) + "\""
    ]

    print(f"Running {definition.algorithm} in container {definition.docker_tag}")
    print(" ".join(map(str, cmd)))

    client = docker.from_env()
    container = client.containers.run(
       definition.docker_tag,
        " ".join(map(str, cmd)),
        volumes={
            os.path.abspath("benchmark"): {"bind": "/home/app/benchmark", "mode": "ro"},
            os.path.abspath("data"): {"bind": "/home/app/data", "mode": "ro"},
            os.path.abspath("results"): {"bind": "/home/app/results", "mode": "rw"},
        },
        mem_limit=int(100*1e9),
        mem_swappiness=0,
        detach=True
    )

    def stream_logs():
        for line in container.logs(stream=True):
            print(line.decode().rstrip())

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()

    try:
        container.wait(timeout=36000)
    except Exception as e:
        print("Container.wait for container %s failed with exception", container.short_id)
        print(str(e))
    finally:
        print("Removing container")
        container.remove(force=True)


def run_worker(dataset: str, eps: float, minPts: int, queue: multiprocessing.Queue) -> None:
    while not queue.empty():
        definition = queue.get()
        run_docker(dataset, eps, minPts, definition)



def create_workers_and_execute(dataset: str, eps: float, minPts: int, definitions: List[Definition]) -> None:
    """
    Manages the creation, execution, and termination of worker processes based on provided arguments.

    Args:
        definitions (List[Definition]): List of algorithm definitions to be processed.
        args (argparse.Namespace): User provided arguments for running workers.

    Raises:
        Exception: If the level of parallelism exceeds the available CPU count or if batch mode is on with more than
                   one worker.
    """
    #cpu_count = multiprocessing.cpu_count()
    #if args.parallelism > cpu_count - 1:
    #    raise Exception(f"Parallelism larger than {cpu_count - 1}! (CPU count minus one)")

    # if args.batch and args.parallelism > 1:
    #     raise Exception(
    #         f"Batch mode uses all available CPU resources, --parallelism should be set to 1. (Was: {args.parallelism})"
    #     )

    task_queue = multiprocessing.Queue()
    for run in definitions:
        task_queue.put(run)

    try:
        workers = [multiprocessing.Process(target=run_worker, args=(dataset, eps, minPts, task_queue))]
        [worker.start() for worker in workers]
        [worker.join() for worker in workers]
    finally:
        print("Terminating %d workers" % len(workers))
        [worker.terminate() for worker in workers]



def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--dataset',
        metavar='NAME',
        help='the dataset to cluster',
        default='mnist',
        choices=DATASETS.keys()
    )

    parser.add_argument(
        '--eps',
        type=float,
        help='the distance threshold',
    )

    parser.add_argument(
        '--minPts',
        type=int,
        help='threshold neighborhood count to become a core point',
    )

    parser.add_argument(
        '--algorithm',
    )

    parser.add_argument(
        '--list-algorithms',
        action='store_true',
        help="list available algorithms"
    )

    parser.add_argument(
        '--prepare',
        action='store_true',
        help='only prepare the dataset'
    )

    parser.add_argument(
        '--nodocker',
        action='store_true',
        help='run algorithm locally'
    )

    parser.add_argument(
        '--arguments',
        help="only for internal use"
    )

    args = parser.parse_args()


    if args.list_algorithms:
        list_algorithms()
        exit(0)

    # if args.algorithm:
    #     runs = [args.algorithm]
    # elif args.nodocker:
    #     runs = list(ALGORITHMS.keys()) # this only works within the docker container?
    # else:
    #     runs = list_by_available_tag()

    definitions = list(get_definitions())

    if args.algorithm:
        definitions = [d for d in definitions if d.algorithm == args.algorithm]

    print(definitions)

    # get definitions here

    ds = DATASETS[args.dataset]
    ds['prepare']()

    if args.prepare:
        exit(0)

    X = get_dataset(args.dataset)
    X = np.array(ds['data'](X))

    eps = ds['eps']
    if args.eps:
        eps = args.eps

    minPts = ds['minPts']
    if args.minPts:
        minPts = args.minPts

    if args.nodocker:
        assert args.arguments
        arguments = json.loads(args.arguments)
        # find the correct definition file
        for d in definitions:
            if d.algorithm == args.algorithm and d.arguments == arguments:
                break
        else:
            raise Exception("Couldn't find the right config file!")

        runner = instantiate_algorithm(d)

        time, (labels, corepoints, borderpoints) = run_experiment(X, runner, eps, minPts)
        #print(labels)
        num_clusters = len(set([x for x in labels if x >= 0]))
        print(f"Found {num_clusters} clusters and identified {len(corepoints)} core points.")
        attrs = {
            "time": time,
            "ds": args.dataset,
            "eps": eps,
            "minPts": minPts,
            "algo": d.algorithm,
            "params": str(runner)
        }
        attrs.update(runner.get_additional())
        store_results(args.dataset, eps, minPts, d.algorithm, repr(runner), attrs, labels, corepoints, borderpoints)
    else:
        create_workers_and_execute(args.dataset, eps, minPts, definitions)
