import json
import os
import re
import traceback
from typing import Any, Optional, Set, Tuple, Iterator
import h5py


def build_result_filepath(dataset_name: Optional[str] = None, 
                          eps: Optional[float] = None,
                          minPts: Optional[int] = None,
                          algorithm: Optional[any] = None,
                          arguments: Optional[Any] = None) -> str:
    d = ["results"]
    if dataset_name:
        d.append(dataset_name)
    if eps:
        d.append(str(eps))
    if minPts:
        d.append(str(minPts))
    if algorithm:
        d.append(algorithm)
        #data = definition.arguments + query_arguments
        #d.append(re.sub(r"\W+", "_", json.dumps(data, sort_keys=True)).strip("_") + ".hdf5")
        if arguments:
            d.append(arguments+".hdf5")
        else:
            d.append("run.hdf5")
    return os.path.join(*d)


def store_results(dataset_name: str, eps: float, minPts: int, algorithm: str, 
        arguments: str, attrs, labels, core_point_indices, border_points):
    filename = build_result_filepath(dataset_name, eps, minPts, algorithm, arguments)
    directory, _ = os.path.split(filename)

    print(f"storing in {filename}")

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with h5py.File(filename, "w") as f:
        for k, v in attrs.items():
            print(k, v)
            f.attrs[k] = v
        f.create_dataset("labels", (len(labels),), "i", labels )
        f.create_dataset("corepoints", (len(core_point_indices),), "i", core_point_indices)
        f.create_dataset("borderpoints", (len(border_points),), "i", border_points)
    


def load_all_results(dataset: Optional[str] = None, prefix: str = ".") -> Iterator[Tuple[h5py.File]]:
    for root, _, files in os.walk(os.path.join(prefix, build_result_filepath(dataset))):
        for filename in files:
            if os.path.splitext(filename)[-1] != ".hdf5":
                continue
            try:
                with h5py.File(os.path.join(root, filename), "r+") as f:
                    yield f
            except Exception:
                print(f"Was unable to read {filename}")
                traceback.print_exc()


def get_unique_algorithms() -> Set[str]:
    """
    Retrieves unique algorithm names from the results.

    Returns:
        set: A set of unique algorithm names.
    """
    algorithms = set()
    for properties, _ in load_all_results():
        algorithms.add(properties["algo"])
    return algorithms