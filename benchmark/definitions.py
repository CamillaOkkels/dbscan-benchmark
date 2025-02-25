from dataclasses import dataclass
from typing import List, Any, Union, Dict
from itertools import product
import glob
import importlib
import os
import yaml
from enum import Enum

from benchmark.algorithms.base.module import BaseDBSCAN

@dataclass
class Definition:
    algorithm: str
    constructor: str
    module: str
    docker_tag: str
    arguments: List[Any]

def instantiate_algorithm(definition: Definition) -> BaseDBSCAN:
    """
    Create a `BaseDBSCAN` from a definition.
     
    Args:
        definition (Definition): An object containing information about the algorithm.

    Returns:
        BaseDBSCAN: Instantiated algorithm

    Note:
        The constructors for the algorithm definition are generally located at 
        benchmark/algorithms/*/module.py.
    """
    print(f"Trying to instantiate benchmark.algorithms.{definition.module}.{definition.constructor}({definition.arguments})")
    module = importlib.import_module(f"benchmark.algorithms.{definition.module}.module")
    constructor = getattr(module, definition.constructor)
    return constructor(*definition.arguments)

class InstantiationStatus(Enum):
    """Possible status of instantiating an algorithm from a python module import."""
    AVAILABLE = 0
    NO_CONSTRUCTOR = 1
    NO_MODULE = 2



def algorithm_status(definition: Definition) -> InstantiationStatus:
    """
    Determine the instantiation status of the algorithm based on its python module and constructor.

    Attempts to find the Python class constructor based on the definition's module path and 
    constructor name.

    Args:
        definition (Definition): The algorithm definition containing module and constructor.

    Returns:
        InstantiationStatus: The status of the algorithm instantiation.
    """
    try:
        module = importlib.import_module(f"{definition.module}.module")
        if hasattr(module, definition.constructor):
            return InstantiationStatus.AVAILABLE
        else:
            return InstantiationStatus.NO_CONSTRUCTOR
    except ImportError:
        return InstantiationStatus.NO_MODULE


def _generate_combinations(args: Union[List[Any], Dict[Any, Any]]) -> List[Union[List[Any], Dict[Any, Any]]]:
    """
    Generate combinations of elements from args, either the list or combinations of key-value pairs in a dict.

    Args:
        args (Union[List[Any], Dict[Any, Any]]): Input list or dict to generate combinations from.

    Returns:
        List[Union[List[Any], Dict[Any, Any]]]: List of combinations generated from input.

    Raises:
        TypeError: If input is neither a list nor a dict.
    """

    if isinstance(args, list):
        args = [el if isinstance(el, list) else [el] for el in args]
        return [list(x) for x in product(*args)]
    elif isinstance(args, dict):
        flat = []
        for k, v in args.items():
            if isinstance(v, list):
                flat.append([(k, el) for el in v])
            else:
                flat.append([(k, v)])
        return [dict(x) for x in product(*flat)]
    else:
        raise TypeError(f"No args handling exists for {type(args).__name__}")


def get_config_files(base_dir: str = "benchmarks/algorithms") -> List[str]:
    """Get config files for all algorithms."""
    config_files = glob.glob(os.path.join(base_dir, "*", "config.yml"))
    return list(
        set(config_files) - {f"{base_dir}/base/config.yml"}
    )

def load_configs(base_dir: str = "benchmark/algorithms") -> Dict[str, Any]:
    """Load algorithm configurations for a given point_type."""
    config_files = get_config_files(base_dir=base_dir)
    configs = {}
    for config_file in config_files:
        with open(config_file, 'r') as stream:
            try:
                config_data = yaml.safe_load(stream)
                algorithm_name = os.path.basename(os.path.dirname(config_file))
                configs[algorithm_name] = config_data
            except yaml.YAMLError as e:
                print(f"Error loading YAML from {config_file}: {e}")
    return configs

def _get_definitions(base_dir: str = "benchmark/algorithms") -> Dict[str, Dict[str, Any]]:
    """Load algorithm configurations for a given point_type."""
    config_files = get_config_files(base_dir=base_dir)
    configs = {}
    for config_file in config_files:
        with open(config_file, 'r') as stream:
            try:
                config_data = yaml.safe_load(stream)
                algorithm_name = os.path.basename(os.path.dirname(config_file))
                configs[algorithm_name] = config_data
            except yaml.YAMLError as e:
                print(f"Error loading YAML from {config_file}: {e}")
    return configs

def _get_algorithm_definitions(dataset: str = "mnist") -> Dict[str, Dict[str, Any]]:
    """Get algorithm definitions for a specific point type and distance metric.
    
    If an algorithm has an 'any', it is also included.

    Returns: A mapping from the algorithm name (not the algorithm class), to the algorithm definitions, i.e.:

    """
    configs = load_configs()
    definitions = {}

    # param `_` is filename, not specific name
    for _, config in configs.items():
        c = []
        print(config)
        c.append(config)
        for cc in c:
            definitions[cc.pop("name")] = cc

    return definitions

def list_algorithms(base_dir: str = "benchmark/algorithms") -> None:
    """
    Output (to stdout), a list of all algorithms, with their supported point types and metrics.
    
    Args:
        base_dir (str, optional): The base directory where the algorithms are stored. 
                                  Defaults to "benchmark/algorithms".
    """
    definitions = _get_definitions(base_dir)

    print("The following algorithms are supported...")
    for algorithm in definitions:
        print(definitions[algorithm]['name'], definitions[algorithm])

def prepare_args(args: Any) -> List:
    """For an Algorithm's run group, prepare arguments. 
    
    An `arg_groups` is preferenced over an `args` key.
    
    Args:
        run_group (Dict[str, Any]): The run group containing argument definitions.

    Returns:
        List: A list of prepared arguments.

    Raises:
        ValueError: If the structure of the run group is not recognized.
    """
    return _generate_combinations(args)
   


def create_definitions_from_algorithm(name: str, algo: Dict[str, Any]) -> List[Definition]:
    """
    Create definitions from an indvidual algorithm. An algorithm (e.g. srrdbscan) can have multiple
     definitions based on various run groups (see config.ymls for clear examples). 
    
    Args:
        name (str): Name of the algorithm.
        algo (Dict[str, Any]): Dictionary with algorithm parameters.
    
    Raises:
        Exception: If the algorithm does not define "docker_tag", "module" or "constructor" properties.
    
    Returns:
        List[Definition]: A list of definitions created from the algorithm.
    """
    required_properties = ["docker-image", "module", "constructor"]
    missing_properties = [prop for prop in required_properties if prop not in algo]
    if missing_properties:
        raise ValueError(f"Algorithm {name} is missing the following properties: {', '.join(missing_properties)}")
    
    base_args = algo.get("base_args", [])
    
    definitions = []
    args = prepare_args(algo.get('args', []))

    for arg_group in args:
        current_args = []
        current_args.extend(base_args)
        if isinstance(arg_group, list):
            current_args.extend(arg_group)
        else:
            current_args.append(arg_group)
        
        definitions.append(
            Definition(
                algorithm=name,
                docker_tag=algo["docker-image"],
                module=algo["module"],
                constructor=algo["constructor"],
                arguments=current_args,
            )
        )
    return definitions

def get_definitions() -> List[Definition]:
    algorithm_definitions = _get_algorithm_definitions()

    definitions: List[Definition] = []

    # Map this for each config.yml
    for (name, algo) in algorithm_definitions.items():
        definitions.extend(
            create_definitions_from_algorithm(name, algo)
        )
        

    return definitions