import sys
import pathlib
import copy

import numpy as np
import optax

# Ensure repository root is on sys.path so local `tests` package is importable
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
	sys.path.insert(0, str(repo_root))

from energnn.graph import Edge, EdgeStructure, Graph, GraphStructure, GraphShape, collate_graphs
from energnn.graph.jax import JaxGraph
from energnn.problem import Problem, ProblemBatch, ProblemLoader, ProblemMetadata
from energnn.trainer import SimpleTrainer

from tests.utils import TestProblemGenerator

pb_generator = TestProblemGenerator(seed=7, n_max=4)
problem = pb_generator.generate_problem()

print(problem.context_structure)

context, _ = problem.get_context()
print(context)