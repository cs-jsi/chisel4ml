import numpy as np
from ortools.sat.python import cp_model

from chisel4ml.accelerator import ACCELERATORS
from chisel4ml.lbir.services_pb2 import Accelerator


class VarArraySolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables, lbir_model, accels):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.variables = variables
        self.solution_list = []
        self.lbir_model = lbir_model
        self.accels = accels

    def name_to_la(self, name):
        # key = f"l{lay_ind}-a{accel_ind}"
        l_str, a_str = name.split("-")
        return int(l_str[1:]), int(a_str[1:])

    def on_solution_callback(self):
        solution_raw = np.zeros((len(self.lbir_model.layers), len(self.accels)))
        for layer_vars in self.variables:
            for var in layer_vars:
                if self.Value(var) == 1:
                    l_ind, a_ind = self.name_to_la(var.Name())
                    solution_raw[l_ind][a_ind] = 1
        solution = []
        for lay in solution_raw:
            acc_ind = lay.tolist().index(1)
            solution.append(self.accels[acc_ind])
        self.solution_list.append(solution)


class SolutionSpace:
    def __init__(self, lbir_model, accel_list=ACCELERATORS):
        self.lbir_model = lbir_model
        self.model_cp = cp_model.CpModel()
        self.vars_cp = []

        # Setup the base variables
        for lay_ind, layer in enumerate(lbir_model.layers):
            self.vars_cp.append([])
            for accel_ind, accel in enumerate(accel_list):
                key = f"l{lay_ind}-a{accel_ind}"
                if layer.WhichOneof("sealed_value_optional") in accel.layers:
                    self.vars_cp[lay_ind].append(self.model_cp.NewBoolVar(key))

        # Only 1 accelerator per layer
        for layer_vars in self.vars_cp:
            self.model_cp.Add(sum(var for var in layer_vars) == 1)

        self.solver = cp_model.CpSolver()
        self.solution_collector = VarArraySolutionCollector(
            self.vars_cp, lbir_model, accel_list
        )

    def __iter__(self):
        for sol in self.solutions:
            yield sol

    def solve(self):
        return self.solver.Solve(self.model_cp, self.solution_collector)


def _create_accelerator(solution_layers):
    solution, layers = solution_layers
    acc = Accelerator()
    acc.name = type(solution).__name__
    layer_mod = layers
    if not hasattr(layers, "__iter__"):
        layer_mod = [layers]
    for layer in layer_mod:
        acc.layers.append(layer)
    return acc


def solution_to_accelerators(solution, lbir_layers):
    assert len(solution) == len(lbir_layers)
    solution_layers = zip(solution, lbir_layers)
    accels = list(map(_create_accelerator, solution_layers))
    # merging ProcessingElementCombToSeq
    new_accels = []
    for accel in accels:
        if len(new_accels) == 0:
            new_accels.append(accel)
        else:
            if (
                accel.name == "ProcessingElementCombToSeq"
                and new_accels[-1].name == "ProcessingElementCombToSeq"
            ):
                for _ in accel.layers:
                    new_accels[-1].layers.append(accel.layers.pop())
            else:
                new_accels.append(accel)
    return new_accels
