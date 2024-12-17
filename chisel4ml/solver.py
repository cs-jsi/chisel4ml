import numpy as np
from google.protobuf import text_format
from ortools.sat.python import cp_model

from chisel4ml.accelerator import ACCELERATORS
from chisel4ml.lbir.services_pb2 import Accelerator


class _VarArraySolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, vars_cp, lbir_model, accel_list, solution_list):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.vars_cp = vars_cp
        self.solution_list = solution_list
        self.lbir_model = lbir_model
        self.accel_list = accel_list

    def name_to_la(self, name):
        # key = f"l{lay_ind}-a{accel_ind}"
        l_str, a_str = name.split("-")
        return int(l_str[1:]), int(a_str[1:])

    def acc_data_to_srvc(self, acc_data, ind):
        accel = Accelerator()
        accel.name = acc_data.name
        accel.layers.extend([self.lbir_model.layers[ind]])
        return accel

    def on_solution_callback(self):
        solution_raw = np.zeros((len(self.lbir_model.layers), len(self.accel_list)))
        for layer_vars in self.vars_cp:
            for var in layer_vars:
                if self.Value(var) == 1:
                    l_ind, a_ind = self.name_to_la(var.Name())
                    solution_raw[l_ind][a_ind] = 1
        solution = []
        for ind, lay in enumerate(solution_raw):
            acc_ind = lay.tolist().index(1)
            acc_data = self.accel_list[acc_ind]
            solution.append(self.acc_data_to_srvc(acc_data, ind))

        # merging ProcessingElementCombToSeq
        merged_solution = []
        for accel in solution:
            if len(merged_solution) == 0:
                merged_solution.append(accel)
            else:
                if (
                    accel.name == "ProcessingElementCombToSeq"
                    and merged_solution[-1].name == "ProcessingElementCombToSeq"
                ):
                    for _ in accel.layers:
                        merged_solution[-1].layers.append(accel.layers.pop())
                else:
                    merged_solution.append(accel)
        self.solution_list.append(merged_solution)


class SolutionSpace:
    def __init__(self, lbir_model, accel_list=ACCELERATORS):
        self.lbir_model = lbir_model
        self.model_cp = cp_model.CpModel()
        self.vars_cp = []
        self.solution_list = []

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
        self.solution_collector = _VarArraySolutionCollector(
            self.vars_cp, lbir_model, accel_list, self.solution_list
        )

    def __iter__(self):
        for sol in self.solution_list:
            yield sol

    def solve(self):
        return self.solver.Solve(self.model_cp, self.solution_collector)

    def __repr__(self):
        repr = ""
        for solution in self.solution_list:
            for accel in solution:
                repr += text_format.MessageToString(
                    accel, use_short_repeated_primitives=True
                )
        return repr

    def __str__(self):
        str = ""
        for sol_id, solution in enumerate(self.solution_list):
            str += f"Solution {sol_id}:\n"
            lstr = "-> ("
            for accel in solution:
                str += f"-> {accel.name} ->"
                for layer in accel.layers:
                    name = layer.WhichOneof("sealed_value_optional")
                    lstr += f"{name}, "
                lstr = lstr[:-2] + ") ->\n"
            str += "\n" + lstr + "\n"
        return str
