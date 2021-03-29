"""Train artificial neural network using backpropagation.

Code by Tyler N. Thieding (tnt36)
Written for CSDS 435: Data Mining (Spring 2021)

"""

import math
import sys
from argparse import ArgumentParser
from typing import Dict, List, Tuple

EXIT_CODE_SUCCESS = 0


class Neuron:

    def __init__(self, neuron_id: int, initial_bias: float) -> None:
        self.id = neuron_id
        self.bias = initial_bias

        self._inputs: List[Tuple[float, "Neuron"]] = []  # tuples of weights and neurons

    def add_input(self, initial_weight: float, neuron: "Neuron") -> None:
        self._inputs.append((initial_weight, neuron))

    @property
    def output(self) -> float:
        net_input = self.bias

        for input_weight, input_neuron in self._inputs:
            net_input += input_weight * input_neuron.output

        return 1 / (1 + math.exp(-net_input))

    def weight_from(self, neuron_id: int) -> float:
        """Get weight from specified neuron to this neuron."""
        for input_weight, input_neuron in self._inputs:
            if input_neuron.id == neuron_id:
                return input_weight

        raise ValueError(f"no input neuron with ID {neuron_id}")


class ConstantNeuron(Neuron):

    def __init__(self, neuron_id: int, output_value: float):
        super().__init__(neuron_id, 0.0)  # bias value ignored for static input neurons
        self._value = output_value

    @property
    def output(self) -> float:
        return self._value


def construct_initial_neural_network(i1: float, i2: float) -> Dict[int, "Neuron"]:
    """Create initial neural network with provided weights."""
    n1 = ConstantNeuron(neuron_id=1, output_value=i1)
    n2 = ConstantNeuron(neuron_id=2, output_value=i2)

    n3 = Neuron(neuron_id=3, initial_bias=0.1)
    n3.add_input(initial_weight=0.1, neuron=n1)
    n3.add_input(initial_weight=-0.2, neuron=n2)

    n4 = Neuron(neuron_id=4, initial_bias=0.2)
    n4.add_input(initial_weight=0, neuron=n1)
    n4.add_input(initial_weight=0.2, neuron=n2)

    n5 = Neuron(neuron_id=5, initial_bias=0.5)
    n5.add_input(initial_weight=0.3, neuron=n1)
    n5.add_input(initial_weight=-0.4, neuron=n2)

    n6 = Neuron(neuron_id=6, initial_bias=-0.1)
    n6.add_input(initial_weight=-0.4, neuron=n3)
    n6.add_input(initial_weight=0.1, neuron=n4)
    n6.add_input(initial_weight=0.6, neuron=n5)

    n7 = Neuron(neuron_id=7, initial_bias=0.6)
    n7.add_input(initial_weight=0.2, neuron=n3)
    n7.add_input(initial_weight=-0.1, neuron=n4)
    n7.add_input(initial_weight=-0.2, neuron=n5)

    return {1: n1, 2: n2, 3: n3, 4: n4, 5: n5, 6: n6, 7: n7}


def dump_weights_and_bias(neurons_map: Dict[int, "Neuron"], title: str) -> None:
    """Dump weights and biases to console."""
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print()

    print(f"w13 = {neurons_map[3].weight_from(1)}")
    print(f"w14 = {neurons_map[4].weight_from(1)}")
    print(f"w15 = {neurons_map[5].weight_from(1)}")

    print(f"w23 = {neurons_map[3].weight_from(2)}")
    print(f"w24 = {neurons_map[4].weight_from(2)}")
    print(f"w25 = {neurons_map[5].weight_from(2)}")

    print(f"w36 = {neurons_map[6].weight_from(3)}")
    print(f"w37 = {neurons_map[7].weight_from(3)}")

    print(f"w46 = {neurons_map[6].weight_from(4)}")
    print(f"w47 = {neurons_map[7].weight_from(4)}")

    print(f"w56 = {neurons_map[6].weight_from(5)}")
    print(f"w57 = {neurons_map[7].weight_from(5)}")

    for neuron_id in range(3, 8):  # iterates from 3 to 7 inclusive
        print(f"θ{neuron_id}  = {neurons_map[neuron_id].bias}")

    print()
    print()


def main() -> int:
    """Train artificial neural network using backpropagation."""
    argument_parser = ArgumentParser(description=main.__doc__)
    argument_parser.add_argument("i1", type=float, help="neuron 1 input")
    argument_parser.add_argument("i2", type=float, help="neuron 2 input")
    argument_parser.add_argument("assigned_class", help="output class (valid choices are nail or screw)")
    argument_parser.add_argument("--iterations", default=1, type=int, help="training iterations to run")
    args = argument_parser.parse_args()

    if args.assigned_class.lower() == "nail":
        c6 = 1
        c7 = 0
    elif args.assigned_class.lower() == "screw":
        c6 = 0
        c7 = 1
    else:
        raise ValueError(f"unknown class {args.assigned_class}")

    neurons_map = construct_initial_neural_network(args.i1, args.i2)
    dump_weights_and_bias(neurons_map, "Initial Weights and Biases")

    for iteration in range(1, args.iterations + 1):
        # Start by calculating error. (The neuron class includes conveniences for getting output from net input.)
        # First, make a mapping from neuron ID to error for the output layer.
        errors = {
            7: neurons_map[7].output * (1 - neurons_map[7].output) * (c7 - neurons_map[7].output),
            6: neurons_map[6].output * (1 - neurons_map[6].output) * (c6 - neurons_map[6].output),
        }

        # Then, update the error map to include the hidden layer.
        for hidden_layer_id in range(3, 6):  # iterates from 3 to 5 inclusive
            weighted_summation = 0
            for output_layer_id in range(6, 8):  # iterates from 6 to 7 inclusive
                weighted_summation += (
                    errors[output_layer_id] * neurons_map[output_layer_id].weight_from(hidden_layer_id)
                )

            errors[hidden_layer_id] = (
                    neurons_map[hidden_layer_id].output * (1 - neurons_map[hidden_layer_id].output) * weighted_summation
            )

        # TODO: Calculate updated

        dump_weights_and_bias(neurons_map, title=f"Iteration {iteration}")

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
