"""Generate decision tree for tennis data set.

Code by Tyler N. Thieding (tnt36)
Written for CSDS 435: Data Mining (Spring 2021)

"""

import math
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

EXIT_CODE_SUCCESS = 0


def log2_zero(x) -> float:
    """Take base-2 logarithm of x and return 0 if domain error occurs."""
    try:
        log_value = math.log2(x)
    except ValueError:
        log_value = 0.0

    return log_value


@dataclass()
class DataRecord:

    """Represent a row of the training data set."""

    outlook: str
    temperature: int
    humidity: int
    windy: bool
    play: bool

    def __hash__(self) -> int:
        return hash(
            (self.outlook, self.temperature, self.humidity, self.windy, self.play)
        )


class TreeNode:

    """Represent node in a decision tree."""

    def __init__(self, label: Optional[str] = None):
        self.label: Optional[str] = label
        self.branches: List[TreeEdge] = []


class TreeEdge:

    """Represent an edge in the decision tree."""

    def __init__(
        self,
        origin: TreeNode,
        destination: Optional[TreeNode],
        condition: Optional[str] = None,
    ):
        self.origin = origin
        self.destination = destination
        self.condition = condition


def make_training_data_set() -> Set[DataRecord]:
    """Make a set containing all rows from the table."""
    return {
        DataRecord(
            outlook="sunny", temperature=85, humidity=85, windy=False, play=False
        ),
        DataRecord(
            outlook="sunny", temperature=80, humidity=90, windy=True, play=False
        ),
        DataRecord(
            outlook="overcast", temperature=83, humidity=86, windy=False, play=True
        ),
        DataRecord(
            outlook="rainy", temperature=70, humidity=96, windy=False, play=True
        ),
        DataRecord(
            outlook="rainy", temperature=68, humidity=80, windy=False, play=True
        ),
        DataRecord(
            outlook="rainy", temperature=65, humidity=70, windy=True, play=False
        ),
        DataRecord(
            outlook="overcast", temperature=64, humidity=65, windy=True, play=True
        ),
        DataRecord(
            outlook="sunny", temperature=72, humidity=95, windy=False, play=False
        ),
        DataRecord(
            outlook="sunny", temperature=69, humidity=70, windy=False, play=True
        ),
        DataRecord(
            outlook="rainy", temperature=75, humidity=80, windy=False, play=True
        ),
        DataRecord(outlook="sunny", temperature=75, humidity=70, windy=True, play=True),
        DataRecord(
            outlook="overcast", temperature=72, humidity=90, windy=True, play=True
        ),
        DataRecord(
            outlook="overcast", temperature=81, humidity=75, windy=False, play=True
        ),
        DataRecord(
            outlook="rainy", temperature=71, humidity=91, windy=True, play=False
        ),
    }


def entropy(data_set: Set[DataRecord]) -> float:
    """Calculate entropy on whether the game is played."""
    entropy_value = 0.0

    played = [record for record in data_set if record.play]
    not_played = [record for record in data_set if not record.play]

    proportion_played = len(played) / len(data_set)
    proportion_not_played = len(not_played) / len(data_set)

    entropy_value -= proportion_played * log2_zero(proportion_played)
    entropy_value -= proportion_not_played * log2_zero(proportion_not_played)

    return entropy_value


def _gain_continuous(
    data_set: Set[DataRecord], attribute_name: str
) -> Tuple[float, float]:
    """Calculate Ires and corresponding split point based on a continuous attribute."""
    sorted_data = sorted(data_set, key=lambda record: getattr(record, attribute_name))

    smallest_ires = 1.0
    split_point_for_smallest = 0.0

    # Note: When iterating, don't consider sets for the first and last possible split points since they will contain
    # sets of one record on one of the sides of the split.
    for index in range(1, len(sorted_data) - 2):
        first_value = getattr(sorted_data[index], attribute_name)
        second_value = getattr(sorted_data[index + 1], attribute_name)
        candidate_split_point = (first_value + second_value) / 2

        num_played_below_split = len(
            [
                record
                for record in data_set
                if record.play
                and getattr(record, attribute_name) <= candidate_split_point
            ]
        )
        num_not_played_below_split = len(
            [
                record
                for record in data_set
                if not record.play
                and getattr(record, attribute_name) <= candidate_split_point
            ]
        )
        num_played_above_split = len(
            [
                record
                for record in data_set
                if record.play
                and getattr(record, attribute_name) > candidate_split_point
            ]
        )
        num_not_played_above_split = len(
            [
                record
                for record in data_set
                if not record.play
                and getattr(record, attribute_name) > candidate_split_point
            ]
        )

        num_below = num_played_below_split + num_not_played_below_split
        num_above = num_played_above_split + num_not_played_above_split

        entropy_below = 0.0
        entropy_below -= (num_played_below_split / num_below) * log2_zero(
            num_played_below_split / num_below
        )
        entropy_below -= (num_not_played_below_split / num_below) * log2_zero(
            num_not_played_below_split / num_below
        )

        entropy_above = 0.0
        entropy_above -= (num_played_above_split / num_above) * log2_zero(
            num_played_above_split / num_above
        )
        entropy_above -= (num_not_played_above_split / num_above) * log2_zero(
            num_not_played_above_split / num_above
        )

        candidate_ires = 0.0
        candidate_ires += (num_below / len(data_set)) * entropy_below
        candidate_ires += (num_above / len(data_set)) * entropy_above

        if candidate_ires < smallest_ires:
            smallest_ires = candidate_ires
            split_point_for_smallest = candidate_split_point

    return entropy(data_set) - smallest_ires, split_point_for_smallest


def gain_outlook(data_set: Set[DataRecord]) -> float:
    """Calculate gain based on condition outlook."""
    ires_value = 0.0

    sunny = {record for record in data_set if record.outlook == "sunny"}
    proportion_sunny = len(sunny) / len(data_set)
    ires_value += proportion_sunny * entropy(sunny)

    overcast = {record for record in data_set if record.outlook == "overcast"}
    proportion_overcast = len(overcast) / len(data_set)
    ires_value += proportion_overcast * entropy(overcast)

    rainy = {record for record in data_set if record.outlook == "rainy"}
    proportion_rainy = len(rainy) / len(data_set)
    ires_value += proportion_rainy * entropy(rainy)

    return entropy(data_set) - ires_value


def gain_temperature(data_set: Set[DataRecord]) -> Tuple[float, float]:
    """Calculate gain and split point based on temperature."""
    return _gain_continuous(data_set, attribute_name="temperature")


def gain_humidity(data_set: Set[DataRecord]) -> Tuple[float, float]:
    """Calculate gain and split point based on humidity."""
    return _gain_continuous(data_set, attribute_name="humidity")


def gain_windy(data_set: Set[DataRecord]) -> float:
    """Calculate Ires based on it's windy."""
    ires_value = 0.0

    windy = {record for record in data_set if record.windy}
    proportion_windy = len(windy) / len(data_set)
    ires_value += proportion_windy * entropy(windy)

    not_windy = {record for record in data_set if not record.windy}
    proportion_not_windy = len(not_windy) / len(data_set)
    ires_value += proportion_not_windy * entropy(not_windy)

    return entropy(data_set) - ires_value


def _generate_decision_tree_for_condition(
    parent_node: TreeNode,
    subset: Set[DataRecord],
    attributes: Set[str],
    selected_attribute: str,
    condition: str,
    played_count: int,
    not_played_count: int,
    heuristic_functions: Dict[str, Callable],
) -> None:
    """Utility function to grow a branch and recurse with error handling and branch creation."""
    # If there's not a subset, attach a leaf labeled with the most common class int the samples.
    if not subset:
        child = TreeNode()
        if played_count > not_played_count:
            child.label = "Play"
        else:
            child.label = "Don't Play"

    # Otherwise, recurse and attach the returned node.
    else:
        child = generate_decision_tree(
            subset,
            attributes.difference({selected_attribute}),
            heuristic_functions,
        )

    parent_node.branches.append(TreeEdge(parent_node, child, condition.capitalize()))


def generate_decision_tree(
    data_set: Set[DataRecord],
    attributes: Set[str],
    heuristic_functions: Dict[str, Callable],
) -> TreeNode:
    node = TreeNode()

    # If all samples are of the same class, then return a labeled leaf node.
    if all([record.play for record in data_set]):
        node.label = "Play"
        return node

    if all([not record.play for record in data_set]):
        node.label = "Don't Play"
        return node

    # If no attributes remain, then see what class is most common.
    played_count = len([record for record in data_set if record.play])
    not_played_count = len([record for record in data_set if not record.play])

    if not attributes:
        if played_count > not_played_count:
            node.label = "Play"
        else:
            node.label = "Don't Play"

        return node

    # Find highest gain value.
    gain_value_map = {}
    split_points = {}

    if "outlook" in attributes:
        gain_value_map["outlook"] = heuristic_functions["outlook"](data_set)
    if "temperature" in attributes:
        gain, split = heuristic_functions["temperature"](data_set)
        gain_value_map["temperature"] = gain
        split_points["temperature"] = split
    if "humidity" in attributes:
        gain, split = heuristic_functions["humidity"](data_set)
        gain_value_map["humidity"] = gain
        split_points["humidity"] = split
    if "windy" in attributes:
        gain_value_map["windy"] = heuristic_functions["windy"](data_set)

    selected_attribute = max(gain_value_map, key=gain_value_map.get)
    node.label = selected_attribute.capitalize()

    # Handle the continuous case.
    if selected_attribute in ["temperature", "humidity"]:
        node.label += f" (Split @ {split_points[selected_attribute]})"

        _generate_decision_tree_for_condition(
            parent_node=node,
            subset={
                record
                for record in data_set
                if getattr(record, selected_attribute)
                <= split_points[selected_attribute]
            },
            attributes=attributes,
            selected_attribute=selected_attribute,
            condition="<=",
            played_count=played_count,
            not_played_count=not_played_count,
            heuristic_functions=heuristic_functions,
        )

        _generate_decision_tree_for_condition(
            parent_node=node,
            subset={
                record
                for record in data_set
                if getattr(record, selected_attribute)
                > split_points[selected_attribute]
            },
            attributes=attributes,
            selected_attribute=selected_attribute,
            condition=">",
            played_count=played_count,
            not_played_count=not_played_count,
            heuristic_functions=heuristic_functions,
        )

    # Handle the discrete case.
    if selected_attribute in ["outlook", "windy"]:
        for condition in {getattr(record, selected_attribute) for record in data_set}:
            _generate_decision_tree_for_condition(
                parent_node=node,
                subset={
                    record
                    for record in data_set
                    if getattr(record, selected_attribute) == condition
                },
                attributes=attributes,
                selected_attribute=selected_attribute,
                condition=str(condition),
                played_count=played_count,
                not_played_count=not_played_count,
                heuristic_functions=heuristic_functions,
            )

    return node


def dump_tree(node: TreeNode, current_depth: int = 0) -> None:
    """Dump tree representation to console."""
    indentation = 4 * current_depth * " "
    print(indentation + node.label)

    for branch in [b for b in node.branches if b.destination]:
        print(indentation + 2 * " " + branch.condition)
        dump_tree(branch.destination, current_depth + 1)


def main() -> int:
    """Generate decision tree for tennis data set."""
    argument_parser = ArgumentParser(description=main.__doc__)
    argument_parser.add_argument(
        "heuristic",
        help="selection heuristic (valid choices are information_gain)",
    )
    args = argument_parser.parse_args()

    starting_attributes = {"outlook", "temperature", "humidity", "windy"}
    training_data_set = make_training_data_set()

    if args.heuristic.lower() == "information_gain":
        information_gain_functions = {
            "outlook": gain_outlook,
            "temperature": gain_temperature,
            "humidity": gain_humidity,
            "windy": gain_windy,
        }
    else:
        raise ValueError(f"unknown selection heuristic {args.heuristic}")

    tree = generate_decision_tree(
        training_data_set, starting_attributes, information_gain_functions
    )
    dump_tree(tree)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
