"""Generate decision tree for tennis data set.

Code by Tyler N. Thieding (tnt36)
Written for CSDS 435: Data Mining (Spring 2021)

"""

import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

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


def _ires_continuous(
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

    return smallest_ires, split_point_for_smallest


def ires_outlook(data_set: Set[DataRecord]) -> float:
    """Calculate Ires based on condition outlook."""
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

    return ires_value


def ires_temperature(data_set: Set[DataRecord]) -> Tuple[float, float]:
    """Calculate Ires and split point based on temperature."""
    return _ires_continuous(data_set, attribute_name="temperature")


def ires_humidity(data_set: Set[DataRecord]) -> Tuple[float, float]:
    """Calculate Ires and split point based on humidity."""
    return _ires_continuous(data_set, attribute_name="humidity")


def ires_windy(data_set: Set[DataRecord]) -> float:
    """Calculate Ires based on it's windy."""
    ires_value = 0.0

    windy = {record for record in data_set if record.windy}
    proportion_windy = len(windy) / len(data_set)
    ires_value += proportion_windy * entropy(windy)

    not_windy = {record for record in data_set if not record.windy}
    proportion_not_windy = len(not_windy) / len(data_set)
    ires_value += proportion_not_windy * entropy(not_windy)

    return ires_value


def _generate_decision_tree_for_condition(
    parent_node: TreeNode,
    subset: Set[DataRecord],
    attributes: Set[str],
    selected_attribute: str,
    condition: str,
    played_count: int,
    not_played_count: int,
) -> None:
    """Utility function to grow a branch and recurse with error handling."""
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
            subset, attributes.difference({selected_attribute})
        )

    parent_node.branches.append(TreeEdge(parent_node, child, condition.capitalize()))


# TODO: When generalizing to support different selection criteria, change attribute list to be a tuple of names and
#  test functions. Any cleaner ideas?
def generate_decision_tree(data_set: Set[DataRecord], attributes: Set[str]) -> TreeNode:
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

    # Find lowest Ires value.
    # TODO: This area could use some cleanup to be less hard coded.
    ires_value_map = {}
    split_points = {}

    if "outlook" in attributes:
        ires_value_map["outlook"] = ires_outlook(data_set)
    if "temperature" in attributes:
        ires, split = ires_temperature(data_set)
        ires_value_map["temperature"] = ires
        split_points["temperature"] = split
    if "humidity" in attributes:
        ires, split = ires_humidity(data_set)
        ires_value_map["humidity"] = ires
        split_points["humidity"] = split
    if "windy" in attributes:
        ires_value_map["windy"] = ires_windy(data_set)

    selected_attribute = min(ires_value_map, key=ires_value_map.get)
    node.label = selected_attribute.capitalize()

    # TODO: Like above, this area could use some cleanup to be less hard coded.
    if selected_attribute == "outlook":
        for condition in ["sunny", "overcast", "rainy"]:
            _generate_decision_tree_for_condition(
                parent_node=node,
                subset={record for record in data_set if record.outlook == condition},
                attributes=attributes,
                selected_attribute=selected_attribute,
                condition=condition,
                played_count=played_count,
                not_played_count=not_played_count,
            )

    if selected_attribute == "temperature":
        condition = "<= " + str(split_points["temperature"])
        _generate_decision_tree_for_condition(
            parent_node=node,
            subset={
                record
                for record in data_set
                if record.temperature <= split_points["temperature"]
            },
            attributes=attributes,
            selected_attribute=selected_attribute,
            condition=condition,
            played_count=played_count,
            not_played_count=not_played_count,
        )

        condition = "> " + str(split_points["temperature"])
        _generate_decision_tree_for_condition(
            parent_node=node,
            subset={
                record
                for record in data_set
                if record.temperature > split_points["temperature"]
            },
            attributes=attributes,
            selected_attribute=selected_attribute,
            condition=condition,
            played_count=played_count,
            not_played_count=not_played_count,
        )

    if selected_attribute == "humidity":
        condition = "<= " + str(split_points["humidity"])
        _generate_decision_tree_for_condition(
            parent_node=node,
            subset={
                record
                for record in data_set
                if record.humidity <= split_points["humidity"]
            },
            attributes=attributes,
            selected_attribute=selected_attribute,
            condition=condition,
            played_count=played_count,
            not_played_count=not_played_count,
        )

        condition = "> " + str(split_points["humidity"])
        _generate_decision_tree_for_condition(
            parent_node=node,
            subset={
                record
                for record in data_set
                if record.humidity > split_points["humidity"]
            },
            attributes=attributes,
            selected_attribute=selected_attribute,
            condition=condition,
            played_count=played_count,
            not_played_count=not_played_count,
        )

    if selected_attribute == "windy":
        for condition in [True, False]:
            _generate_decision_tree_for_condition(
                parent_node=node,
                subset={record for record in data_set if record.windy == condition},
                attributes=attributes,
                selected_attribute=selected_attribute,
                condition=str(condition),
                played_count=played_count,
                not_played_count=not_played_count,
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
    starting_attributes = {"outlook", "temperature", "humidity", "windy"}
    training_data_set = make_training_data_set()

    tree = generate_decision_tree(training_data_set, starting_attributes)
    dump_tree(tree)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
