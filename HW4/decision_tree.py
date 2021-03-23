"""Generate decision tree for tennis data set.

Code by Tyler N. Thieding (tnt36)
Written for CSDS 435: Data Mining (Spring 2021)

"""

import math
import sys
from dataclasses import dataclass
from typing import Set

EXIT_CODE_SUCCESS = 0


@dataclass()
class DataRecord:

    """Represent a row of the training data set."""

    outlook: str
    temperature: int
    humidity: int
    windy: bool
    play: bool

    def __hash__(self) -> int:
        return hash((self.outlook, self.temperature, self.humidity, self.windy, self.play))


def make_training_data_set() -> Set[DataRecord]:
    """Make a set containing all rows from the table."""
    return {
        DataRecord(outlook="sunny", temperature=85, humidity=85, windy=False, play=False),
        DataRecord(outlook="sunny", temperature=80, humidity=90, windy=True, play=False),
        DataRecord(outlook="overcast", temperature=83, humidity=86, windy=False, play=True),
        DataRecord(outlook="rainy", temperature=70, humidity=96, windy=False, play=True),
        DataRecord(outlook="rainy", temperature=68, humidity=80, windy=False, play=True),
        DataRecord(outlook="rainy", temperature=65, humidity=70, windy=True, play=False),
        DataRecord(outlook="overcast", temperature=64, humidity=65, windy=True, play=True),
        DataRecord(outlook="sunny", temperature=72, humidity=95, windy=False, play=False),
        DataRecord(outlook="sunny", temperature=69, humidity=70, windy=False, play=True),
        DataRecord(outlook="rainy", temperature=75, humidity=80, windy=False, play=True),
        DataRecord(outlook="sunny", temperature=75, humidity=70, windy=True, play=True),
        DataRecord(outlook="overcast", temperature=72, humidity=90, windy=True, play=True),
        DataRecord(outlook="overcast", temperature=81, humidity=75, windy=False, play=True),
        DataRecord(outlook="rainy", temperature=71, humidity=91, windy=True, play=False),
    }


def entropy(data_set: Set[DataRecord]) -> float:
    """Calculate entropy on whether the game is played."""
    entropy_value = 0.0

    played = [record for record in data_set if record.play]
    not_played = [record for record in data_set if not record.play]

    proportion_played = len(played) / len(data_set)
    proportion_not_played = len(not_played) / len(data_set)

    try:
        entropy_value -= proportion_played * math.log(proportion_played, 2)
    except ValueError:
        pass  # treat log of zero as zero

    try:
        entropy_value -= proportion_not_played * math.log(proportion_not_played, 2)
    except ValueError:
        pass  # treat log of zero as zero

    return entropy_value


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


def main() -> int:
    """Generate decision tree for tennis data set."""
    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
