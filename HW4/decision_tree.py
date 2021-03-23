"""Generate decision tree for tennis data set.

Code by Tyler N. Thieding (tnt36)
Written for CSDS 435: Data Mining (Spring 2021)

"""

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
        DataRecord(outlook="sunny", temperature=85, humidity=85, windy=False, play=True),
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


def main() -> int:
    """Generate decision tree for tennis data set."""

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
