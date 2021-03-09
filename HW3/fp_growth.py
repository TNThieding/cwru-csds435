"""Apply FP-Growth algorithm to transaction set for association rule mining.

Code by Tyler N. Thieding (tnt36)
Written for CSDS 435: Data Mining (Spring 2021)

"""

import os
import sys
from argparse import ArgumentParser
from typing import Dict, List, Optional

EXIT_CODE_SUCCESS = 0
TRANSACTIONS_FILE_PATH = os.path.join(os.path.dirname(__file__), "table_1.csv")


class FpNode:

    """FP-tree node."""

    def __init__(self, parent: Optional["FpNode"] = None) -> None:
        self.item: Optional[str] = None
        self.count = 0

        self.children: List["FpNode"] = []
        self.parent = parent

    def __repr__(self) -> str:
        return f"FpNode(item={self.item}, count={self.count}, children={repr(self.children)})"

    def add_transaction(self, items: List[str]) -> None:
        if items:
            next_item = items.pop(0)

            child_node_for_item = None
            for child in self.children:
                if child.item == next_item:
                    child_node_for_item = child

            if not child_node_for_item:
                child_node_for_item = FpNode(parent=self)
                child_node_for_item.item = next_item
                self.children.append(child_node_for_item)

            child_node_for_item.count += 1
            child_node_for_item.add_transaction(items)


def make_support_count_map(transactions: List[List[str]]) -> Dict[str, int]:
    """Make support count map from transactions."""
    support_counts = {}

    for transaction in transactions:
        for item in transaction:
            try:
                support_counts[item] += 1
            except KeyError:
                support_counts[item] = 1

    return support_counts


def main() -> int:
    """Apply FP-Growth algorithm to transaction set for association rule mining."""
    # Get arguments from command line. By default, use assignment parameters and data table.
    argument_parser = ArgumentParser(description=main.__doc__)
    argument_parser.add_argument("--transactions", default=TRANSACTIONS_FILE_PATH, metavar="PATH",
                                 help="transactions in comma-separated values form")
    argument_parser.add_argument("--min_sup", default=0.4, type=float, help="minimum support value")
    argument_parser.add_argument("--min_conf", default=0.66, type=float, help="minimum confidence value")
    args = argument_parser.parse_args()

    # Read data from specified transactions file.
    transactions_set = []
    with open(args.transactions, "r") as transactions_fd:
        for transaction_line in transactions_fd.readlines():
            transactions_set.append([
                transaction_item for transaction_item in transaction_line.rstrip().split(",") if transaction_item
            ])

    # Make populated support counts map and empty node link map.
    support_counts = make_support_count_map(transactions_set)
    node_links = {item_id: [] for item_id in support_counts.keys()}

    # Construct FP-tree from transactions.
    fp_tree_root = FpNode()
    for transaction in transactions_set:
        # Recall that items must be sorted from highest to lowest support count.
        sorted_transaction = sorted(transaction, key=lambda item: support_counts[item], reverse=True)
        fp_tree_root.add_transaction(sorted_transaction)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
