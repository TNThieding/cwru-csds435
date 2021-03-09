"""Apply FP-Growth algorithm to transaction set for association rule mining.

Code by Tyler N. Thieding (tnt36)
Written for CSDS 435: Data Mining (Spring 2021)

"""

import os
import sys
from argparse import ArgumentParser
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

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

    def add_transaction(self, items: List[str], node_links: Dict[str, List["FpNode"]]) -> None:
        """Add transaction to tree to build it up."""
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
                node_links[next_item].append(child_node_for_item)

            child_node_for_item.count += 1
            child_node_for_item.add_transaction(items, node_links)

    def is_single_path(self) -> bool:
        """Determine if tree contains a single path."""
        if not self.children:
            return_value = True
        elif len(self.children) > 1:
            return_value = False
        else:
            child = self.children[0]  # guaranteed to have exactly 1 by if and elif clauses above
            return_value = child.is_single_path()

        return return_value

    def get_conditional_pattern_base(self) -> Dict[Tuple[str], int]:
        """Get a single conditional pattern base and its count as a single-element dictionary."""
        parent_items: List[str] = []
        current_node = self.parent

        while current_node.item:
            parent_items.append(current_node.item)
            current_node = current_node.parent

        if parent_items:
            return_value = {tuple(reversed(parent_items)): self.count}
        else:
            return_value = {}

        return return_value


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


def extract_frequent_patterns(conditional_fp_root: FpNode, base_item: str) -> List[Tuple[str]]:
    """Extract frequent patterns from a conditional FP-tree for an item."""
    assert conditional_fp_root.is_single_path()

    current_node = conditional_fp_root
    items = {base_item}

    while True:
        if current_node.item:
            items.add(current_node.item)

        if not current_node.children:
            break
        else:
            current_node = current_node.children[0]

    patterns = []

    # Only get patterns of length 2 or larger since patterns of length 1 can't be used to construct rules.
    for subsequence_length in range(2, len(items) + 1):
        patterns += combinations(items, subsequence_length)

    return patterns


def mine(node_links: Dict[str, List[FpNode]], item_names: List[str]) -> Set[Tuple[str]]:
    """Recursively mine frequent patterns from an FP-tree via its node links."""
    # Generate conditional pattern bases.
    conditional_pattern_bases = {item_id: {} for item_id in item_names}
    for item_id, item_nodes in node_links.items():
        for item_node in item_nodes:
            conditional_pattern_bases[item_id].update(item_node.get_conditional_pattern_base())

    # Use conditional pattern bases to make conditional FP-trees.
    patterns = set()
    for item_id, item_conditional_pattern_bases in conditional_pattern_bases.items():
        conditional_tree = FpNode()
        conditional_tree_node_links = {item_id: [] for item_id in item_names}

        for conditional_pattern_base in item_conditional_pattern_bases:
            conditional_tree.add_transaction(list(conditional_pattern_base), conditional_tree_node_links)

        if conditional_tree.is_single_path():
            for pattern in extract_frequent_patterns(conditional_tree, item_id):
                patterns.add(pattern)
        else:
            # TODO: Recurse!
            pass

    return patterns


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
        fp_tree_root.add_transaction(sorted_transaction, node_links)

    print(mine(node_links, item_names=list(support_counts.keys())))

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
