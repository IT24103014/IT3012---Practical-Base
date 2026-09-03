# logic_engine.py
"""A simple knowledge base for storing facts and rules (Practical 5: Knowledge-Based Agents)."""


class KnowledgeBase:
    """A propositional-logic knowledge base.

    Stores unique string facts in a set and implication rules as
    (premises, conclusion) tuples, where premises is a list of fact strings.
    """

    def __init__(self):
        self.facts = set()      # Unique string facts, e.g. {"TargetVisible"}
        self.rules = []         # Rules as ( [premise_list], "conclusion_string" )

    def tell_fact(self, fact_string: str) -> None:
        """Add a single fact string to the knowledge base."""
        self.facts.add(fact_string)

    def tell_rule(self, premise_list: list, conclusion_string: str) -> None:
        """Add a rule of the form (premises -> conclusion)."""
        self.rules.append((premise_list, conclusion_string))

    def clear_facts(self) -> None:
        """Remove all facts from the knowledge base."""
        self.facts.clear()

    def forward_chain(self) -> None:
        """Deduce new facts by repeatedly applying rules (data-driven forward chaining)."""
        new_facts_added = True

        while new_facts_added:
            new_facts_added = False

            for premises, conclusion in self.rules:
                if conclusion not in self.facts:
                    # Modus Ponens check: all premises must be known facts
                    if all(premise in self.facts for premise in premises):
                        self.facts.add(conclusion)
                        new_facts_added = True

