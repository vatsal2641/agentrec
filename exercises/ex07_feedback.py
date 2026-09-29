"""Exercise 07: feedback -> user state. Reference: memory/user_state.py.

Rules to implement:
  click, or rating >= 4 : add item to history (no duplicates); remove from disliked
  rating <= 2           : remove from history; add to disliked
  skip                  : skip_counts[item] += 1
  blocked()             : disliked ∪ {items skipped >= 2 times}
"""


class State:
    def __init__(self):
        self.history, self.disliked, self.skips = [], set(), {}

    def apply(self, event_type: str, item: int, value: float | None = None) -> None:
        raise NotImplementedError

    def blocked(self) -> set[int]:
        raise NotImplementedError


if __name__ == "__main__":
    s = State()
    s.apply("click", 1); s.apply("click", 1); s.apply("rating", 2, 5)
    assert s.history == [1, 2]
    s.apply("rating", 1, 1)
    assert s.history == [2] and s.disliked == {1}
    s.apply("skip", 3); assert 3 not in s.blocked()
    s.apply("skip", 3); assert s.blocked() == {1, 3}
    s.apply("rating", 1, 4); assert 1 not in s.blocked() and s.history == [2, 1]
    print("all tests passed. Question: why should a single skip NOT block an item?")
