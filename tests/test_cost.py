from fraud.cost import flag_nothing_cost, total_cost


def test_hand_computed():
    # one fraud caught (TP), one fraud missed worth $50 (FN), one false alarm costing $5 (FP)
    # total = 50 + 5 = 55
    y = [1, 1, 0, 0]
    pred = [1, 0, 1, 0]
    amt = [100, 50, 20, 10]
    assert total_cost(y, pred, amt, fp_cost=5) == 55.0
    # adding tp_cost=2: TP is caught fraud at index 0 → +2
    assert total_cost(y, pred, amt, fp_cost=5, tp_cost=2) == 57.0


def test_flag_nothing():
    # frauds are at index 0 and 2 with amounts 10 and 5
    assert flag_nothing_cost([1, 0, 1], [10, 99, 5]) == 15.0


def test_all_flagged_no_fraud():
    # no frauds → FN cost = 0, TP cost = 0, FP cost for all non-frauds
    assert total_cost([0, 0], [1, 1], [10, 20], fp_cost=3) == 6.0


def test_no_flags():
    # no alerts → only FN cost (missed frauds)
    assert total_cost([1, 1, 0], [0, 0, 0], [30, 70, 50], fp_cost=5) == 100.0
