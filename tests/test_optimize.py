"""06장 탐색 테스트 — 최종 테스트 구간 차단, Walk-forward 창, 목적함수, 탐색 공간."""

import numpy as np
import optuna
import pandas as pd
import pytest

from vwap_trading.optimize.search import (FINAL_TEST_START, SPACE, FinalTestLeak, check_before_final, optimize,
                                          score, suggest, walk_forward_windows)


def test_final_test_period_is_blocked():
    check_before_final(["2026-08-31"])
    with pytest.raises(FinalTestLeak):
        check_before_final(["2026-03-02", "2026-09-01"])
    sel = pd.DataFrame({"symbol": ["000001"], "tradable_from": [pd.Timestamp("2026-09-15 09:30")]})
    with pytest.raises(FinalTestLeak):
        optimize({}, sel, n_trials=1)


def test_walk_forward_windows_never_touch_final_period():
    months = [str(m) for m in pd.period_range("2025-11", "2026-10", freq="M")]
    w = walk_forward_windows(months, train=4, test=1)
    assert w[0] == (["2025-11", "2025-12", "2026-01", "2026-02"], ["2026-03"])
    assert w[-1][1] == ["2026-08"]
    assert len(w) == 6
    for train, test in w:
        assert max(train) < min(test)                                    # 학습은 항상 검증보다 앞
        assert pd.Period(test[-1]).end_time.date() < FINAL_TEST_START


def test_score_kinds():
    net = np.array([0.01, -0.02, 0.0, 0.005])
    assert score(net, "total", min_trades=1) == pytest.approx(-0.005)
    assert score(net, "mean", min_trades=1) == pytest.approx(-0.00125)
    assert score(net, "lcb", min_trades=1) == pytest.approx(net.mean() - net.std(ddof=1) / 2)
    assert score(net, "lcb", min_trades=5) == -1.0                       # 최소 거래 미달


def test_suggest_stays_in_space_and_keeps_exit_rules():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(50):
        trial = study.ask()
        p = suggest(trial)
        values = {"N": p.vwap.N, "k": p.vwap.k, "theta": p.vwap.theta, "m": p.vwap.m, "delta": p.delta, "rho": p.rho}
        for name, (lo, hi, _) in SPACE.items():
            assert lo <= values[name] <= hi
        assert (p.lam, p.tau, p.h, p.eps, p.cooldown) == (1.5, 2.0, 30, 0.2, 10)    # 위험관리 규칙은 고정
        study.tell(trial, 0.0)


def test_final_evaluation_still_blocks_same_month():
    """09장 최종 평가는 limit 을 평가 달 첫날로 옮길 뿐, 그 달 데이터로 학습하는 것은 여전히 막는다."""
    from datetime import date
    check_before_final(["2026-09-30"], limit=date(2026, 10, 1))
    with pytest.raises(FinalTestLeak):
        check_before_final(["2026-10-01"], limit=date(2026, 10, 1))
