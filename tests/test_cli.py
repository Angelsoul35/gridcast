from __future__ import annotations

import argparse

from gridcast import cli


def test_cli_drift(capsys):
    cli.cmd_drift(argparse.Namespace(seed=7))
    assert "drifted" in capsys.readouterr().out


def test_cli_backtest(capsys):
    cli.cmd_backtest(argparse.Namespace(days=90, seed=1, folds=2, horizon=48))
    assert "mape" in capsys.readouterr().out


def test_cli_demo(capsys):
    cli.cmd_demo(argparse.Namespace(days=90))
    assert "should_retrain" in capsys.readouterr().out
