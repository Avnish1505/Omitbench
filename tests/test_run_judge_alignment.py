"""
Regression tests for the corpus-alignment bug: scripts/run_judge.py used to
call `experiment.build(repos, ...)` once with the FULL --repos list.
experiment.build() creates exactly one random.Random(seed) and reuses it
across every repo passed in, shuffling each repo's commits from wherever
that shared RNG happened to land -- so only the FIRST repo in the list ever
matched results/shards/*.jsonl, which scripts/run_shards.sh generates by
calling `omitbench.experiment` as one SUBPROCESS PER REPO (a fresh
random.Random(seed) each time). Every other repo silently drew a different
sample of commits, breaking analyze.py's assumption that every detector
(including a judge) sees the same iid coverage -- before a single dollar is
spent on a real sweep.

`scripts/run_judge.py::build_instances()` fixes this by calling build() once
per repo, exactly mirroring run_shards.sh. `check_alignment()` is the
runtime guard that would have caught the original bug: it hard-fails unless
a judge corpus build's iids are a subset of the shards'.

No network, no real corpus needed -- omitbench.experiment.build and
run_judge.corpus_variant_counts are both monkeypatched.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")  # run_judge.py has no package __init__

import run_judge as RJ  # noqa: E402
from omitbench import experiment as E  # noqa: E402
from omitbench import mutate as M  # noqa: E402


def _fake_build_one_repo_per_call(seen_calls):
    """Simulates the real defect precisely: a fresh, deterministic sequence
    per (repo, seed) IF called with exactly one repo per invocation (what
    run_shards.sh/experiment.main() do, and what build_instances() must
    reproduce); a DIFFERENT, seed-drifted sequence if more than one repo is
    passed in a single call (the bug).
    """
    def fake(repos, per_repo, seed, scan_cap):
        seen_calls.append(tuple(repos))
        insts = []
        for offset, repo in enumerate(repos):
            # "drift": every repo after the first in a multi-repo call gets
            # a different id sequence than a solo call with the same seed
            # would -- modelling the shared-RNG bug without needing git.
            drift = 0 if len(repos) == 1 else offset
            for k in range(per_repo):
                iid = f"{repo}-seed{seed}-drift{drift}-{k}"
                insts.append((iid, repo, "spec", {}, "x::y", ["x::y"],
                              [("CLEAN", {})]))
        return insts
    return fake


def test_build_instances_calls_build_once_per_repo(monkeypatch):
    seen_calls = []
    monkeypatch.setattr(E, "build", _fake_build_one_repo_per_call(seen_calls))

    RJ.build_instances(["repoA", "repoB", "repoC"], per_repo=2, seed=0, scan_cap=10)

    assert seen_calls == [("repoA",), ("repoB",), ("repoC",)], (
        "build_instances must call experiment.build() ONCE PER REPO, not "
        "once with the full repo list -- that's the fix for the shared-RNG "
        "bug this file guards against")


def test_build_instances_matches_a_solo_call_per_repo(monkeypatch):
    """The actual regression: instances for repoB from build_instances()
    must be identical to a solo build([repoB], ...) call -- the multi-repo
    single-call pattern this replaces would NOT have matched for repoB."""
    seen_calls = []
    monkeypatch.setattr(E, "build", _fake_build_one_repo_per_call(seen_calls))

    combined = RJ.build_instances(["repoA", "repoB"], per_repo=2, seed=0, scan_cap=10)
    solo_b = E.build(["repoB"], per_repo=2, seed=0, scan_cap=10)

    combined_b_iids = sorted(i[0] for i in combined if i[1] == "repoB")
    solo_b_iids = sorted(i[0] for i in solo_b)
    assert combined_b_iids == solo_b_iids


def test_check_alignment_passes_when_judge_iids_are_a_subset(monkeypatch, capsys):
    monkeypatch.setattr(RJ, "corpus_variant_counts",
                         lambda: (set(), {"a": set(), "b": set(), "c": set()}, {}))
    insts = [("a", "r", "s", {}, "t", [], []), ("b", "r", "s", {}, "t", [], [])]

    RJ.check_alignment(insts, limit=20)  # must not raise / exit

    out = capsys.readouterr().out
    assert "alignment check passed" in out
    assert "iid count, results/shards/*.jsonl   3" in out
    assert "iid count, 20-instance judge build   2" in out
    assert "intersection                        2" in out


def test_check_alignment_hard_fails_when_judge_iids_are_not_a_subset(monkeypatch):
    """The exact failure mode the original bug would have produced: a judge
    iid absent from the shards must hard-fail, not warn."""
    monkeypatch.setattr(RJ, "corpus_variant_counts",
                         lambda: (set(), {"a": set()}, {}))
    insts = [("a", "r", "s", {}, "t", [], []),
             ("not-in-shards", "r", "s", {}, "t", [], [])]

    try:
        RJ.check_alignment(insts, limit=20)
        assert False, "expected SystemExit on a non-subset alignment check"
    except SystemExit as e:
        assert "ALIGNMENT CHECK FAILED" in str(e)


def test_build_instances_clears_mutate_memos_between_repos(monkeypatch):
    """CLAUDE.md: 'mutate.py memoises AST parses in module-level dicts. A
    multi-repo run OOMs without clearing them between repos... this is why
    run_shards.sh uses one process per repo.' build_instances() runs every
    repo in ONE process (no subprocess boundary), so it must do the clearing
    experiment.py's own multi-repo entry point does -- this caught a real
    growing-RSS OOM risk found while running this check against the real
    8-repo corpus."""
    def fake_build(repos, per_repo, seed, scan_cap):
        # simulate what a real build does to the memos: populate them
        M._TREE_MEMO[len(M._TREE_MEMO)] = object()
        M._DEF_MEMO[len(M._DEF_MEMO)] = {}
        M._CALL_MEMO[(len(M._CALL_MEMO),)] = []
        M._TARGETS_MEMO[len(M._TARGETS_MEMO)] = frozenset()
        return []

    monkeypatch.setattr(E, "build", fake_build)
    for memo in (M._TREE_MEMO, M._DEF_MEMO, M._CALL_MEMO, M._TARGETS_MEMO):
        memo.clear()

    RJ.build_instances(["repoA", "repoB", "repoC"], per_repo=1, seed=0, scan_cap=1)

    for memo in (M._TREE_MEMO, M._DEF_MEMO, M._CALL_MEMO, M._TARGETS_MEMO):
        assert len(memo) == 0, (
            f"{memo} was not cleared after the last repo -- memos must be "
            f"cleared after EVERY repo, including the last, so a second "
            f"build_instances() call in the same process (e.g. the "
            f"check_alignment pre-flight followed by the real sweep) "
            f"doesn't inherit stale state")


def test_check_alignment_respects_limit(monkeypatch, capsys):
    """Only the first `limit` instances count as 'the judge run' -- matches
    the '20-instance judge run' framing this check was specified with."""
    monkeypatch.setattr(RJ, "corpus_variant_counts",
                         lambda: (set(), {f"i{n}": set() for n in range(50)}, {}))
    insts = [(f"i{n}", "r", "s", {}, "t", [], []) for n in range(50)]

    RJ.check_alignment(insts, limit=5)
    out = capsys.readouterr().out
    assert "iid count, 5-instance judge build   5" in out
