"""
eds_validate.py
===============
Does the Evidence Defensibility Score predict what a human decides?

WHY THIS SCRIPT IS THE PROJECT'S ONLY REAL EXPERIMENT
Everything else the system does can be demonstrated: the detector has a number,
the rules fire or abstain, the portal shows a challan. EDS is the one claim that
cannot be demonstrated, only tested. Inventing a score and asserting it captures
"would this survive review" is worth nothing by itself -- the score has to be
measured against the judgement it claims to anticipate.

THE MEASURE
AUC, computed from ranks. It answers exactly the question worth asking: given
one package a reviewer approved and one they rejected, how often does the
approved one score higher? 0.5 is a coin flip and the score is worthless. Around
0.7 it is useful. Above 0.8 it is doing real work.

AUC is the right statistic here rather than accuracy because it needs no
threshold. The bands (80 / 50) are a policy choice that can be moved; whether
the underlying score ORDERS packages the way a human does is the property that
either exists or does not, and AUC measures that alone.

WHY RANKS AND NOT A LIBRARY
scikit-learn is not installed and pandas in this environment is broken by a
numpy ABI mismatch. The rank form of AUC -- the Mann-Whitney U statistic over
n_pos * n_neg -- is a dozen lines, handles ties correctly by averaging ranks,
and removes a dependency from a result that has to be reproducible by whoever
reads the report.

WHAT COUNTS AS WHICH LABEL
Approved is positive, rejected is negative. Escalated is NEITHER and is excluded
from the primary analysis: it is a reviewer declining to decide, and folding it
into either class would be inventing an opinion they refused to give. It is
reported separately, and a secondary analysis treats escalated-or-rejected as
"not issuable" for anyone who wants that framing.

ONLY THE LATEST DECISION PER REVIEWER COUNTS
decisions.jsonl is append-only, so a reviewer who changed their mind appears
twice. The last word is the one measured; the history stays in the log.

HONEST LIMITS, WHICH BELONG IN THE REPORT
Student reviewers are not traffic officers. The sample is small. The packages
come from one camera and mostly one rule. None of that invalidates the result,
but every one of them bounds it, and the script prints them so they cannot be
quietly dropped between the console and the write-up.

USAGE
    python scripts/eds_validate.py
    python scripts/eds_validate.py --decisions data/review/decisions.jsonl
    python scripts/eds_validate.py --out results/eds
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# statistics, from first principles
# ---------------------------------------------------------------------------

def ranks(values: list[float]) -> list[float]:
    """Ranks with ties averaged -- the tie handling AUC depends on."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(a random positive outranks a random negative), ties counting half.

    The Mann-Whitney U form: sum the positives' ranks over the pooled sample,
    subtract the minimum possible, divide by the number of pairs.
    """
    if not pos or not neg:
        return None
    pooled = pos + neg
    r = ranks(pooled)
    rank_sum = sum(r[:len(pos)])
    n1, n2 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n2)


def bootstrap_ci(pos: list[float], neg: list[float], iters: int = 4000,
                 seed: int = 7) -> tuple[float, float] | None:
    """Percentile CI by resampling. Small samples deserve a visible interval."""
    if len(pos) < 2 or len(neg) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        a = [rng.choice(pos) for _ in pos]
        b = [rng.choice(neg) for _ in neg]
        v = auc(a, b)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def roc_points(pos: list[float], neg: list[float]) -> list[tuple[float, float]]:
    """(FPR, TPR) across every threshold the data actually contains."""
    thresholds = sorted(set(pos + neg), reverse=True)
    pts = [(0.0, 0.0)]
    for t in thresholds:
        tp = sum(1 for v in pos if v >= t)
        fp = sum(1 for v in neg if v >= t)
        pts.append((fp / len(neg), tp / len(pos)))
    pts.append((1.0, 1.0))
    return pts


def cohens_kappa(a: dict, b: dict) -> float | None:
    """Agreement between two reviewers beyond chance, on shared packages."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return None
    labels = sorted({a[k] for k in shared} | {b[k] for k in shared})
    n = len(shared)
    observed = sum(1 for k in shared if a[k] == b[k]) / n
    expected = sum((sum(1 for k in shared if a[k] == L) / n) *
                   (sum(1 for k in shared if b[k] == L) / n) for L in labels)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


# ---------------------------------------------------------------------------

def load(decisions: Path) -> list[dict]:
    if not decisions.is_file():
        return []
    out = []
    for line in decisions.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def latest_per_reviewer(rows: list[dict]) -> dict:
    """{(officer, evidence_id): row} keeping only each reviewer's last word."""
    out = {}
    for r in rows:                      # file order is chronological
        out[(r.get('officer', 'anon'), r.get('evidence_id'))] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--decisions', default=str(ROOT / 'data' / 'review' / 'decisions.jsonl'))
    ap.add_argument('--out', default=str(ROOT / 'results' / 'eds'))
    ap.add_argument('--no-chart', action='store_true')
    args = ap.parse_args()

    rows = load(Path(args.decisions))
    if not rows:
        raise SystemExit(
            f'No decisions in {args.decisions}.\n'
            'Review some packages first:\n'
            '    python portal/server.py --blind --shuffle')

    latest = latest_per_reviewer(rows)
    decisions = list(latest.values())

    by_decision = Counter(r['decision'] for r in decisions)
    reviewers = sorted({r.get('officer', 'anon') for r in decisions})

    def scores(kind):
        return [float(r['eds_score']) for r in decisions
                if r['decision'] == kind and r.get('eds_score') is not None]

    approved, rejected, escalated = scores('approved'), scores('rejected'), scores('escalated')

    print('=' * 70)
    print('EDS VALIDATION')
    print('=' * 70)
    print(f'decisions file : {args.decisions}')
    print(f'rows           : {len(rows)} logged, {len(decisions)} after keeping '
          f'each reviewer\'s latest')
    print(f'reviewers      : {len(reviewers)} ({", ".join(reviewers)})')
    print(f'decisions      : ' + ', '.join(f'{k} {v}' for k, v in sorted(by_decision.items())))

    if len(approved) < 2 or len(rejected) < 2:
        print('\nNOT ENOUGH DATA to measure anything.')
        print(f'  approved with a score: {len(approved)}')
        print(f'  rejected with a score: {len(rejected)}')
        print('\nAUC needs at least two of each. Aim for 150+ reviewed packages')
        print('spanning a range of evidence quality -- a set where everything is')
        print('obviously good or obviously bad measures nothing either.')
        raise SystemExit(1)

    # -- primary result -----------------------------------------------------
    a = auc(approved, rejected)
    ci = bootstrap_ci(approved, rejected)
    mean_a = sum(approved) / len(approved)
    mean_r = sum(rejected) / len(rejected)

    print('\n' + '-' * 70)
    print('PRIMARY: does EDS rank approved above rejected?')
    print('-' * 70)
    print(f'  n approved     : {len(approved)}   mean EDS {mean_a:.1f}')
    print(f'  n rejected     : {len(rejected)}   mean EDS {mean_r:.1f}')
    print(f'  separation     : {mean_a - mean_r:+.1f} points')
    print(f'  AUC            : {a:.3f}' + (f'   95% CI [{ci[0]:.3f}, {ci[1]:.3f}]'
                                           if ci else '   (CI needs more data)'))
    verdict = ('no better than chance' if a < 0.6 else
               'weak' if a < 0.7 else
               'useful' if a < 0.8 else
               'strong')
    print(f'  reading        : {verdict}')
    if ci and ci[0] <= 0.5:
        print('  NOTE: the interval includes 0.5, so chance is not excluded.')

    # -- do the bands line up? ---------------------------------------------
    print('\n' + '-' * 70)
    print('BANDS: is the auto / review / drop triage doing its job?')
    print('-' * 70)
    bands = defaultdict(Counter)
    for r in decisions:
        if r.get('eds_verdict'):
            bands[r['eds_verdict']][r['decision']] += 1
    print(f'  {"band":8s} {"approved":>9s} {"rejected":>9s} {"escalated":>10s} '
          f'{"approval rate":>14s}')
    for band in ('auto', 'review', 'drop'):
        c = bands.get(band, Counter())
        tot = c['approved'] + c['rejected']
        rate = f'{100.0 * c["approved"] / tot:.0f}%' if tot else '--'
        print(f'  {band:8s} {c["approved"]:>9d} {c["rejected"]:>9d} '
              f'{c["escalated"]:>10d} {rate:>14s}')
    print('  A working triage approves most of `auto` and almost none of `drop`.')

    # -- where should the threshold actually sit? ---------------------------
    print('\n' + '-' * 70)
    print('THRESHOLDS: what each cut-off would cost')
    print('-' * 70)
    print(f'  {"cut":>5s} {"precision":>10s} {"recall":>8s} '
          f'{"approved kept":>14s} {"rejected let through":>21s}')
    for t in (40, 50, 60, 70, 80, 90):
        tp = sum(1 for v in approved if v >= t)
        fp = sum(1 for v in rejected if v >= t)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / len(approved)
        print(f'  {t:>5d} {prec:>10.2f} {rec:>8.2f} '
              f'{tp:>7d}/{len(approved):<6d} {fp:>12d}/{len(rejected):<8d}')

    # -- agreement between reviewers ---------------------------------------
    kappas = []
    if len(reviewers) >= 2:
        per = defaultdict(dict)
        for (off, eid), r in latest.items():
            per[off][eid] = r['decision']
        print('\n' + '-' * 70)
        print('AGREEMENT between reviewers (Cohen\'s kappa)')
        print('-' * 70)
        for i, x in enumerate(reviewers):
            for y in reviewers[i + 1:]:
                k = cohens_kappa(per[x], per[y])
                shared = len(set(per[x]) & set(per[y]))
                if k is None:
                    print(f'  {x} vs {y}: too little overlap ({shared} shared)')
                else:
                    kappas.append(k)
                    print(f'  {x} vs {y}: kappa {k:+.2f} over {shared} shared package(s)')
        print('  Humans disagreeing with each other bounds how well ANY score')
        print('  can predict them -- that ceiling belongs in the write-up.')

    # -- time on screen -----------------------------------------------------
    secs = [float(r.get('seconds_on_screen') or 0) for r in decisions
            if r.get('seconds_on_screen')]
    if secs:
        secs_sorted = sorted(secs)
        print(f'\n  median time on screen: {secs_sorted[len(secs_sorted) // 2]:.1f}s '
              f'over {len(secs)} decision(s)')

    # -- write it out -------------------------------------------------------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        'n_decisions': len(decisions), 'reviewers': reviewers,
        'counts': dict(by_decision),
        'approved': {'n': len(approved), 'mean_eds': round(mean_a, 2)},
        'rejected': {'n': len(rejected), 'mean_eds': round(mean_r, 2)},
        'escalated': {'n': len(escalated)},
        'auc': round(a, 4),
        'auc_ci95': [round(ci[0], 4), round(ci[1], 4)] if ci else None,
        'kappas': [round(k, 3) for k in kappas],
        'bands': {b: dict(c) for b, c in bands.items()},
        'limits': [
            'Reviewers are students, not serving traffic officers.',
            'Packages come from a single camera and largely a single rule.',
            'Sample size is small; see the confidence interval.',
        ],
    }
    (out_dir / 'eds_validation.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(f'\nwrote {out_dir / "eds_validation.json"}')

    if not args.no_chart:
        try:
            chart(approved, rejected, a, ci, out_dir / 'eds_roc.png')
            print(f'wrote {out_dir / "eds_roc.png"}')
        except Exception as e:                       # charts are never the point
            print(f'(chart skipped: {e})')

    print('\n' + '=' * 70)
    print('LIMITS -- carry these into the report, do not drop them')
    for L in result['limits']:
        print('  * ' + L)
    print('=' * 70)


def chart(approved, rejected, a, ci, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    BG, FG, ACC, OK, NO = '#16404D', '#FBF5DD', '#78633F', '#3B6B50', '#A6402F'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(FG)
        ax.tick_params(colors=FG, labelsize=10)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)

    pts = roc_points(approved, rejected)
    ax1.plot([p[0] for p in pts], [p[1] for p in pts], color=ACC, lw=2.4,
             marker='o', ms=3)
    ax1.plot([0, 1], [0, 1], color=FG, ls='--', lw=1, alpha=.45)
    label = f'AUC {a:.3f}' + (f'\n95% CI [{ci[0]:.2f}, {ci[1]:.2f}]' if ci else '')
    ax1.text(.96, .08, label, ha='right', color=FG, fontsize=12,
             fontweight='bold', transform=ax1.transAxes)
    ax1.set_xlabel('false positive rate')
    ax1.set_ylabel('true positive rate')
    ax1.set_title('Does EDS rank approvals above rejections?',
                  color=FG, fontsize=12, pad=12, fontweight='bold')

    bins = [i * 10 for i in range(11)]
    ax2.hist([rejected, approved], bins=bins, stacked=False,
             color=[NO, OK], edgecolor=FG, linewidth=.6,
             label=[f'rejected (n={len(rejected)})', f'approved (n={len(approved)})'])
    ax2.set_xlabel('Evidence Defensibility Score')
    ax2.set_ylabel('packages')
    ax2.set_title('Score distribution by officer decision',
                  color=FG, fontsize=12, pad=12, fontweight='bold')
    ax2.legend(facecolor=BG, edgecolor=FG, labelcolor=FG, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=BG)
    plt.close(fig)


if __name__ == '__main__':
    main()
