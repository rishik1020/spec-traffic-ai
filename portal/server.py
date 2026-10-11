"""
portal/server.py
================
The officer review portal: the human half of the loop.

WHY THIS EXISTS
Every claim the project makes rests on a human approving the challan. Until now
that human was hypothetical -- the pipeline wrote evidence packages to a folder
and nobody ever looked at them. This is where someone looks, and where their
decision is recorded.

It does two jobs at once, which is why it was worth building before anything
else that remains:

  1. It is the demonstration. An evidence package as raw JSON persuades nobody;
     the same package laid out as a draft challan -- offence, section, amount,
     clip, reason -- is the whole thesis on one screen.
  2. It is the instrument for the EDS validation experiment. Hiding the score
     and recording decisions turns "we invented a score" into "we measured that
     the score predicts human judgement". One build, two deliverables.

WHY THE STANDARD LIBRARY AND NOT FLASK
Flask is not installed, the deadline is days away, and pandas in this
environment is already broken by a numpy ABI mismatch -- adding dependencies
before a demo is how demos die. `http.server` serves a handful of pages
perfectly well, and "runs with python alone" is a better property for a project
someone else has to reproduce.

BLINDING IS A START-UP FLAG, NOT A TOGGLE
`--blind` hides the Evidence Defensibility Score everywhere, with no way to
reveal it from the browser. A reviewer who sees the score anchors on it, and an
anchored reviewer measures nothing.

THE THREE DECISIONS
Approve, escalate, reject. Escalation is not a softer rejection: it sends the
package to a human who watches the footage and writes a citation -- the reason
the challan stands or falls, in their words. That citation is the audit record,
and under s.65B of the Evidence Act the human attestation is what makes the
electronic record admissible at all.

WHAT CANNOT BE APPROVED
Rules whose evidence is inadmissible by construction -- vision speed, lane
discipline -- carry `blocks_challan` in the penalty schedule. The portal
refuses to issue a challan for them no matter what is clicked. A system that
can be clicked into issuing an inadmissible fine does not have a human in the
loop; it has a human holding the bag.

USAGE
    python portal/server.py                      # normal review
    python portal/server.py --blind --shuffle    # EDS validation run
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from senti.core.penalties import Schedule                       # noqa: E402

TEMPLATES = ROOT / 'templates'
STATIC = ROOT / 'static'

MIME = {
    '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8', '.json': 'application/json',
    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
    '.mp4': 'video/mp4', '.webm': 'video/webm',
}

STATUSES = ('pending', 'escalated', 'approved', 'rejected')

REJECT_REASONS = [
    'Vehicle not identifiable',
    'Violation not clearly visible',
    'Wrong vehicle attributed',
    'Not actually a violation',
    'Evidence clip too short or unclear',
    'Signal state unclear',
    'Other',
]


# ---------------------------------------------------------------------------
# evidence store
# ---------------------------------------------------------------------------

class Store:
    """Evidence packages on disk, plus the decisions made about them.

    Decisions are written in TWO places on purpose. The package's own
    evidence.json is updated so a folder stays self-describing when copied
    anywhere, and an append-only decisions.jsonl keeps the full history --
    including a reviewer changing their mind, which the package alone would
    overwrite and lose. The experiment needs the history; the challan only
    needs the latest.
    """

    def __init__(self, evidence_root: Path, review_root: Path) -> None:
        self.root = evidence_root
        self.review_root = review_root
        self.log = review_root / 'decisions.jsonl'

    def packages(self) -> list[dict]:
        out = []
        if not self.root.is_dir():
            return out
        for folder in sorted(self.root.iterdir()):
            meta = folder / 'evidence.json'
            if not meta.is_file():
                continue
            try:
                rec = json.loads(meta.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                continue
            rec['_folder'] = folder.name
            rec.setdefault('evidence_id', folder.name)
            rec.setdefault('review', {})
            rec['review'].setdefault('status', 'pending')
            out.append(rec)
        return out

    def get(self, eid: str) -> dict | None:
        for p in self.packages():
            if p['evidence_id'] == eid:
                return p
        return None

    def decide(self, eid: str, decision: str, officer: str, plate: str = '',
               reason: str = '', citation: str = '', seconds: float = 0.0) -> bool:
        pkg = self.get(eid)
        if pkg is None:
            return False
        meta = self.root / pkg['_folder'] / 'evidence.json'
        rec = json.loads(meta.read_text(encoding='utf-8'))
        prev = (rec.get('review') or {}).get('status', 'pending')

        now = datetime.now().isoformat(timespec='seconds')
        review = dict(rec.get('review') or {})
        review.update(status=decision, officer=officer or 'anonymous',
                      decision_at=now, plate_number=plate or None,
                      rejection_reason=reason or None)
        if citation:
            # Keep every citation, not just the last. An escalation that is
            # later approved has two humans on the record, and both belong in
            # the chain of custody.
            review.setdefault('citations', []).append(
                {'by': officer or 'anonymous', 'at': now, 'text': citation})
        rec['review'] = review
        meta.write_text(json.dumps(rec, indent=2), encoding='utf-8')

        self.review_root.mkdir(parents=True, exist_ok=True)
        with self.log.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps({
                'evidence_id': eid, 'decision': decision, 'from_status': prev,
                'officer': officer or 'anonymous', 'at': now,
                'seconds_on_screen': round(seconds, 1),
                'plate_number': plate or None, 'rejection_reason': reason or None,
                'citation': citation or None,
                # Recorded here so the analysis never has to trust that the
                # score was unchanged between the decision and the measurement.
                'eds_score': (rec.get('defensibility') or {}).get('score'),
                'eds_verdict': (rec.get('defensibility') or {}).get('verdict'),
                'rule': (rec.get('violation') or {}).get('rule'),
            }) + '\n')
        return True


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render(name: str, **ctx) -> str:
    """Substitute {{key}} in a template file.

    Not a template engine and not trying to be. Keeping the HTML in real files
    means the look can be reworked without touching Python, which is the only
    property that matters here.
    """
    text = (TEMPLATES / name).read_text(encoding='utf-8')
    for k, v in ctx.items():
        text = text.replace('{{' + k + '}}', str(v))
    return re.sub(r'\{\{\w+\}\}', '', text)


def esc(s) -> str:
    return html.escape(str(s if s is not None else ''))


def eds_chip(pkg: dict, blind: bool) -> str:
    if blind:
        return '<span class="chip chip-blind">score hidden</span>'
    d = pkg.get('defensibility') or {}
    if d.get('score') is None:
        return ''
    v = d.get('verdict', 'review')
    return (f'<span class="chip chip-{esc(v)}">EDS {d["score"]:.0f}'
            f'<small>{esc(v)}</small></span>')


def status_chip(pkg: dict) -> str:
    st = (pkg.get('review') or {}).get('status', 'pending')
    return f'<span class="chip chip-{esc(st)}">{esc(st)}</span>'


def blind_banner(blind: bool) -> str:
    if not blind:
        return ''
    return ('<div class="banner">Blind review &mdash; the defensibility score is '
            'hidden, so your judgement measures the evidence rather than the score.</div>')


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------

def queue_view(store: Store, blind: bool, officer: str, shuffle: bool,
               tab: str) -> str:
    pkgs = store.packages()
    counts = {s: 0 for s in STATUSES}
    for p in pkgs:
        st = (p.get('review') or {}).get('status', 'pending')
        counts[st] = counts.get(st, 0) + 1

    shown = [p for p in pkgs
             if tab == 'all' or (p.get('review') or {}).get('status') == tab]

    if shuffle:
        # Seeded on the reviewer: order effects are real (people get stricter
        # as they tire), so each person gets a DIFFERENT order, but the same
        # person gets a stable one across reloads.
        random.Random(officer or 'anon').shuffle(shown)
    else:
        shown.sort(key=lambda p: -(p.get('defensibility') or {}).get('score', 0))

    rows = []
    for p in shown:
        v = p.get('violation', {})
        rows.append(f'''
          <a class="row" href="/review/{esc(p['evidence_id'])}">
            <div class="row-main">
              <div class="row-rule">{esc(str(v.get('rule', '?')).replace('_', ' '))}</div>
              <div class="row-reason">{esc(p.get('reason_trace', ''))[:160]}</div>
              <div class="row-meta">{esc(v.get('vehicle_class', '?'))}
                &middot; track #{esc(v.get('track_id', '?'))}
                &middot; {esc(p.get('camera_id', ''))}</div>
            </div>
            <div class="row-side">{status_chip(p)}{eds_chip(p, blind)}</div>
          </a>''')

    if not rows:
        rows = [f'''<div class="empty">
            <h3>Nothing {esc(tab)}</h3>
            <p>Generate evidence by running the pipeline over some footage:</p>
            <code>python run_senti.py --source data/videos/kolkata.mp4</code>
          </div>''']

    tabs = ''.join(
        f'<a class="tab{" on" if tab == t else ""}" href="/?tab={t}">'
        f'{t.title()}<b>{counts.get(t, 0) if t != "all" else len(pkgs)}</b></a>'
        for t in ('pending', 'escalated', 'approved', 'rejected', 'all'))

    done = counts['approved'] + counts['rejected']
    return render('queue.html', officer=esc(officer),
                  blind_banner=blind_banner(blind), tabs=tabs, rows=''.join(rows),
                  progress=int(100 * done / max(1, len(pkgs))),
                  done=done, total=len(pkgs))


def challan_view(store: Store, eid: str, blind: bool, officer: str,
                 schedule: Schedule) -> str | None:
    pkg = store.get(eid)
    if pkg is None:
        return None

    v = pkg.get('violation', {})
    ctx = pkg.get('context', {})
    art = pkg.get('artifacts', {})
    rev = pkg.get('review') or {}
    d = pkg.get('defensibility') or {}
    rule = v.get('rule', '')
    pen = schedule.for_rule(rule)

    # -- media --------------------------------------------------------------
    media = []
    if art.get('clip'):
        media.append(f'''<figure class="clip"><video controls autoplay muted loop
            src="/file/{esc(eid)}/{esc(art['clip'])}"></video>
            <figcaption>Evidence clip &mdash; approach, violation, departure</figcaption></figure>''')
    # The offending vehicle, boxed. Drawn as an overlay in PERCENTAGES rather
    # than burned into the jpg: the original frame stays untouched evidence,
    # and the box tracks the image at whatever width the browser renders it.
    #
    # Only on the violation frame. The approach frame is a DIFFERENT moment --
    # the vehicle is somewhere else in it -- so a box at the violation
    # coordinates would point at empty road and quietly mislead the officer.
    box_html = ''
    fw, fh = (pkg.get('frame_size') or [0, 0])[:2] or (0, 0)
    bbox = v.get('bbox_xyxy') or []
    if fw and fh and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        box_html = (
            f'<span class="vbox" style="left:{100*x1/fw:.2f}%;top:{100*y1/fh:.2f}%;'
            f'width:{100*(x2-x1)/fw:.2f}%;height:{100*(y2-y1)/fh:.2f}%">'
            f'<i>{esc(str(rule).replace("_", " "))} &middot; #{esc(v.get("track_id", ""))}</i>'
            f'</span>')

    stills = []
    for key, cap in (('violation_frame', 'Moment of violation'),
                     ('approach_frame', 'Approach')):
        if art.get(key):
            overlay = box_html if key == 'violation_frame' else ''
            stills.append(f'''<figure class="shot"><span class="shot-wrap">
                <img src="/file/{esc(eid)}/{esc(art[key])}" alt="{esc(cap)}">{overlay}</span>
                <figcaption>{esc(cap)}</figcaption></figure>''')
    if stills:
        media.append('<div class="stills">' + ''.join(stills) + '</div>')
    if not media:
        media.append('<p class="muted">No media written for this package.</p>')

    # -- the money ----------------------------------------------------------
    verified_mark = ('<span class="verified">verified</span>' if pen.verified
                     else '<span class="unverified">unverified &mdash; check the '
                          'Telangana notification before issuing</span>')
    consequences = ''.join(f'<li>{esc(c)}</li>' for c in pen.consequences)
    if consequences:
        consequences = f'<ul class="consequences">{consequences}</ul>'

    blocked = ''
    if pen.blocks_challan:
        blocked = ('<div class="blocked"><b>Cannot be issued as a challan.</b> '
                   + esc(pen.notes.split('.')[0]) + '. Approving records the '
                   'officer\'s agreement for analysis only.</div>')

    # -- EDS ----------------------------------------------------------------
    if blind:
        eds_block = ''
    else:
        dims = d.get('dimensions_used') or ['plate', 'visibility',
                                            'track_integrity', 'rule_margin',
                                            'context']
        bars = []
        for dim in dims:
            val = float(d.get(dim, 0.0))
            weak = ' weak' if dim == d.get('weakest') else ''
            bars.append(f'''<div class="bar-row{weak}">
                <span class="bar-label">{esc(dim.replace('_', ' '))}</span>
                <span class="bar"><i style="width:{val*100:.0f}%"></i></span>
                <span class="bar-val">{val:.2f}</span></div>''')
        omitted = ''
        if not d.get('plate_available', False):
            omitted = ('<p class="muted">Scored <b>without</b> the plate '
                       'dimension &mdash; plate recognition is not implemented, '
                       'so its weight is redistributed across the rest.</p>')
        eds_block = f'''<section class="panel">
            <h2>Evidence defensibility</h2>
            <div class="eds-head"><div class="eds-score">{d.get('score', 0):.0f}<small>/100</small></div>
              {eds_chip(pkg, False)}</div>
            {''.join(bars)}
            <p class="muted">Weakest: <b>{esc(str(d.get('weakest', '')).replace('_', ' '))}</b>
               &mdash; what would be attacked first if contested.</p>
            {omitted}
          </section>'''

    facts = [
        ('Vehicle', v.get('vehicle_class', '')),
        ('Track', f"#{v.get('track_id', '')}"),
        ('Camera', pkg.get('camera_id', '')),
        ('Recorded', (pkg.get('timing') or {}).get('recorded_at_ist', '')),
        ('Signal', f"{ctx.get('signal_state', '')} "
                   f"({float(ctx.get('signal_confidence', 0) or 0):.2f} px fraction)"
                   if ctx.get('signal_state') not in (None, '', 'unknown') else ''),
        ('Detector confidence', f"{float(v.get('detection_confidence', 0) or 0):.2f}"),
        ('Frames tracked', ctx.get('track_frames', '')),
    ]
    fact_rows = ''.join(f'<div class="fact"><dt>{esc(k)}</dt><dd>{esc(val)}</dd></div>'
                        for k, val in facts if str(val).strip())

    # -- prior decisions ----------------------------------------------------
    history = ''
    cites = rev.get('citations') or []
    if cites:
        items = ''.join(
            f'<li><b>{esc(c.get("by"))}</b> <span class="muted">{esc(c.get("at"))}</span>'
            f'<p>{esc(c.get("text"))}</p></li>' for c in cites)
        history = f'<section class="panel"><h2>Citations</h2><ul class="cites">{items}</ul></section>'

    st = rev.get('status', 'pending')
    decided = ''
    if st in ('approved', 'rejected', 'escalated'):
        decided = (f'<div class="decided">Currently <b>{esc(st)}</b>'
                   + (f' by {esc(rev.get("officer"))}' if rev.get('officer') else '')
                   + (f' at {esc(rev.get("decision_at"))}' if rev.get('decision_at') else '')
                   + '. Deciding again replaces it.</div>')

    # An escalated package is being looked at by the second human, so the
    # citation field is required rather than optional.
    cite_required = ' required' if st == 'escalated' else ''
    cite_hint = ('Write why this stands or falls. Required at this stage.'
                 if st == 'escalated'
                 else 'Optional note recorded with your decision.')

    return render(
        'challan.html',
        eid=esc(eid), officer=esc(officer), blind_banner=blind_banner(blind),
        rule=esc(str(rule).replace('_', ' ')),
        offence=esc(pen.offence or str(rule).replace('_', ' ').title()),
        section=esc(pen.section or v.get('mv_act_section', '')),
        amount=esc(pen.display_amount), verified_mark=verified_mark,
        consequences=consequences, blocked=blocked,
        reason_trace=esc(pkg.get('reason_trace', '')),
        media=''.join(media), eds_block=eds_block, facts=fact_rows,
        detail=esc(json.dumps(pkg.get('detail', {}), indent=2)),
        reasons=''.join(f'<option value="{esc(r)}">{esc(r)}</option>'
                        for r in REJECT_REASONS),
        decided=decided, history=history,
        cite_required=cite_required, cite_hint=esc(cite_hint),
        plate_value=esc(rev.get('plate_number') or ''),
    )


def monitor_view(store: Store, monitor_root: Path, officer: str,
                 which: str | None) -> str:
    """Playback of an annotated run, with its violations as a timeline.

    PLAYBACK, NOT LIVE, AND DELIBERATELY SO
    Streaming a live decode into a browser is the expensive, fragile part, and
    a demo that depends on a network camera staying up is a demo that fails in
    front of a panel. The pipeline writes the annotated video it actually
    produced plus the timeline of what it found, and this replays them
    together. Nothing here is staged: the boxes are the ones the engine drew
    and the events are the ones the rules fired. Live ingestion already runs
    through the same code path when it is wanted.
    """
    sets = sorted(monitor_root.glob('*.events.json')) if monitor_root.is_dir() else []
    if not sets:
        empty = ('<div class="empty"><h3>No annotated run yet</h3>'
                 '<p>Produce one with:</p><code>python run_senti.py --source '
                 'data/videos/kolkata.mp4 --annotate data/monitor/kolkata.mp4'
                 '</code></div>')
        return render('monitor.html', officer=esc(officer), options='',
                      video='', events=empty, stats='', camera='')

    def stem_of(p):
        return p.name[:-len('.events.json')]

    chosen = next((p for p in sets if stem_of(p) == which), sets[0])
    data = json.loads(chosen.read_text(encoding='utf-8'))
    events = data.get('events', [])

    # Which packages exist, so an event links to its challan when one was written
    known = {p['evidence_id']: p for p in store.packages()}

    rows = []
    for e in events:
        eid = e.get('evidence_id', '')
        pkg = known.get(eid)
        chip = status_chip(pkg) if pkg else '<span class="chip">no package</span>'
        href = '/review/' + urllib.parse.quote(eid) if pkg else '#'
        rows.append(
            '<li class="ev" data-t="{t}">'
            '<button class="ev-seek" type="button">'
            '<span class="ev-t">{t:.1f}s</span>'
            '<span class="ev-body"><span class="ev-rule">{rule}</span>'
            '<span class="ev-meta">{veh} &middot; track #{tid}</span></span>'
            '</button>'
            '<a class="ev-open" href="{href}">{chip}</a></li>'.format(
                t=float(e.get('seconds', 0)),
                rule=esc(str(e.get('rule', '')).replace('_', ' ')),
                veh=esc(e.get('vehicle_class', '')),
                tid=esc(e.get('track_id', '')),
                href=href, chip=chip))

    options = ''.join(
        '<option value="{v}"{sel}>{v}</option>'.format(
            v=esc(stem_of(p)), sel=' selected' if p == chosen else '')
        for p in sets)

    by_rule = {}
    for e in events:
        by_rule[e.get('rule', '?')] = by_rule.get(e.get('rule', '?'), 0) + 1
    stats = '<div class="stat"><b>{n}</b><span>violations</span></div>'.format(
        n=len(events))
    stats += ''.join(
        '<div class="stat"><b>{n}</b><span>{r}</span></div>'.format(
            n=n, r=esc(r.replace('_', ' ')))
        for r, n in sorted(by_rule.items(), key=lambda kv: -kv[1]))

    return render('monitor.html', officer=esc(officer), options=options,
                  camera=esc(data.get('camera_id', '')),
                  video='/monitor-file/' + esc(data.get('video', '')),
                  events='<ul class="events">' + ''.join(rows) + '</ul>',
                  stats=stats)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    store: Store
    schedule: Schedule
    monitor_root: Path
    blind: bool = False
    shuffle: bool = False

    def log_message(self, fmt, *args):
        pass

    def _officer(self) -> str:
        m = re.search(r'officer=([^;]+)', self.headers.get('Cookie', ''))
        return urllib.parse.unquote(m.group(1)) if m else ''

    def _send(self, body: bytes, ctype: str, code: int = 200, extra=None):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text: str, code: int = 200, extra=None):
        self._send(text.encode('utf-8'), 'text/html; charset=utf-8', code, extra)

    def _redirect(self, where: str, extra=None):
        self.send_response(303)
        self.send_header('Location', where)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path, qs = url.path, urllib.parse.parse_qs(url.query)

        if path.startswith('/static/'):
            return self._static(STATIC / path[len('/static/'):])

        if path.startswith('/monitor-file/'):
            name = urllib.parse.unquote(path[len('/monitor-file/'):])
            return self._static(self.monitor_root / name)

        if path.startswith('/file/'):
            parts = path[len('/file/'):].split('/', 1)
            if len(parts) != 2:
                return self._html('<h1>404</h1>', 404)
            pkg = self.store.get(urllib.parse.unquote(parts[0]))
            if pkg is None:
                return self._html('<h1>404</h1>', 404)
            return self._static(self.store.root / pkg['_folder'] /
                                urllib.parse.unquote(parts[1]))

        officer = self._officer()
        if not officer and path != '/who':
            return self._redirect('/who')

        if path == '/who':
            return self._html(render('who.html'))
        if path == '/':
            tab = (qs.get('tab') or ['pending'])[0]
            if tab not in STATUSES + ('all',):
                tab = 'pending'
            return self._html(queue_view(self.store, self.blind, officer,
                                         self.shuffle, tab))
        if path == '/monitor':
            return self._html(monitor_view(self.store, self.monitor_root,
                                           officer, (qs.get('set') or [None])[0]))
        if path.startswith('/review/'):
            eid = urllib.parse.unquote(path[len('/review/'):])
            page = challan_view(self.store, eid, self.blind, officer, self.schedule)
            if page is None:
                return self._html('<h1>404 &mdash; no such package</h1>', 404)
            return self._html(page)
        return self._html('<h1>404</h1>', 404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0) or 0)
        form = urllib.parse.parse_qs(self.rfile.read(length).decode('utf-8'))
        path = urllib.parse.urlparse(self.path).path

        def field(k, default=''):
            return (form.get(k) or [default])[0].strip()

        if path == '/who':
            name = field('officer') or 'anonymous'
            return self._redirect('/', {
                'Set-Cookie': f'officer={urllib.parse.quote(name)}; Path=/; Max-Age=2592000'})

        if path == '/decide':
            decision = field('decision')
            if decision not in ('approved', 'escalated', 'rejected'):
                return self._html('<h1>400 &mdash; bad decision</h1>', 400)
            eid = field('evidence_id')
            self.store.decide(eid, decision, self._officer(),
                              plate=field('plate_number'),
                              reason=field('rejection_reason'),
                              citation=field('citation'),
                              seconds=float(field('seconds', '0') or 0))
            nxt = self._next_pending(eid)
            return self._redirect(f'/review/{urllib.parse.quote(nxt)}' if nxt else '/')

        return self._html('<h1>404</h1>', 404)

    def _next_pending(self, after: str) -> str | None:
        """Hand the reviewer the next undecided package.

        A queue that bounces back to a list after every decision makes a
        200-package experiment feel like 200 decisions. This makes it feel like
        one task.
        """
        pkgs = self.store.packages()
        if self.shuffle:
            random.Random(self._officer() or 'anon').shuffle(pkgs)
        pending = [p['evidence_id'] for p in pkgs
                   if (p.get('review') or {}).get('status') == 'pending'
                   and p['evidence_id'] != after]
        return pending[0] if pending else None

    def _static(self, p: Path):
        try:
            p = p.resolve()
            if not (str(p).startswith(str(STATIC.resolve()))
                    or str(p).startswith(str(self.store.root.resolve()))
                    or str(p).startswith(str(self.monitor_root.resolve()))):
                return self._html('<h1>403</h1>', 403)
            body = p.read_bytes()
        except (OSError, ValueError):
            return self._html('<h1>404</h1>', 404)
        return self._send(body, MIME.get(p.suffix.lower(), 'application/octet-stream'))


def main() -> None:
    ap = argparse.ArgumentParser(description='SPEC Traffic AI officer review portal')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--evidence', default=str(PROJECT / 'data' / 'evidence'))
    ap.add_argument('--review-dir', default=str(PROJECT / 'data' / 'review'))
    ap.add_argument('--penalties', default=str(PROJECT / 'config' / 'penalties.yaml'))
    ap.add_argument('--monitor-dir', default=str(PROJECT / 'data' / 'monitor'))
    ap.add_argument('--blind', action='store_true',
                    help='hide the defensibility score (EDS validation runs)')
    ap.add_argument('--shuffle', action='store_true',
                    help='randomise order per reviewer, to cancel order effects')
    args = ap.parse_args()

    Handler.store = Store(Path(args.evidence), Path(args.review_dir))
    Handler.schedule = Schedule.load(args.penalties)
    Handler.monitor_root = Path(args.monitor_dir)
    Handler.blind = args.blind
    Handler.shuffle = args.shuffle

    n = len(Handler.store.packages())
    print(f'  evidence : {args.evidence}  ({n} package(s))')
    print(f'  penalties: {len(Handler.schedule.penalties)} row(s), '
          f'schedule_verified={Handler.schedule.schedule_verified}')
    if args.blind:
        print('  MODE     : BLIND -- the score is hidden and cannot be revealed')
    if args.shuffle:
        print('  ORDER    : shuffled per reviewer')
    print(f'\n  open http://localhost:{args.port}\n')

    try:
        ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')
        sys.exit(0)


if __name__ == '__main__':
    main()
