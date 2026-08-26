# SPEC Traffic AI — Review-II Talk Track

*Thursday 27 August 2026 · panel of two, one is your mentor*
*Deck: `docs/SPEC Traffic AI - Evaluation 2 FINAL.pptx` — **20 slides***

---

## THE THREE NUMBERS — memorise these

| | |
|---|---|
| **0.703** | test mAP@50, all 21 classes, held-out split |
| **0.888** | auto-rickshaw — the class COCO does not have |
| **78.7%** | DriveIndia's published baseline (context, not your score) |

If you remember nothing else, remember those. Everything else you can read off a slide.

---

## SLIDE-BY-SLIDE (~35–40 sec each, ~13 min total)

### 1 · Title
> *"SPEC Traffic AI — traffic violation detection built for Indian road conditions, where every detection is checked for whether it could actually be enforced."*

### 2 · Recap
> *"Evaluation 1 was a proposal. Since then we've trained the detector and implemented the full pipeline end to end — it runs on real footage today."*

Two sentences. Don't re-present Eval-1.

### 3 · Literature Review ⭐ *(the comparison table)*
Don't read the table. Walk the **last column** — that is the whole argument.
> *"Five representative works. Detection on Indian roads is close to solved — DriveIndia publishes 78.7%. End-to-end enforcement pipelines already exist. And automated enforcement genuinely works: the Cochrane review of thirty-five studies puts crash reductions between eight and forty-nine percent. But look at the right-hand column. Every one of them stops at 'did we detect it'. Not one asks whether the resulting challan would survive being contested."*

Footer line if they want the India case: *"Overspeeding is in 68.4% of Indian accidents, and 54,568 people died without a helmet in 2023."*

### 4 · Where It Breaks Down ⭐
This is the failure evidence. Deliver it as six facts, no commentary.
> *"Bengaluru's AI helmet detection is 75 to 80 percent raw — it reaches 99.9 only because every fine is revalidated by hand. Kerala issued four hundred and twenty-eight crore in challans and collected seventy-six. Ren reports ninety-eight percent accuracy with no held-out split and no field deployment at all. BMD-45 shows a detector at 83.8 percent in-domain collapsing to 33.6 on real Indian CCTV. ANPR misreads 28.5 percent of plates in field conditions. And without a Section 65B certificate, none of the footage is admissible in the first place."*

### 5 · Research Gap
Three numbered points. The third is the one that matters — admissibility never enters the CV pipeline.

### 6 · Objectives
Land the first one hard:
> *"COCO-trained detectors have no auto-rickshaw class at all. They are structurally incapable of enforcing against a large share of Indian traffic."*

### 7 · What We Built
Don't read it. Walk the pipeline out loud, and end on the last bullet:
> *"…and the same detection pass now also drives signal timing. I'll come back to that."*

### 8 · How It Works
Four stages, one line each. Then the design claim:
> *"One shared perception pass, many independent consumers. Adding a violation is one file; deploying to a new camera is one config file."*

### 9 · Results ⭐
**Slow down. This is 25% of your marks.**
> *"0.703 across all 21 classes on the held-out test split — data the model never saw during training or model selection. Three classes had under 100 examples and couldn't be learned; over the 18 adequately supported it's 0.800. Over the seven classes we actually enforce on, 0.868."*

Then: *"DriveIndia's own baseline is 78.7% on the full 67,000-image dataset. We used 12,000 images on a 6 GB laptop GPU."*

### 10 · Generalisation Gap ⭐
> *"Same weights, three splits. Validation and test agree almost exactly, which tells us the split is representative. Most published work in this area reports training accuracy — the blue bar. We report the green one."*

That sentence is your Methodology marks. Say it clearly.

### 11 · Per-Class Accuracy
> *"Auto-rickshaw is at 0.888 — above our overall mean. That's the whole justification for using an Indian dataset."*

> *"And the interesting one: a speed bump learned from 124 examples beats a commercial vehicle with 4,280. Visual distinctiveness matters more than sample count."*

### 12 · Challenges
Deliver as a **finding**, not an apology:
> *"Testing on real footage exposed a viewpoint gap. DriveIndia is dashcam footage. Enforcement cameras look down from above, and from above a car roof and a bus roof are both large rectangles. BMD-45 measures the same collapse independently — 83.8 percent in-domain down to 33.6 across domains. Our number is valid in the dashcam domain and does not transfer directly to overhead CCTV."*

### 13 · Domain Gap Image
> *"This is that failure, visible. Cars, motorcycles, auto-rickshaws and pedestrians correct — large vehicles from above, wrong."*

### 14 · Novelty ⭐⭐ — Evidence Defensibility Score
**The most important slide.**
> *"Detection confidence tells you how sure the model is that something is a motorcycle. It tells you nothing about whether the resulting challan would survive being contested. Those come apart constantly. Our Evidence Defensibility Score rates every detection on five dimensions before an officer sees it — and names the weak one. Above 80 it's pre-filled, 50 to 80 goes to review, below 50 it's dropped rather than risk fining someone wrongly."*

### 15 · Adaptive Signal Timing ⭐⭐ *(new)*
Open by naming the shift:
> *"A challan punishes a driver after the fact. The same camera can stop the queue forming in the first place — and it costs us nothing, because the detection pass has already run."*

Then the two decisions worth defending:
> *"We measure congestion in Passenger Car Units, not vehicle counts. Forty motorcycles and forty buses are not equal demand — the buses need roughly six times the green. Western adaptive systems count vehicles because their traffic is homogeneous enough to get away with it."*

> *"And the split uses Webster's method, not a neural network. Four lines of arithmetic that a traffic engineer can check by hand. Same argument as the defensibility score — a decision that will be questioned has to be explainable."*

Close on the honesty:
> *"Past a flow ratio of about 0.9 Webster's cycle runs to infinity — that is the formula telling you the junction is over capacity and no timing fixes it. Our controller stops optimising, drains the longest queue instead, and records which mode it used. And it recommends only. It cannot actuate a signal head, deliberately: a bad plan isn't a wrong challan, it's a collision."*

### 16 · Green Follows the Queue *(new — the chart)*
Point at the two panels.
> *"Left is what the camera measured, right is what the controller recommended. North-south holds 88 percent of the standing queue and its green moves from 37 seconds to 57 as the queue builds. East-west never drops below its 12-second minimum — that minimum is a pedestrian's crossing time, so it is not negotiable. And the change is rate-limited to ten seconds a cycle, because drivers learn a junction's rhythm and a plan that lurches causes the late-amber running we're trying to catch."*

### 17 · Future Actions
Three points, briskly. Emphasise the EDS validation experiment — it is a real result you can produce.

### 18 · Conclusion
> *"Existing systems detect violations. SPEC Traffic AI predicts whether the violation it detected can actually be enforced — and uses the same perception pass to recommend how the junction should be timed."*

### 19 · References
Don't read it. It exists so the panel can see the numbers are sourced.

### 20 · Thank You

---

## QUESTIONS YOU WILL GET

**"Why is your accuracy below the 78.7% baseline?"**
> *"Three reasons: we trained on 12,000 of the 67,000 images, on a 6 GB laptop GPU, and our mean includes three classes with under 100 instances that can't be learned. Over adequately-supported classes we're at 0.800, and over the enforcement-relevant classes 0.868."*

**"Everyone does YOLO violation detection. What's new here?"**
> *"The detection isn't new and we don't claim it is. What's new is scoring whether the evidence would hold up, and using the same perception pass for signal control instead of only punishment."*

**"Is the viewpoint domain gap really absent from the literature?"**
> *"Not entirely, and that strengthens it. BMD-45 measures cross-domain collapse independently — 83.8% in-domain to 33.6% on real Indian CCTV. We found the same effect on our own data before seeing that paper. What is still absent is any test of viewpoint transfer in the violation-detection literature specifically."*

**"Why cite Ren if you're criticising it?"**
> *"Because it's the closest prior system and it's honest about its design. Our criticism is narrow: it reports accuracy with no held-out split and no field deployment. That's exactly the practice we corrected in our own methodology."*

**"Have you tested on real CCTV?"**
> *"Not on municipal CCTV — that needs a traffic-authority partnership. We tested on real traffic footage and a live network camera stream through the identical code path. That testing is what revealed the domain gap."*

**"Can it run in real time?"**
> *"Yes on ingestion — we measured a live stream at 43 fps. Detection on our laptop GPU runs around 12 fps at full resolution; a deployment box or a smaller model closes that. Real systems use one edge box per junction."*

**"Has the signal controller been tested on a real junction?"**
> *"No — we don't have footage of a signalised junction with two arms and the signal head in frame yet, and that's the honest blocker. The controller itself is verified against scripted demand across four scenarios, including one where an arm is completely empty and still gets served every cycle. The measurement half runs end to end on real footage today."*

**"How would you validate the EDS?"**
> *"Give around 200 evidence packages to a reviewer blind to the score, and measure whether EDS predicts their approve/reject decisions — report the AUC."*

**"Which violations actually work today?"**
> *"Wrong-way end to end with evidence packages; over-speeding and lane discipline as screening signals. Triple riding, red-light and stop-line are next — the perception they need already works. Helmet is the only one needing an additional model."*

---

## DO NOT SAY

- ❌ "It's 91% accurate" — that was the old 7-class model, not comparable
- ❌ "It detects speeding" — vision speed is a **screening signal**; legal speed needs radar
- ❌ "It controls traffic signals" — say **recommends**. It cannot actuate, by design
- ❌ "ANPR is 98% accurate" — real-world is 70–85%
- ❌ "It works on CCTV" — you found the domain gap; say so instead
- ❌ Kerala "issued only 3,000 challans" — **wrong**, see the correction below

---

## ⚠️ CORRECTIONS APPLIED TO THE DECK (26 Aug)

Four figures in earlier drafts did not survive checking. All are fixed in the
current deck; know why, in case the panel saw an older version.

**Kerala.** An earlier draft said *"over one lakh violations, about three
thousand challans."* That was a first-week news report about startup glitches.
Verified one-year funnel:

```
66.41 lakh violations detected
64.72 lakh challans ISSUED          <- issuance works fine
Rs 428.4 crore billed
Rs  76.7 crore collected  (18%)     <- the real failure is COLLECTION
```

Issuance is not Kerala's bottleneck; **collection** is — and collection is
administrative, not something EDS fixes. Be honest about that, then pivot:

> **Bengaluru is the anchor that actually fits.** Its ITMS detects helmet
> violations at 75–80% raw and reaches 99.9% only by **manually revalidating
> every fine before issue.** That manual revalidation is exactly the burden the
> Evidence Defensibility Score triages.

**Ren (2024).** An earlier draft attributed *"98.09% on benchmark, 83% caught in
deployment"* to it. Neither number is in the paper. What Ren actually reports:
**>96.9% across eight violation types, ~98% average, 99.7% on red-light** — and
critically, **no held-out split and no field deployment**. The corrected slide
says that.

**BMD-45.** An earlier draft said *"dashcam-trained model scores 0.46 mAP on real
CCTV."* The paper's actual finding is stronger and differently framed: a
**UA-DETRAC-trained** detector scores **33.6% mAP@0.50:0.95** on Indian CCTV
versus **83.8%** trained in-domain — a 2.5× collapse. UA-DETRAC is itself
surveillance footage, not dashcam, so transfer fails even between two CCTV
datasets — and your dashcam→elevated finding remains your own.

**Cochrane.** 35 studies, not 28. The 8–49% crash-reduction range is correct.

---

## ⚠️ ONE UNVERIFIED NUMBER

The **28.5% ANPR misread** rate [5] is widely attributed to Rhead et al. (2012),
but the primary is paywalled and we could not read it directly. It is a **UK,
2012** figure. If pressed, say exactly that and frame it as a floor:

> *"That's UK field conditions in 2012. Indian plates are less standardised and
> more often damaged, so we treat it as an optimistic floor, not a target."*

---

## IF ASKED FOR A DEMO

```bash
python run_senti.py --source data/videos/kolkata.mp4 --show
```

Then open an evidence folder and show `evidence.json` — the reason trace and the
EDS breakdown. **That JSON is more impressive than the video**, because it is the
thing nobody else produces.

For the signal controller, no footage needed:

```bash
python scripts/simulate_signal.py --scenario imbalanced
```

`python run_senti.py --list-rules` shows the rule interface if they ask about
extensibility.

---

## PRE-FLIGHT — Wednesday night

- [ ] Two-page document filled, printed, **signed**
- [ ] **Check the abstract word count** — it is now 242 words. If the form caps
      it lower, cut the sentence starting *"Confirmed violations trigger…"*
- [ ] **Check the header** — the form still reads *"REVIEW – I"* and
      *"30th and 31st July 2026"*. Change it if your department expects Review-II
- [ ] Deck printed (hard copy required) — **20 slides now**
- [ ] Page through slides 3, 16 and 19 on a real screen: the table, the chart and
      the two-column references are new layouts and want a human eye
- [ ] Deck on a USB stick **and** emailed to yourself
- [ ] Laptop charged + charger packed
- [ ] Test the demo once, cold, on battery
- [ ] Delete the stray `SPEC Traffic AI - Evaluation 2.pptx` and
      `… FINAL 1.pptx` so you don't grab the wrong file

---

## FINAL NOTE

You have a trained model with honest metrics, a working end-to-end pipeline, a
public repository, a genuine technical finding, and a second output — signal
timing — built on the same perception pass at zero extra inference cost. That is
well beyond typical Review-II scope.

The domain gap is a strength, not a weakness. You found it by testing properly.
Most projects never look.
