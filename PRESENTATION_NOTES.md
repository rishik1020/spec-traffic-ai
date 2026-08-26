# SPEC Traffic AI — Review-II Talk Track

*Thursday 27 August 2026 · panel of two, one is your mentor*

---

## THE THREE NUMBERS — memorise these

| | |
|---|---|
| **0.703** | test mAP@50, all 21 classes, held-out split |
| **0.888** | auto-rickshaw — the class COCO does not have |
| **78.7%** | DriveIndia's published baseline (context, not your score) |

If you remember nothing else, remember those. Everything else you can read off a slide.

---

## SLIDE-BY-SLIDE (~45 sec each, ~11 min total)

### 1 · Title
Name the project and the one-line claim.
> *"SPEC Traffic AI — traffic violation detection built for Indian road conditions, where every detection is checked for whether it could actually be enforced."*

### 2 · Recap
> *"Evaluation 1 was a proposal. Since then we've trained the detector and implemented the full pipeline end to end — it runs on real footage today."*

Keep it to two sentences. Don't re-present Eval-1.

### 3 · Objectives
Three points. Land the first one hard:
> *"COCO-trained detectors have no auto-rickshaw class at all. They are structurally incapable of enforcing against a large share of Indian traffic. That's why we needed Indian training data."*

### 4 · What We Built
Don't read the paragraph. Walk the pipeline out loud:
> *"Video comes in from a file or a live camera through one interface. The detector finds 21 Indian classes. ByteTrack gives each vehicle an identity. The rule engine judges violations. A rolling buffer cuts the evidence clip. Everything is on GitHub."*

### 5 · How It Works
Point at the four stages. One line each. Then the design claim:
> *"One shared perception pass, many independent rules. Adding a violation is one file; deploying to a new camera is one config file — not a new codebase."*

### 6 · Results ⭐
**Slow down here.** This is 25% of your marks.
> *"0.703 across all 21 classes on the held-out test split — data the model never saw during training or model selection. Three classes had under 100 examples and couldn't be learned; over the 18 that were adequately supported it's 0.800. Over the seven classes we actually enforce on, 0.868."*

Then: *"DriveIndia's own published baseline is 78.7% on the full 67,000-image dataset. We used 12,000 images on a 6 GB laptop GPU."*

### 7 · Generalisation Gap ⭐
> *"Same weights, three splits. Validation and test agree almost exactly, which tells us the split is representative. Most published work in this area reports training accuracy — the blue bar. We report the green one."*

That sentence is your Methodology marks. Say it clearly.

### 8 · Per-Class Accuracy
Two things:
> *"Auto-rickshaw is at 0.888 — above our overall mean. That's the whole justification for using an Indian dataset."*

> *"And the interesting one: a speed bump learned from 124 examples scores higher than a commercial vehicle with 4,280. Visual distinctiveness matters more than sample count."*

### 9 · Challenges
Deliver this as a **finding**, not an apology:
> *"Testing on real footage exposed a viewpoint gap. DriveIndia is dashcam footage from autonomous-driving research. Enforcement cameras look down from above. From above, a car roof and a bus roof are both large rectangles — so the model confuses them. Our number is valid in the dashcam domain and doesn't transfer directly to overhead CCTV. That's not in any of the papers we reviewed."*

### 10 · Domain Gap Image
Point at the misclassified vehicles.
> *"This is that failure, visible. Cars, motorcycles, auto-rickshaws and pedestrians correct — large vehicles from above, wrong."*

### 11 · Novelty ⭐⭐
**The most important slide. Lead with Kerala.**
> *"In 2023 Kerala deployed 726 AI cameras. In the opening days they detected over one lakh violations — and issued about three thousand challans. Detection was never the bottleneck. Enforceability was."*

Pause. Then:
> *"Detection confidence tells you how sure the model is that something is a motorcycle. It tells you nothing about whether the resulting challan would survive being contested. Those come apart constantly. Our Evidence Defensibility Score rates every detection on five dimensions before an officer sees it — and names the weak one. Above 80 it's pre-filled, 50 to 80 goes to review, below 50 it's dropped rather than risk fining someone wrongly."*

### 12 · Future Actions
Three points, briskly. Emphasise the EDS validation experiment — it's a real result you can produce.

### 13 · Conclusion
Close on the one-liner:
> *"Existing systems detect violations. SPEC Traffic AI predicts whether the violation it detected can actually be enforced."*

### 14 · Thank You

---

## QUESTIONS YOU WILL GET

**"Why is your accuracy below the 78.7% baseline?"**
> *"Three reasons: we trained on 12,000 of the 67,000 images, on a 6 GB laptop GPU, and our mean includes three classes with under 100 instances that can't be learned. Over adequately-supported classes we're at 0.800, and over the enforcement-relevant classes 0.868."*

**"Everyone does YOLO violation detection. What's new here?"**
> *"The detection isn't new and we don't claim it is — DriveIndia publishes its own baseline. What's new is scoring whether the evidence would hold up. Kerala's system detected a lakh of violations and issued three thousand challans. Nobody in this literature models that gap."*

**"Have you tested on real CCTV?"**
> *"Not on municipal CCTV — that needs a traffic-authority partnership. We tested on real traffic footage and on a live network camera stream using the identical code path. That testing is what revealed the viewpoint domain gap, which is our main technical finding."*

**"Can it run in real time?"**
> *"Yes on the ingestion side — we measured a live stream at 43 fps. Detection on our laptop GPU runs around 12 fps at full resolution; a deployment box or a smaller model closes that. Real systems use one edge box per junction."*

**"What about privacy / wrongful fines?"**
> *"No challan is issued without a human approving it. Every detection carries a plain-language reason trace, so an officer sees why — not just a confidence score. And signal state is read by colour thresholding rather than a neural network specifically so it can be explained when contested."*

**"Why not just buy a commercial system?"**
> *"Kerala did. It detected a lakh of violations and issued three thousand challans. The gap wasn't detection."*

**"How would you validate the EDS?"**
> *"Give around 200 evidence packages to a reviewer blind to the score, and measure whether EDS predicts their approve/reject decisions — report the AUC. If it does, we've shown a machine can anticipate why enforcement evidence fails."*

**"Which violations actually work today?"**
> *"Wrong-way is implemented end to end and produces evidence packages. Triple riding, red-light and stop-line are next — the perception they need already works: pedestrian at 0.874, motorcycle at 0.950, traffic light at 0.764. Helmet is the only one needing an additional model."*

---

## DO NOT SAY

- ❌ "It's 91% accurate" — that was the old 7-class model, not comparable
- ❌ "It detects speeding" — vision speed is a **screening signal**; legal speed needs radar
- ❌ "It controls traffic signals" — you cannot actuate safety-critical hardware
- ❌ "ANPR is 98% accurate" — real-world is 70–85%
- ❌ "It works on CCTV" — you found the domain gap; say so instead

**Every one of these is a trap you've already avoided in the deck. Don't walk into it verbally.**

---

## IF ASKED FOR A DEMO

```bash
cd "C:\Users\Rishik Reddy\Desktop\senti-traffic"
python run_senti.py --source data/videos/kolkata.mp4 --show
```

Then open an evidence folder and show `evidence.json` — the reason trace and the EDS breakdown. **That JSON is more impressive than the video**, because it's the thing nobody else produces.

`python run_senti.py --list-rules` shows the rule interface if they ask about extensibility.

---

## PRE-FLIGHT — Wednesday night

- [ ] Two-page document filled, printed, **signed**
- [ ] Deck printed (hard copy required)
- [ ] Deck on a USB stick **and** emailed to yourself
- [ ] Laptop charged + charger packed
- [ ] Test the demo once, cold, on battery
- [ ] Delete the stray `SPEC Traffic AI - Evaluation 2.pptx` so you don't grab the wrong file

---

## FINAL NOTE

You have a trained model with honest metrics, a working end-to-end pipeline, a public repository, and a genuine technical finding that isn't in the literature you reviewed. That is **well beyond** typical Review-II scope.

The domain gap is a strength, not a weakness. You found it by testing properly. Most projects never look.
