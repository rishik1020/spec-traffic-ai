# SPEC Traffic AI: An Explainable Computer Vision Platform for Indian Traffic Violation Detection with Human-in-the-Loop E-Challan Verification

**Capstone Project-I · School of Technology · Woxsen University**

Rishik Reddy P (23WU0102149) · Rohan Reddy (23WU0102169) · Hari Ramaneti (23WU0102135)
Mentor: Prof. Meher Gayatri

---

## Abstract

India records the highest number of road-accident fatalities worldwide, yet manual
enforcement cannot scale to urban vehicle density and existing closed-circuit
television infrastructure remains confined to passive monitoring. Detectors trained
on general-purpose Western datasets contain no auto-rickshaw class, rendering them
structurally incapable of enforcing violations against a substantial share of Indian
traffic. This project develops SPEC Traffic AI, a modular enforcement platform
designed for Indian road conditions. A single shared perception pass performs vehicle
and number-plate detection, ByteTrack assigns persistent identities, and an
extensible rule engine evaluates violations as independent geometric modules
configured per camera. Confirmed violations trigger a rolling frame buffer producing
an evidence package containing the approach, violation and departure sequence
together with a plain-language reason trace. A novel Evidence Defensibility Score
assesses whether each detection would withstand legal challenge and routes it for
officer verification accordingly. The same perception pass drives a second output:
congestion is measured per junction arm in Passenger Car Units rather than vehicle
counts, and Webster's method converts it into a recommended green-time split, issued
as an advisory plan the system deliberately cannot actuate. Evaluation established a
detection accuracy of 0.703 mAP@50 on a held-out test split and identified a
quantified viewpoint domain gap between dashcam training imagery and
enforcement-camera geometry. The platform enables authorities to scale enforcement
while ensuring no challan is issued without human approval under the Motor Vehicles
Act.

**Keywords:** Traffic Violation Detection; Computer Vision; Intelligent Transportation
Systems; Human-in-the-Loop Verification; Explainable AI; Adaptive Signal Control

---

## 1. Introduction

### 1.1 Background

Road safety represents one of India's most persistent public-health failures. The
Ministry of Road Transport and Highways reports that over-speeding contributed to
68.4 per cent of road accidents during 2023, and that 54,568 persons were killed
while not wearing protective headgear [8]. Enforcement capacity has not kept pace
with vehicle growth: manual policing cannot observe every junction continuously, and
the extensive closed-circuit television networks installed under smart-city
programmes remain largely passive, recording incidents rather than acting upon them.

The economic case for automation is established, yet operational experience shows
that detection volume alone is insufficient. The Safe Kerala deployment recorded
66.41 lakh violations in its first year and issued 64.72 lakh challans valued at
approximately 428 crore rupees, of which 76.7 crore rupees was collected. The
shortfall is administrative rather than technical, but it demonstrates that a
violation detected is not a penalty enforced, and that the stages following detection
determine whether an enforcement system produces any deterrent effect.

### 1.2 Problem Statement

Two deficiencies prevent existing computer-vision systems from closing this gap.
First, detectors trained on general-purpose datasets such as COCO contain no
auto-rickshaw class and are therefore structurally unable to enforce against a
substantial proportion of Indian traffic. Second, and less widely acknowledged,
published violation-detection systems optimise for whether an offence was detected
rather than whether the resulting evidence could survive legal challenge. Under
Section 65B of the Indian Evidence Act, since superseded by Section 63 of the
Bharatiya Sakshya Adhiniyam 2023, electronic records are inadmissible without
certification [13]. Evidence quality is consequently a statutory precondition rather
than a refinement.

### 1.3 Scope

This project delivers a working detection and evidence-generation pipeline for Indian
road conditions, an officer review portal implementing human-in-the-loop challan
approval, and an advisory adaptive signal-timing module. Automatic number-plate
recognition, multi-camera deployment and integration with departmental challan
systems lie outside the present scope and are addressed as future work.

---

## 2. Technology Review

### 2.1 Detection on Indian Roads

Indian-road object detection is substantially solved as a research problem.
DriveIndia provides a large annotated dataset of diverse Indian traffic scenes and
reports a baseline of 78.7 per cent mAP@50 [1]. More recently, the Urban Vision
Hackathon dataset (UVH-26) released 26,646 high-resolution frames sampled from
approximately 2,800 operational Safe-City closed-circuit cameras in Bengaluru,
annotated across fourteen India-specific vehicle classes, together with six
fine-tuned detection models [4]. Both employ the YOLOv11 architecture [11], whose
principal enhancements concern feature aggregation and inference efficiency.
Persistent vehicle identity is supplied by ByteTrack, which associates every
detection box rather than discarding low-confidence candidates, and which is
constructed from a Kalman filter and Hungarian assignment rather than a learned
model [10].

### 2.2 End-to-End Enforcement Systems

Complete pipelines already exist. Ravish et al. describe a system performing helmet
detection, triple-riding identification, number-plate recognition and automated
ticket generation [2]. Ren reports a violation-detection system combining computer
vision with human-computer interaction, achieving accuracies exceeding 96.9 per cent
across eight violation categories and 99.7 per cent for red-light running [3].
These figures are obtained under laboratory conditions; the study reports neither a
held-out evaluation split nor deployment against operational camera feeds, so the
extent to which they transfer to field conditions remains undetermined.

Both systems treat number-plate recognition as a solved subsystem. Field measurement
contradicts this. Rhead et al. examined automatic number-plate recognition against
real-world plate conditions and documented substantial misread rates arising from
non-standard fonts, physical damage, mounting angle and illumination [9]. Because a
challan is issued to a registered vehicle rather than to a bounding box, plate
recognition accuracy bounds the usable output of any enforcement pipeline
irrespective of detection performance.

### 2.3 Effectiveness of Automated Enforcement

Evidence that automated enforcement reduces harm is strong. A Cochrane systematic
review of thirty-five studies found crash reductions between 8 and 49 per cent in
the vicinity of camera sites [5]. A subsequent meta-analysis attributed a 51 per cent
reduction in fatal crashes to spot cameras and a 56 per cent reduction in
killed-or-seriously-injured crashes to section control [6].

### 2.4 Signal Control and Calibration

Adaptive signal timing rests on Webster's method, which derives an optimum cycle
length from the ratio of demand flow to saturation flow and remains the basis of
signal design practice, including Indian Roads Congress guidance [7], [12]. Camera
calibration for surveillance applications has been approached through
three-dimensional model alignment [15] and through automatic methods for wide-angle
lenses [16], both of which bear directly on scaling any geometric rule beyond a
single installation.

A consideration specific to Indian conditions concerns how demand is measured.
Western adaptive systems typically count vehicles, an approximation acceptable where
the traffic stream is comparatively homogeneous. Indian traffic is not: an approach
carrying forty motorcycles and one carrying forty buses impose substantially
different demands on junction capacity. Indian Roads Congress guidance addresses this
through Passenger Car Unit equivalency factors, which weight each vehicle class by
its effective consumption of road capacity [12]. Demand measurement intended for
Indian junctions must therefore be expressed in Passenger Car Units rather than in
raw vehicle counts.

---

## 3. Review Summary, Research Gap and Objectives

### 3.1 Review Summary and Limitations of Prior Work

Three limitations recur across the reviewed literature. First, reported accuracies
are frequently obtained without a held-out evaluation split; Ren reports laboratory
accuracy with neither a held-out split nor field deployment [3], and Ravish et al.
present a working prototype without quantified generalisation [2]. Second, viewpoint
transfer is rarely tested. BMD-45 measured a detector achieving 83.8 per cent mAP
in-domain falling to 33.6 per cent across domains on Indian closed-circuit footage
[14], a collapse absent from the violation-detection literature. Third, and most
consequentially, no reviewed system models whether its own evidence would be
admissible. Operational experience supports this concern: Bengaluru's Intelligent
Traffic Management System reports helmet detection at 75–80 per cent accuracy,
reaching 99.9 per cent only through manual revalidation of every fine prior to issue
[17].

### 3.2 Research Gap

The gap addressed is therefore not detection accuracy but evidential defensibility.
Detection confidence expresses certainty that an object belongs to a class; it
expresses nothing about whether a resulting challan would survive contest. No
published system carries an evidence-quality metric or a chain-of-custody concept
within the vision pipeline itself.

### 3.3 Objectives

1. To train and evaluate a detector on Indian road data recognising vehicle classes
   absent from general-purpose models, reporting performance on a held-out split.
2. To construct a modular rule engine in which each violation is an independent,
   calibrated geometric module operating over one shared perception pass.
3. To design and implement a quantitative measure of evidential defensibility that
   triages detections for officer review before any challan is issued.
4. To demonstrate that the same perception pass can additionally support advisory
   junction-signal timing without additional inference cost.

---

## 4. Methodology

### 4.1 System Architecture

The architecture separates perception from judgement. A single detection pass
produces frame results consumed independently by the rule engine and by the
signal-demand module, as shown in Figure 1. Because detection is the computationally
expensive stage and executes once, adding the signal-timing consumer incurred no
measurable per-frame cost.

Rules are self-registering modules declaring the classes they inspect and the
calibration they require. Adding a violation is a single file; deploying to a new
camera is a single configuration file.

### 4.2 Temporal Consistency and Abstention

Two design decisions dominate false-positive behaviour. Temporal consistency is
implemented in the rule base class rather than in individual rules, requiring a
condition to hold for a defined number of consecutive frames on the same track
before a violation is emitted. Placing it in the base class means every rule
inherits it rather than reimplementing it inconsistently.

Abstention is treated as a first-class outcome. Where a vehicle's road-contact point
falls outside every declared lane, the wrong-way rule returns no verdict rather than
assuming a default direction. Where no homography exists, no speed is reported. Where
insufficient traffic has been observed, the signal controller recommends the existing
fixed plan. For a system terminating in a monetary penalty, silence is inexpensive
whereas a wrongful challan is not.

### 4.3 Calibration by Measurement

Geometric rules depend on per-camera calibration: lane polygons, permitted headings,
stop lines and a pixel-to-metre homography. Permitted heading proved the critical
parameter. Entered in reverse, the wrong-way rule inverts, flagging every lawful
vehicle while ignoring actual offenders, and the rule cannot detect this condition
internally.

A still frame does not convey motion, so heading was derived from the footage
instead. Vehicle tracks are accumulated, each track's net direction computed, and the
dominant heading per lane reported together with a coherence measure. Where traffic
within a polygon does not share a direction, the tool reports incoherence rather than
selecting a majority, indicating that the polygon spans opposing carriageways and
must be redrawn.

### 4.4 Evidence Packaging

A violation occupies an instant, whereas evidence must establish context. The system
maintains a rolling frame buffer and, upon confirmation, writes a package spanning
the approach, the violation and the departure. The package contains the video clip,
the frame at the moment of violation with the offending vehicle delineated, an
earlier approach frame, the cropped number plate where one was located, and a
machine-readable record carrying the statutory section, the quantitative detail
underlying the verdict, and a plain-language reason trace stating in ordinary words
why the vehicle was flagged.

The reason trace is a deliberate design decision. An officer presented only with a
confidence value cannot assess whether a detection is sound, whereas a statement that
a vehicle travelled a measured number of degrees against the permitted direction of
its lane, sustained over a measured distance, can be disputed on its own terms.
Penalties are drawn from a configurable schedule carrying the statutory section and a
verification flag displayed alongside every amount. Offences whose evidence is
inadmissible by construction, such as vision-derived speed, are prevented from being
approved as challans irrespective of operator action.

### 4.5 Evidence Defensibility Score

Each violation is scored across five weighted dimensions: plate readability (0.30),
subject visibility (0.20), track integrity (0.20), rule margin (0.15) and contextual
certainty (0.15). Scores at or above 80 are pre-filled for approval, 50 to 79 are
routed for review, and those below 50 are dropped. Where a dimension cannot be
measured, its weight is redistributed proportionally across the remainder and the
contributing dimensions are recorded, so that the score spans its full range and the
omission remains visible.

Plate readability is measured from the image rather than taken from detector
confidence, since the two answer different questions. Readability combines plate
width in source pixels, sharpness by variance of the Laplacian, and contrast,
multiplied by detector confidence so that either uncertainty about the object or
illegibility of the characters reduces the score.

### 4.6 Explainability by Construction

Signal state is determined by hue-saturation-value thresholding over a calibrated
signal-head region rather than by a learned classifier, and the resulting pixel
fraction is recorded in the evidence. The statement that a given proportion of pixels
within the signal head matched red is defensible under examination in a way that an
unexplained network output is not. The same reasoning governs the choice of Webster's
method for signal timing: four lines of arithmetic with physically meaningful terms
can be verified by the traffic engineer who must authorise the plan.

---

## 5. Implementation and Results

### 5.1 Detector Training and Evaluation

A YOLOv11-small model was fine-tuned on DriveIndia for 50 epochs on a 6 GB laptop
GPU. The class set was reduced from 28 to 21, as seven classes contained between zero
and twenty-two instances and could not be learned. Performance on the official
held-out test split is reported in Table 1.

**Table 1: Detector performance across data splits**

| Split | mAP@50 | mAP@50–95 | Precision | Recall |
|---|---|---|---|---|
| Train | 0.850 | 0.752 | 0.832 | 0.763 |
| Validation | 0.716 | 0.598 | 0.721 | 0.679 |
| **Test** | **0.703** | 0.606 | 0.620 | 0.671 |

Restricting the mean to the eighteen adequately supported classes yields 0.800, and
to the seven enforcement-relevant classes 0.868. Auto-rickshaw, absent from COCO,
reached 0.888. The train-to-test generalisation gap of 0.147 is reported rather than
omitted; the close agreement between validation and test figures indicates a
representative split.

### 5.2 The Viewpoint Domain Gap

Testing on elevated footage revealed systematic misclassification of large vehicles.
Comparison against a detector trained on enforcement-camera imagery quantified the
effect. On an identical frame at identical confidence threshold, the DriveIndia
fine-tune produced 11 detections while the UVH-26 model produced 52, with the former
labelling a sport-utility vehicle and a sedan as buses at confidences of 0.91 and
0.85 respectively (Figure 2). Across six sampled frames the totals were 33 and 330.
This independently corroborates the cross-domain collapse reported for BMD-45 [14],
and the UVH-26 model was consequently adopted as the default detector for elevated
cameras.

### 5.3 Calibration and False-Positive Reduction

The calibration methodology described in Section 4.3 was validated directly. An
initial hand-entered calibration produced fifteen wrong-way violations. Measurement
reported one lane coherent at 0.92 with complete agreement, and a second incoherent
at 0.42 with 70 per cent of tracks opposing the declared heading; eleven of the
fifteen violations originated in the latter. With the heading corrected and default
thresholds restored, fifteen violations reduced to one, at 173 degrees against the
flow, by a vehicle moving contrary to eleven of twenty-one tracked vehicles.

### 5.4 Rule Verification Without Footage

Three rules could not be exercised by available footage. Rather than leave them
unverified, twenty-two assertions were constructed covering both halves of each
rule's contract: that it fires on the offence, and that it abstains on the lawful
case most resembling it. All twenty-two pass.

### 5.5 Detection of Riders and Plates

Two measured limitations are reported. Across 1,467 motorcycle detections spanning
two datasets, the detector never placed two persons on the same motorcycle, because
the training annotation convention treats a rider as part of the motorcycle
boundary rather than as a separate pedestrian. Triple riding and helmet compliance
are therefore unreachable without rider-annotated data.

Plate availability proved similarly constrained. Of twenty-one violations, three
carried a detected plate, at widths of 17, 23 and 41 pixels, all below the threshold
at which characters remain legible. Sampling forty frames from operational Bengaluru
enforcement cameras produced a median plate width of 36 pixels, with only 30 per cent
at or above 45 pixels. This indicates that the camera observing an offence and the
camera identifying the vehicle are necessarily different instruments.

### 5.6 Officer Review Portal and Signal Timing

The portal presents captured vehicles as a filterable gallery, each violation as a
draft challan with statutory section and penalty, and annotated playback of processed
runs. Three decisions are available: approval, rejection, and escalation to a second
reviewer who records a citation in their own words. Decisions are written to the evidence
package, so that a folder remains self-describing wherever it is copied, and to an
append-only log retaining the complete history including reversals. The log records
the defensibility score at the moment of decision and the time the reviewer spent on
screen, both of which are required for subsequent validation of the score against
human judgement.

Escalation is not a weaker form of rejection. It routes the package to a second
reviewer who examines the footage and records a citation in their own words, and
citations accumulate rather than overwrite, so that a package escalated and
subsequently approved carries two human attestations. Given that certification of
electronic records depends upon human attestation, retaining both is a requirement of
the chain of custody rather than an interface convenience.

The signal controller was verified across four scripted demand scenarios. Green time
follows measured queue, increasing from 37 to 57 seconds as demand built; an empty
arm continued to receive its minimum allocation each cycle; and beyond practical
capacity the controller reported the condition rather than returning an unusable
cycle length.

### 5.7 Score Distribution Across the Evidence Set

Twenty-one evidence packages were generated from a single camera by varying rule
sensitivity across four passes while holding calibration fixed, producing a range of
evidence quality rather than a uniformly strong set. Defensibility scores distributed
across all three bands: five qualified for automatic preparation, twelve for officer
review and four for rejection.

The distinction between varying sensitivity and varying calibration is material.
Sensitivity expresses how readily a rule reacts, and relaxing it yields genuinely
marginal evidence suitable for evaluating a triage mechanism. Calibration expresses a
claim about the physical scene, and relaxing it would yield confidently incorrect
evidence of an entirely different character.

The three packages carrying detected plates received the lowest scores in the set, at
33.7, 38.0 and 41.7. In each, the violation is plainly visible while the vehicle
remains unidentifiable, and the score reflects this without manual labelling. This
constitutes a concrete instance of detection confidence and evidential value
diverging, which is the central proposition of the project.

### 5.8 Computational Performance

Detection latency was measured at 18.0 ms per frame at 1280-pixel inference on an
RTX 4050 laptop GPU, corresponding to 55.6 frames per second, against 304.2 ms on
CPU. Real-time operation at full resolution is therefore achievable on
consumer-grade hardware.

---

## 6. Conclusion and Future Scope

### 6.1 Conclusion

This project demonstrates that the limiting factor in automated traffic enforcement
is not detection accuracy but evidential defensibility. A detector was trained and
honestly evaluated, achieving 0.703 mAP@50 on a held-out split and 0.868 across
enforcement-relevant classes. A modular rule engine, an evidence packaging pipeline
and an officer review portal were implemented and verified. The Evidence
Defensibility Score provides a quantitative basis for triaging detections before
human review, and the plate-readability analysis demonstrates concretely that
detection confidence and evidential value diverge.

Equally significant are the negative results. The viewpoint domain gap, the
unreachability of rider detection under prevailing annotation conventions, and the
scarcity of legible plates in general-view footage are each quantified rather than
asserted, and each carries a specific remedy.

The architectural claim is likewise supported by measurement. Adding adaptive signal
timing as a second consumer of the existing perception pass incurred no additional
per-frame inference, demonstrating that a shared perception layer allows enforcement
and traffic management to be served by a single installation. This bears on
deployment economics: a municipality installing cameras for enforcement obtains
congestion measurement without further hardware, and the recommendations produced are
auditable arithmetic rather than opaque output.

The treatment of uncertainty is the most transferable contribution. Rules abstain
outside declared geometry, speeds are withheld in the absence of a homography, the
signal controller defers to the existing plan until sufficient traffic has been
observed, and the defensibility score redistributes weight across the dimensions it
could measure while recording which contributed. In an application whose output is a
monetary penalty levied upon a member of the public, a system that declines to answer
is preferable to one that answers incorrectly.

### 6.2 Future Scope

Four directions follow. Automatic number-plate recognition with multi-frame voting
should be implemented against a dedicated plate camera per lane, accepting the
70–85 per cent accuracy achievable in field conditions [9] rather than the figures
commonly advertised. Automatic calibration, extending the heading measurement
reported here toward lane-polygon inference, is the step that converts operation on
one camera into operation across many [15], [16]. Rider-annotated training data would
restore triple-riding and helmet detection. Finally, the Evidence Defensibility Score
should be validated against the approve-and-reject decisions of serving enforcement
officers, reporting the area under the receiver operating characteristic curve.

---

## References

[1] S. Kumar et al., "DriveIndia: An Object Detection Dataset for Diverse Indian Traffic Scenes," arXiv:2507.19912, 2025.

[2] R. Ravish, S. Rangaswamy et al., "Intelligent Traffic Violation Detection," in Proc. IEEE Global Conf. Advancement Technol. (GCAT), 2021, doi: 10.1109/GCAT52182.2021.9587520.

[3] Y. Ren, "Intelligent Vehicle Violation Detection System Under Human–Computer Interaction and Computer Vision," Int. J. Computational Intelligence Systems, vol. 17, no. 40, 2024, doi: 10.1007/s44196-024-00427-6.

[4] A. Sharma et al., "The Urban Vision Hackathon Dataset and Models: Towards Image Annotations and Accurate Vision Models for Indian Traffic," arXiv:2511.02563, 2025.

[5] C. Wilson, C. Willis, J. K. Hendrikz, R. Le Brocque, and N. Bellamy, "Speed cameras for the prevention of road traffic injuries and deaths," Cochrane Database of Systematic Reviews, no. 11, Art. no. CD004607, 2010.

[6] A. Høye, "Speed cameras, section control, and kangaroo jumps — a meta-analysis," Accident Analysis & Prevention, vol. 73, pp. 200–208, 2015.

[7] F. V. Webster, "Traffic Signal Settings," Road Research Technical Paper No. 39, HMSO, London, 1958.

[8] Ministry of Road Transport and Highways, Government of India, "Road Accidents in India 2023," New Delhi, 2025.

[9] M. Rhead, R. Gurney, S. Ramalingam, and N. Cohen, "Accuracy of automatic number plate recognition (ANPR) and real world UK number plate problems," in Proc. IEEE 46th Int. Carnahan Conf. Security Technology, 2012, doi: 10.1109/CCST.2012.6393574.

[10] Y. Zhang, P. Sun, Y. Jiang, D. Yu et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," arXiv:2110.06864, 2021.

[11] R. Khanam and M. Hussain, "YOLOv11: An Overview of the Key Architectural Enhancements," arXiv:2410.17725, 2024.

[12] Indian Roads Congress, "IRC:93 — Guidelines on Design and Installation of Road Traffic Signals," New Delhi.

[13] Indian Evidence Act 1872, s. 65B; Bharatiya Sakshya Adhiniyam 2023, s. 63.

[14] A. Sharma et al., "BMD-45: A Large-Scale CCTV Vehicle Detection Dataset for Urban Traffic in Developing Cities," arXiv:2604.24419, 2026.

[15] J. Sochor, R. Juránek, and A. Herout, "Traffic Surveillance Camera Calibration by 3D Model Bounding Box Alignment," arXiv:1702.06451, 2017.

[16] A. G. Jain and N. Saunier, "Autocamera Calibration for Traffic Surveillance Cameras with Wide Angle Lenses," arXiv:2001.07243, 2020.

[17] Bengaluru Traffic Police, "Intelligent Traffic Management System — reported helmet-detection accuracy and manual revalidation practice."

---

## Figures referenced

- **Figure 1:** System architecture — one perception pass, two consumers
  (`results/driveindia/` — to be drawn)
- **Figure 2:** Viewpoint domain gap — same frame, 11 detections against 52
  (`cmp_ours.jpg` / `cmp_uvh.jpg`)
- **Figure 3:** Generalisation gap across splits (`generalisation_gap.png`)
- **Figure 4:** Per-class accuracy (`per_class.png`)
- **Figure 5:** Officer review portal — challan draft
- **Figure 6:** Green time following measured queue (`signal_split.png`)
