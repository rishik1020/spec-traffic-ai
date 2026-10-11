"""
senti.perception.plates
=======================
Find the number plate, and attach it to the vehicle it belongs to.

WHY THIS IS THE MOST IMPORTANT MISSING PIECE
A challan is issued to a VEHICLE, not to a bounding box. Detect an offence
perfectly, and without a plate you still have nobody to fine. That is the whole
distance between "detects" and "enforces", and it is why the Evidence
Defensibility Score weights `plate` at 0.30 -- more than any other dimension.

Until now that weight scored nothing: plate_conf was always 0, every package
was scored without it, and the EDS `auto` band was unreachable. This closes
that hole.

WHY DETECTION BUT NOT RECOGNITION
This module finds the plate and crops it. It does not read the characters, and
that is a deliberate stopping point rather than unfinished work.

Real-world ANPR runs 70-85%, not the 98% vendors advertise -- night glare,
dust, damaged plates, two-wheeler occlusion. A wrong plate number is worse than
no plate number: it is a fine sent to an innocent registered owner, and it is
precisely the failure the Evidence Defensibility Score exists to catch. So the
system crops the plate, scores how readable it is, and puts the crop in front
of the officer who confirms the characters. The human reads; the machine
measures how fair it was to ask.

That is the same bargain the rest of the system makes. Signal state is colour
thresholding rather than a classifier so it can be explained when contested.
Here, the plate is evidence an officer can check rather than a string a model
asserted.

WHICH VEHICLE A PLATE BELONGS TO
The one whose box contains most of it. `overlap_fraction` is asymmetric on
purpose -- a plate is tiny and a bus is large, so IoU is useless here while
"what fraction of the PLATE lies inside this vehicle" is exactly right. Where
two vehicles overlap, the plate goes to whichever contains more of it, and ties
go to the smaller box, because a plate sits on the car in front rather than the
bus behind it.

MODEL
morsetechlab/yolov11-license-plate-detection (AGPL-3.0), fetched by
scripts/fetch_plates.py. Same framework as the vehicle detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..core.types import Detection

DEFAULT_WEIGHTS = (Path(__file__).resolve().parent.parent.parent
                   / 'models' / 'plates' / 'license-plate-v1s.pt')


class PlateDetector:
    """Finds plates in a frame and assigns each to a vehicle."""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS,
                 device: Optional[str] = None, imgsz: int = 1280,
                 conf: float = 0.25, min_overlap: float = 0.6) -> None:
        from ultralytics import YOLO

        self.path = Path(weights)
        self.available = self.path.is_file()
        if not self.available:
            print(f'[plates] {self.path.name} not found -- plate detection OFF. '
                  f'Fetch it with: python scripts/fetch_plates.py')
            self.model = None
            return

        self.model = YOLO(str(self.path))
        if device:
            self.model.to(device)
        self.imgsz = imgsz
        # Low on purpose. A plate the model is unsure about is still worth
        # cropping -- the officer decides whether it is readable, and the
        # confidence becomes the EDS plate score rather than a gate.
        self.conf = conf
        self.min_overlap = min_overlap

    # ---------------------------------------------------------------------

    def attach(self, frame, detections: list[Detection]) -> int:
        """Find plates and write them onto the vehicles they belong to.

        Mutates `detections` in place and returns how many were matched. A
        vehicle with no plate found keeps plate_conf 0.0, which is honest: it
        means the evidence cannot identify anyone, and EDS should say so.
        """
        if self.model is None or not detections:
            return 0

        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                 verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return 0

        vehicles = [d for d in detections if d.is_vehicle]
        if not vehicles:
            return 0

        matched = 0
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            conf = float(b.conf)
            plate = Detection(cls_name='number_plate', confidence=conf,
                              xyxy=(x1, y1, x2, y2))

            best, best_ov = None, 0.0
            for v in vehicles:
                ov = plate.overlap_fraction(v)
                # On a tie prefer the SMALLER box: a plate belongs to the car
                # in front, not the bus behind it.
                if ov > best_ov or (ov == best_ov and best is not None
                                    and v.area < best.area):
                    best, best_ov = v, ov

            if best is None or best_ov < self.min_overlap:
                continue
            # one plate per vehicle -- keep the most confident
            if conf > best.plate_conf:
                best.plate_xyxy = (x1, y1, x2, y2)
                best.plate_conf = conf
                matched += 1
        return matched

    @staticmethod
    def readability(frame, det: Detection) -> tuple[float, dict]:
        """How readable is this plate, 0-1, and why.

        WHY THIS IS NOT THE DETECTOR'S CONFIDENCE
        The detector answers "is that a plate", which is a different question
        from "can anyone read it". The first package this was tested on scored
        0.57 confidence on a crop that is a dark red blur -- genuinely a plate,
        genuinely unreadable. Feeding that 0.57 into the Evidence Defensibility
        Score as the plate dimension would have rated unreadable evidence as
        middling, which is exactly the failure the score exists to prevent.

        So readability is measured from the image itself, on three things a
        human squinting at a crop actually depends on:

          width    a plate under ~45px across in the source frame cannot be
                   read by anyone, however confident the detector was
          sharpness variance of the Laplacian -- the standard blur measure,
                   near zero on a motion-smeared plate
          contrast spread between the plate's light background and dark
                   characters; a washed-out or shadowed plate has little

        All three are explainable in a contested hearing, which is the same
        reason signal state is colour thresholding rather than a classifier.
        """
        import cv2
        import numpy as np

        if det.plate_xyxy is None:
            return 0.0, {'reason': 'no plate found'}

        x1, y1, x2, y2 = (int(v) for v in det.plate_xyxy)
        h, w = frame.shape[:2]
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0:
            return 0.0, {'reason': 'empty crop'}

        px_w = x2 - x1
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # A full-size Indian plate is ~500mm wide; 45px is about the floor at
        # which characters survive at all, 140px is comfortable.
        w_score = max(0.0, min(1.0, (px_w - 45.0) / (140.0 - 45.0)))
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        s_score = max(0.0, min(1.0, sharp / 220.0))
        spread = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        c_score = max(0.0, min(1.0, spread / 90.0))

        # Width gates legibility: a 20px plate is unreadable no matter how
        # sharp and contrasty those 20 pixels are.
        legible = w_score * (0.45 * s_score + 0.35 * c_score + 0.20)

        # ...and the detector's confidence gates the whole thing, because
        # legibility alone measures the wrong property. Tested against real
        # Bengaluru enforcement frames, the highest-scoring "plate" in the
        # sample was the BENGALURU SARIGE livery panel on a city bus: large,
        # sharp, high-contrast, perfectly legible, and not a number plate.
        # Scoring that 1.00 would put bus paintwork in front of an officer as
        # the thing identifying the vehicle to be fined.
        #
        # Multiplying is the honest combination. Either half failing should
        # sink the score: unsure it is a plate, or sure but unreadable, both
        # mean the evidence cannot identify anybody.
        score = float(det.plate_conf) * legible
        return round(max(0.0, min(1.0, score)), 3), {
            'legibility': round(legible, 3),
            'plate_px_width': px_w,
            'detector_conf': round(det.plate_conf, 3),
            'sharpness': round(sharp, 1),
            'contrast': round(spread, 1),
            'width_score': round(w_score, 3),
            'sharpness_score': round(s_score, 3),
            'contrast_score': round(c_score, 3),
        }

    @staticmethod
    def crop(frame, det: Detection, pad: int = 4, upscale: int = 4):
        """The plate image an officer actually reads.

        Upscaled because a plate is often 30px wide in the source frame, and
        a human squinting at 30px is being asked to guess. Cubic interpolation
        does not add information, but it does stop the browser from
        nearest-neighbour'ing it into mush.
        """
        if det.plate_xyxy is None:
            return None
        import cv2

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in det.plate_xyxy)
        crop = frame[max(0, y1 - pad):min(h, y2 + pad),
                     max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size == 0:
            return None
        if upscale > 1:
            crop = cv2.resize(crop, (crop.shape[1] * upscale,
                                     crop.shape[0] * upscale),
                              interpolation=cv2.INTER_CUBIC)
        return crop
