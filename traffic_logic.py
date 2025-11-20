"""
traffic_logic_fixed.py

Corrected TrafficLightController implementing:
 - Default all RED
 - If one lane >=15 and all others <5 -> that lane GREEN (preempts current green)
 - If all lanes <5 -> pick lane with most cars (even 1 car)
 - Only one green at a time
 - Max green = 25s -> then YELLOW 5s (blocking)
 - YELLOW (5s) is shown on transitions both for expiry and when switching lanes
 - Adaptive lane regions (4 quadrants by default)
 - Optional helper to set YOLO model class names so car/ped IDs are inferred

Usage examples:
  python traffic_logic_fixed.py --video 0
  python traffic_logic_fixed.py --video traffic_intersection.mp4
"""

import time
from typing import Dict, Tuple, List, Optional, Sequence
import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False


class TrafficLightController:
    def __init__(self, num_lanes: int = 4, frame_shape: Tuple[int, int] = (640, 480)):
        """
        frame_shape: (width, height)
        """
        self.num_lanes = num_lanes
        self.frame_shape = frame_shape
        self.lane_regions = self.define_adaptive_lane_regions(frame_shape)

        # lane_states stores 'red'|'green'|'yellow'
        self.lane_states: Dict[int, str] = {i: 'red' for i in range(1, num_lanes + 1)}

        # runtime state
        self.frame_count = 0
        self.current_green_lane: Optional[int] = None
        self.green_start_time: Optional[float] = None

        self.max_green_time = 25.0  # seconds
        self.yellow_duration = 5.0  # seconds (blocking)

        # class id configuration (inferred or set by user)
        # By default assume car=0, person=1 (typical COCO style), but allow custom mapping
        self.vehicle_class_ids: Sequence[int] = (0,)   # treat these ids as vehicles/cars
        self.pedestrian_class_ids: Sequence[int] = (1,)  # treat these ids as pedestrians

        print(f"🛣️ Lane regions defined for {frame_shape[0]}x{frame_shape[1]} resolution")
        for lane_id, region in self.lane_regions.items():
            print(f"   Lane {lane_id}: {region}")

    def define_adaptive_lane_regions(self, frame_shape: Tuple[int, int]):
        width, height = frame_shape
        mid_x = width // 2
        mid_y = height // 2

        # Quadrants: 1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right
        return {
            1: [(0, 0), (mid_x, 0), (mid_x, mid_y), (0, mid_y)],
            2: [(mid_x, 0), (width, 0), (width, mid_y), (mid_x, mid_y)],
            3: [(0, mid_y), (mid_x, mid_y), (mid_x, height), (0, height)],
            4: [(mid_x, mid_y), (width, mid_y), (width, height), (mid_x, height)],
        }

    def update_frame_shape(self, frame_shape: Tuple[int, int]):
        if frame_shape != self.frame_shape:
            self.frame_shape = frame_shape
            self.lane_regions = self.define_adaptive_lane_regions(frame_shape)
            print(f"🔄 Updated lane regions for {frame_shape[0]}x{frame_shape[1]}")

    def set_model_names(self, names: Dict[int, str]):
        """
        Provide model.names mapping (index -> label string) to let controller
        automatically infer vehicle and pedestrian class ids.
        It will include classes whose name contains 'car','truck','bus','vehicle' as vehicles,
        and 'person'/'pedestrian' as pedestrian class ids.
        """
        vehicle_ids = []
        ped_ids = []
        for idx, name in names.items():
            lname = str(name).lower()
            if any(k in lname for k in ('car', 'truck', 'bus', 'vehicle', 'van', 'taxi', 'auto')):
                vehicle_ids.append(idx)
            if any(k in lname for k in ('person', 'pedestrian', 'people')):
                ped_ids.append(idx)

        if len(vehicle_ids) > 0:
            self.vehicle_class_ids = tuple(vehicle_ids)
        if len(ped_ids) > 0:
            self.pedestrian_class_ids = tuple(ped_ids)

        print(f"Configured vehicle class ids: {self.vehicle_class_ids}, pedestrian ids: {self.pedestrian_class_ids}")

    def is_point_in_region(self, x: float, y: float, region: List[Tuple[int, int]]) -> bool:
        region_array = np.array(region, dtype=np.int32)
        result = cv2.pointPolygonTest(region_array, (int(x), int(y)), False)
        return result >= 0

    def process_detections(self, detections, frame_shape: Tuple[int, int] = None):
        """
        Convert model detections into per-lane car counts & pedestrian flags.
        Expects Ultralytics results-like input (iterable of Results, each with .boxes).
        """

        if frame_shape:
            # frame_shape as (width, height) expected
            self.update_frame_shape(frame_shape)

        lane_car_counts = {i: 0 for i in range(1, self.num_lanes + 1)}
        pedestrian_detections = {i: False for i in range(1, self.num_lanes + 1)}

        if detections is None:
            # nothing detected
            self.frame_count += 1
            return lane_car_counts, pedestrian_detections

        total_detections = 0
        car_detections = 0

        for result in detections:
            if not hasattr(result, 'boxes') or result.boxes is None:
                continue

            for box in result.boxes:
                try:
                    # Ultralytics boxes typically have .cls and .conf and .xyxy
                    class_id = int(box.cls[0]) if hasattr(box, 'cls') else None
                    confidence = float(box.conf[0]) if hasattr(box, 'conf') else None
                    # xyxy may be on cpu or already numpy
                    xyxy = box.xyxy[0]
                    try:
                        bbox = xyxy.cpu().numpy() if hasattr(xyxy, 'cpu') else np.array(xyxy)
                    except Exception:
                        bbox = np.array(xyxy)
                except Exception:
                    # skip malformed
                    continue

                center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
                center_y = (float(bbox[1]) + float(bbox[3])) / 2.0

                total_detections += 1

                # find which lane this center is in
                object_lane = None
                for lane_id, region in self.lane_regions.items():
                    if self.is_point_in_region(center_x, center_y, region):
                        object_lane = lane_id
                        break

                if object_lane is None:
                    # not inside any lane region -> ignore
                    continue

                # decide if this is vehicle or pedestrian by class_id
                if class_id is not None:
                    if class_id in self.vehicle_class_ids:
                        lane_car_counts[object_lane] += 1
                        car_detections += 1
                    elif class_id in self.pedestrian_class_ids:
                        pedestrian_detections[object_lane] = True
                    else:
                        # class not recognized as vehicle or pedestrian -> ignore
                        pass
                else:
                    # if no class id available, fallback: treat as vehicle
                    lane_car_counts[object_lane] += 1
                    car_detections += 1

        # debug for early frames
        if self.frame_count < 10 and total_detections > 0:
            print(f" Frame {self.frame_count}: {car_detections} cars detected in lanes: {lane_car_counts}")

        self.frame_count += 1
        return lane_car_counts, pedestrian_detections

    def control_traffic_lights(self, lane_car_counts: Dict[int, int], pedestrian_detections: Dict[int, bool]) -> Dict[int, str]:
        """
        Main control logic implementing the rules with blocking yellow (5s).
        Returns lane_states dict mapping lane_id -> 'red'|'yellow'|'green'
        """

        current_time = time.time()
        previous_green = self.current_green_lane

        # 1) If there's an active green, check expiry or preemption conditions
        if self.current_green_lane is not None:
            elapsed = current_time - (self.green_start_time or current_time)

            # If green has expired -> perform expiry yellow (blocking), then clear current green to re-evaluate
            if elapsed >= self.max_green_time:
                print(f"⏰ Green expired for lane {self.current_green_lane} (elapsed {elapsed:.1f}s). Switching to YELLOW for {self.yellow_duration}s.")
                # show yellow
                self.lane_states = {i: 'red' for i in range(1, self.num_lanes + 1)}
                self.lane_states[self.current_green_lane] = 'yellow'
                # blocking yellow
                time.sleep(self.yellow_duration)
                # after yellow -> set to red and clear
                print(f"🟨 Yellow finished for lane {self.current_green_lane}. Setting RED and re-evaluating.")
                self.lane_states[self.current_green_lane] = 'red'
                self.current_green_lane = None
                self.green_start_time = None
            else:
                # Check for Rule A preemption: if any other lane meets Rule A and it's not the current green lane,
                # preempt current lane (show yellow, then switch).
                preempt_lane = None
                for lane_id, count in lane_car_counts.items():
                    if lane_id == self.current_green_lane:
                        continue
                    if count >= 15:
                        others_small = all((c < 5) for lid, c in lane_car_counts.items() if lid != lane_id)
                        if others_small:
                            preempt_lane = lane_id
                            break

                if preempt_lane is not None:
                    # Preempt: show yellow on previous, then set previous to red and pick preempt lane
                    print(f"⚠️ Preempting lane {self.current_green_lane} for lane {preempt_lane} (Rule A triggered). Showing YELLOW for {self.yellow_duration}s.")
                    self.lane_states = {i: 'red' for i in range(1, self.num_lanes + 1)}
                    self.lane_states[self.current_green_lane] = 'yellow'
                    time.sleep(self.yellow_duration)
                    print(f"🟨 Preemption yellow finished. Switching to lane {preempt_lane}.")
                    self.lane_states[self.current_green_lane] = 'red'
                    # clear so selection below behaves like "no current green"
                    self.current_green_lane = None
                    self.green_start_time = None
                    # continue to selection below (no immediate return)

                else:
                    # keep current green (no expiry & no preemption)
                    states = {i: 'red' for i in range(1, self.num_lanes + 1)}
                    states[self.current_green_lane] = 'green'
                    return states

        # 2) No active green now: choose next lane based on rules

        chosen_lane: Optional[int] = None

        # Rule A: If a lane has >=15 and all others <5 -> pick that lane
        for lane_id, count in lane_car_counts.items():
            if count >= 15:
                others_small = all((c < 5) for lid, c in lane_car_counts.items() if lid != lane_id)
                if others_small:
                    chosen_lane = lane_id
                    break

        # Rule B: if all lanes <5 -> pick the lane with most cars
        if chosen_lane is None:
            if all(c < 5 for c in lane_car_counts.values()):
                chosen_lane = max(lane_car_counts, key=lane_car_counts.get)
            else:
                # fallback: pick lane with most cars (general)
                chosen_lane = max(lane_car_counts, key=lane_car_counts.get)

        if chosen_lane is None:
            # no choice -> keep all red
            self.lane_states = {i: 'red' for i in range(1, self.num_lanes + 1)}
            return self.lane_states

        # If we are switching from a previous lane (which might still be set), ensure we show yellow on that previous lane
        # This covers cases where previous green was cleared due to expiry/preemption earlier in this method,
        # but also if for some reason previous_green is still set (defensive).
        if previous_green is not None and previous_green != chosen_lane:
            # Show yellow on previous_green (defensive: only if it's still logically active)
            print(f"🔁 Switching from lane {previous_green} to lane {chosen_lane}. Showing YELLOW for {self.yellow_duration}s.")
            self.lane_states = {i: 'red' for i in range(1, self.num_lanes + 1)}
            self.lane_states[previous_green] = 'yellow'
            time.sleep(self.yellow_duration)
            print(f"🟨 Yellow finished for previous lane {previous_green}. Now activating lane {chosen_lane}.")
            # ensure previous is red
            self.lane_states[previous_green] = 'red'

        # Activate chosen lane
        self.current_green_lane = chosen_lane
        self.green_start_time = time.time()

        states = {i: 'red' for i in range(1, self.num_lanes + 1)}
        states[chosen_lane] = 'green'

        print(f"🟢 Activating lane {chosen_lane} as GREEN (cars: {lane_car_counts.get(chosen_lane,0)})")
        self.lane_states = states
        return states

    def visualize_traffic_state(self, frame, lane_car_counts: Dict[int, int], pedestrian_detections: Dict[int, bool], lane_states: Dict[int, str]):
        debug_frame = frame.copy()
        overlay = debug_frame.copy()

        for lane_id, region in self.lane_regions.items():
            state = lane_states.get(lane_id, 'red')
            color = self.get_lane_color(state)
            pts = np.array(region, dtype=np.int32)

            # Draw filled polygon (semi-transparent handled below)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(debug_frame, [pts], True, color, 2)

            # Put lane info
            center_x = sum([pt[0] for pt in region]) // len(region)
            center_y = sum([pt[1] for pt in region]) // len(region)

            info_text = f"L{lane_id}: {lane_car_counts.get(lane_id,0)} cars"
            cv2.putText(debug_frame, info_text, (center_x - 60, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            state_text = lane_states.get(lane_id, 'red').upper()
            cv2.putText(debug_frame, state_text, (center_x - 60, center_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Merge overlay with transparency
        cv2.addWeighted(overlay, 0.18, debug_frame, 0.82, 0, debug_frame)

        total_cars = sum(lane_car_counts.values())
        cv2.putText(debug_frame, f"Total Cars: {total_cars}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return debug_frame

    def get_lane_color(self, state: str) -> Tuple[int, int, int]:
        colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255)
        }
        return colors.get(state, (255,255,255))


# ---------------------
# Example main loop (kept from original but using corrected controller)
# ---------------------
def main(video_source: str = 0, model_path: str = 'yolo11n.pt'):
    # Open video
    cap = cv2.VideoCapture(int(video_source) if str(video_source).isdigit() else str(video_source))
    if not cap.isOpened():
        print(f"Couldn't open video source: {video_source}")
        return

    # Create controller after reading a single frame to know resolution
    ret, frame = cap.read()
    if not ret:
        print("Couldn't read first frame from source")
        return

    h, w = frame.shape[:2]
    controller = TrafficLightController(num_lanes=4, frame_shape=(w, h))

    # Load YOLO model if available; else, we will not run detection
    model = None
    if ULTRALYTICS_AVAILABLE:
        try:
            model = YOLO(model_path)
            print("Loaded YOLO model:", model_path)
            # If model has names, provide them to controller to auto-configure class ids
            if hasattr(model, 'names') and isinstance(model.names, dict):
                controller.set_model_names(model.names)
        except Exception as e:
            print("Couldn't load YOLO model:", e)
            model = None
    else:
        print("Ultralytics not available; running without detection (will show empty lanes).")

    fps_t0 = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Run detection (if model available)
        detections = None
        if model is not None:
            # model returns a list of Results; we wrap it in a list for processing
            # Keep device logic minimal to avoid errors
            try:
                results = model(frame, imgsz=640)
            except Exception:
                results = model(frame)
            detections = results

        lane_car_counts, pedestrian_detections = controller.process_detections(detections, frame_shape=(w, h))

        lane_states = controller.control_traffic_lights(lane_car_counts, pedestrian_detections)

        vis = controller.visualize_traffic_state(frame, lane_car_counts, pedestrian_detections, lane_states)

        cv2.imshow('Traffic Controller', vis)

        # Basic FPS display
        if frame_idx % 30 == 0:
            fps = 30.0 / max(1e-6, time.time() - fps_t0)
            fps_t0 = time.time()
            print(f"Frame {frame_idx} - approx FPS: {fps:.1f}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='0', help='Video file path or camera index (default 0)')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='YOLO model path (optional)')
    args = parser.parse_args()

    main(video_source=args.video, model_path=args.model)
