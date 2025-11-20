import cv2
import os
import time
import numpy as np

# Use the corrected controller file name you already have
from traffic_logic import TrafficLightController

# ultralytics import - keep as-is
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False


class VideoTrafficTester:
    def __init__(self, model_path: str, conf: float = 0.25, imgsz: int = 640):
        model_path = model_path.replace('\\', '/')
        print(f" Loading model from: {model_path}")

        # Load model with fallback
        if not os.path.exists(model_path):
            print(f" Model not found: {model_path}")
            print(" Using default YOLO model for testing (yolo11n.pt)")
            self.model = YOLO("yolo11n.pt") if ULTRALYTICS_AVAILABLE else None
        else:
            try:
                self.model = YOLO(model_path) if ULTRALYTICS_AVAILABLE else None
                print(" Custom model loaded successfully!")
            except Exception as e:
                print(f" Error loading custom model: {e}")
                print(" Falling back to default YOLO model (yolo11n.pt)")
                self.model = YOLO("yolo11n.pt") if ULTRALYTICS_AVAILABLE else None

        # controller
        self.controller = TrafficLightController()
        # If model has names, configure controller so it knows which class ids are vehicles/pedestrians
        if self.model is not None and hasattr(self.model, 'names'):
            try:
                self.controller.set_model_names(self.model.names)
            except Exception:
                # defensive: some model objects store names differently
                print(" Warning: couldn't auto-configure model names into controller")

        # detection options
        self.conf = conf
        self.imgsz = imgsz

        self.is_running = False

    def test_video(self, video_path: str, output_path: str = None):
        video_path = video_path.replace('\\', '/')
        if not os.path.exists(video_path):
            print(f" Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f" Could not open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        print(f" Video Info: {width}x{height} @ {fps:.2f}fps, frames={total_frames}")

        writer = None
        if output_path:
            output_path = output_path.replace('\\', '/')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f" Output will be saved to: {output_path}")

        self.is_running = True
        frame_count = 0
        start_time = time.time()

        paused = False
        skip_frames = 0

        print("\n Starting video inference...")
        print("   Controls: 'q'=quit, 'p'=pause, 's'=skip 30 frames, 'r'=reset")

        while self.is_running:
            if not paused and skip_frames <= 0:
                ret, frame = cap.read()
                if not ret:
                    print(" End of video reached")
                    break

                frame_count += 1

                processed_frame = self.process_frame(frame, frame_count)

                # Progress overlay
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0.0
                cv2.putText(processed_frame, f"Progress: {progress:.1f}%", (10, height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames or '?'}", (10, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('Traffic System - Video Test', processed_frame)

                if writer:
                    writer.write(processed_frame)

            # handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('s'):
                skip_frames = 30
                print("Skipping 30 frames")
            elif key == ord('r'):
                # reset controller but keep model mapping
                self.controller = TrafficLightController()
                if hasattr(self.model, 'names'):
                    self.controller.set_model_names(self.model.names)
                print("System reset")

            if skip_frames > 0:
                skip_frames -= 1

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nProcessed {frame_count} frames in {total_time:.2f}s, avg FPS: {avg_fps:.2f}")

    def process_frame(self, frame, frame_number):
        """
        Runs YOLO inference with debug prints and passes detections to controller.
        """
        try:
            # Defensive: ensure model exists
            if self.model is None:
                # no model — just visualize lanes
                lane_counts, ped = self.controller.process_detections(None, frame_shape=(frame.shape[1], frame.shape[0]))
                states = self.controller.control_traffic_lights(lane_counts, ped)
                out = self.controller.visualize_traffic_state(frame, lane_counts, ped, states)
                return out

            # Run YOLO with explicit confidence and imgsz
            # Note: Ultralytics YOLO returns a Results object (or list-like). We keep the raw results to forward.
            try:
                results = self.model(frame, imgsz=self.imgsz, conf=self.conf)
            except TypeError:
                # older/newer signatures may not accept named args
                results = self.model(frame)

            # Debug: inspect model results to see what classes/boxes are found
            boxes_total = 0
            classes_seen = {}
            # results may be a Results object (single), or a list; handle both
            results_list = results if isinstance(results, (list, tuple)) else [results]
            for r in results_list:
                # r.boxes might be an iterable or have xyxy / cls; handle gracefully
                try:
                    n = len(r.boxes)
                except Exception:
                    # fallback: attempt to access r.boxes.data if available
                    try:
                        n = len(getattr(r, 'boxes').xyxy)
                    except Exception:
                        n = 0
                boxes_total += n

                # try to inspect classes
                try:
                    for b in r.boxes:
                        try:
                            cid = int(b.cls[0])
                            classes_seen[cid] = classes_seen.get(cid, 0) + 1
                        except Exception:
                            pass
                except Exception:
                    pass

            print(f"[Frame {frame_number}] model returned {boxes_total} boxes; class histogram: {classes_seen}")

            # Provide frame dims to controller so lane regions are correct (width, height)
            frame_shape = (frame.shape[1], frame.shape[0])
            lane_car_counts, pedestrian_detections = self.controller.process_detections(results_list, frame_shape=frame_shape)

            # Debug: show what controller thinks
            print(f" -> controller lane counts: {lane_car_counts}")

            lane_states = self.controller.control_traffic_lights(lane_car_counts, pedestrian_detections)

            processed_frame = self.enhanced_visualization(frame, lane_car_counts, pedestrian_detections, lane_states, results_list, frame_number)

            return processed_frame

        except Exception as e:
            print(f" Error processing frame {frame_number}: {e}")
            return frame

    def enhanced_visualization(self, frame, lane_car_counts, pedestrian_detections, lane_states, results, frame_number):
        frame = self.controller.visualize_traffic_state(frame, lane_car_counts, pedestrian_detections, lane_states)
        self.add_detection_info(frame, results)
        return frame

    def add_detection_info(self, frame, results):
        # results_list guaranteed by process_frame
        if results and len(results) > 0:
            r = results[0]
            try:
                boxes = r.boxes
            except Exception:
                boxes = None

            car_count = 0
            person_count = 0
            if boxes is not None:
                for b in boxes:
                    try:
                        cid = int(b.cls[0])
                        # get label if available
                        lbl = self.model.names[cid] if hasattr(self.model, 'names') and cid in self.model.names else str(cid)
                        lbl_l = lbl.lower()
                        if any(k in lbl_l for k in ("car", "truck", "bus", "van", "auto", "vehicle")):
                            car_count += 1
                        elif "person" in lbl_l:
                            person_count += 1
                    except Exception:
                        continue

            cv2.putText(frame, f"Detected Cars: {car_count} | Persons: {person_count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
 
    def test_image(self, image_path: str, output_path: str = None):
        """
        Run inference on a single image and optionally save the output.
        """
        image_path = image_path.replace('\\', '/')
        if not os.path.exists(image_path):
            print(f" Image file not found: {image_path}")
            return

        frame = cv2.imread(image_path)
        if frame is None:
            print(f" Could not read image: {image_path}")
            return

        processed_frame = self.process_frame(frame, frame_number=1)

        # Show the image
        cv2.imshow("Traffic System - Image Inference", processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save if requested
        if output_path:
            output_path = output_path.replace('\\', '/')
            cv2.imwrite(output_path, processed_frame)
            print(f"Processed image saved to: {output_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test Traffic System with Video or Image')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model (pt)')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--output', type=str, help='Optional output file to save annotated video or image')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='YOLO image size')
    args = parser.parse_args()

    tester = VideoTrafficTester(args.model, conf=args.conf, imgsz=args.imgsz)

    if args.video:
        tester.test_video(args.video, args.output)
    elif args.image:
        tester.test_image(args.image, args.output)
    else:
        print("Please provide either a --video or --image path.")