"""
GelSight Video Viewer with Difference Statistics and Marker Displacement Tracking
2x2 sensors, each showing 3 images: RGB, Heatmap, Marker
Reads from recorded files in the same structure as the recording script
"""

import os
import cv2
import numpy as np
import glob
from datetime import datetime
import json
import sys

# Add path to SDK utilities if needed
sys.path.append('/home/guret/Documents/gsrobotics')  # Adjust this path as needed

try:
    from utilities.marker_tracker import MarkerTracker
    from config import GSConfig
    SDK_AVAILABLE = True
    print("✓ SDK utilities imported successfully")
except ImportError as e:
    print(f"✗ SDK utilities not available: {e}")
    print("Will use simplified marker tracking")
    SDK_AVAILABLE = False

class MarkerTrackerWrapper:
    """Wrapper for official SDK MarkerTracker or fallback implementation"""
    
    def __init__(self, camera_id, config=None):
        self.camera_id = camera_id
        self.config = config
        self.sdk_tracker = None
        self.initial_marker_center = None
        self.initialized = False
        
        # Optical flow parameters (same as SDK)
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Tracking state
        self.Ox = None  # Original x positions (columns)
        self.Oy = None  # Original y positions (rows)
        self.nct = 0    # Number of markers
        self.p0 = None  # Initial positions for optical flow
        self.old_gray = None  # Reference grayscale image
        
        # Visualization
        self.marker_size = 3
        self.vector_scale = 3.0
        
        print(f"[Tracker {camera_id}] Using {'SDK' if SDK_AVAILABLE else 'fallback'} tracker")
    
    def initialize_with_sdk(self, frame):
        """Initialize using official SDK MarkerTracker"""
        try:
            # Convert to float32 and normalize as SDK expects
            img = np.float32(frame) / 255.0
            
            # Initialize SDK tracker
            self.sdk_tracker = MarkerTracker(img, self.config)
            self.initial_marker_center = self.sdk_tracker.initial_marker_center
            
            if self.initial_marker_center is not None and len(self.initial_marker_center) > 0:
                # SDK returns centers as (row, col) = (y, x)
                self.Ox = self.initial_marker_center[:, 1]  # columns (x)
                self.Oy = self.initial_marker_center[:, 0]  # rows (y)
                self.nct = len(self.initial_marker_center)
                
                # Prepare p0 for optical flow (shape: n, 1, 2)
                self.p0 = np.zeros((self.nct, 1, 2), dtype=np.float32)
                for i in range(self.nct):
                    self.p0[i, 0, 0] = self.Ox[i]  # x
                    self.p0[i, 0, 1] = self.Oy[i]  # y
                
                # Store reference grayscale
                self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.initialized = True
                
                print(f"[Tracker {self.camera_id}] SDK initialized {self.nct} markers")
                return True
        except Exception as e:
            print(f"[Tracker {self.camera_id}] SDK initialization failed: {e}")
        
        return False
    
    def initialize_fallback(self, frame):
        """Fallback initialization using feature detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try to find markers (darker spots on brighter background)
        # Apply adaptive threshold to find dark markers
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        markers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 100:  # Filter by size
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    markers.append([cx, cy])
        
        if len(markers) > 0:
            markers = np.array(markers, dtype=np.float32)
            self.initial_marker_center = markers
            
            # markers are (x, y)
            self.Ox = markers[:, 0]  # x
            self.Oy = markers[:, 1]  # y
            self.nct = len(markers)
            
            # Prepare p0
            self.p0 = np.zeros((self.nct, 1, 2), dtype=np.float32)
            for i in range(self.nct):
                self.p0[i, 0, 0] = self.Ox[i]
                self.p0[i, 0, 1] = self.Oy[i]
            
            self.old_gray = gray.copy()
            self.initialized = True
            
            print(f"[Tracker {self.camera_id}] Fallback initialized {self.nct} markers")
            return True
        
        print(f"[Tracker {self.camera_id}] Fallback found no markers")
        return False
    
    def initialize(self, frame, use_sdk=True):
        """Initialize marker tracking with reference frame"""
        if use_sdk and SDK_AVAILABLE:
            success = self.initialize_with_sdk(frame)
            if success:
                return True
        
        # Fallback if SDK fails or not available
        return self.initialize_fallback(frame)
    
    def track(self, current_frame):
        """Track markers from reference to current frame"""
        if not self.initialized or self.old_gray is None or self.p0 is None:
            return None, None
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_gray, current_gray, self.p0, None, **self.lk_params
        )
        
        if p1 is not None and st is not None:
            # Select good points
            good_new = p1[st.ravel() == 1]
            good_old = self.p0[st.ravel() == 1]
            
            if len(good_new) > 0:
                # Calculate statistics
                displacements = good_new - good_old
                magnitudes = np.sqrt(np.sum(displacements**2, axis=1))
                
                stats = {
                    'num_markers': len(good_new),
                    'mean_displacement': np.mean(magnitudes),
                    'max_displacement': np.max(magnitudes),
                    'total_displacement': np.sum(magnitudes),
                    'std_displacement': np.std(magnitudes),
                    'displacements': displacements
                }
                
                return good_new, stats
        
        return None, None
    
    def draw_tracking(self, frame, tracked_positions, stats=None):
        """Draw markers and displacement vectors on frame"""
        if not self.initialized or tracked_positions is None:
            return frame
        
        result = frame.copy()
        
        # Draw original markers (reference positions)
        for i in range(self.nct):
            if i < len(tracked_positions):
                # Original position
                orig_x, orig_y = int(self.Ox[i]), int(self.Oy[i])
                
                # Current tracked position
                curr_pos = tracked_positions[i].ravel()
                curr_x, curr_y = int(curr_pos[0]), int(curr_pos[1])
                
                # Draw arrow from original to current (scaled for visibility)
                dx = curr_x - orig_x
                dy = curr_y - orig_y
                
                # Scale vector for better visibility
                end_x = int(orig_x + dx * self.vector_scale)
                end_y = int(orig_y + dy * self.vector_scale)
                
                # Draw arrow
                cv2.arrowedLine(result, (orig_x, orig_y), (end_x, end_y),
                              (0, 255, 0), 2, tipLength=0.3)
                
                # Draw original marker (small circle)
                cv2.circle(result, (orig_x, orig_y), self.marker_size, (255, 0, 0), -1)
                
                # Draw current marker position
                cv2.circle(result, (curr_x, curr_y), self.marker_size, (0, 0, 255), -1)
        
        # Add statistics text
        if stats is not None:
            y_offset = 30
            cv2.putText(result, f"Markers: {stats['num_markers']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
            cv2.putText(result, f"Mean Disp: {stats['mean_displacement']:.2f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
            cv2.putText(result, f"Max Disp: {stats['max_displacement']:.2f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result
    
    def reset(self):
        """Reset tracker state"""
        self.initialized = False
        self.initial_marker_center = None
        self.Ox = None
        self.Oy = None
        self.nct = 0
        self.p0 = None
        self.old_gray = None

class GelSightViewerWithStats:
    def __init__(self, base_path=None):
        # Use the same path structure as the recording script
        if base_path is None:
            self.base_path = os.path.join(os.path.expanduser("~"), "Desktop", "gelsight_recordings")
        else:
            self.base_path = base_path
            
        print(f"Looking for recordings in: {self.base_path}")
        
        # First, check if there are any recording folders
        if not os.path.exists(self.base_path):
            print(f"ERROR: Base path does not exist: {self.base_path}")
            print("Please create recordings using the live script first.")
            self.recordings = []
        else:
            self.recordings = self.get_all_recordings()
            
        self.current_recording_idx = 0 if self.recordings else -1
        self.caps = [None, None, None, None]
        self.references = [None, None, None, None]
        self.playing = True
        self.threshold = 30
        self.scale = 2.0
        self.current_frame = 0
        self.total_frames = 0
        
        # Load SDK configuration
        self.config = self.load_config()
        
        # Marker tracking
        self.marker_trackers = [MarkerTrackerWrapper(i+1, self.config) for i in range(4)]
        self.marker_stats = []
        
        # Statistics tracking
        self.diff_stats = []  # Store difference statistics per frame
        
        # Debug mode
        self.debug_mode = True
        self.auto_initialize = True  # Auto-init markers on first frame
        
        # Display layout - BIGGER size for larger window (same as vizu_video.py)
        self.display_size = (320, 240)  # Width, Height
    
    def load_config(self):
        """Load SDK configuration"""
        try:
            # Try to load from default location
            config_path = "default_config.json"
            if os.path.exists(config_path):
                gs_config = GSConfig(config_path)
                print(f"✓ Loaded config from {config_path}")
                return gs_config.config
            else:
                # Create minimal config
                print("⚠ Using minimal default config")
                return None
        except Exception as e:
            print(f"✗ Config load failed: {e}, using defaults")
            return None
    
    def get_all_recordings(self):
        """Get all recordings sorted by timestamp (most recent first)"""
        print("Scanning for recordings...")
        recording_folders = []
        
        # Look for folders with "recording_" prefix or named folders
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            if os.path.isdir(item_path):
                # Check if this folder contains camera files
                camera_files = glob.glob(os.path.join(item_path, "*.avi"))
                camera_files += glob.glob(os.path.join(item_path, "*.mp4"))
                camera_files += glob.glob(os.path.join(item_path, "*.mov"))
                
                if len(camera_files) >= 1:  # At least one camera file
                    recording_folders.append(item)
        
        # Sort by modification time (most recent first)
        recording_folders.sort(key=lambda x: os.path.getmtime(os.path.join(self.base_path, x)), reverse=True)
        
        # Display found recordings
        for i, folder in enumerate(recording_folders):
            # Count camera files in this folder
            camera_files = glob.glob(os.path.join(self.base_path, folder, "*.avi"))
            camera_files += glob.glob(os.path.join(self.base_path, folder, "*.mp4"))
            camera_files += glob.glob(os.path.join(self.base_path, folder, "*.mov"))
            print(f"  {i+1}. {folder} ({len(camera_files)} video files)")
        
        return recording_folders
    
    def find_camera_file(self, folder_path, camera_id):
        """Find camera file for a specific camera ID, handling different naming conventions"""
        # Try different naming patterns
        patterns = [
            f"*cam{camera_id}.avi",
            f"*cam{camera_id}.mp4",
            f"*cam{camera_id}.mov",
            f"camera_{camera_id}.avi",
            f"camera_{camera_id}.mp4",
            f"camera_{camera_id}.mov",
            f"*_camera{camera_id}.avi",
            f"*_camera{camera_id}.mp4",
            f"*_camera{camera_id}.mov",
        ]
        
        for pattern in patterns:
            files = glob.glob(os.path.join(folder_path, pattern))
            if files:
                return files[0]  # Return the first matching file
        
        # If no pattern matches, try to find any file with camera number
        all_video_files = glob.glob(os.path.join(folder_path, "*"))
        for file in all_video_files:
            filename = os.path.basename(file).lower()
            # Look for patterns like cam1, camera1, _1., etc.
            if (f"cam{camera_id}" in filename or 
                f"camera{camera_id}" in filename or 
                f"_{camera_id}." in filename):
                return file
        
        return None
    
    def open_current_recording(self):
        """Open the current recording for all cameras"""
        if self.current_recording_idx < 0 or self.current_recording_idx >= len(self.recordings):
            print("No recordings available!")
            return False
        
        recording_folder = self.recordings[self.current_recording_idx]
        recording_path = os.path.join(self.base_path, recording_folder)
        print(f"\nOpening recording: {recording_folder}")
        
        # Close current videos
        for i, cap in enumerate(self.caps):
            if cap is not None:
                cap.release()
                self.caps[i] = None
        
        # Open new videos for all cameras
        success = True
        for cam_id in range(1, 5):
            video_path = self.find_camera_file(recording_path, cam_id)
            
            if video_path and os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    self.caps[cam_id-1] = cap
                    
                    # Get first frame as reference for this camera
                    ret, frame = cap.read()
                    if ret:
                        self.references[cam_id-1] = frame.copy()
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        
                        # Check frame properties
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        print(f"✓ Camera {cam_id}: {width}x{height}, {fps:.1f} FPS, {frame_count} frames")
                        print(f"  File: {os.path.basename(video_path)}")
                        
                        # Reset marker tracker for this camera
                        self.marker_trackers[cam_id-1].reset()
                    else:
                        print(f"✗ Camera {cam_id}: Could not read first frame")
                        self.references[cam_id-1] = None
                        success = False
                else:
                    print(f"✗ Camera {cam_id}: Failed to open {video_path}")
                    self.references[cam_id-1] = None
                    success = False
            else:
                print(f"✗ Camera {cam_id}: No video file found in {recording_path}")
                self.references[cam_id-1] = None
        
        # Reset frame counter
        self.current_frame = 0
        self.diff_stats = []
        self.marker_stats = []
        
        # Auto-initialize markers if enabled (EXACTLY like vizu_video.py)
        if self.auto_initialize:
            print("Auto-initializing markers from first reference frames...")
            for i in range(4):
                if self.references[i] is not None:
                    tracker = self.marker_trackers[i]
                    if not tracker.initialized:
                        tracker.initialize(self.references[i], use_sdk=SDK_AVAILABLE)
        
        return success
    
    def next_recording(self):
        """Switch to next recording"""
        if len(self.recordings) <= 1:
            return False
        
        # Move to next recording
        self.current_recording_idx = (self.current_recording_idx + 1) % len(self.recordings)
        
        # Open new recording
        success = self.open_current_recording()
        
        return success
    
    def previous_recording(self):
        """Switch to previous recording"""
        if len(self.recordings) <= 1:
            return False
        
        # Move to previous recording
        self.current_recording_idx = (self.current_recording_idx - 1) % len(self.recordings)
        
        # Open new recording
        success = self.open_current_recording()
        
        return success
    
    def calculate_difference(self, frame, reference, camera_idx):
        """Calculate difference with detailed statistics"""
        if frame is None or reference is None:
            return None, None
        
        # Grayscale difference
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.absdiff(frame_gray, ref_gray)
        
        # Calculate statistics
        stats = {
            'frame': self.current_frame,
            'camera': camera_idx,
            'gray_mean': np.mean(diff_gray),
            'gray_max': np.max(diff_gray),
            'gray_std': np.std(diff_gray),
            'pixels_above_threshold': np.sum(diff_gray >= self.threshold),
            'total_pixels': diff_gray.size,
            'percent_above_threshold': np.sum(diff_gray >= self.threshold) / diff_gray.size * 100
        }
        
        self.diff_stats.append(stats)
        
        # Create heatmap visualization
        diff_scaled = cv2.convertScaleAbs(diff_gray, alpha=self.scale, beta=0)
        heatmap = cv2.applyColorMap(diff_scaled, cv2.COLORMAP_JET)
        
        return heatmap, stats
    
    def calculate_marker_displacement(self, frame, reference, camera_idx):
        """Calculate marker displacement between frame and reference"""
        if frame is None:
            return None, None
        
        tracker = self.marker_trackers[camera_idx]
        
        # Initialize tracker if not already done (EXACTLY like vizu_video.py)
        if not tracker.initialized and reference is not None:
            success = tracker.initialize(reference, use_sdk=SDK_AVAILABLE)
            if not success:
                return None, None
        
        # Track markers
        tracked_positions, stats = tracker.track(frame)
        
        if tracked_positions is not None and stats is not None:
            # Add frame info to stats
            stats['frame'] = self.current_frame
            stats['camera'] = camera_idx
            self.marker_stats.append(stats)
            
            return tracked_positions, stats
        
        return None, None
    
    def create_sensor_row(self, camera_idx, rgb_frame, heatmap_frame, marker_frame):
        """Create a row of 3 images for one sensor, rotated 90 degrees (EXACTLY like vizu_video.py)"""
        # Resize all frames to display size
        rgb_resized = cv2.resize(rgb_frame, self.display_size) if rgb_frame is not None else None
        heatmap_resized = cv2.resize(heatmap_frame, self.display_size) if heatmap_frame is not None else None
        marker_resized = cv2.resize(marker_frame, self.display_size) if marker_frame is not None else None
        
        # Rotate all images 90 degrees clockwise
        if rgb_resized is not None:
            rgb_resized = cv2.rotate(rgb_resized, cv2.ROTATE_90_CLOCKWISE)
        if heatmap_resized is not None:
            heatmap_resized = cv2.rotate(heatmap_resized, cv2.ROTATE_90_CLOCKWISE)
        if marker_resized is not None:
            marker_resized = cv2.rotate(marker_resized, cv2.ROTATE_90_CLOCKWISE)
        
        # ADDED: Rotate first two sensors (Camera 1 and 2) by 180 degrees
        if camera_idx in [0, 1]:  # Camera 1 (index 0) and Camera 2 (index 1)
            if rgb_resized is not None:
                rgb_resized = cv2.rotate(rgb_resized, cv2.ROTATE_180)
            if heatmap_resized is not None:
                heatmap_resized = cv2.rotate(heatmap_resized, cv2.ROTATE_180)
            if marker_resized is not None:
                marker_resized = cv2.rotate(marker_resized, cv2.ROTATE_180)
        
        # Create placeholders if needed (EXACTLY like vizu_video.py)
        if rgb_resized is None:
            rgb_resized = np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)
            rgb_resized = cv2.rotate(rgb_resized, cv2.ROTATE_90_CLOCKWISE)
            if camera_idx in [0, 1]:  # Apply 180 rotation to placeholders too
                rgb_resized = cv2.rotate(rgb_resized, cv2.ROTATE_180)
        
        if heatmap_resized is None:
            heatmap_resized = np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)
            heatmap_resized = cv2.rotate(heatmap_resized, cv2.ROTATE_90_CLOCKWISE)
            if camera_idx in [0, 1]:  # Apply 180 rotation to placeholders too
                heatmap_resized = cv2.rotate(heatmap_resized, cv2.ROTATE_180)
        
        if marker_resized is None:
            marker_resized = np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)
            marker_resized = cv2.rotate(marker_resized, cv2.ROTATE_90_CLOCKWISE)
            if camera_idx in [0, 1]:  # Apply 180 rotation to placeholders too
                marker_resized = cv2.rotate(marker_resized, cv2.ROTATE_180)
        
        # Add titles to each view
        cv2.putText(rgb_resized, f"Cam {camera_idx+1} RGB", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(heatmap_resized, "Heatmap", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(marker_resized, "Markers", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Create a row of 3 images
        sensor_row = np.hstack([rgb_resized, heatmap_resized, marker_resized])
        
        return sensor_row
    
    def run(self):
        """Main visualization loop"""
        # Open first recording
        if self.current_recording_idx >= 0:
            success = self.open_current_recording()
            if not success:
                print("ERROR: Failed to open recording!")
                return
        else:
            print("ERROR: No recordings found!")
            print(f"Please make sure you have recordings in: {self.base_path}")
            print("You can create recordings using the live script.")
            return
        
        print("\n" + "="*80)
        print("GEL SIGHT 3-VIEW VIDEO PLAYBACK")
        print("="*80)
        print("\n2x2 sensors, each showing 3 images: RGB, Heatmap, Marker Tracking")
        print(f"SDK Available: {'YES' if SDK_AVAILABLE else 'NO (using fallback)'}")
        print(f"Reading from: {self.base_path}")
        print("="*80)
        
        print("\nQUICK CONTROLS:")
        print("  SPACE: Play/Pause")
        print("  N: Next recording")
        print("  P: Previous recording")
        print("  Q: Quit")
        print("  R: Reset reference and re-initialize markers")
        print("  +/-: Adjust threshold")
        print("  s/S: Adjust scale")
        print("  D: Toggle debug mode")
        print("="*80 + "\n")
        
        print("[INFO] Press 'R' on a frame with visible gel markers to initialize tracking")
        
        while True:
            frames = []
            
            # Read frames
            for i, cap in enumerate(self.caps):
                if cap is not None:
                    if self.playing:
                        ret, frame = cap.read()
                        if not ret:
                            # Loop back to beginning
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.current_frame = 0
                            ret, frame = cap.read()
                        
                        if ret:
                            frames.append(frame)
                            if i == 0:
                                self.current_frame += 1
                        else:
                            frames.append(None)
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                        ret, frame = cap.read()
                        frames.append(frame if ret else None)
                else:
                    frames.append(None)
            
            # Create sensor rows (3 images each, all rotated 90 degrees)
            sensor_rows = []
            
            for i in range(4):
                if frames[i] is not None:
                    # 1. Original RGB frame
                    rgb_frame = frames[i]
                    
                    # 2. Heatmap frame
                    heatmap_frame = None
                    if self.references[i] is not None:
                        heatmap, _ = self.calculate_difference(frames[i], self.references[i], i)
                        if heatmap is not None:
                            heatmap_frame = heatmap
                    
                    # 3. Marker tracking frame
                    marker_frame = None
                    tracked_positions, marker_stats = self.calculate_marker_displacement(
                        frames[i], self.references[i], i
                    )
                    
                    if tracked_positions is not None and marker_stats is not None:
                        # Create marker visualization frame
                        tracker = self.marker_trackers[i]
                        marker_frame = tracker.draw_tracking(frames[i].copy(), tracked_positions, marker_stats)
                    
                    # Create a row of 3 images for this sensor (all rotated 90°)
                    sensor_row = self.create_sensor_row(i, rgb_frame, heatmap_frame, marker_frame)
                    sensor_rows.append(sensor_row)
                    
                else:
                    # Placeholder for missing camera (also rotated) - EXACTLY like vizu_video.py
                    placeholder = np.zeros((self.display_size[1], self.display_size[0]*3, 3), dtype=np.uint8)
                    placeholder = cv2.rotate(placeholder, cv2.ROTATE_90_CLOCKWISE)
                    
                    # ADDED: Rotate first two sensors by 180 degrees
                    if i in [0, 1]:
                        placeholder = cv2.rotate(placeholder, cv2.ROTATE_180)
                    
                    cv2.putText(placeholder, f"No Camera {i+1}", 
                               (placeholder.shape[1]//2 - 80, placeholder.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    sensor_rows.append(placeholder)
            
            # Arrange 4 sensors in 2x2 layout (EXACTLY like vizu_video.py)
            # Each sensor shows 3 images side-by-side
            # After rotation: each image is 240x320 (was 320x240, rotated 90°)
            # 3 images side-by-side: 720x320 per sensor row
            
            # First row: Sensor 1 and Sensor 2 side-by-side
            first_row = np.hstack([sensor_rows[0], sensor_rows[1]])
            
            # Second row: Sensor 3 and Sensor 4 side-by-side
            second_row = np.hstack([sensor_rows[2], sensor_rows[3]])
            
            # Stack rows vertically
            full_grid = np.vstack([first_row, second_row])
            
            # Add separator lines between sensors
            # Each sensor row is 3 images * 240 width = 720 pixels
            cv2.line(full_grid, (720, 0), (720, 320), (255, 255, 255), 2)
            cv2.line(full_grid, (0, 320), (1440, 320), (255, 255, 255), 2)
            cv2.line(full_grid, (720, 320), (720, 640), (255, 255, 255), 2)
            
            # Add separator lines between images within each sensor
            for row in range(2):
                for sensor in range(2):
                    # Vertical separator between RGB and Heatmap
                    x_pos = (sensor * 3 + 1) * 240
                    y_start = row * 320
                    y_end = (row + 1) * 320
                    cv2.line(full_grid, (x_pos, y_start), (x_pos, y_end), (200, 200, 200), 1)
                    
                    # Vertical separator between Heatmap and Markers
                    x_pos = (sensor * 3 + 2) * 240
                    cv2.line(full_grid, (x_pos, y_start), (x_pos, y_end), (200, 200, 200), 1)
            
            # Add status bar
            rec_name = self.recordings[self.current_recording_idx] if self.current_recording_idx >= 0 else "None"
            initialized_trackers = sum(1 for t in self.marker_trackers if t.initialized)
            
            status = (f"Recording: {rec_name[:15]}... | Frame: {self.current_frame} | "
                     f"Markers: {initialized_trackers}/4 init | Thresh: {self.threshold} | Scale: {self.scale:.1f}x")
            cv2.putText(full_grid, status, (10, full_grid.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Make the window resizable
            window_name = "GelSight: 2x2 Sensors, 3 Views Each (RGB | Heatmap | Markers) - Video Playback"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1440, 640)
            cv2.imshow(window_name, full_grid)
            
            # Handle keys
            key = cv2.waitKey(30 if self.playing else 0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.playing = not self.playing
                print(f"[DEBUG] Playing: {self.playing}")
            elif key == ord('n') or key == ord('N'):
                if self.next_recording():
                    print(f"[DEBUG] Switched to next recording")
            elif key == ord('p') or key == ord('P'):
                if self.previous_recording():
                    print(f"[DEBUG] Switched to previous recording")
            elif key == ord('r'):
                print(f"[DEBUG] Resetting references and markers...")
                for i in range(4):
                    if frames[i] is not None:
                        self.references[i] = frames[i].copy()
                        tracker = self.marker_trackers[i]
                        tracker.reset()
                        success = tracker.initialize(self.references[i], use_sdk=SDK_AVAILABLE)
                        if success:
                            print(f"[DEBUG] Camera {i+1}: Initialized {tracker.nct} markers")
                        else:
                            print(f"[DEBUG] Camera {i+1}: Marker initialization failed")
                
                print("[DEBUG] All references and markers reset")
                self.diff_stats = []
                self.marker_stats = []
            elif key == ord('d') or key == ord('D'):
                self.debug_mode = not self.debug_mode
                print(f"[DEBUG] Debug mode: {'ENABLED' if self.debug_mode else 'DISABLED'}")
            elif key == ord('+'):
                self.threshold = min(self.threshold + 5, 100)
                print(f"[DEBUG] Threshold: {self.threshold}")
            elif key == ord('-'):
                self.threshold = max(self.threshold - 5, 1)
                print(f"[DEBUG] Threshold: {self.threshold}")
            elif key == ord('s'):
                self.scale = min(self.scale + 0.5, 10.0)
                print(f"[DEBUG] Scale: {self.scale:.1f}x")
            elif key == ord('S'):
                self.scale = max(self.scale - 0.5, 0.5)
                print(f"[DEBUG] Scale: {self.scale:.1f}x")
        
        # Cleanup
        for cap in self.caps:
            if cap:
                cap.release()
        cv2.destroyAllWindows()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GelSight Video Playback Viewer")
    parser.add_argument("--path", type=str, default=None,
                       help="Path to recordings directory (default: ~/Desktop/gelsight_recordings)")
    
    args = parser.parse_args()
    
    if args.path:
        base_path = args.path
    else:
        base_path = os.path.join(os.path.expanduser("~"), "Desktop", "gelsight_recordings")
    
    viewer = GelSightViewerWithStats(base_path=base_path)
    viewer.run()

if __name__ == "__main__":
    main()
