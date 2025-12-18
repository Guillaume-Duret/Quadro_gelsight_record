import os
import subprocess
import re
import cv2
import numpy as np
from datetime import datetime
import time
import threading
import queue
import csv

os.environ["KIVY_NO_ARGS"] = "1"
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.popup import Popup
from kivy.uix.modalview import ModalView
from kivy.core.window import Window
from kivy.config import Config
from kivy.metrics import dp
from kivy.graphics.texture import Texture

from utilities.gelsightmini import GelSightMini
from utilities.image_processing import add_fps_count_overlay, rescale
from utilities.ui_components import ConnectingOverlay, FileChooserPopup
from config import ConfigModel
from utilities.logger import log_message

Config.set("input", "mouse", "mouse,multitouch_on_demand")

# ============================================================================
# DEFAULT GELSIGHT DEVICE MAPPING - Using fixed indices [5, 3, 7, 0]
# ============================================================================
GELSIGHT_DEVICE_MAP = {
    1: {"serial": "28LT-JX5J", "name": "GelSight #1", "default_idx": 5},
    2: {"serial": "28MR-JCYM", "name": "GelSight #2", "default_idx": 3},
    3: {"serial": "28RW-Y70E", "name": "GelSight #3", "default_idx": 7},
    4: {"serial": "2BXB-ETNM", "name": "GelSight #4", "default_idx": 0}
}

# Forward declaration for type hints
class QuadGelsightMini:
    """Forward declaration for type hints"""
    pass


class RecordingNamePopup(Popup):
    """Popup for entering recording name when stopping recording"""
    def __init__(self, on_name_selected_callback, on_cancel_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.title = "Name Your Recording"
        self.size_hint = (0.8, 0.4)
        self.auto_dismiss = False
        self.on_name_selected_callback = on_name_selected_callback
        self.on_cancel_callback = on_cancel_callback
        
        # Create layout
        layout = BoxLayout(orientation="vertical", padding=10, spacing=10)
        
        # Instructions
        instructions = Label(
            text="Enter a name for your recording\n(optional - leave blank for timestamp)",
            size_hint_y=None,
            height=40,
            halign="center"
        )
        layout.add_widget(instructions)
        
        # Text input for recording name
        self.name_input = TextInput(
            hint_text="e.g., Experiment_1, Test_Run, etc.",
            multiline=False,
            size_hint_y=None,
            height=40
        )
        layout.add_widget(self.name_input)
        
        # Button container
        button_layout = BoxLayout(orientation="horizontal", size_hint_y=None, height=50, spacing=10)
        
        # Cancel button
        cancel_btn = Button(text="Cancel", background_color=(0.8, 0.2, 0.2, 1))
        cancel_btn.bind(on_press=self.on_cancel_pressed)
        button_layout.add_widget(cancel_btn)
        
        # OK button
        ok_btn = Button(text="OK", background_color=(0.2, 0.6, 0.2, 1))
        ok_btn.bind(on_press=self.on_ok_pressed)
        button_layout.add_widget(ok_btn)
        
        layout.add_widget(button_layout)
        
        self.content = layout
        
    def on_ok_pressed(self, instance):
        """Handle OK button press"""
        recording_name = self.name_input.text.strip()
        self.on_name_selected_callback(recording_name)
        self.dismiss()
        
    def on_cancel_pressed(self, instance):
        """Handle Cancel button press"""
        if self.on_cancel_callback:
            self.on_cancel_callback()
        self.dismiss()


class QuadTopBar(BoxLayout):
    """Custom top bar for mapping GelSight devices to video indices with default indices"""
    def __init__(self, on_device_selected_callback=None, auto_mapping=None, 
                 all_cameras_detected=False, **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=None, height=dp(120), 
                         spacing=dp(5), padding=[dp(5), dp(5), dp(5), dp(5)], **kwargs)
        self.on_device_selected_callback = on_device_selected_callback
        self.auto_mapping = auto_mapping or {}
        self.all_cameras_detected = all_cameras_detected
        
        self.available_indices = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        
        # Create 4 selection boxes - one for each GelSight
        self.video_inputs = []
        
        for gelsight_id in range(1, 5):
            # Get device info from the map
            device_info = GELSIGHT_DEVICE_MAP.get(gelsight_id, {})
            display_name = device_info.get("name", f"GelSight {gelsight_id}")
            default_idx = device_info.get("default_idx", -1)
            
            # Container for each GelSight selection
            device_container = BoxLayout(orientation="vertical", size_hint=(None, None), 
                                         size=(dp(120), dp(110)))
            
            # GelSight name
            name_label = Label(
                text=f"{display_name}",
                size_hint_y=None,
                height=dp(20),
                font_size='11sp',
                halign="center",
                bold=True,
                color=(0.1, 0.3, 0.6, 1)
            )
            device_container.add_widget(name_label)
            
            # Default index
            status_label = Label(
                text=f"Default: /dev/video{default_idx}",
                size_hint_y=None,
                height=dp(18),
                font_size='9sp',
                halign="center",
                color=(0.2, 0.6, 0.2, 1)
            )
            device_container.add_widget(status_label)
            
            # Manual text input for video device index (pre-filled with default)
            video_input = TextInput(
                text=str(default_idx) if default_idx >= 0 else "",
                hint_text=f"Video index",
                size_hint=(None, None),
                size=(dp(110), dp(25)),
                font_size='11sp',
                multiline=False,
                write_tab=False,
                background_color=(0.9, 1.0, 0.9, 1) if default_idx >= 0 else (1, 1, 1, 0.9),
                foreground_color=(0, 0, 0, 1)
            )
            video_input.gelsight_id = gelsight_id  # Store which GelSight this is for
            video_input.bind(text=self.on_video_input_change)
            self.video_inputs.append(video_input)
            device_container.add_widget(video_input)
            
            # Dropdown with available indices
            spinner = Spinner(
                text=str(default_idx) if default_idx >= 0 else "Select",
                values=(["Select"] if default_idx < 0 else [str(default_idx)]) + self.available_indices,
                size_hint=(None, None),
                size=(dp(110), dp(25)),
                font_size='10sp'
            )
            spinner.gelsight_id = gelsight_id
            spinner.bind(text=self.create_spinner_handler(gelsight_id))
            device_container.add_widget(spinner)
            
            self.add_widget(device_container)
            
            # Add vertical separator between devices (except after last one)
            if gelsight_id < 4:
                separator = Label(text="|", size_hint=(None, None), 
                                 size=(dp(5), dp(80)), color=(0.7, 0.7, 0.7, 1))
                self.add_widget(separator)

        # Add start button
        start_container = BoxLayout(orientation="vertical", size_hint=(None, None),
                                   size=(dp(150), dp(110)))
        
        self.start_btn = Button(
            text="Start All Cameras",
            size_hint=(None, None),
            size=(dp(140), dp(40)),
            background_color=(0.2, 0.5, 0.2, 1),
            disabled=not self.all_cameras_detected,
            font_size='12sp'
        )
        self.start_btn.bind(on_press=self.on_start_pressed)
        start_container.add_widget(self.start_btn)
        
        # Instructions
        instructions = Label(
            text="Using default indices\n[5, 3, 7, 0]",
            size_hint_y=None,
            height=dp(40),
            font_size='10sp',
            color=(0.3, 0.3, 0.3, 1),
            halign="center",
            valign="middle"
        )
        start_container.add_widget(instructions)
        
        self.add_widget(start_container)

    def create_spinner_handler(self, gelsight_id):
        """Create a closure to handle spinner selection for a specific GelSight"""
        def handler(spinner, text):
            if text != "Select":
                # Find the corresponding text input and set its value
                for video_input in self.video_inputs:
                    if video_input.gelsight_id == gelsight_id:
                        video_input.text = text
                        self.on_video_input_change(video_input, text)
                        break
        return handler

    def on_video_input_change(self, text_input, text):
        # Check if all inputs have values
        all_filled = all(input.text.strip() != "" for input in self.video_inputs)
        
        # Check for valid integer inputs
        all_valid = True
        for input_widget in self.video_inputs:
            try:
                val = input_widget.text.strip()
                if val:
                    int_val = int(val)  # Try to convert to integer
                    if int_val < 0:
                        all_valid = False
            except ValueError:
                all_valid = False
                break
        
        if all_filled and all_valid:
            self.start_btn.disabled = False
            self.start_btn.background_color = (0.2, 0.8, 0.2, 1)  # Bright green
        else:
            self.start_btn.disabled = True
            self.start_btn.background_color = (0.2, 0.5, 0.2, 1)  # Dark green

    def on_start_pressed(self, instance):
        if self.on_device_selected_callback:
            device_mapping = {}
            
            for video_input in self.video_inputs:
                try:
                    gelsight_id = video_input.gelsight_id
                    idx_text = video_input.text.strip()
                    
                    if not idx_text:
                        return
                    
                    video_idx = int(idx_text)
                    device_mapping[gelsight_id] = video_idx
                    
                except ValueError as e:
                    return
            
            self.on_device_selected_callback(device_mapping)


class CameraThread:
    """Thread for each camera to capture frames without blocking GUI"""
    def __init__(self, camera_id, video_index, width=640, height=480, border_fraction=0.1):
        self.camera_id = camera_id
        self.video_index = video_index
        self.width = width
        self.height = height
        self.border_fraction = border_fraction
        
        # Thread control
        self.running = False
        self.thread = None
        
        # Frame storage - IMPORTANT: GelSightMini returns RGB frames
        self.current_frame_rgb = None  # RGB for display
        self.current_diff_rgb = None   # RGB for display
        self.reference_frame_rgb = None  # RGB for processing
        
        # Statistics
        self.fps = 0.0
        self.frame_count = 0
        self.diff_mean = 0.0
        self.diff_max = 0.0
        
        # Statistics history for saving
        self.stats_history = []
        
        # Control parameters
        self.diff_threshold = 30
        self.diff_scale = 2.0
        self.zoom_factor = 1.0
        
        # Thread safety
        self.lock = threading.Lock()
        self.frame_ready = threading.Event()
        
        # Camera object
        self.camera = None
        
        # FPS calculation
        self.frame_times = []
        
        # Reference control flags
        self.set_reference_flag = False
        self.reset_reference_flag = False
        
        # Recording
        self.recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.recording_frame_count = 0
        self.recording_filepath = None
        self.recording_folder = None
        
    def start(self):
        """Start the camera thread"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the camera thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self._cleanup()
            
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        # Initialize camera
        self.camera = GelSightMini(
            target_width=self.width,
            target_height=self.height,
            border_fraction=self.border_fraction,
        )
        
        try:
            if self.video_index >= 0:
                self.camera.select_device(self.video_index)
            self.camera.start()
            
            # Wait for camera to initialize
            time.sleep(1.0)
            
            # Main capture loop
            while self.running:
                start_time = time.time()
                
                try:
                    # Get frame from camera (GelSightMini returns RGB)
                    frame_rgb = self.camera.update(1/30.0)
                    
                    if frame_rgb is None:
                        time.sleep(0.01)
                        continue
                    
                    # Apply transformations
                    frame_rgb = cv2.flip(frame_rgb, 1)  # Horizontal mirror
                    if self.camera_id < 3:  # First three cameras rotated 180
                        frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
                    
                    # Check for reference control flags
                    with self.lock:
                        if self.set_reference_flag:
                            # Store reference in RGB format
                            self.reference_frame_rgb = frame_rgb.copy()
                            self.set_reference_flag = False
                            
                        if self.reset_reference_flag:
                            self.reference_frame_rgb = None
                            self.reset_reference_flag = False
                    
                    # Calculate difference if reference exists
                    diff_frame_rgb = None
                    diff_mean = 0.0
                    diff_max = 0.0
                    
                    with self.lock:
                        reference_exists = self.reference_frame_rgb is not None
                    
                    if reference_exists:
                        with self.lock:
                            ref_frame_rgb = self.reference_frame_rgb
                        
                        # Convert RGB frames to grayscale for difference calculation
                        current_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                        ref_gray = cv2.cvtColor(ref_frame_rgb, cv2.COLOR_RGB2GRAY)
                        
                        # Calculate absolute difference
                        diff = cv2.absdiff(current_gray, ref_gray)
                        
                        # Calculate statistics
                        diff_mean = np.mean(diff)
                        diff_max = np.max(diff)
                        
                        # Store stats
                        with self.lock:
                            self.diff_mean = diff_mean
                            self.diff_max = diff_max
                        
                        # Create heatmap (COLORMAP_JET returns BGR, convert to RGB)
                        diff_scaled = cv2.convertScaleAbs(diff, alpha=self.diff_scale, beta=0)
                        diff_frame_bgr = cv2.applyColorMap(diff_scaled, cv2.COLORMAP_JET)
                        diff_frame_rgb = cv2.cvtColor(diff_frame_bgr, cv2.COLOR_BGR2RGB)
                        
                        # Add statistics overlay (use white text on RGB)
                        cv2.putText(diff_frame_rgb, f"M:{diff_mean:.1f}", (5, 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        cv2.putText(diff_frame_rgb, f"T:{self.diff_threshold}", (5, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # Store stats in history
                        stats_entry = {
                            'timestamp': datetime.now(),
                            'camera_id': self.camera_id,
                            'frame_count': self.frame_count,
                            'diff_mean': diff_mean,
                            'diff_max': diff_max,
                            'fps': self.fps
                        }
                        with self.lock:
                            self.stats_history.append(stats_entry)
                            # Keep only last 1000 entries
                            if len(self.stats_history) > 1000:
                                self.stats_history = self.stats_history[-1000:]
                    
                    # Apply zoom to the display frame
                    display_frame_rgb = frame_rgb.copy()
                    if self.zoom_factor != 1.0:
                        display_frame_rgb = rescale(display_frame_rgb, scale=self.zoom_factor)
                    
                    # Apply zoom to difference frame if it exists
                    if diff_frame_rgb is not None and self.zoom_factor != 1.0:
                        diff_frame_rgb = rescale(diff_frame_rgb, scale=self.zoom_factor)
                    
                    # Add FPS overlay to display frame
                    # Calculate FPS
                    self.frame_times.append(start_time)
                    # Keep only last 30 frames
                    self.frame_times[:] = [t for t in self.frame_times if start_time - t < 1.0]
                    fps = len(self.frame_times)
                    
                    # Add FPS text to display frame (white text on RGB)
                    cv2.putText(display_frame_rgb, f"FPS: {fps:.1f}", (5, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Store frames and update statistics
                    with self.lock:
                        self.current_frame_rgb = display_frame_rgb
                        self.current_diff_rgb = diff_frame_rgb
                        self.fps = fps
                        self.frame_count += 1
                    
                    # Recording - convert RGB to BGR for OpenCV video writer
                    with self.lock:
                        if self.recording and self.video_writer is not None:
                            # Convert RGB to BGR for recording
                            frame_bgr = cv2.cvtColor(display_frame_rgb, cv2.COLOR_RGB2BGR)
                            self.video_writer.write(frame_bgr)
                            self.recording_frame_count += 1
                    
                    # Signal that new frame is ready
                    self.frame_ready.set()
                    
                except Exception as e:
                    print(f"Camera {self.camera_id} error: {e}")
                    time.sleep(0.1)
                
                # Control frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, 1/30.0 - elapsed)
                time.sleep(sleep_time)
                
        except Exception as e:
            print(f"Camera {self.camera_id} thread error: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up camera resources"""
        # Stop recording
        self.stop_recording()
        
        # Release camera if it exists
        if self.camera:
            try:
                # Check if camera has a camera attribute (OpenCV camera object)
                if hasattr(self.camera, 'camera') and self.camera.camera:
                    self.camera.camera.release()
                # Also check for any other cleanup needed
                if hasattr(self.camera, 'release'):
                    self.camera.release()
            except Exception as e:
                print(f"Error cleaning up camera {self.camera_id}: {e}")
                
    def set_reference(self):
        """Set current frame as reference"""
        with self.lock:
            self.set_reference_flag = True
                
    def reset_reference(self):
        """Reset reference frame"""
        with self.lock:
            self.reset_reference_flag = True
            
    def update_params(self, threshold, scale, zoom):
        """Update processing parameters"""
        with self.lock:
            self.diff_threshold = threshold
            self.diff_scale = scale
            self.zoom_factor = zoom
            
    def start_recording(self, filepath=None):
        """Start recording video"""
        with self.lock:
            if self.recording:
                return False
            
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"camera_{self.camera_id}_recording_{timestamp}.avi"
            
            try:
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(
                    filepath, 
                    fourcc, 
                    30.0,  # FPS
                    (self.width, self.height)
                )
                self.recording = True
                self.recording_start_time = datetime.now()
                self.recording_frame_count = 0
                self.recording_filepath = filepath
                return True
            except Exception as e:
                print(f"Failed to start recording for camera {self.camera_id}: {e}")
                self.video_writer = None
                return False
    
    def stop_recording(self, new_name=None):
        """Stop recording video"""
        with self.lock:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            if self.recording:
                # Rename file if a new name is provided
                if new_name is not None and self.recording_filepath:
                    try:
                        # Extract directory and extension
                        dir_path = os.path.dirname(self.recording_filepath)
                        base_name = os.path.basename(self.recording_filepath)
                        name_no_ext, ext = os.path.splitext(base_name)
                        
                        # Create new filename
                        if new_name.strip():  # If name is not empty
                            # Clean the name for filename use
                            clean_name = re.sub(r'[^\w\-_\. ]', '_', new_name.strip())
                            new_filename = f"{clean_name}_cam{self.camera_id}{ext}"
                        else:
                            # Keep original name if no new name provided
                            new_filename = base_name
                        
                        new_filepath = os.path.join(dir_path, new_filename)
                        
                        # Rename the file if names are different
                        if new_filepath != self.recording_filepath:
                            os.rename(self.recording_filepath, new_filepath)
                            print(f"Camera {self.camera_id}: Recording renamed to: {new_filename}")
                            self.recording_filepath = new_filepath
                        else:
                            print(f"Camera {self.camera_id}: Recording saved as: {base_name}")
                            
                    except Exception as e:
                        print(f"Error renaming recording for camera {self.camera_id}: {e}")
                elif self.recording_filepath:
                    print(f"Camera {self.camera_id}: Recording saved as: {os.path.basename(self.recording_filepath)}")
            
            self.recording = False
            self.recording_start_time = None
            
    def get_recording_status(self):
        """Get recording status"""
        with self.lock:
            return {
                'recording': self.recording,
                'start_time': self.recording_start_time,
                'frame_count': self.recording_frame_count,
                'filepath': self.recording_filepath
            }
            
    def get_current_frame_rgb(self):
        """Get current frame in RGB format (thread-safe)"""
        with self.lock:
            if self.current_frame_rgb is not None:
                return self.current_frame_rgb.copy()
            return None
            
    def get_current_diff_rgb(self):
        """Get current difference frame in RGB format (thread-safe)"""
        with self.lock:
            if self.current_diff_rgb is not None:
                return self.current_diff_rgb.copy()
            return None
            
    def get_stats(self):
        """Get camera statistics (thread-safe)"""
        with self.lock:
            return {
                'fps': self.fps,
                'frame_count': self.frame_count,
                'diff_mean': self.diff_mean,
                'diff_max': self.diff_max,
                'has_reference': self.reference_frame_rgb is not None
            }
            
    def get_stats_history(self):
        """Get statistics history (thread-safe)"""
        with self.lock:
            return self.stats_history.copy()
            
    def clear_stats_history(self):
        """Clear statistics history"""
        with self.lock:
            self.stats_history = []


class CameraManager:
    """Manages multiple camera threads"""
    def __init__(self, width=640, height=480, border_fraction=0.1, device_mapping=None):
        self.width = width
        self.height = height
        self.border_fraction = border_fraction
        self.device_mapping = device_mapping or {}
        self.camera_threads = {}
        self.recording_folder = None
        self.is_recording = False
        
    def start_cameras(self):
        """Start all camera threads"""
        for camera_id in range(1, 5):
            video_idx = self.device_mapping.get(camera_id, -1)
            if video_idx < 0:
                continue
                
            # Create camera thread
            camera_thread = CameraThread(
                camera_id=camera_id,
                video_index=video_idx,
                width=self.width,
                height=self.height,
                border_fraction=self.border_fraction
            )
            
            camera_thread.start()
            self.camera_threads[camera_id] = camera_thread
            
        # Wait for cameras to initialize
        time.sleep(1.0)
        
    def stop_cameras(self):
        """Stop all camera threads"""
        for camera_thread in self.camera_threads.values():
            camera_thread.stop()
        self.camera_threads.clear()
        
    def get_camera_thread(self, camera_id):
        """Get camera thread by ID"""
        return self.camera_threads.get(camera_id)
        
    def set_reference_for_all(self):
        """Set reference for all cameras"""
        for camera_thread in self.camera_threads.values():
            camera_thread.set_reference()
            
    def reset_reference_for_all(self):
        """Reset reference for all cameras"""
        for camera_thread in self.camera_threads.values():
            camera_thread.reset_reference()
            
    def update_params_for_all(self, threshold, scale, zoom):
        """Update parameters for all cameras"""
        for camera_thread in self.camera_threads.values():
            camera_thread.update_params(threshold, scale, zoom)
            
    def start_recording_for_all(self, base_folder=None):
        """Start recording for all cameras"""
        if self.is_recording:
            return self.recording_folder
            
        if base_folder is None:
            base_folder = os.path.join(os.path.expanduser("~"), "Desktop", "gelsight_recordings")
        
        os.makedirs(base_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_folder = os.path.join(base_folder, f"recording_{timestamp}")
        os.makedirs(self.recording_folder, exist_ok=True)
        
        all_started = True
        for camera_id, camera_thread in self.camera_threads.items():
            recording_file = os.path.join(self.recording_folder, f"camera_{camera_id}.avi")
            success = camera_thread.start_recording(recording_file)
            if not success:
                all_started = False
                print(f"Failed to start recording for camera {camera_id}")
        
        if all_started:
            self.is_recording = True
            print(f"Recording started in folder: {self.recording_folder}")
        else:
            # If any camera failed to start recording, stop all
            self.stop_recording_for_all()
            
        return self.recording_folder
        
    def stop_recording_for_all(self, recording_name=None):
        """Stop recording for all cameras with optional name"""
        if not self.is_recording:
            return self.recording_folder
        
        # First stop all recordings
        for camera_thread in self.camera_threads.values():
            camera_thread.stop_recording(recording_name)
        
        self.is_recording = False
        
        # If a name was provided and folder exists, rename the folder
        if recording_name and self.recording_folder and os.path.exists(self.recording_folder):
            try:
                # Clean the name for folder use
                clean_name = re.sub(r'[^\w\-_\. ]', '_', recording_name.strip())
                if clean_name:  # If name is not empty
                    # Get parent directory
                    parent_dir = os.path.dirname(self.recording_folder)
                    
                    # Extract timestamp from current folder name
                    current_folder_name = os.path.basename(self.recording_folder)
                    # Check if folder name follows pattern "recording_TIMESTAMP"
                    if current_folder_name.startswith("recording_"):
                        timestamp = current_folder_name[10:]  # Remove "recording_" prefix
                    else:
                        timestamp = current_folder_name
                    
                    # Create new folder name
                    new_folder_name = f"{clean_name}_{timestamp}"
                    new_folder_path = os.path.join(parent_dir, new_folder_name)
                    
                    # Rename the folder
                    os.rename(self.recording_folder, new_folder_path)
                    self.recording_folder = new_folder_path
                    print(f"Recording folder renamed to: {new_folder_name}")
                else:
                    print(f"Recording saved in folder: {self.recording_folder}")
                    
            except Exception as e:
                print(f"Error renaming recording folder: {e}")
        
        return self.recording_folder
            
    def is_any_recording(self):
        """Check if any camera is recording"""
        return self.is_recording
        
    def save_stats_to_csv(self, filepath=None):
        """Save statistics from all cameras to a CSV file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(os.path.expanduser("~"), "Desktop", f"gelsight_stats_{timestamp}.csv")
        
        try:
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'camera_id', 'frame_count', 'diff_mean', 'diff_max', 'fps']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                
                # Collect all stats from all cameras
                all_stats = []
                for camera_id, camera_thread in self.camera_threads.items():
                    stats_history = camera_thread.get_stats_history()
                    for stat in stats_history:
                        # Convert datetime to string
                        stat_copy = stat.copy()
                        stat_copy['timestamp'] = stat['timestamp'].strftime("%Y-%m-%d %H:%M:%S.%f")
                        all_stats.append(stat_copy)
                
                # Sort by timestamp
                all_stats.sort(key=lambda x: x['timestamp'])
                
                # Write all stats
                for stat in all_stats:
                    writer.writerow(stat)
            
            print(f"Statistics saved to {filepath}")
            return True
        except Exception as e:
            print(f"Failed to save statistics: {e}")
            return False


class DualView2x2LiveViewWidget(BoxLayout):
    def __init__(self, main_app, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.main_app = main_app
        
        # Difference visualization parameters
        self.difference_threshold = 30
        self.difference_scale = 2.0
        
        # Status bar
        self.status_bar = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(35),
                                   padding=[dp(10), 0, dp(10), 0])
        
        self.status_labels = []
        for gelsight_id in range(1, 5):
            device_info = GELSIGHT_DEVICE_MAP.get(gelsight_id, {})
            display_name = device_info.get("name", f"GelSight {gelsight_id}")
            
            status_label = Label(
                text=f"{display_name}: Initializing...",
                size_hint_x=0.25,
                font_size='10sp',
                color=(0.6, 0.6, 0.6, 1)
            )
            self.status_labels.append(status_label)
            self.status_bar.add_widget(status_label)
        
        self.add_widget(self.status_bar)
        
        # Mode indicator
        mode_container = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(30),
                                  padding=[dp(10), 0, dp(10), 0])
        
        self.mode_label = Label(
            text="Display: RGB | Difference (2x2 Grid) - Threaded Version",
            size_hint_x=1.0,
            font_size='11sp',
            color=(0.2, 0.6, 0.8, 1),
            bold=True
        )
        mode_container.add_widget(self.mode_label)
        self.add_widget(mode_container)
        
        # Create 2x2 grid for cameras
        self.grid_layout = BoxLayout(orientation="vertical", size_hint_y=0.8, spacing=0)
        
        # First row: Camera 1 and 2
        row1 = BoxLayout(orientation="horizontal", size_hint_y=0.5, spacing=0)
        
        # Camera 1 container (RGB + Diff)
        cam1_container = BoxLayout(orientation="horizontal", size_hint_x=0.5, spacing=0)
        self.cam1_rgb = Image()
        self.cam1_diff = Image()
        cam1_container.add_widget(self.cam1_rgb)
        cam1_container.add_widget(self.cam1_diff)
        row1.add_widget(cam1_container)
        
        # Camera 2 container (RGB + Diff)
        cam2_container = BoxLayout(orientation="horizontal", size_hint_x=0.5, spacing=0)
        self.cam2_rgb = Image()
        self.cam2_diff = Image()
        cam2_container.add_widget(self.cam2_rgb)
        cam2_container.add_widget(self.cam2_diff)
        row1.add_widget(cam2_container)
        
        self.grid_layout.add_widget(row1)
        
        # Second row: Camera 3 and 4
        row2 = BoxLayout(orientation="horizontal", size_hint_y=0.5, spacing=0)
        
        # Camera 3 container (RGB + Diff)
        cam3_container = BoxLayout(orientation="horizontal", size_hint_x=0.5, spacing=0)
        self.cam3_rgb = Image()
        self.cam3_diff = Image()
        cam3_container.add_widget(self.cam3_rgb)
        cam3_container.add_widget(self.cam3_diff)
        row2.add_widget(cam3_container)
        
        # Camera 4 container (RGB + Diff)
        cam4_container = BoxLayout(orientation="horizontal", size_hint_x=0.5, spacing=0)
        self.cam4_rgb = Image()
        self.cam4_diff = Image()
        cam4_container.add_widget(self.cam4_rgb)
        cam4_container.add_widget(self.cam4_diff)
        row2.add_widget(cam4_container)
        
        self.grid_layout.add_widget(row2)
        
        # Store image widgets in a list for easy access
        self.rgb_widgets = [self.cam1_rgb, self.cam2_rgb, self.cam3_rgb, self.cam4_rgb]
        self.diff_widgets = [self.cam1_diff, self.cam2_diff, self.cam3_diff, self.cam4_diff]
        
        self.add_widget(self.grid_layout)
        
        # Control panel
        control_panel = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(50),
                                 spacing=dp(10), padding=[dp(10), dp(5), dp(10), dp(5)])
        
        # Threshold control
        diff_label = Label(
            text="Diff Threshold:",
            size_hint_x=0.15,
            font_size='10sp'
        )
        control_panel.add_widget(diff_label)
        
        self.diff_slider = Slider(
            min=1, max=100, value=self.difference_threshold,
            size_hint_x=0.3
        )
        self.diff_slider.bind(value=self.on_diff_threshold_change)
        control_panel.add_widget(self.diff_slider)
        
        self.threshold_label = Label(
            text=f"{self.difference_threshold}",
            size_hint_x=0.1,
            font_size='11sp',
            color=(0.8, 0.6, 0.2, 1)
        )
        control_panel.add_widget(self.threshold_label)
        
        # Scale control
        scale_label = Label(
            text="Scale:",
            size_hint_x=0.1,
            font_size='10sp'
        )
        control_panel.add_widget(scale_label)
        
        self.scale_slider = Slider(
            min=0.5, max=5.0, value=self.difference_scale,
            size_hint_x=0.2
        )
        self.scale_slider.bind(value=self.on_scale_change)
        control_panel.add_widget(self.scale_slider)
        
        self.scale_label = Label(
            text=f"{self.difference_scale:.1f}",
            size_hint_x=0.1,
            font_size='11sp',
            color=(0.6, 0.8, 0.2, 1)
        )
        control_panel.add_widget(self.scale_label)
        
        # Reference status
        self.ref_status_label = Label(
            text="Waiting for stable reference...",
            size_hint_x=0.25,
            font_size='11sp',
            color=(0.8, 0.6, 0.2, 1),
            bold=True
        )
        control_panel.add_widget(self.ref_status_label)
        
        # Set reference button
        self.set_ref_btn = Button(
            text="Set Reference Now",
            size_hint_x=0.15,
            background_color=(0.2, 0.6, 0.8, 1)
        )
        self.set_ref_btn.bind(on_press=self.set_reference_frames)
        control_panel.add_widget(self.set_ref_btn)
        
        # Reset reference button
        self.reset_ref_btn = Button(
            text="Reset Reference",
            size_hint_x=0.15,
            background_color=(0.3, 0.5, 0.3, 1)
        )
        self.reset_ref_btn.bind(on_press=self.reset_reference_frames)
        control_panel.add_widget(self.reset_ref_btn)
        
        self.add_widget(control_panel)
        
        # Statistics panel
        stats_panel = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(60),
                               spacing=dp(5), padding=[dp(10), dp(2), dp(10), dp(2)])
        
        self.stats_labels = []
        for cam_idx in range(4):
            stats_box = BoxLayout(orientation="vertical", size_hint_x=0.25, spacing=dp(2))
            
            cam_label = Label(
                text=f"Cam {cam_idx+1}",
                size_hint_y=None,
                height=dp(20),
                font_size='10sp',
                color=(0.1, 0.3, 0.6, 1),
                bold=True
            )
            stats_box.add_widget(cam_label)
            
            # Reference status
            ref_label = Label(
                text="Ref: Waiting",
                size_hint_y=None,
                height=dp(15),
                font_size='9sp',
                color=(0.8, 0.2, 0.2, 1)
            )
            stats_box.add_widget(ref_label)
            
            # Diff stats
            diff_label = Label(
                text="Diff: N/A",
                size_hint_y=None,
                height=dp(15),
                font_size='9sp',
                color=(0.4, 0.4, 0.4, 1)
            )
            stats_box.add_widget(diff_label)
            
            # Frame counter
            frame_label = Label(
                text="FPS: 0",
                size_hint_y=None,
                height=dp(15),
                font_size='9sp',
                color=(0.4, 0.4, 0.4, 1)
            )
            stats_box.add_widget(frame_label)
            
            stats_panel.add_widget(stats_box)
            self.stats_labels.append({
                'ref': ref_label,
                'diff': diff_label,
                'frame': frame_label
            })
        
        self.add_widget(stats_panel)
        
        # Bottom controls
        bottom_panel = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(40))
        
        # Zoom control
        zoom_box = BoxLayout(orientation="horizontal", size_hint_x=0.3)
        zoom_box.add_widget(Label(text="Zoom:", size_hint_x=0.3))
        self.zoom_slider = Slider(min=0.5, max=3.0, value=1.0, size_hint_x=0.5)
        self.zoom_slider.bind(value=self.on_zoom_value_change)
        zoom_box.add_widget(self.zoom_slider)
        self.zoom_label = Label(text="1.0x", size_hint_x=0.2)
        zoom_box.add_widget(self.zoom_label)
        bottom_panel.add_widget(zoom_box)
        
        # Action buttons
        action_box = BoxLayout(orientation="horizontal", size_hint_x=0.7, spacing=dp(5))
        
        # Recording button
        self.recording_btn = Button(
            text="Start Recording",
            background_color=(0.8, 0.2, 0.2, 1)  # Red for start
        )
        self.recording_btn.bind(on_press=self.toggle_recording)
        action_box.add_widget(self.recording_btn)
        
        # Screenshot button
        self.screenshot_btn = Button(
            text="Save Screenshots",
            background_color=(0.2, 0.6, 0.2, 1)
        )
        self.screenshot_btn.bind(on_press=self.take_screenshots)
        action_box.add_widget(self.screenshot_btn)
        
        # Save Stats button
        self.save_stats_btn = Button(
            text="Save Stats",
            background_color=(0.3, 0.6, 0.3, 1)
        )
        self.save_stats_btn.bind(on_press=self.save_difference_stats)
        action_box.add_widget(self.save_stats_btn)
        
        bottom_panel.add_widget(action_box)
        
        self.add_widget(bottom_panel)
        
        # Initialization
        self.screenshot_folder = os.path.join(os.path.expanduser("~"), "Desktop")
        self.event = None
        self.start_time = time.time()
        
        # Update timer for UI refresh rate
        self.ui_refresh_rate = 30  # Hz
        
        # Recording status
        self.recording = False
        
        # Recording name popup
        self.recording_popup = None
        self.recording_popup_open = False
        
        Window.bind(on_key_down=self.on_key_down)

    def on_diff_threshold_change(self, instance, value):
        self.difference_threshold = int(value)
        self.threshold_label.text = str(int(value))
        if self.main_app.camera_manager:
            self.main_app.camera_manager.update_params_for_all(
                self.difference_threshold,
                self.difference_scale,
                self.zoom_slider.value
            )

    def on_scale_change(self, instance, value):
        self.difference_scale = value
        self.scale_label.text = f"{value:.1f}"
        if self.main_app.camera_manager:
            self.main_app.camera_manager.update_params_for_all(
                self.difference_threshold,
                self.difference_scale,
                self.zoom_slider.value
            )

    def on_zoom_value_change(self, instance, value):
        self.zoom_label.text = f"{value:.1f}x"
        if self.main_app.camera_manager:
            self.main_app.camera_manager.update_params_for_all(
                self.difference_threshold,
                self.difference_scale,
                value
            )

    def start(self):
        if not self.event:
            self.event = Clock.schedule_interval(self.update, 1 / self.ui_refresh_rate)

    def stop(self):
        if self.event:
            self.event.cancel()
            self.event = None
        
        # Stop camera threads
        if self.main_app.camera_manager:
            self.main_app.camera_manager.stop_cameras()

    def set_reference_frames(self, instance):
        """Set reference frames to current frames"""
        if self.main_app.camera_manager:
            self.main_app.camera_manager.set_reference_for_all()
            
        # Update status
        self.ref_status_label.text = "Setting reference..."
        self.ref_status_label.color = (0.8, 0.8, 0.2, 1)

    def reset_reference_frames(self, instance):
        """Reset reference frames"""
        if self.main_app.camera_manager:
            self.main_app.camera_manager.reset_reference_for_all()
            
        # Update status
        self.ref_status_label.text = "References reset - Set new reference"
        self.ref_status_label.color = (0.8, 0.6, 0.2, 1)

    def toggle_recording(self, instance):
        """Toggle recording for all cameras"""
        if not self.main_app.camera_manager:
            return
            
        if self.recording:
            # Show popup to ask for recording name
            self.show_recording_name_popup()
        else:
            # Start recording
            recording_folder = self.main_app.camera_manager.start_recording_for_all()
            if recording_folder:
                self.recording_btn.text = "Stop Recording"
                self.recording_btn.background_color = (0.2, 0.8, 0.2, 1)  # Green
                self.recording = True
                print(f"Recording started in folder: {recording_folder}")

    def show_recording_name_popup(self):
        """Show popup to ask for recording name"""
        # Prevent multiple popups
        if self.recording_popup_open:
            return
            
        self.recording_popup_open = True
        self.recording_popup = RecordingNamePopup(
            on_name_selected_callback=self.on_recording_name_selected,
            on_cancel_callback=self.on_recording_popup_cancelled
        )
        self.recording_popup.bind(on_dismiss=self.on_recording_popup_dismissed)
        self.recording_popup.open()

    def on_recording_popup_dismissed(self, instance):
        """Handle popup dismissal"""
        self.recording_popup_open = False
        self.recording_popup = None

    def on_recording_popup_cancelled(self):
        """Handle popup cancellation"""
        # User cancelled, don't stop recording
        self.recording_popup_open = False
        self.recording_popup = None
        print("Recording rename cancelled, continuing recording...")

    def on_recording_name_selected(self, recording_name):
        """Handle recording name selection"""
        self.recording_popup_open = False
        
        # Stop recording with the provided name
        recording_folder = self.main_app.camera_manager.stop_recording_for_all(recording_name)
        self.recording_btn.text = "Start Recording"
        self.recording_btn.background_color = (0.8, 0.2, 0.2, 1)  # Red
        self.recording = False
        
        if recording_name and recording_name.strip():
            print(f"Recording saved as: '{recording_name}' in {recording_folder}")
            self.ref_status_label.text = f"Recording saved: {recording_name[:20]}..."
            self.ref_status_label.color = (0.2, 0.8, 0.2, 1)
        else:
            print(f"Recording saved in folder: {recording_folder}")
            self.ref_status_label.text = "Recording saved ✓"
            self.ref_status_label.color = (0.2, 0.8, 0.2, 1)
        
        # Reset status after 3 seconds
        Clock.schedule_once(lambda dt: self.reset_recording_status(), 3.0)

    def reset_recording_status(self):
        """Reset recording status label"""
        self.ref_status_label.text = "Ready"
        self.ref_status_label.color = (0.8, 0.6, 0.2, 1)

    def update_texture_from_rgb(self, image_widget, rgb_frame):
        """Update Kivy image texture from RGB numpy array"""
        if rgb_frame is None:
            return
            
        try:
            # Create texture from RGB frame
            texture = Texture.create(
                size=(rgb_frame.shape[1], rgb_frame.shape[0]), 
                colorfmt="rgb"
            )
            texture.blit_buffer(rgb_frame.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
            texture.flip_vertical()
            image_widget.texture = texture
        except Exception as e:
            # Silently handle texture creation errors
            pass

    def update(self, dt):
        current_time = time.time()
        
        # Check if we have camera manager
        if not self.main_app.camera_manager:
            return
            
        # Check recording status
        is_recording = self.main_app.camera_manager.is_any_recording()
        
        # Update UI based on recording state
        if is_recording != self.recording:
            self.recording = is_recording
            if is_recording:
                self.recording_btn.text = "Stop Recording"
                self.recording_btn.background_color = (0.2, 0.8, 0.2, 1)  # Green
            else:
                self.recording_btn.text = "Start Recording"
                self.recording_btn.background_color = (0.8, 0.2, 0.2, 1)  # Red
            
        # Update all cameras
        refs_set = 0
        for camera_idx in range(4):
            camera_thread = self.main_app.camera_manager.get_camera_thread(camera_idx + 1)
            if not camera_thread:
                continue
                
            # Get frames from camera thread (already in RGB format)
            rgb_frame = camera_thread.get_current_frame_rgb()
            diff_frame = camera_thread.get_current_diff_rgb()
            stats = camera_thread.get_stats()
            
            # Update textures if frames are available
            if rgb_frame is not None:
                self.update_texture_from_rgb(self.rgb_widgets[camera_idx], rgb_frame)
                
            if diff_frame is not None:
                self.update_texture_from_rgb(self.diff_widgets[camera_idx], diff_frame)
            else:
                # Show waiting message if no difference frame
                waiting_frame = np.zeros((self.main_app.camera_height, self.main_app.camera_width, 3), dtype=np.uint8)
                message = "Waiting for reference"
                color = (255, 255, 255)  # White in RGB
                
                # Center text
                text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = (waiting_frame.shape[1] - text_size[0]) // 2
                text_y = (waiting_frame.shape[0] + text_size[1]) // 2
                
                cv2.putText(waiting_frame, message, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                self.update_texture_from_rgb(self.diff_widgets[camera_idx], waiting_frame)
            
            # Update status label
            device_info = GELSIGHT_DEVICE_MAP.get(camera_idx + 1, {})
            display_name = device_info.get("name", f"GelSight {camera_idx + 1}")
            video_idx = self.main_app.video_device_map.get(camera_idx + 1, "?")
            
            status_text = f"{display_name}: /dev/video{video_idx}"
            if stats['has_reference']:
                status_text += " ✓"
                self.status_labels[camera_idx].color = (0.2, 0.8, 0.2, 1)
                refs_set += 1
            else:
                status_text += " (no ref)"
                self.status_labels[camera_idx].color = (0.8, 0.6, 0.2, 1)
            
            self.status_labels[camera_idx].text = status_text
            
            # Update statistics labels
            if stats['has_reference']:
                self.stats_labels[camera_idx]['ref'].text = "Ref: Set ✓"
                self.stats_labels[camera_idx]['ref'].color = (0.2, 0.8, 0.2, 1)
                
                self.stats_labels[camera_idx]['diff'].text = f"Diff: {stats['diff_mean']:.1f}"
                self.stats_labels[camera_idx]['diff'].color = (0.2, 0.6, 0.2, 1)
            else:
                self.stats_labels[camera_idx]['ref'].text = "Ref: Not set"
                self.stats_labels[camera_idx]['ref'].color = (0.8, 0.2, 0.2, 1)
                
                self.stats_labels[camera_idx]['diff'].text = "Diff: N/A"
                self.stats_labels[camera_idx]['diff'].color = (0.6, 0.6, 0.6, 1)
            
            # Update FPS
            self.stats_labels[camera_idx]['frame'].text = f"FPS: {stats['fps']:.0f}"
        
        # Update reference status
        if refs_set == 4:
            self.ref_status_label.text = "All references set ✓"
            self.ref_status_label.color = (0.2, 0.8, 0.2, 1)
        elif refs_set > 0:
            self.ref_status_label.text = f"References: {refs_set}/4 set"
            self.ref_status_label.color = (0.8, 0.8, 0.2, 1)
        else:
            self.ref_status_label.text = "Waiting for stable reference..."
            self.ref_status_label.color = (0.8, 0.6, 0.2, 1)

    def take_screenshots(self, instance=None):
        """Take screenshots from all cameras"""
        if not self.main_app.camera_manager:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_folder = os.path.join(self.screenshot_folder, f"gelsight_screenshots_{timestamp}")
        os.makedirs(screenshot_folder, exist_ok=True)
        
        for camera_idx in range(4):
            camera_thread = self.main_app.camera_manager.get_camera_thread(camera_idx + 1)
            if camera_thread:
                rgb_frame = camera_thread.get_current_frame_rgb()
                diff_frame = camera_thread.get_current_diff_rgb()
                
                if rgb_frame is not None:
                    # Save RGB frame (convert to BGR for OpenCV save)
                    rgb_path = os.path.join(screenshot_folder, f"camera_{camera_idx + 1}_rgb.png")
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(rgb_path, bgr_frame)
                    
                if diff_frame is not None:
                    # Save diff frame (already in RGB)
                    diff_path = os.path.join(screenshot_folder, f"camera_{camera_idx + 1}_diff.png")
                    # Convert to BGR for saving
                    bgr_diff = cv2.cvtColor(diff_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(diff_path, bgr_diff)
        
        print(f"Screenshots saved to {screenshot_folder}")

    def save_difference_stats(self, instance):
        """Save difference statistics to a CSV file"""
        if not self.main_app.camera_manager:
            return
            
        success = self.main_app.camera_manager.save_stats_to_csv()
        if success:
            self.ref_status_label.text = "Statistics saved ✓"
            self.ref_status_label.color = (0.2, 0.8, 0.2, 1)
            # Reset after 2 seconds
            Clock.schedule_once(lambda dt: self.reset_status_label(), 2.0)

    def reset_status_label(self):
        """Reset the status label to default"""
        self.ref_status_label.text = "Ready"
        self.ref_status_label.color = (0.8, 0.6, 0.2, 1)

    def on_key_down(self, window, key, *args):
        if key == 32:  # Space key
            self.take_screenshots()
        elif key == 100:  # 'd' key
            self.save_difference_stats(None)
        elif key == 114:  # 'r' key
            self.reset_reference_frames(None)
        elif key == 115:  # 's' key
            self.set_reference_frames(None)
        elif key == 116:  # 't' key for recording toggle
            self.toggle_recording(None)
        elif key == 113:  # 'q' key
            self.stop()
            Window.close()


# Now define the actual QuadGelsightMini class with its full implementation
class QuadGelsightMini(App):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        
        # Use default indices
        self.auto_mapping = auto_detect_gelsight_video_devices()
        
        # Check if all 4 cameras have default indices
        self.all_cameras_detected = len(self.auto_mapping) == 4
        
        # Camera manager for threading
        self.camera_manager = None
        
        # Store video device mapping
        self.video_device_map = {}
        
        # Flag to track if we should auto-start
        self.auto_start_scheduled = False
        
        # Default camera parameters
        self.camera_width = 640
        self.camera_height = 480
        self.border_fraction = 0.1
        
        # If config is provided, try to extract parameters
        if config:
            try:
                # Try to access as ConfigParser object
                self.camera_width = config.getint('camera', 'width')
                self.camera_height = config.getint('camera', 'height')
                self.border_fraction = config.getfloat('camera', 'border_fraction')
            except (AttributeError, KeyError, ValueError):
                # Try to access as object with attributes
                try:
                    self.camera_width = config.camera_width
                    self.camera_height = config.camera_height
                    self.border_fraction = config.border_fraction
                except AttributeError:
                    # Use defaults
                    pass

    def build(self):
        self.title = "Gelsight Mini Quad Viewer - Threaded Version"
        self.loading_overlay = None
        root = BoxLayout(orientation="vertical")
        
        # Create custom top bar for 4 cameras with default indices
        self.top_bar = QuadTopBar(
            on_device_selected_callback=self.on_quad_device_selected,
            auto_mapping=self.auto_mapping,
            all_cameras_detected=self.all_cameras_detected
        )
        root.add_widget(self.top_bar)
        
        self.live_view = DualView2x2LiveViewWidget(main_app=self)
        root.add_widget(self.live_view)
        
        # If all cameras have default indices, schedule auto-start
        if self.all_cameras_detected:
            Clock.schedule_once(lambda dt: self.auto_start_cameras(), 0.5)
        
        return root
    
    def auto_start_cameras(self):
        """Automatically start all cameras with default indices"""
        if not self.all_cameras_detected or self.auto_start_scheduled:
            return
            
        self.auto_start_scheduled = True
        
        # Use the default mapping
        device_mapping = self.auto_mapping
        self.on_quad_device_selected(device_mapping)

    def show_overlay(self, message):
        if not self.loading_overlay:
            self.loading_overlay = ConnectingOverlay(message=message)
            self.loading_overlay.open()

    def hide_overlay(self):
        if self.loading_overlay:
            self.loading_overlay.dismiss()
            self.loading_overlay = None

    def on_quad_device_selected(self, device_mapping):
        """
        device_mapping is a dict: {gelsight_id: video_device_index, ...}
        """
        if isinstance(device_mapping, dict) and len(device_mapping) == 4:
            # Stop existing cameras if any
            if self.camera_manager:
                self.camera_manager.stop_cameras()
                self.camera_manager = None
            
            # Store the mapping for reference
            self.video_device_map = device_mapping.copy()
            
            # Check if all values are integers
            valid_mapping = True
            for gelsight_id, video_idx in device_mapping.items():
                if not isinstance(video_idx, int) or video_idx < 0:
                    valid_mapping = False
                    break
            
            if valid_mapping:
                self.restart_camera_streams(device_mapping)
        else:
            log_message(f"Error: Expected device mapping dict, got {device_mapping}")

    def restart_camera_streams(self, device_mapping):
        # Show loading overlay
        self.show_overlay("Starting camera threads...")
        
        # Create and start camera manager with appropriate parameters
        self.camera_manager = CameraManager(
            width=self.camera_width,
            height=self.camera_height,
            border_fraction=self.border_fraction,
            device_mapping=device_mapping
        )
        self.camera_manager.start_cameras()
        
        Clock.schedule_once(lambda dt: self.finish_device_selection(), 0.5)

    def finish_device_selection(self):
        self.hide_overlay()
        self.live_view.start()

    def on_stop(self):
        """Clean up when app stops"""
        if self.camera_manager:
            self.camera_manager.stop_cameras()
        return super().on_stop()


def auto_detect_gelsight_video_devices():
    """
    Auto-detect using default indices [5, 3, 7, 0]
    Returns a dictionary: {gelsight_id: video_index, ...}
    """
    auto_mapping = {}
    
    # Use default indices from GELSIGHT_DEVICE_MAP
    for gelsight_id, info in GELSIGHT_DEVICE_MAP.items():
        default_idx = info.get("default_idx", -1)
        if default_idx >= 0:
            auto_mapping[gelsight_id] = default_idx
    
    return auto_mapping


if __name__ == "__main__":
    import argparse
    from config import GSConfig

    parser = argparse.ArgumentParser(
        description="Run the GelSight Mini Quad Viewer with threading for better FPS."
    )
    parser.add_argument(
        "--gs-config",
        type=str,
        default=None,
        help="Path to the JSON configuration file. If not provided, default config is used.",
    )
    
    # Display startup info
    print("\n" + "="*60)
    print("GelSight Quad Viewer - Threaded Version")
    print("="*60)
    print("Each camera runs in its own thread for better performance")
    print("Default camera indices: [5, 3, 7, 0]")
    print("\nKEYBOARD SHORTCUTS:")
    print("  SPACE: Save screenshots")
    print("  R: Reset reference")
    print("  S: Set reference")
    print("  T: Toggle recording (Start/Stop)")
    print("  D: Save difference statistics")
    print("  Q: Quit application")
    print("="*60 + "\n")
    
    args = parser.parse_args()
    
    # Load config if provided
    config_obj = None
    if args.gs_config:
        try:
            gs_config = GSConfig(args.gs_config)
            config_obj = gs_config.config
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using default configuration")
    
    # Add error handling for the main application
    try:
        app = QuadGelsightMini(config=config_obj)
        app.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
