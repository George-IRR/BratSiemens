"""Streamlit UI for visualizing detection results."""

import streamlit as st
import requests
import time
from threading import Thread
import sys
import os
import base64

# Ensure we can import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from image_utils import ImageProcessor
from config import FLASK_PORT


class StreamlitUI:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.flask_url = f"http://localhost:{FLASK_PORT}"
        self.server_running = False

    def check_server_status(self):
        """Check if Flask server is running."""
        try:
            response = requests.get(f"{self.flask_url}/status", timeout=2)
            return response.status_code == 200
        except:
            return False

    def setup_page(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="BLE Detection Monitor",
            page_icon="🔍",
            layout="wide"
        )
        
        st.title("🔍 BLE Detection Monitor")
        st.markdown("Real-time object detection with BLE communication")
        st.markdown(f"**Flask Server:** `{self.flask_url}`")

    def check_flask_server(self):
        """Check Flask server status and update UI."""
        if self.check_server_status():
            self.server_running = True
            st.success(f"✅ Connected to Flask server at {self.flask_url}")
        else:
            self.server_running = False
            st.error(f"❌ Cannot connect to Flask server at {self.flask_url}")
            st.info("💡 Make sure the Flask server is running on port 5001")

    def fetch_detection_data(self):
        """Fetch latest detection data from Flask server."""
        if not self.server_running:
            return {}
            
        try:
            response = requests.get(f"{self.flask_url}/get", timeout=3)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.warning("🔄 Connection lost to Flask server")
            self.server_running = False
            return {}
        except requests.exceptions.Timeout:
            st.warning("⏱️ Request timeout")
            return {}
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            return {}

    def fetch_raw_data(self):
        """Fetch latest raw image data from Flask server."""
        if not self.server_running:
            return {}
            
        try:
            response = requests.get(f"{self.flask_url}/get_raw", timeout=3)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.warning("🔄 Connection lost to Flask server")
            self.server_running = False
            return {}
        except requests.exceptions.Timeout:
            st.warning("⏱️ Request timeout")
            return {}
        except Exception as e:
            st.error(f"❌ Error fetching raw data: {e}")
            return {}

    def display_server_status(self):
        """Display server status in sidebar."""
        with st.sidebar:
            st.header("🖥️ Server Status")
            
            # Check server status
            if st.button("🔄 Check Server"):
                self.check_flask_server()
            
            if self.server_running:
                st.success("✅ Flask Server Online")
                try:
                    response = requests.get(f"{self.flask_url}/status", timeout=2)
                    status_data = response.json()
                    
                    # Basic server info
                    st.write(f"**Detection Count:** {status_data.get('detection_count', 0)}")
                    timestamp = status_data.get('timestamp', 0)
                    if timestamp:
                        st.write(f"**Last Update:** {time.ctime(timestamp)}")
                    
                    # BLE Status
                    st.subheader("📡 BLE Status")
                    ble_available = status_data.get('ble_available', False)
                    if ble_available:
                        st.success("✅ BLE Handler Available")
                        ble_status = status_data.get('ble_status', {})
                        
                        # BLE connection status
                        connected = ble_status.get('connected', False)
                        if connected:
                            st.success("🔗 BLE Connected")
                        else:
                            st.warning("⚠️ BLE Disconnected")
                        
                        # Arm status
                        arm_idle = ble_status.get('arm_idle', False)
                        if arm_idle:
                            st.success("🤖 Arm Ready")
                        else:
                            st.warning("🤖 Arm Busy")
                        
                        # Queue status
                        has_queued = ble_status.get('has_queued_payload', False)
                        if has_queued:
                            st.info("📋 Command Queued")
                        else:
                            st.info("📋 No Queued Commands")
                            
                    else:
                        st.error("❌ BLE Handler Not Available")
                    
                    # Check arm ready status
                    try:
                        ready_response = requests.get(f"{self.flask_url}/ready", timeout=2)
                        ready_data = ready_response.json()
                        arm_ready = ready_data.get('ready', False)
                        if arm_ready:
                            st.success("🎯 Ready for Commands")
                        else:
                            st.warning("⏳ Processing Command")
                    except:
                        pass
                        
                except Exception as e:
                    st.error(f"Error getting status: {e}")
            else:
                st.error("❌ Flask Server Offline")
                st.markdown(f"**Expected URL:** {self.flask_url}")
                st.markdown("**Troubleshooting:**")
                st.markdown("1. Check if Flask server is running")
                st.markdown("2. Verify port 5001 is not blocked")
                st.markdown("3. Try restarting the Flask server")

    def display_detection_info(self, data):
        """Display detection information in sidebar."""
        with st.sidebar:
            st.header("📊 Detection Info")
            
            # Check if it's raw image or processed
            is_raw = data.get('raw_image', False)
            
            if is_raw:
                st.info("📸 **RAW IMAGE MODE**")
                st.write("Currently showing raw camera feed")
                # Show image dimensions
                if 'crop_shape' in data:
                    w, h = data['crop_shape']
                    st.metric("Image Size", f"{w}×{h}")
                # Show timestamp
                if 'timestamp' in data:
                    st.write(f"**Timestamp:** {data['timestamp']}")
            else:
                # Normal detection info
                detections = data.get("detections", [])
                st.metric("Detections Count", len(detections))
                
                if detections:
                    st.subheader("🎯 Detected Objects")
                    for i, det in enumerate(detections):
                        with st.expander(f"Object {i+1}: {det.get('class', 'Unknown')}"):
                            st.write(f"**Confidence:** {det.get('confidence', 0):.3f}")
                            if 'center_px' in det:
                                cx, cy = det['center_px']
                                st.write(f"**Position:** ({cx:.0f}, {cy:.0f})")
                            
                            # Show class mapping
                            class_name = det.get('class', 'Unknown')
                            from config import CLASS_ID
                            class_id = CLASS_ID.get(class_name, 'Unknown')
                            st.write(f"**Class ID:** {class_id}")
                
                # Crop shape info
                if 'crop_shape' in data:
                    w, h = data['crop_shape']
                    st.metric("Crop Size", f"{w}×{h}")
                    
            # Always show timestamp if available
            if 'timestamp' in data and not is_raw:
                st.write(f"**Timestamp:** {data['timestamp']}")
                
            # Instructions
            st.markdown("---")
            st.markdown("**💡 Instructions:**")
            if is_raw:
                st.markdown("• Type 'raw' in Pi script to disable raw mode")
                st.markdown("• Raw images are not processed by BLE")
            else:
                st.markdown("• Type 'raw' in Pi script to enable raw mode")
                st.markdown("• Processed images are sent to BLE arm")

    def display_controls(self):
        """Display control panel."""
        st.header("🎛️ Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            auto_refresh = st.checkbox("Auto Refresh", value=False)
        
        with col2:
            refresh_rate = st.slider("Refresh Rate (s)", 0.5, 5.0, 1.0)
        
        with col3:
            if st.button("🗑️ Clear Data"):
                if self.server_running:
                    try:
                        requests.post(f"{self.flask_url}/clear", timeout=3)
                        st.success("✅ Data cleared!")
                        time.sleep(1)
                        st.rerun()
                    except:
                        st.error("❌ Failed to clear data")
                else:
                    st.error("❌ Server not connected")
        
        with col4:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        return auto_refresh, refresh_rate

    def run(self):
        """Run the Streamlit application."""
        self.setup_page()
        
        # Initial server check
        self.check_flask_server()
        
        # Server status in sidebar
        self.display_server_status()
        
        # Control panel
        auto_refresh, refresh_rate = self.display_controls()
        
        # Main content area
        st.header("📷 Detection Display")
        
        # Create tabs for different image types
        tab1, tab2 = st.tabs(["🎯 Processed Images", "📸 Raw Images"])
        
        with tab1:
            st.subheader("Processed Images with Detections")
            processed_placeholder = st.empty()
            processed_status = st.empty()
        
        with tab2:
            st.subheader("Raw Camera Images")
            raw_placeholder = st.empty()
            raw_status = st.empty()
        
        # Manual refresh or auto-refresh
        if auto_refresh and self.server_running:
            st.info("🔄 Auto-refresh enabled")
            
            # Create a placeholder for refresh counter
            refresh_counter = st.empty()
            
            # Auto-refresh loop
            for i in range(100):  # Limit to prevent infinite loop
                # Fetch both processed and raw data
                processed_data = self.fetch_detection_data()
                raw_data = self.fetch_raw_data()
                
                # Display processed images
                if processed_data and processed_data.get('detections'):
                    img, status = self.image_processor.process_detection_data(processed_data)
                    if img is not None:
                        processed_placeholder.image(
                            img, 
                            caption="Latest Detection", 
                            use_container_width=True
                        )
                        processed_status.success(f"📊 {status}")
                        # Display detection info in sidebar
                        self.display_detection_info(processed_data)
                    else:
                        processed_placeholder.info("📷 Waiting for processed image data...")
                        processed_status.info("⏳ No processed images")
                else:
                    processed_placeholder.info("� Waiting for detection data...")
                    processed_status.info("⏳ No detections available")
                
                # Display raw images
                if raw_data and raw_data.get('raw_image', False):
                    raw_img, raw_status_msg = self.image_processor.process_detection_data(raw_data)
                    if raw_img is not None:
                        raw_placeholder.image(
                            raw_img, 
                            caption="Raw Camera Feed", 
                            use_container_width=True
                        )
                        raw_status.success(f"📸 {raw_status_msg}")
                        # Display raw info in sidebar when we have raw data
                        self.display_detection_info(raw_data)
                    else:
                        raw_placeholder.info("📷 No raw image data...")
                        raw_status.info("⏳ No raw images")
                else:
                    raw_placeholder.info("📷 Waiting for raw images...")
                    raw_status.info("⏳ Switch Pi to raw mode to see images here")
                
                # Show refresh counter
                refresh_counter.text(f"🔄 Refresh #{i+1} - Next in {refresh_rate}s")
                
                time.sleep(refresh_rate)
        else:
            # Manual refresh mode
            if self.server_running:
                # Fetch both types of data
                processed_data = self.fetch_detection_data()
                raw_data = self.fetch_raw_data()
                
                # Display processed images
                if processed_data and processed_data.get('detections'):
                    img, status = self.image_processor.process_detection_data(processed_data)
                    if img is not None:
                        processed_placeholder.image(
                            img, 
                            caption="Latest Detection", 
                            use_container_width=True
                        )
                        processed_status.success(f"📊 {status}")
                        self.display_detection_info(processed_data)
                    else:
                        processed_placeholder.info("📷 No processed image available")
                        processed_status.info("⏳ No processed images")
                else:
                    processed_placeholder.info("📷 No detection data available")
                    processed_status.info("� Send detection data to Flask server")
                
                # Display raw images
                if raw_data and raw_data.get('raw_image', False):
                    raw_img, raw_status_msg = self.image_processor.process_detection_data(raw_data)
                    if raw_img is not None:
                        raw_placeholder.image(
                            raw_img, 
                            caption="Raw Camera Feed", 
                            use_container_width=True
                        )
                        raw_status.success(f"📸 {raw_status_msg}")
                        # Display raw info in sidebar when we have raw data
                        self.display_detection_info(raw_data)
                    else:
                        raw_placeholder.info("📷 No raw image available")
                        raw_status.info("⏳ No raw images")
                else:
                    raw_placeholder.info("📷 No raw images available")
                    raw_status.info("💡 Type 'raw' in Pi script to enable raw mode")
            else:
                st.warning("⚠️ Please ensure Flask server is running and try refreshing the page")


# Entry point when run directly
if __name__ == "__main__":
    ui = StreamlitUI()
    ui.run()
