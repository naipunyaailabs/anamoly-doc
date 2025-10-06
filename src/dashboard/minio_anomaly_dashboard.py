#!/usr/bin/env python3
"""
MinIO-Enabled Anomaly Detection Dashboard
Two-column layout with anomaly details on left and MinIO screenshots on right
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
from PIL import Image
from pymongo import MongoClient
import base64
from io import BytesIO
from minio_client import MinIOAnomalyStorage
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="MinIO Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .anomaly-card {
        border-left: 5px solid #ff4b4b;
        background-color: #fff5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        cursor: pointer;
    }
    .anomaly-card:hover {
        background-color: #ffecec;
    }
    .anomaly-high { background-color: #f8d7da; color: #721c24; }
    .anomaly-medium { background-color: #fff3cd; color: #856404; }
    .anomaly-low { background-color: #d1ecf1; color: #0c5460; }
    .anomaly-critical { background-color: #f5c6cb; color: #721c24; font-weight: bold; }
    .header {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .section-header {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .anomaly-icon {
        font-size: 1.5rem;
        margin-right: 10px;
    }
    .evidence-section {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .timestamp {
        color: #6c757d;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .corner-badge {
        position: fixed;
        top: 10px;
        right: 10px;
        background-color: #4CAF50;
        color: white;
        padding: 10px 15px;
        border-radius: 20px;
        font-weight: bold;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        z-index: 1000;
    }
    .selected-anomaly {
        border-left: 5px solid #007bff;
        background-color: #e3f2fd;
    }
    .frame-info {
        background-color: #e3f2fd;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Corner badge
st.markdown('<div class="corner-badge">🔍 MinIO Anomaly Detection</div>', unsafe_allow_html=True)

# Session state initialization
if 'selected_anomaly' not in st.session_state:
    st.session_state.selected_anomaly = None
if 'minio_client' not in st.session_state:
    try:
        # Initialize with environment variables or defaults
        st.session_state.minio_client = MinIOAnomalyStorage(
            endpoint="storage.docapture.com",
            access_key="QtvliJKH7cwHqYH6Egk1",
            secret_key="LqOhilkLztbJowpNvQ0rNvJlKjvMKiDFnlFlVpSi",
            secure=True
        )
        st.session_state.minio_client.initialize_bucket()
        st.success("MinIO storage initialized successfully")
    except Exception as e:
        st.warning(f"Could not initialize MinIO storage: {str(e)}")
        # Print more detailed error information
        st.info("MinIO connection details:")
        st.info(f"- Endpoint: {os.getenv('MINIO_ENDPOINT', 'storage.docapture.com')}")
        st.info("- Secure: True")
        st.info("- Region: us-east-1")
        st.session_state.minio_client = None

class MongoDBAnomalyReader:
    def __init__(self, mongo_uri=None, db_name=None, collection_name=None):
        # Use environment variables or defaults
        mongo_uri = mongo_uri or os.getenv("MONGO_URI", "mongodb://admin:wRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c@69.62.83.244:27017/")
        db_name = db_name or os.getenv("DB_NAME", "crowd_db")
        collection_name = collection_name or os.getenv("COLLECTION_NAME", "crowd_anomalies")
        
        self.client = None
        self.db = None
        self.collection = None
        try:
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            # Test connection
            self.client.server_info()
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            print("MongoDB connected successfully")
        except Exception as e:
            st.warning(f"Could not connect to MongoDB: {e}")
            self.client = None
            self.db = None
            self.collection = None
        
    def get_recent_anomalies(self, limit=100):
        """Get recent anomalies from MongoDB"""
        # Check if MongoDB is available
        if self.collection is None:
            return []
        try:
            # Get recent documents sorted by timestamp
            documents = list(self.collection.find().sort("timestamp", -1).limit(limit))
            return documents
        except Exception as e:
            st.error(f"Error reading from MongoDB: {e}")
            return []
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client is not None:
            self.client.close()

def categorize_anomalies(anomaly_log):
    """Categorize anomalies by type"""
    standing_anomalies = []
    phone_anomalies = []
    empty_chair_anomalies = []
    document_anomalies = []
    
    for entry in anomaly_log:
        anomalies = entry.get('anomaly', [])
        person_id = entry.get('person', -1)
        
        if 'standing' in anomalies:
            standing_anomalies.append({
                'person_id': person_id,
                'entry': entry
            })
        if 'phone' in anomalies:
            phone_anomalies.append({
                'person_id': person_id,
                'entry': entry
            })
        if 'empty_chair' in anomalies:
            empty_chair_anomalies.append({
                'person_id': person_id,
                'entry': entry
            })
        if 'unattended_document' in anomalies:
            document_anomalies.append({
                'person_id': person_id,
                'entry': entry
            })
    
    return standing_anomalies, phone_anomalies, empty_chair_anomalies, document_anomalies

def format_timestamp(timestamp):
    """Format timestamp for display"""
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return str(timestamp)

def format_video_time(video_time):
    """Format video time for display"""
    if video_time:
        return str(video_time)
    return "N/A"

def display_anomaly_list(anomalies_data):
    """Display the list of anomalies in the left column"""
    st.subheader("Anomaly List")
    
    if not anomalies_data:
        st.info("No anomalies detected")
        return
    
    # Group anomalies by type
    standing_anomalies = [a for a in anomalies_data if 'standing' in a.get('anomaly_types', [])]
    phone_anomalies = [a for a in anomalies_data if 'phone' in a.get('anomaly_types', [])]
    empty_chair_anomalies = [a for a in anomalies_data if 'empty_chair' in a.get('anomaly_types', [])]
    document_anomalies = [a for a in anomalies_data if 'unattended_document' in a.get('anomaly_types', [])]
    
    # Display tabs for different anomaly types
    tab1, tab2, tab3, tab4 = st.tabs(["🚶 Standing", "📱 Phone Usage", "🪑 Empty Chairs", "📄 Documents"])
    
    with tab1:
        if standing_anomalies:
            for anomaly in standing_anomalies:
                is_selected = (st.session_state.selected_anomaly is not None and 
                              st.session_state.selected_anomaly.get('_id') == anomaly.get('_id'))
                
                card_class = "anomaly-card selected-anomaly" if is_selected else "anomaly-card"
                
                with st.container():
                    st.markdown(f"""
                    <div class="{card_class}" style="cursor: pointer;">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <strong>Frame {anomaly.get('frame_id', 'N/A')}</strong>
                            </div>
                            <div>
                                {format_video_time(anomaly.get('video_time'))}
                            </div>
                        </div>
                        <div style="margin-top: 0.5rem;">
                            Person ID: {anomaly.get('person_id', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Make the card clickable
                    if st.button("", key=f"select_{anomaly.get('_id')}", 
                                help="Click to view details", 
                                use_container_width=True):
                        st.session_state.selected_anomaly = anomaly
                        st.rerun()
        else:
            st.info("No standing anomalies detected")
    
    with tab2:
        if phone_anomalies:
            for anomaly in phone_anomalies:
                is_selected = (st.session_state.selected_anomaly is not None and 
                              st.session_state.selected_anomaly.get('_id') == anomaly.get('_id'))
                
                card_class = "anomaly-card selected-anomaly" if is_selected else "anomaly-card"
                
                with st.container():
                    st.markdown(f"""
                    <div class="{card_class}" style="cursor: pointer;">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <strong>Frame {anomaly.get('frame_id', 'N/A')}</strong>
                            </div>
                            <div>
                                {format_video_time(anomaly.get('video_time'))}
                            </div>
                        </div>
                        <div style="margin-top: 0.5rem;">
                            Person ID: {anomaly.get('person_id', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Make the card clickable
                    if st.button("", key=f"select_{anomaly.get('_id')}_phone", 
                                help="Click to view details", 
                                use_container_width=True):
                        st.session_state.selected_anomaly = anomaly
                        st.rerun()
        else:
            st.info("No phone usage anomalies detected")
    
    with tab3:
        if empty_chair_anomalies:
            for anomaly in empty_chair_anomalies:
                is_selected = (st.session_state.selected_anomaly is not None and 
                              st.session_state.selected_anomaly.get('_id') == anomaly.get('_id'))
                
                card_class = "anomaly-card selected-anomaly" if is_selected else "anomaly-card"
                
                with st.container():
                    st.markdown(f"""
                    <div class="{card_class}" style="cursor: pointer;">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <strong>Frame {anomaly.get('frame_id', 'N/A')}</strong>
                            </div>
                            <div>
                                {format_video_time(anomaly.get('video_time'))}
                            </div>
                        </div>
                        <div style="margin-top: 0.5rem;">
                            Chair Detection
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Make the card clickable
                    if st.button("", key=f"select_{anomaly.get('_id')}_chair", 
                                help="Click to view details", 
                                use_container_width=True):
                        st.session_state.selected_anomaly = anomaly
                        st.rerun()
        else:
            st.info("No empty chair anomalies detected")
            
    with tab4:
        if document_anomalies:
            for anomaly in document_anomalies:
                is_selected = (st.session_state.selected_anomaly is not None and 
                              st.session_state.selected_anomaly.get('_id') == anomaly.get('_id'))
                
                card_class = "anomaly-card selected-anomaly" if is_selected else "anomaly-card"
                
                with st.container():
                    st.markdown(f"""
                    <div class="{card_class}" style="cursor: pointer;">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <strong>Frame {anomaly.get('frame_id', 'N/A')}</strong>
                            </div>
                            <div>
                                {format_video_time(anomaly.get('video_time'))}
                            </div>
                        </div>
                        <div style="margin-top: 0.5rem;">
                            Document Detection
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Make the card clickable
                    if st.button("", key=f"select_{anomaly.get('_id')}_document", 
                                help="Click to view details", 
                                use_container_width=True):
                        st.session_state.selected_anomaly = anomaly
                        st.rerun()
        else:
            st.info("No document anomalies detected")

def display_anomaly_details(anomaly_data):
    """Display detailed information about the selected anomaly in the right column"""
    if not anomaly_data:
        st.info("Select an anomaly from the list to view details")
        return
    
    st.subheader("Anomaly Details")
    
    # Display basic information
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Frame ID", anomaly_data.get('frame_id', 'N/A'))
        st.metric("Person ID", anomaly_data.get('person_id', 'N/A') if anomaly_data.get('person_id') != -1 else 'N/A')
    with col2:
        st.metric("Video Time", format_video_time(anomaly_data.get('video_time')))
        st.metric("Anomaly Types", ', '.join(anomaly_data.get('anomaly_types', [])))
    
    # Display frame information
    st.markdown(f'<div class="frame-info">📍 Frame ID: {anomaly_data.get("frame_id", "N/A")} | 🕐 Video Time: {format_video_time(anomaly_data.get("video_time"))}</div>', unsafe_allow_html=True)
    
    # Display screenshot from MinIO if available
    st.subheader("Evidence Screenshots")
    
    if st.session_state.minio_client is not None and anomaly_data.get('minio_object_names'):
        minio_objects = anomaly_data.get('minio_object_names', [])
        
        if minio_objects:
            # Display all screenshots (not limited to 9)
            cols = st.columns(3)  # Create 3 columns for better display
            for i, obj in enumerate(minio_objects):
                try:
                    with cols[i % 3]:
                        object_name = obj.get('object_name')
                        anomaly_type = obj.get('anomaly_type')
                        
                        # Get presigned URL for the screenshot
                        url = st.session_state.minio_client.get_screenshot_url(object_name)
                        st.image(url, caption=f"{anomaly_type.title()} Anomaly (Frame {anomaly_data.get('frame_id', 'N/A')})", use_container_width=True)
                        
                        # Show frame information
                        st.caption(f"Frame: {anomaly_data.get('frame_id', 'N/A')} | Time: {format_video_time(anomaly_data.get('video_time'))}")
                        
                        # Provide download button
                        st.download_button(
                            label=f"Download {anomaly_type.title()} Screenshot",
                            data=st.session_state.minio_client.get_screenshot_data(object_name),
                            file_name=f"{anomaly_type}_{anomaly_data.get('frame_id')}.jpg",
                            mime="image/jpeg",
                            key=f"download_{object_name}"
                        )
                except Exception as e:
                    st.error(f"Error displaying screenshot: {str(e)}")
        else:
            st.info("No screenshots available in MinIO for this anomaly")
    elif anomaly_data.get('screenshot_data'):
        # Fallback to base64 screenshot if MinIO is not available
        try:
            screenshot_data = anomaly_data.get('screenshot_data')
            if screenshot_data:
                # Decode base64 image
                image_data = base64.b64decode(screenshot_data)
                image = Image.open(BytesIO(image_data))
                st.image(image, caption="Anomaly Screenshot", use_container_width=True)
                
                # Show frame information
                st.caption(f"Frame: {anomaly_data.get('frame_id', 'N/A')} | Time: {format_video_time(anomaly_data.get('video_time'))}")
                
                # Provide download button
                st.download_button(
                    label="Download Screenshot",
                    data=image_data,
                    file_name=f"anomaly_{anomaly_data.get('frame_id')}.jpg",
                    mime="image/jpeg"
                )
            else:
                st.info("No screenshot available for this anomaly")
        except Exception as e:
            st.error(f"Error displaying screenshot: {str(e)}")
    else:
        st.info("No evidence screenshots available for this anomaly")

def display_all_minio_images():
    """Display all images from MinIO"""
    st.subheader("All MinIO Images")
    
    if st.session_state.minio_client is None:
        st.info("MinIO client not available")
        return
    
    try:
        # List all screenshots from MinIO
        all_screenshots = []
        
        # Get screenshots for each anomaly type
        for anomaly_type in ["standing", "phone", "empty_chair", "unattended_document"]:
            try:
                screenshots = st.session_state.minio_client.list_screenshots_by_anomaly(anomaly_type)
                for screenshot in screenshots:
                    all_screenshots.append({
                        "object_name": screenshot,
                        "anomaly_type": anomaly_type
                    })
            except Exception as e:
                st.warning(f"Could not list {anomaly_type} screenshots: {str(e)}")
        
        if not all_screenshots:
            st.info("No screenshots found in MinIO")
            return
        
        # Display all screenshots
        st.info(f"Found {len(all_screenshots)} screenshots in MinIO")
        
        # Create columns for display
        cols = st.columns(4)  # 4 columns for better display
        
        for i, screenshot in enumerate(all_screenshots):
            try:
                with cols[i % 4]:
                    object_name = screenshot["object_name"]
                    anomaly_type = screenshot["anomaly_type"]
                    
                    # Extract frame info from object name if possible
                    # Object names are in format: anomaly_type/frame_timestamp_uniqueid.jpg
                    parts = object_name.split("/")
                    if len(parts) > 1:
                        filename = parts[1]
                        frame_info = filename.split("_")[0] if "_" in filename else "N/A"
                    else:
                        frame_info = "N/A"
                    
                    # Get presigned URL for the screenshot
                    url = st.session_state.minio_client.get_screenshot_url(object_name)
                    st.image(url, caption=f"{anomaly_type.title()} (Frame {frame_info})", use_container_width=True)
                    
                    # Show frame information
                    st.caption(f"Frame: {frame_info} | Type: {anomaly_type}")
                    
                    # Provide download button
                    st.download_button(
                        label=f"Download",
                        data=st.session_state.minio_client.get_screenshot_data(object_name),
                        file_name=f"{anomaly_type}_{frame_info}.jpg",
                        mime="image/jpeg",
                        key=f"download_all_{object_name}"
                    )
            except Exception as e:
                st.error(f"Error displaying screenshot {screenshot['object_name']}: {str(e)}")
                
    except Exception as e:
        st.error(f"Error retrieving MinIO images: {str(e)}")

def main():
    """Main Streamlit application"""
    st.markdown('<div class="header"><h1>🔍 MinIO Anomaly Detection Dashboard</h1><p>Two-column layout with MinIO storage integration</p></div>', unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        st.header("⚙️ MongoDB Settings")
        mongo_uri = st.text_input("MongoDB URI", os.getenv("MONGO_URI", "mongodb://admin:wRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c@69.62.83.244:27017/"), 
                                 help="MongoDB connection string")
        db_name = st.text_input("Database Name", os.getenv("DB_NAME", "crowd_db"), 
                               help="MongoDB database name")
        collection_name = st.text_input("Collection Name", os.getenv("COLLECTION_NAME", "crowd_anomalies"), 
                                       help="MongoDB collection name")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state.selected_anomaly = None
            st.rerun()
        
        # Add a button to show all MinIO images
        if st.button("🖼️ Show All MinIO Images", use_container_width=True):
            st.session_state.show_all_minio = not st.session_state.get('show_all_minio', False)
            st.rerun()
    
    # Show all MinIO images if requested
    if st.session_state.get('show_all_minio', False):
        display_all_minio_images()
        st.divider()
    
    # Main content area with two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Initialize MongoDB reader
        try:
            mongo_reader = MongoDBAnomalyReader(mongo_uri, db_name, collection_name)
            
            # Get recent anomalies
            recent_anomalies = mongo_reader.get_recent_anomalies(100)
            
            if not recent_anomalies:
                st.info("No anomaly data available in MongoDB.")
                mongo_reader.close_connection()
                return
            
            # Process anomalies for display
            anomalies_data = []
            for doc in recent_anomalies:
                anomaly_log = doc.get('anomaly_log', [])
                
                # Extract individual anomalies
                for entry in anomaly_log:
                    anomalies = entry.get('anomaly', [])
                    person_id = entry.get('person', -1)
                    
                    anomalies_data.append({
                        '_id': str(doc.get('_id')),
                        'frame_id': doc.get('frame_id'),
                        'timestamp': doc.get('timestamp'),
                        'video_time': doc.get('video_time'),  # Add video time
                        'person_id': person_id,
                        'anomaly_types': anomalies,
                        'anomaly_log_entry': entry,
                        'screenshot_data': doc.get('screenshot_data', ''),  # This will be empty now
                        'minio_object_names': doc.get('minio_object_names', [])
                    })
            
            # Display anomaly list
            display_anomaly_list(anomalies_data)
            
            mongo_reader.close_connection()
            
        except Exception as e:
            st.error(f"Error connecting to MongoDB: {e}")
            st.info("Make sure MongoDB is running and accessible at the provided URI")
    
    with col2:
        # Display details of selected anomaly
        display_anomaly_details(st.session_state.selected_anomaly)

if __name__ == "__main__":
    main()