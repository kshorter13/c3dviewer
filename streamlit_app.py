import streamlit as st
import numpy as np
import pandas as pd
import io
import traceback

# Try to import plotly with error handling
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError as e:
    PLOTLY_AVAILABLE = False
    st.error(f"Plotly import failed: {str(e)}")
    st.error("Please ensure 'plotly' is in your requirements.txt file")
    st.stop()  # Stop execution if plotly is not available

# Try to import C3D library - if not available, we'll handle it gracefully
try:
    import c3d
    C3D_AVAILABLE = True
except ImportError:
    C3D_AVAILABLE = False
    st.sidebar.warning("⚠️ C3D library not available. Using demo data only.")
except Exception as e:
    C3D_AVAILABLE = False
    st.sidebar.error(f"Error importing C3D library: {str(e)}")

def calculate_angle_3d(p1, p2, p3):
    """
    Calculate angle at p2 formed by points p1-p2-p3 in 3D space
    Returns angle in degrees
    """
    # Create vectors
    v1 = p1 - p2  # Vector from p2 to p1
    v2 = p3 - p2  # Vector from p2 to p3
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Clamp to [-1, 1] to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def load_c3d_file(uploaded_file):
    """
    Load C3D file and extract marker data
    """
    try:
        if not C3D_AVAILABLE:
            st.error("C3D library not available. Please install it using: pip install c3d")
    def load_c3d_file_alternative(uploaded_file):
    """
    Alternative C3D loading method using different approach
    """
    try:
        bytes_data = uploaded_file.read()
        file_obj = io.BytesIO(bytes_data)
        
        reader = c3d.Reader(file_obj)
        
        # Get basic info
        frame_rate = reader.header.frame_rate
        
        # Alternative method: read all frames at once
        frames = list(reader.read_frames())
        
        if not frames:
            st.error("No frames found in C3D file")
            return None, None, None
        
        # Get marker labels
        marker_labels = []
        for label in reader.point_labels:
            cleaned_label = label.strip().replace('\x00', '')
            if cleaned_label:
                marker_labels.append(cleaned_label)
        
        # Process frames
        marker_data = []
        for frame_idx, frame_tuple in enumerate(frames):
            points = frame_tuple[0] if len(frame_tuple) > 0 else None
            
            frame_data = {'frame': frame_idx}
            
            if points is not None and hasattr(points, 'shape') and len(points.shape) >= 2:
                for marker_idx, label in enumerate(marker_labels):
                    if marker_idx < points.shape[1] and points.shape[0] >= 3:
                        try:
                            x, y, z = float(points[0, marker_idx]), float(points[1, marker_idx]), float(points[2, marker_idx])
                            frame_data[f"{label}_X"] = x if not np.isnan(x) else 0.0
                            frame_data[f"{label}_Y"] = y if not np.isnan(y) else 0.0
                            frame_data[f"{label}_Z"] = z if not np.isnan(z) else 0.0
                        except:
                            frame_data[f"{label}_X"] = 0.0
                            frame_data[f"{label}_Y"] = 0.0
                            frame_data[f"{label}_Z"] = 0.0
            
            marker_data.append(frame_data)
        
        df = pd.DataFrame(marker_data)
        return df, marker_labels, frame_rate
        
    except Exception as e:
        st.error(f"Alternative loading method also failed: {str(e)}")
        return None, None, None
        
        # Read the uploaded file
        bytes_data = uploaded_file.read()
        
        # Create a temporary file-like object
        file_obj = io.BytesIO(bytes_data)
        
        # Read C3D file
        reader = c3d.Reader(file_obj)
        
        # Debug information
        st.info(f"📊 C3D File Info:")
        st.info(f"  • Frame rate: {reader.header.frame_rate} Hz")
        st.info(f"  • Point count: {reader.header.point_count}")
        st.info(f"  • Frame count: {reader.header.last_frame - reader.header.first_frame + 1}")
        
        # Extract marker data
        marker_data = []
        marker_labels = []
        frame_rate = reader.header.frame_rate
        
        # Get marker labels and clean them
        for i, label in enumerate(reader.point_labels):
            cleaned_label = label.strip().replace('\x00', '')  # Remove null characters
            if cleaned_label:  # Only add non-empty labels
                marker_labels.append(cleaned_label)
        
        st.info(f"  • Markers found: {len(marker_labels)}")
        if len(marker_labels) > 0:
            st.info(f"  • First few markers: {marker_labels[:5]}")
        
        # Extract point data
        for frame_idx, frame_data_tuple in enumerate(reader.read_frames()):
            # Handle different return formats from read_frames()
            if len(frame_data_tuple) >= 2:
                points = frame_data_tuple[0]
                analog = frame_data_tuple[1] if len(frame_data_tuple) > 1 else None
            else:
                points = frame_data_tuple[0]
                analog = None
            
            frame_data = {'frame': frame_idx}
            
            # Check if points data is valid
            if points is not None and len(points.shape) >= 2:
                for marker_idx, label in enumerate(marker_labels):
                    if marker_idx < points.shape[1]:
                        try:
                            x, y, z = points[:3, marker_idx]
                            # Check for valid data (not NaN or extremely large values)
                            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                                frame_data[f"{label}_X"] = float(x)
                                frame_data[f"{label}_Y"] = float(y)
                                frame_data[f"{label}_Z"] = float(z)
                            else:
                                frame_data[f"{label}_X"] = 0.0
                                frame_data[f"{label}_Y"] = 0.0
                                frame_data[f"{label}_Z"] = 0.0
                        except (IndexError, ValueError) as e:
                            # Handle invalid marker data
                            frame_data[f"{label}_X"] = 0.0
                            frame_data[f"{label}_Y"] = 0.0
                            frame_data[f"{label}_Z"] = 0.0
            
            marker_data.append(frame_data)
            
            # Limit frames for very large files (optional safety measure)
            if frame_idx > 10000:  # Limit to ~100 seconds at 100Hz
                st.warning(f"⚠️ Large file detected. Loaded first {frame_idx} frames.")
                break
        
        df = pd.DataFrame(marker_data)
        
        # Remove empty columns if any
        df = df.dropna(axis=1, how='all')
        
        return df, marker_labels, frame_rate
        
    except Exception as e:
        st.error(f"Error loading C3D file: {str(e)}")
        
        # More detailed error information
        error_details = traceback.format_exc()
        
        with st.expander("🔍 Detailed Error Information"):
            st.code(error_details)
            st.markdown("""
            **Common C3D File Issues:**
            - File format not supported by the c3d library
            - Corrupted or incomplete file
            - Different C3D version/format than expected
            - File contains no valid marker data
            
            **Try:**
            1. Export your file in a different C3D format from Qualisys
            2. Check if the file opens in other C3D viewers
            3. Use the demo data to test the application features
            """)
        
        return None, None, None

def create_mock_data():
    """
    Create mock biomechanical data for demonstration
    """
    np.random.seed(42)
    n_frames = 100
    
    # Simulate walking motion with some realistic marker positions
    time = np.linspace(0, 2, n_frames)  # 2 seconds of data
    
    # Create mock markers for a simple leg model
    markers = ['Hip', 'Knee', 'Ankle', 'Toe', 'Head', 'Shoulder']
    
    data = {'frame': range(n_frames)}
    
    for i, marker in enumerate(markers):
        # Create some realistic motion patterns
        base_x = i * 0.1
        base_y = 0.8 + i * 0.2
        base_z = 1.0 + i * 0.1
        
        # Add some cyclical motion (like walking)
        x = base_x + 0.05 * np.sin(2 * np.pi * time * 2)  # 2 Hz walking
        y = base_y + 0.1 * np.sin(2 * np.pi * time * 2 + i * np.pi/3)
        z = base_z + 0.02 * np.cos(2 * np.pi * time * 2)
        
        data[f"{marker}_X"] = x
        data[f"{marker}_Y"] = y
        data[f"{marker}_Z"] = z
    
    df = pd.DataFrame(data)
    return df, markers, 100  # 100 Hz sample rate

def plot_3d_markers(df, markers, frame_idx):
    """
    Create 3D plot of marker positions for a specific frame
    """
    fig = go.Figure()
    
    # Plot each marker
    for marker in markers:
        x_col = f"{marker}_X"
        y_col = f"{marker}_Y"
        z_col = f"{marker}_Z"
        
        if x_col in df.columns:
            x = df.iloc[frame_idx][x_col]
            y = df.iloc[frame_idx][y_col]
            z = df.iloc[frame_idx][z_col]
            
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers+text',
                name=marker,
                text=[marker],
                textposition="top center",
                marker=dict(size=8)
            ))
    
    fig.update_layout(
        title=f'3D Marker Positions - Frame {frame_idx}',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='cube'
        ),
        height=600
    )
    
    return fig

def plot_trajectory(df, markers, selected_markers):
    """
    Plot trajectory of selected markers over time
    """
    fig = go.Figure()
    
    for marker in selected_markers:
        x_col = f"{marker}_X"
        y_col = f"{marker}_Y"
        z_col = f"{marker}_Z"
        
        if x_col in df.columns:
            fig.add_trace(go.Scatter3d(
                x=df[x_col],
                y=df[y_col],
                z=df[z_col],
                mode='lines',
                name=f"{marker} trajectory",
                line=dict(width=4)
            ))
    
    fig.update_layout(
        title='Marker Trajectories',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)'
        ),
        height=600
    )
    
    return fig

def calculate_angles_over_time(df, marker1, marker2, marker3):
    """
    Calculate angles over all frames for three markers
    """
    angles = []
    frames = []
    
    for idx in df.index:
        try:
            p1 = np.array([df.iloc[idx][f"{marker1}_X"], 
                          df.iloc[idx][f"{marker1}_Y"], 
                          df.iloc[idx][f"{marker1}_Z"]])
            p2 = np.array([df.iloc[idx][f"{marker2}_X"], 
                          df.iloc[idx][f"{marker2}_Y"], 
                          df.iloc[idx][f"{marker2}_Z"]])
            p3 = np.array([df.iloc[idx][f"{marker3}_X"], 
                          df.iloc[idx][f"{marker3}_Y"], 
                          df.iloc[idx][f"{marker3}_Z"]])
            
            angle = calculate_angle_3d(p1, p2, p3)
            angles.append(angle)
            frames.append(idx)
        except:
            continue
    
    return frames, angles

# Streamlit App
def main():
    st.set_page_config(page_title="C3D Biomechanical Data Viewer", layout="wide")
    
    st.title("🏃‍♂️ C3D Biomechanical Data Viewer")
    st.markdown("Upload a C3D file to visualize marker positions and calculate joint angles")
    
    # Sidebar for file upload and controls
    st.sidebar.header("Data Input")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload C3D file", 
        type=['c3d'],
        help="Upload a C3D file containing biomechanical marker data"
    )
    
    # Initialize data
    df = None
    markers = []
    frame_rate = 100
    
    if uploaded_file is not None:
        with st.spinner("Loading C3D file..."):
            df, markers, frame_rate = load_c3d_file(uploaded_file)
    else:
        # Use mock data for demonstration
        if not C3D_AVAILABLE:
            st.sidebar.info("💡 C3D library not available. Demo data automatically loaded.")
            df, markers, frame_rate = create_mock_data()
        else:
            st.sidebar.info("No file uploaded. Click below to load demo data.")
            if st.sidebar.button("Load Demo Data"):
                df, markers, frame_rate = create_mock_data()
    
    if df is not None and len(markers) > 0:
        st.success(f"Data loaded successfully! {len(markers)} markers, {len(df)} frames")
        
        # Display basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Markers", len(markers))
        with col2:
            st.metric("Number of Frames", len(df))
        with col3:
            st.metric("Frame Rate", f"{frame_rate} Hz")
        
        # Marker selection and frame control
        st.sidebar.header("Visualization Controls")
        
        # Frame selector
        max_frame = len(df) - 1
        current_frame = st.sidebar.slider("Frame", 0, max_frame, 0)
        
        # Auto-play option
        auto_play = st.sidebar.checkbox("Auto-play")
        if auto_play:
            current_frame = st.sidebar.number_input(
                "Auto-play will cycle through frames", 
                min_value=0, 
                max_value=max_frame, 
                value=current_frame
            )
        
        # Main visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(["3D View", "Trajectories", "Angle Analysis", "Data Table"])
        
        with tab1:
            st.header("3D Marker Positions")
            
            # Create and display 3D plot
            fig_3d = plot_3d_markers(df, markers, current_frame)
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Show marker coordinates for current frame
            st.subheader("Marker Coordinates (Current Frame)")
            coord_data = []
            for marker in markers:
                x_col = f"{marker}_X"
                y_col = f"{marker}_Y"
                z_col = f"{marker}_Z"
                if x_col in df.columns:
                    coord_data.append({
                        'Marker': marker,
                        'X': f"{df.iloc[current_frame][x_col]:.3f}",
                        'Y': f"{df.iloc[current_frame][y_col]:.3f}",
                        'Z': f"{df.iloc[current_frame][z_col]:.3f}"
                    })
            
            st.dataframe(pd.DataFrame(coord_data), use_container_width=True)
        
        with tab2:
            st.header("Marker Trajectories")
            
            # Marker selection for trajectories
            selected_markers = st.multiselect(
                "Select markers to visualize trajectories:",
                markers,
                default=markers[:3] if len(markers) >= 3 else markers
            )
            
            if selected_markers:
                fig_traj = plot_trajectory(df, markers, selected_markers)
                st.plotly_chart(fig_traj, use_container_width=True)
        
        with tab3:
            st.header("Joint Angle Analysis")
            
            if len(markers) >= 3:
                st.markdown("Select three markers to calculate the angle at the middle marker:")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    marker1 = st.selectbox("First marker", markers, key="angle_m1")
                with col2:
                    marker2 = st.selectbox("Vertex marker (angle calculated here)", markers, 
                                         index=1 if len(markers) > 1 else 0, key="angle_m2")
                with col3:
                    marker3 = st.selectbox("Third marker", markers, 
                                         index=2 if len(markers) > 2 else 0, key="angle_m3")
                
                if marker1 != marker2 and marker2 != marker3 and marker1 != marker3:
                    # Calculate current angle
                    try:
                        p1 = np.array([df.iloc[current_frame][f"{marker1}_X"], 
                                      df.iloc[current_frame][f"{marker1}_Y"], 
                                      df.iloc[current_frame][f"{marker1}_Z"]])
                        p2 = np.array([df.iloc[current_frame][f"{marker2}_X"], 
                                      df.iloc[current_frame][f"{marker2}_Y"], 
                                      df.iloc[current_frame][f"{marker2}_Z"]])
                        p3 = np.array([df.iloc[current_frame][f"{marker3}_X"], 
                                      df.iloc[current_frame][f"{marker3}_Y"], 
                                      df.iloc[current_frame][f"{marker3}_Z"]])
                        
                        current_angle = calculate_angle_3d(p1, p2, p3)
                        
                        st.metric(
                            f"Current Angle ({marker1}-{marker2}-{marker3})", 
                            f"{current_angle:.1f}°"
                        )
                        
                        # Calculate angles over time
                        frames, angles = calculate_angles_over_time(df, marker1, marker2, marker3)
                        
                        if angles:
                            # Plot angle over time
                            fig_angle = go.Figure()
                            fig_angle.add_trace(go.Scatter(
                                x=frames,
                                y=angles,
                                mode='lines',
                                name=f"{marker1}-{marker2}-{marker3}",
                                line=dict(width=2)
                            ))
                            
                            # Add current frame indicator
                            fig_angle.add_vline(
                                x=current_frame, 
                                line_dash="dash", 
                                line_color="red",
                                annotation_text="Current Frame"
                            )
                            
                            fig_angle.update_layout(
                                title=f"Joint Angle Over Time: {marker1}-{marker2}-{marker3}",
                                xaxis_title="Frame",
                                yaxis_title="Angle (degrees)",
                                height=400
                            )
                            
                            st.plotly_chart(fig_angle, use_container_width=True)
                            
                            # Angle statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean Angle", f"{np.mean(angles):.1f}°")
                            with col2:
                                st.metric("Min Angle", f"{np.min(angles):.1f}°")
                            with col3:
                                st.metric("Max Angle", f"{np.max(angles):.1f}°")
                            with col4:
                                st.metric("Range", f"{np.max(angles) - np.min(angles):.1f}°")
                    
                    except Exception as e:
                        st.error(f"Error calculating angle: {str(e)}")
                else:
                    st.warning("Please select three different markers for angle calculation.")
            else:
                st.warning("Need at least 3 markers for angle calculation.")
        
        with tab4:
            st.header("Raw Data")
            
            # Display options
            show_all = st.checkbox("Show all data", value=False)
            
            if show_all:
                st.dataframe(df, use_container_width=True)
            else:
                # Show just current frame
                st.subheader(f"Frame {current_frame} Data")
                frame_data = df.iloc[current_frame:current_frame+1]
                st.dataframe(frame_data, use_container_width=True)
            
            # Download processed data
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name="biomechanical_data.csv",
                mime="text/csv"
            )
    
    else:
        st.info("Please upload a C3D file or load demo data to begin analysis.")
        
        # Show requirements
        st.markdown("""
        ### Requirements
        To use this application with real C3D files, you need to install the required Python packages:
        
        ```bash
        pip install streamlit plotly pandas numpy c3d
        ```
        
        ### Streamlit Cloud Deployment
        This app can be deployed on Streamlit Cloud! Create a `requirements.txt` file with the dependencies above.
        
        **Note**: If the `c3d` library fails to install on Streamlit Cloud, the app will automatically 
        use demo data to demonstrate all functionality.
        
        ### Features
        - **3D Visualization**: View marker positions in 3D space
        - **Trajectory Analysis**: Track marker movement over time
        - **Joint Angle Calculation**: Calculate angles between three markers
        - **Interactive Controls**: Navigate through frames and select markers
        - **Data Export**: Download processed data as CSV
        """)

if __name__ == "__main__":
    main()
