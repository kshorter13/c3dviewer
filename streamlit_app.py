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
    st.stop()

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

def debug_c3d_structure(uploaded_file):
    """
    Debug function to inspect the actual C3D file structure
    """
    try:
        uploaded_file.seek(0)
        bytes_data = uploaded_file.read()
        file_obj = io.BytesIO(bytes_data)
        
        reader = c3d.Reader(file_obj)
        
        st.markdown("### 🔍 **Detailed C3D File Structure Analysis**")
        
        # Show header information
        st.markdown("#### Header Information:")
        st.write(f"- Frame rate: {reader.header.frame_rate}")
        st.write(f"- First frame: {reader.header.first_frame}")
        st.write(f"- Last frame: {reader.header.last_frame}")
        st.write(f"- Point count: {reader.header.point_count}")
        st.write(f"- Point scale: {getattr(reader.header, 'point_scale', 'Not found')}")
        st.write(f"- Analog count: {getattr(reader.header, 'analog_count', 'Not found')}")
        st.write(f"- Max interpolation gap: {getattr(reader.header, 'max_interpolation_gap', 'Not found')}")
        
        # Show C3D specification parameters
        st.markdown("#### C3D Parameters (C3D.org spec):")
        if hasattr(reader, 'parameters'):
            important_params = {}
            for key, param in reader.parameters.items():
                if any(keyword in key.upper() for keyword in ['POINT', 'SCALE', 'UNITS', 'RATE', 'LABELS']):
                    try:
                        if hasattr(param, 'float_value'):
                            important_params[key] = f"Float: {param.float_value}"
                        elif hasattr(param, 'int_value'):
                            important_params[key] = f"Int: {param.int_value}"
                        elif hasattr(param, 'string_value'):
                            important_params[key] = f"String: {param.string_value}"
                        else:
                            important_params[key] = f"Raw: {param}"
                    except:
                        important_params[key] = f"Type: {type(param)}"
            
            for key, value in important_params.items():
                st.write(f"- {key}: {value}")
        
        # Check point labels
        st.markdown("#### Point Labels:")
        point_labels = []
        for i, label in enumerate(reader.point_labels):
            cleaned = label.strip().replace('\x00', '')
            if cleaned:
                point_labels.append(f"{i}: '{cleaned}'")
        st.write(point_labels[:10])  # Show first 10 labels
        
        # Show available attributes
        st.markdown("#### Available Reader Attributes:")
        reader_attrs = [attr for attr in dir(reader) if not attr.startswith('_')]
        st.write(reader_attrs)
        
        # Try to access different data structures
        st.markdown("#### Data Structure Investigation:")
        
        # Check if point_data exists
        if hasattr(reader, 'point_data') and reader.point_data is not None:
            st.write(f"✅ point_data found with shape: {reader.point_data.shape}")
            st.write(f"✅ point_data dtype: {reader.point_data.dtype}")
            st.write(f"✅ point_data min value: {reader.point_data.min()}")
            st.write(f"✅ point_data max value: {reader.point_data.max()}")
            st.write(f"✅ point_data mean value: {np.mean(reader.point_data)}")
            
            # Show sample from multiple frames if they exist
            sample_frames = [0, min(10, reader.point_data.shape[0]-1), min(99, reader.point_data.shape[0]-1)] if reader.point_data.shape[0] > 0 else []
            for frame_num in sample_frames:
                if frame_num < reader.point_data.shape[0]:
                    frame_data = reader.point_data[frame_num]
                    st.write(f"✅ Frame {frame_num} raw data shape: {frame_data.shape}")
                    st.write(f"✅ Frame {frame_num} sample values: {frame_data.flatten()[:12]}...")  # First 12 values
                    
                    # Check for potential byte order issues
                    if len(frame_data.shape) >= 2 and frame_data.shape[1] >= 3:
                        first_point = frame_data[0, :3]
                        st.write(f"✅ Frame {frame_num} first point coords: {first_point}")
                        
                        # Try different interpretations
                        if hasattr(reader.header, 'point_scale'):
                            scaled = first_point * reader.header.point_scale
                            st.write(f"✅ Frame {frame_num} scaled coords: {scaled}")
                        
                        # Check if values look like they need different scaling
                        if np.max(np.abs(first_point)) > 10000:
                            st.write(f"✅ Frame {frame_num} values seem very large - might need different scale")
                        elif np.max(np.abs(first_point)) < 0.001:
                            st.write(f"✅ Frame {frame_num} values seem very small - might need different scale")
        else:
            st.write("❌ No point_data attribute found")
        
        # Try reading a few frames to see structure
        st.markdown("#### Frame Reading Test:")
        try:
            frame_count = 0
            sample_frames_to_show = [0, 1, 2, 99]  # Show these specific frames
            for frame_data in reader.read_frames():
                if frame_count in sample_frames_to_show:
                    st.write(f"**Frame {frame_count}:**")
                    st.write(f"  - Type: {type(frame_data)}")
                    st.write(f"  - Length: {len(frame_data) if hasattr(frame_data, '__len__') else 'N/A'}")
                    
                    if isinstance(frame_data, tuple):
                        for i, item in enumerate(frame_data):
                            st.write(f"  - Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'no shape')}")
                            if hasattr(item, 'shape') and len(item.shape) <= 2:
                                st.write(f"    Values: {item}")
                    
                frame_count += 1
                if frame_count > 100:  # Only check first 101 frames
                    break
                    
        except Exception as e:
            st.error(f"Frame reading failed: {e}")
        
        # Check parameters
        st.markdown("#### All Parameters (for debugging):")
        if hasattr(reader, 'parameters'):
            param_count = 0
            for key, param in reader.parameters.items():
                try:
                    param_info = f"- {key}: "
                    if hasattr(param, 'float_value') and param.float_value is not None:
                        param_info += f"float={param.float_value}"
                    elif hasattr(param, 'int_value') and param.int_value is not None:
                        param_info += f"int={param.int_value}"
                    elif hasattr(param, 'string_value') and param.string_value:
                        param_info += f"string='{param.string_value}'"
                    elif hasattr(param, 'bytes_value') and param.bytes_value:
                        param_info += f"bytes={param.bytes_value}"
                    else:
                        param_info += f"type={type(param)}, value={param}"
                    
                    st.write(param_info)
                    param_count += 1
                    
                    # Limit parameter display
                    if param_count > 20:
                        st.write(f"... and {len(reader.parameters) - param_count} more parameters")
                        break
                except Exception as e:
                    st.write(f"- {key}: Error reading parameter - {e}")
            
            if param_count == 0:
                st.write("No parameters found or accessible")
        else:
            st.write("No parameters attribute found")
        
        return reader
        
    except Exception as e:
        st.error(f"Debug failed: {e}")
        st.error(traceback.format_exc())
        return None

def load_c3d_specification_compliant(uploaded_file, show_details=False):
    """
    C3D loading method that follows C3D.org specification exactly
    """
    try:
        uploaded_file.seek(0)
        bytes_data = uploaded_file.read()
        file_obj = io.BytesIO(bytes_data)
        
        reader = c3d.Reader(file_obj)
        
        # Get header information
        header = reader.header
        frame_rate = header.frame_rate
        first_frame = header.first_frame
        last_frame = header.last_frame
        point_count = header.point_count
        
        # Get marker labels
        marker_labels = []
        for i, label in enumerate(reader.point_labels):
            cleaned_label = label.strip().replace('\x00', '').replace(' ', '')
            if cleaned_label:
                marker_labels.append(cleaned_label)
        
        # Extract data from frame iteration
        marker_data = []
        frame_idx = 0
        
        # Show progress message (always show basic progress)
        progress_placeholder = st.empty()
        total_frames = last_frame - first_frame + 1
        progress_placeholder.info(f"🔄 Loading {len(marker_labels)} markers from {total_frames} frames...")
        
        for frame_tuple in reader.read_frames():
            frame_data = {'frame': frame_idx}
            
            if isinstance(frame_tuple, tuple) and len(frame_tuple) >= 2:
                point_data = frame_tuple[1]  # This is the (n, 5) array
                
                # Extract coordinates for each marker
                for marker_idx, label in enumerate(marker_labels):
                    if marker_idx < point_data.shape[0]:
                        try:
                            # Extract X, Y, Z (first 3 columns)
                            x = float(point_data[marker_idx, 0])
                            y = float(point_data[marker_idx, 1])
                            z = float(point_data[marker_idx, 2])
                            residual = float(point_data[marker_idx, 3])
                            
                            # Check if data is valid
                            if residual > 0:
                                frame_data[f"{label}_X"] = x
                                frame_data[f"{label}_Y"] = y
                                frame_data[f"{label}_Z"] = z
                                frame_data[f"{label}_residual"] = residual
                            else:
                                frame_data[f"{label}_X"] = 0.0
                                frame_data[f"{label}_Y"] = 0.0
                                frame_data[f"{label}_Z"] = 0.0
                                frame_data[f"{label}_residual"] = -1.0
                                
                        except Exception:
                            frame_data[f"{label}_X"] = 0.0
                            frame_data[f"{label}_Y"] = 0.0
                            frame_data[f"{label}_Z"] = 0.0
                            frame_data[f"{label}_residual"] = -1.0
                
                # Fill any missing markers with zeros
                for label in marker_labels:
                    if f"{label}_X" not in frame_data:
                        frame_data[f"{label}_X"] = 0.0
                        frame_data[f"{label}_Y"] = 0.0
                        frame_data[f"{label}_Z"] = 0.0
            
            marker_data.append(frame_data)
            frame_idx += 1
            
            # Update progress less frequently for better performance
            if show_details and frame_idx % 200 == 0:
                progress_placeholder.info(f"🔄 Loading... {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")
            elif not show_details and frame_idx % 500 == 0:
                progress_placeholder.info(f"🔄 Loading {len(marker_labels)} markers... {100*frame_idx/total_frames:.0f}% complete")
            
            # Safety limit
            if frame_idx > 10000:
                if show_details:
                    st.warning(f"⚠️ Large file detected. Loaded first {frame_idx} frames.")
                break
        
        # Clear progress message
        progress_placeholder.empty()
        
        # Create DataFrame
        df = pd.DataFrame(marker_data)
        
        # Final validation and success message
        if df is not None and not df.empty:
            coord_columns = [col for col in df.columns if col.endswith(('_X', '_Y', '_Z'))]
            if coord_columns:
                coord_data = df[coord_columns]
                non_zero_count = (coord_data != 0).sum().sum()
                total_count = coord_data.size
                
                # Show concise success message
                st.success(f"✅ **Successfully loaded C3D file!** {len(marker_labels)} markers • {len(df)} frames • {frame_rate} Hz")
                
                # Put detailed info in an expandable section (always available but collapsed)
                with st.expander("📊 **Loading Details** (Click to expand)", expanded=show_details):
                    st.info(f"**File Information:**")
                    st.write(f"  • Frame range: {first_frame} to {last_frame}")
                    st.write(f"  • Frame rate: {frame_rate} Hz")
                    st.write(f"  • Marker count: {len(marker_labels)}")
                    st.write(f"  • Marker labels: {', '.join(marker_labels[:8])}{'...' if len(marker_labels) > 8 else ''}")
                    
                    st.info(f"**Data Quality:**")
                    st.write(f"  • Coordinate columns: {len(coord_columns)}")
                    st.write(f"  • Valid data points: {non_zero_count:,}/{total_count:,} ({100*non_zero_count/total_count:.1f}%)")
                    
                    # Show sample coordinates from key frames
                    st.info(f"**Sample Coordinates:**")
                    key_frames = [0, min(99, len(df)-1), len(df)-1] if len(df) > 1 else [0]
                    first_marker = marker_labels[0] if marker_labels else None
                    
                    if first_marker:
                        for frame_num in key_frames:
                            if frame_num < len(df):
                                frame_sample = df.iloc[frame_num]
                                x_val = frame_sample.get(f"{first_marker}_X", 0)
                                y_val = frame_sample.get(f"{first_marker}_Y", 0)
                                z_val = frame_sample.get(f"{first_marker}_Z", 0)
                                
                                if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)) and isinstance(z_val, (int, float)):
                                    st.write(f"  • Frame {frame_num} {first_marker}: X={x_val:.1f}, Y={y_val:.1f}, Z={z_val:.1f} mm")
                
                return df, marker_labels, frame_rate
            else:
                st.error("❌ No coordinate columns found in the file")
                return None, None, None
        else:
            st.error("❌ Failed to create data from C3D file")
            return None, None, None
            
    except Exception as e:
        st.error(f"❌ **C3D loading failed:** {str(e)}")
        
        # Put detailed error in expandable section
        with st.expander("🔍 **Error Details** (Click to expand)", expanded=show_details):
            st.code(traceback.format_exc())
            st.markdown("""
            **Common Solutions:**
            - Ensure the file is a valid C3D format from Qualisys
            - Try re-exporting the file from your motion capture software  
            - Check that the file isn't corrupted
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

def calculate_movement_bounds(df, markers, padding_percent=15):
    """
    Calculate the min/max bounds for all marker movements with padding
    """
    all_x, all_y, all_z = [], [], []
    
    # Collect all coordinate values across all markers and frames
    for marker in markers:
        x_col = f"{marker}_X"
        y_col = f"{marker}_Y"
        z_col = f"{marker}_Z"
        
        if x_col in df.columns:
            # Get non-zero values (valid coordinates)
            x_values = df[x_col][df[x_col] != 0].values
            y_values = df[y_col][df[y_col] != 0].values
            z_values = df[z_col][df[z_col] != 0].values
            
            all_x.extend(x_values)
            all_y.extend(y_values)
            all_z.extend(z_values)
    
    if not all_x:  # No valid data found
        return None
    
    # Calculate min/max for each axis
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_z), max(all_z)
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    x_padding = x_range * (padding_percent / 100)
    y_padding = y_range * (padding_percent / 100)
    z_padding = z_range * (padding_percent / 100)
    
    bounds = {
        'x_range': [x_min - x_padding, x_max + x_padding],
        'y_range': [y_min - y_padding, y_max + y_padding],
        'z_range': [z_min - z_padding, z_max + z_padding]
    }
    
    return bounds

def plot_3d_markers(df, markers, frame_idx, movement_bounds=None):
    """
    Create 3D plot of marker positions for a specific frame with fixed scaling
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
            
            # Only plot if coordinates are valid (non-zero)
            if x != 0 or y != 0 or z != 0:
                fig.add_trace(go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode='markers+text',
                    name=marker,
                    text=[marker],
                    textposition="top center",
                    marker=dict(size=8),
                    showlegend=True
                ))
    
    # Set up the layout with fixed axis ranges if bounds are provided
    scene_dict = {
        'xaxis_title': 'X (mm)',
        'yaxis_title': 'Y (mm)',
        'zaxis_title': 'Z (mm)',
        'aspectmode': 'cube'
    }
    
    if movement_bounds:
        scene_dict.update({
            'xaxis': dict(range=movement_bounds['x_range'], title='X (mm)'),
            'yaxis': dict(range=movement_bounds['y_range'], title='Y (mm)'),
            'zaxis': dict(range=movement_bounds['z_range'], title='Z (mm)')
        })
    
    fig.update_layout(
        title=f'3D Marker Positions - Frame {frame_idx}',
        scene=scene_dict,
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

def plot_trajectory(df, markers, selected_markers, movement_bounds=None):
    """
    Plot trajectory of selected markers over time with fixed scaling
    """
    fig = go.Figure()
    
    for marker in selected_markers:
        x_col = f"{marker}_X"
        y_col = f"{marker}_Y"
        z_col = f"{marker}_Z"
        
        if x_col in df.columns:
            # Get valid (non-zero) coordinates
            valid_mask = (df[x_col] != 0) | (df[y_col] != 0) | (df[z_col] != 0)
            
            if valid_mask.any():
                fig.add_trace(go.Scatter3d(
                    x=df[x_col][valid_mask],
                    y=df[y_col][valid_mask],
                    z=df[z_col][valid_mask],
                    mode='lines+markers',
                    name=f"{marker} trajectory",
                    line=dict(width=4),
                    marker=dict(size=3)
                ))
    
    # Set up the layout with fixed axis ranges if bounds are provided
    scene_dict = {
        'xaxis_title': 'X (mm)',
        'yaxis_title': 'Y (mm)',
        'zaxis_title': 'Z (mm)',
        'aspectmode': 'cube'
    }
    
    if movement_bounds:
        scene_dict.update({
            'xaxis': dict(range=movement_bounds['x_range'], title='X (mm)'),
            'yaxis': dict(range=movement_bounds['y_range'], title='Y (mm)'),
            'zaxis': dict(range=movement_bounds['z_range'], title='Z (mm)')
        })
    
    fig.update_layout(
        title='Marker Trajectories',
        scene=scene_dict,
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
    
    # Advanced options (always available)
    with st.sidebar.expander("⚙️ Advanced Options"):
        show_debug_info = st.checkbox("Show detailed file structure analysis", value=False, 
                                     help="Display technical details about the C3D file structure")
        show_loading_details = st.checkbox("Show detailed loading progress", value=False,
                                          help="Display step-by-step loading information")
    
    # Initialize data
    df = None
    markers = []
    frame_rate = 100
    
    if uploaded_file is not None:
        with st.spinner("Loading C3D file..."):
            # Show debug section only if requested
            if show_debug_info:
                with st.expander("🔍 **File Structure Analysis** (Click to expand)", expanded=False):
                    debug_reader = debug_c3d_structure(uploaded_file)
            
            # Load the C3D file
            df, markers, frame_rate = load_c3d_specification_compliant(uploaded_file, show_details=show_loading_details)
            
            # If loading fails, fall back to demo data
            if df is None:
                st.warning("⚠️ C3D loading failed. Loading demo data for interface testing.")
                df, markers, frame_rate = create_mock_data()
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
