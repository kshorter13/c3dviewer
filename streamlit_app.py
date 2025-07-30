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

def load_c3d_file_alternative(uploaded_file):
    """
    Alternative C3D loading method using different approach
    """
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        bytes_data = uploaded_file.read()
        file_obj = io.BytesIO(bytes_data)
        
        reader = c3d.Reader(file_obj)
        
        # Get basic info
        frame_rate = reader.header.frame_rate
        
        # Get marker labels
        marker_labels = []
        for label in reader.point_labels:
            cleaned_label = label.strip().replace('\x00', '')
            if cleaned_label:
                marker_labels.append(cleaned_label)
        
        st.info(f"🔄 Alternative method - trying direct data access...")
        
        # Try to access point data directly
        if hasattr(reader, 'point_data') and reader.point_data is not None:
            st.info(f"  • Found point_data attribute with shape: {reader.point_data.shape}")
            point_data = reader.point_data
            
            marker_data = []
            for frame_idx in range(point_data.shape[0]):
                frame_data = {'frame': frame_idx}
                
                for marker_idx, label in enumerate(marker_labels):
                    if marker_idx < point_data.shape[1]:
                        try:
                            # Point data is usually in format [frame, marker, coordinates]
                            coords = point_data[frame_idx, marker_idx, :3]
                            x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                            
                            frame_data[f"{label}_X"] = x if not np.isnan(x) else 0.0
                            frame_data[f"{label}_Y"] = y if not np.isnan(y) else 0.0
                            frame_data[f"{label}_Z"] = z if not np.isnan(z) else 0.0
                        except:
                            frame_data[f"{label}_X"] = 0.0
                            frame_data[f"{label}_Y"] = 0.0
                            frame_data[f"{label}_Z"] = 0.0
                
                marker_data.append(frame_data)
            
            df = pd.DataFrame(marker_data)
            if not df.empty and len([col for col in df.columns if col.endswith('_X')]) > 0:
                return df, marker_labels, frame_rate
        
        # If direct access doesn't work, try iterating through frames again
        st.info("  • Trying frame iteration method...")
        
        frames_list = []
        try:
            for frame_data in reader.read_frames():
                frames_list.append(frame_data)
        except:
            st.error("  • Frame iteration failed")
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
        st.write(f"- Point scale: {reader.header.point_scale}")
        
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
            # Show sample from frame 99 if it exists
            if reader.point_data.shape[0] > 99:
                frame_99_data = reader.point_data[99]
                st.write(f"✅ Frame 99 raw data shape: {frame_99_data.shape}")
                st.write(f"✅ Frame 99 sample values: {frame_99_data}")
        else:
            st.write("❌ No point_data attribute found")
        
        # Try reading a few frames to see structure
        st.markdown("#### Frame Reading Test:")
        try:
            frame_count = 0
            for frame_data in reader.read_frames():
                if frame_count < 3 or frame_count == 99:  # Show first 3 frames and frame 99
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
        st.markdown("#### Parameters:")
        if hasattr(reader, 'parameters'):
            for key, param in reader.parameters.items():
                st.write(f"- {key}: {param}")
        
        return reader
        
    except Exception as e:
        st.error(f"Debug failed: {e}")
        st.error(traceback.format_exc())
        return None

def load_c3d_file_fixed(uploaded_file):
    """
    Fixed C3D loading based on actual file structure
    """
    try:
        uploaded_file.seek(0)
        bytes_data = uploaded_file.read()
        file_obj = io.BytesIO(bytes_data)
        
        reader = c3d.Reader(file_obj)
        
        # Get basic info
        frame_rate = reader.header.frame_rate
        point_scale = getattr(reader.header, 'point_scale', 1.0)
        
        # Get marker labels
        marker_labels = []
        for label in reader.point_labels:
            cleaned_label = label.strip().replace('\x00', '')
            if cleaned_label:
                marker_labels.append(cleaned_label)
        
        st.info(f"🔧 Fixed method - Point scale factor: {point_scale}")
        
        marker_data = []
        
        # Method 1: Try using point_data directly if available
        if hasattr(reader, 'point_data') and reader.point_data is not None:
            st.info("✅ Using direct point_data access")
            point_data = reader.point_data
            
            for frame_idx in range(point_data.shape[0]):
                frame_data = {'frame': frame_idx}
                
                for marker_idx, label in enumerate(marker_labels):
                    if marker_idx < point_data.shape[1]:
                        # Try different coordinate arrangements
                        try:
                            # Most common format: [frame, marker, [x,y,z,residual]]
                            if point_data.shape[2] >= 3:
                                x = point_data[frame_idx, marker_idx, 0] * point_scale
                                y = point_data[frame_idx, marker_idx, 1] * point_scale  
                                z = point_data[frame_idx, marker_idx, 2] * point_scale
                            else:
                                # Alternative format
                                coords = point_data[frame_idx, marker_idx]
                                x, y, z = coords[0] * point_scale, coords[1] * point_scale, coords[2] * point_scale
                            
                            frame_data[f"{label}_X"] = float(x) if not np.isnan(x) else 0.0
                            frame_data[f"{label}_Y"] = float(y) if not np.isnan(y) else 0.0
                            frame_data[f"{label}_Z"] = float(z) if not np.isnan(z) else 0.0
                            
                        except Exception as e:
                            frame_data[f"{label}_X"] = 0.0
                            frame_data[f"{label}_Y"] = 0.0
                            frame_data[f"{label}_Z"] = 0.0
                
                marker_data.append(frame_data)
                
                # Show progress for large files
                if frame_idx % 1000 == 0:
                    st.info(f"  Processing frame {frame_idx}...")
                
                # Safety limit
                if frame_idx > 10000:
                    st.warning(f"⚠️ Large file detected. Loaded first {frame_idx} frames.")
                    break
        
        else:
            # Method 2: Frame-by-frame reading with better parsing
            st.info("✅ Using frame-by-frame reading")
            
            frame_idx = 0
            for frame_data_tuple in reader.read_frames():
                frame_data = {'frame': frame_idx}
                
                # Handle the tuple structure better
                if isinstance(frame_data_tuple, tuple) and len(frame_data_tuple) > 0:
                    points = frame_data_tuple[0]
                    
                    if hasattr(points, 'shape'):
                        # Apply scale factor and handle different arrangements
                        if len(points.shape) == 2:
                            if points.shape[0] >= 3:  # [coordinates, markers] format
                                for marker_idx, label in enumerate(marker_labels):
                                    if marker_idx < points.shape[1]:
                                        x = points[0, marker_idx] * point_scale
                                        y = points[1, marker_idx] * point_scale
                                        z = points[2, marker_idx] * point_scale
                                        
                                        frame_data[f"{label}_X"] = float(x) if not np.isnan(x) else 0.0
                                        frame_data[f"{label}_Y"] = float(y) if not np.isnan(y) else 0.0
                                        frame_data[f"{label}_Z"] = float(z) if not np.isnan(z) else 0.0
                            
                            elif points.shape[1] >= 3:  # [markers, coordinates] format
                                for marker_idx, label in enumerate(marker_labels):
                                    if marker_idx < points.shape[0]:
                                        x = points[marker_idx, 0] * point_scale
                                        y = points[marker_idx, 1] * point_scale
                                        z = points[marker_idx, 2] * point_scale
                                        
                                        frame_data[f"{label}_X"] = float(x) if not np.isnan(x) else 0.0
                                        frame_data[f"{label}_Y"] = float(y) if not np.isnan(y) else 0.0
                                        frame_data[f"{label}_Z"] = float(z) if not np.isnan(z) else 0.0
                
                # Fill missing data with zeros
                for label in marker_labels:
                    if f"{label}_X" not in frame_data:
                        frame_data[f"{label}_X"] = 0.0
                        frame_data[f"{label}_Y"] = 0.0
                        frame_data[f"{label}_Z"] = 0.0
                
                marker_data.append(frame_data)
                frame_idx += 1
                
                # Safety limits
                if frame_idx > 10000:
                    st.warning(f"⚠️ Large file detected. Loaded first {frame_idx} frames.")
                    break
        
        # Create DataFrame
        df = pd.DataFrame(marker_data)
        
        # Validate data extraction
        coord_columns = [col for col in df.columns if col.endswith(('_X', '_Y', '_Z'))]
        if len(coord_columns) > 0:
            coord_data = df[coord_columns]
            non_zero_count = (coord_data != 0).sum().sum()
            total_count = coord_data.size
            
            st.success(f"✅ Fixed method: Extracted {len(coord_columns)} coordinates")
            st.info(f"  • Non-zero values: {non_zero_count}/{total_count} ({100*non_zero_count/total_count:.1f}%)")
            
            # Show frame 99 data if available
            if len(df) > 99:
                frame_99 = df.iloc[99]
                hip_x = frame_99.get('Hip_X', 'Not found')
                st.info(f"  • Frame 99 Hip_X: {hip_x}")
            
            return df, marker_labels, frame_rate
        else:
            st.error("❌ Fixed method: No coordinate columns found")
            return None, None, None
            
    except Exception as e:
        st.error(f"Fixed loading method failed: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None
        
        if not frames_list:
            st.error("  • No frames found")
            return None, None, None
        
        st.info(f"  • Successfully read {len(frames_list)} frames")
        
        # Process frames
        marker_data = []
        for frame_idx, frame_tuple in enumerate(frames_list):
            frame_data = {'frame': frame_idx}
            
            # Try different ways to extract point data
            points = None
            if isinstance(frame_tuple, tuple) and len(frame_tuple) > 0:
                points = frame_tuple[0]
            elif isinstance(frame_tuple, (list, np.ndarray)):
                points = frame_tuple
            
            if frame_idx == 0:
                st.info(f"  • Sample frame data type: {type(points)}")
                if hasattr(points, 'shape'):
                    st.info(f"  • Sample frame shape: {points.shape}")
            
            # Extract coordinates
            if points is not None:
                if hasattr(points, 'shape') and len(points.shape) >= 2:
                    # Standard numpy array format [coordinates, markers]
                    for marker_idx, label in enumerate(marker_labels):
                        if marker_idx < points.shape[1]:
                            try:
                                x, y, z = float(points[0, marker_idx]), float(points[1, marker_idx]), float(points[2, marker_idx])
                                frame_data[f"{label}_X"] = x if not np.isnan(x) else 0.0
                                frame_data[f"{label}_Y"] = y if not np.isnan(y) else 0.0
                                frame_data[f"{label}_Z"] = z if not np.isnan(z) else 0.0
                            except:
                                frame_data[f"{label}_X"] = 0.0
                                frame_data[f"{label}_Y"] = 0.0
                                frame_data[f"{label}_Z"] = 0.0
                
                elif isinstance(points, (list, tuple)):
                    # List/tuple format
                    for marker_idx, label in enumerate(marker_labels):
                        if marker_idx < len(points):
                            try:
                                point = points[marker_idx]
                                if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 3:
                                    x, y, z = float(point[0]), float(point[1]), float(point[2])
                                    frame_data[f"{label}_X"] = x if not np.isnan(x) else 0.0
                                    frame_data[f"{label}_Y"] = y if not np.isnan(y) else 0.0
                                    frame_data[f"{label}_Z"] = z if not np.isnan(z) else 0.0
                                else:
                                    frame_data[f"{label}_X"] = 0.0
                                    frame_data[f"{label}_Y"] = 0.0
                                    frame_data[f"{label}_Z"] = 0.0
                            except:
                                frame_data[f"{label}_X"] = 0.0
                                frame_data[f"{label}_Y"] = 0.0
                                frame_data[f"{label}_Z"] = 0.0
            
            marker_data.append(frame_data)
        
        df = pd.DataFrame(marker_data)
        return df, marker_labels, frame_rate
        
    except Exception as e:
        st.error(f"Alternative loading method failed: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None

def load_c3d_file(uploaded_file):
    """
    Load C3D file and extract marker data
    """
    if not C3D_AVAILABLE:
        st.error("C3D library not available. Please install it using: pip install c3d")
        return None, None, None
    
    try:
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
        
        # Extract point data with better debugging
        st.info("🔍 Analyzing data structure...")
        
        for frame_idx, frame_data_tuple in enumerate(reader.read_frames()):
            # Debug: Show what we're getting in the first frame
            if frame_idx == 0:
                st.info(f"  • Frame data type: {type(frame_data_tuple)}")
                st.info(f"  • Frame data length: {len(frame_data_tuple)}")
                if len(frame_data_tuple) > 0:
                    st.info(f"  • Points data type: {type(frame_data_tuple[0])}")
                    if hasattr(frame_data_tuple[0], 'shape'):
                        st.info(f"  • Points shape: {frame_data_tuple[0].shape}")
                    else:
                        st.info(f"  • Points value: {frame_data_tuple[0]}")
            
            # Handle different return formats from read_frames()
            points = None
            analog = None
            
            if isinstance(frame_data_tuple, tuple) and len(frame_data_tuple) >= 1:
                points = frame_data_tuple[0]
                analog = frame_data_tuple[1] if len(frame_data_tuple) > 1 else None
            else:
                points = frame_data_tuple
            
            frame_data = {'frame': frame_idx}
            
            # Check if points data is valid and extract coordinates
            if points is not None:
                # Handle different point data formats
                if hasattr(points, 'shape') and len(points.shape) >= 2:
                    # Standard numpy array format
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
                            except (IndexError, ValueError):
                                frame_data[f"{label}_X"] = 0.0
                                frame_data[f"{label}_Y"] = 0.0
                                frame_data[f"{label}_Z"] = 0.0
                
                elif isinstance(points, (list, tuple)) and len(points) > 0:
                    # Handle list/tuple format
                    for marker_idx, label in enumerate(marker_labels):
                        if marker_idx < len(points):
                            try:
                                point = points[marker_idx]
                                if isinstance(point, (list, tuple)) and len(point) >= 3:
                                    x, y, z = float(point[0]), float(point[1]), float(point[2])
                                    frame_data[f"{label}_X"] = x if not np.isnan(x) else 0.0
                                    frame_data[f"{label}_Y"] = y if not np.isnan(y) else 0.0
                                    frame_data[f"{label}_Z"] = z if not np.isnan(z) else 0.0
                                else:
                                    frame_data[f"{label}_X"] = 0.0
                                    frame_data[f"{label}_Y"] = 0.0
                                    frame_data[f"{label}_Z"] = 0.0
                            except:
                                frame_data[f"{label}_X"] = 0.0
                                frame_data[f"{label}_Y"] = 0.0
                                frame_data[f"{label}_Z"] = 0.0
                
                else:
                    # Unknown format - try to extract from reader directly
                    try:
                        # Alternative approach: get point data directly from reader
                        point_data = reader.point_data
                        if point_data is not None and hasattr(point_data, 'shape'):
                            if frame_idx < point_data.shape[0]:
                                for marker_idx, label in enumerate(marker_labels):
                                    if marker_idx < point_data.shape[1]:
                                        x, y, z = point_data[frame_idx, marker_idx, :3]
                                        frame_data[f"{label}_X"] = float(x) if not np.isnan(x) else 0.0
                                        frame_data[f"{label}_Y"] = float(y) if not np.isnan(y) else 0.0
                                        frame_data[f"{label}_Z"] = float(z) if not np.isnan(z) else 0.0
                    except:
                        # Fill with zeros if we can't extract data
                        for label in marker_labels:
                            frame_data[f"{label}_X"] = 0.0
                            frame_data[f"{label}_Y"] = 0.0
                            frame_data[f"{label}_Z"] = 0.0
            
            marker_data.append(frame_data)
            
            # Limit frames for very large files (optional safety measure)
            if frame_idx > 10000:  # Limit to ~50 seconds at 200Hz
                st.warning(f"⚠️ Large file detected. Loaded first {frame_idx} frames.")
                break
        
        df = pd.DataFrame(marker_data)
        
        # Remove empty columns if any
        df = df.dropna(axis=1, how='all')
        
        # Validate that we actually got coordinate data
        coord_columns = [col for col in df.columns if col.endswith(('_X', '_Y', '_Z'))]
        if len(coord_columns) == 0:
            st.error("❌ No coordinate data found in extracted dataframe")
            return None, None, None
        
        # Check if we have actual non-zero data
        coord_data = df[coord_columns]
        if coord_data.abs().sum().sum() == 0:
            st.warning("⚠️ All coordinates are zero - this might indicate a data extraction issue")
        else:
            st.success(f"✅ Successfully extracted {len(coord_columns)} coordinate columns")
            # Show sample of first few values
            sample_data = coord_data.head(1)
            st.info(f"  • Sample data: {sample_data.iloc[0].to_dict()}")
        
        return df, marker_labels, frame_rate
        
    except Exception as e:
        st.error(f"Primary C3D loading method failed: {str(e)}")
        
        # Try alternative loading method
        st.info("🔄 Trying alternative loading method...")
        try:
            # Reset the uploaded file for the alternative method
            uploaded_file.seek(0)
            result = load_c3d_file_alternative(uploaded_file)
            if result[0] is not None:
                coord_columns = [col for col in result[0].columns if col.endswith(('_X', '_Y', '_Z'))]
                if len(coord_columns) > 0:
                    return result
        except Exception as alt_error:
            st.error(f"Alternative method also failed: {str(alt_error)}")
        
        # Try third method - raw C3D access
        st.info("🔄 Trying raw data access method...")
        try:
            uploaded_file.seek(0)
            result = load_c3d_file_raw(uploaded_file)
            if result[0] is not None:
                return result
        except Exception as raw_error:
            st.error(f"Raw access method failed: {str(raw_error)}")
        
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
            - Data is stored in an unexpected format
            
            **Try:**
            1. Export your file in a different C3D format from Qualisys
            2. Check if the file opens in other C3D viewers
            3. Use the demo data to test the application features
            4. Try opening the file in Qualisys Track Manager and re-export
            """)
        
        return None, None, None

def load_c3d_file_raw(uploaded_file):
    """
    Raw C3D loading method - tries to access data structures directly
    """
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        bytes_data = uploaded_file.read()
        file_obj = io.BytesIO(bytes_data)
        
        reader = c3d.Reader(file_obj)
        
        st.info("🔄 Raw method - inspecting C3D structure...")
        
        # Get basic info
        frame_rate = reader.header.frame_rate
        first_frame = reader.header.first_frame
        last_frame = reader.header.last_frame
        point_count = reader.header.point_count
        
        # Get marker labels
        marker_labels = []
        for label in reader.point_labels:
            cleaned_label = label.strip().replace('\x00', '')
            if cleaned_label:
                marker_labels.append(cleaned_label)
        
        st.info(f"  • Will extract frames {first_frame} to {last_frame}")
        st.info(f"  • Expected {point_count} points per frame")
        
        # Try accessing data through different methods
        marker_data = []
        
        # Method 1: Try using reader parameters directly
        if hasattr(reader, 'read'):
            try:
                # Reset reader
                reader.seek(first_frame)
                
                for frame_idx in range(last_frame - first_frame + 1):
                    frame_data = {'frame': frame_idx}
                    
                    try:
                        # Try to read one frame
                        frame_info = next(reader.read_frames())
                        
                        if isinstance(frame_info, tuple) and len(frame_info) > 0:
                            points = frame_info[0]
                            
                            # Debug the structure
                            if frame_idx == 0:
                                st.info(f"  • Raw frame data type: {type(points)}")
                                if hasattr(points, 'shape'):
                                    st.info(f"  • Raw frame shape: {points.shape}")
                                elif isinstance(points, (list, tuple)):
                                    st.info(f"  • Raw frame length: {len(points)}")
                            
                            # Extract coordinates based on the actual structure
                            if hasattr(points, 'shape'):
                                if len(points.shape) == 2:  # [coords, markers] or [markers, coords]
                                    if points.shape[0] == 3:  # [3, n_markers] format
                                        for marker_idx, label in enumerate(marker_labels):
                                            if marker_idx < points.shape[1]:
                                                x, y, z = points[0, marker_idx], points[1, marker_idx], points[2, marker_idx]
                                                frame_data[f"{label}_X"] = float(x) if not np.isnan(x) else 0.0
                                                frame_data[f"{label}_Y"] = float(y) if not np.isnan(y) else 0.0
                                                frame_data[f"{label}_Z"] = float(z) if not np.isnan(z) else 0.0
                                    elif points.shape[1] == 3:  # [n_markers, 3] format
                                        for marker_idx, label in enumerate(marker_labels):
                                            if marker_idx < points.shape[0]:
                                                x, y, z = points[marker_idx, 0], points[marker_idx, 1], points[marker_idx, 2]
                                                frame_data[f"{label}_X"] = float(x) if not np.isnan(x) else 0.0
                                                frame_data[f"{label}_Y"] = float(y) if not np.isnan(y) else 0.0
                                                frame_data[f"{label}_Z"] = float(z) if not np.isnan(z) else 0.0
                                
                                elif len(points.shape) == 1:  # Flattened array
                                    # Assume format is [x1, y1, z1, x2, y2, z2, ...]
                                    for marker_idx, label in enumerate(marker_labels):
                                        if (marker_idx + 1) * 3 <= len(points):
                                            x = points[marker_idx * 3]
                                            y = points[marker_idx * 3 + 1]
                                            z = points[marker_idx * 3 + 2]
                                            frame_data[f"{label}_X"] = float(x) if not np.isnan(x) else 0.0
                                            frame_data[f"{label}_Y"] = float(y) if not np.isnan(y) else 0.0
                                            frame_data[f"{label}_Z"] = float(z) if not np.isnan(z) else 0.0
                            
                            # Fill with zeros if we couldn't extract data
                            for label in marker_labels:
                                if f"{label}_X" not in frame_data:
                                    frame_data[f"{label}_X"] = 0.0
                                    frame_data[f"{label}_Y"] = 0.0
                                    frame_data[f"{label}_Z"] = 0.0
                        
                    except StopIteration:
                        break
                    except Exception as e:
                        # Fill frame with zeros if extraction fails
                        for label in marker_labels:
                            frame_data[f"{label}_X"] = 0.0
                            frame_data[f"{label}_Y"] = 0.0
                            frame_data[f"{label}_Z"] = 0.0
                    
                    marker_data.append(frame_data)
                    
                    # Limit processing for very large files
                    if frame_idx > 10000:
                        st.warning(f"⚠️ Large file detected. Loaded first {frame_idx} frames.")
                        break
                
            except Exception as read_error:
                st.error(f"Raw reading failed: {str(read_error)}")
                return None, None, None
        
        # Create DataFrame and validate
        df = pd.DataFrame(marker_data)
        
        # Check if we got any coordinate data
        coord_columns = [col for col in df.columns if col.endswith(('_X', '_Y', '_Z'))]
        if len(coord_columns) == 0:
            st.error("❌ Raw method: No coordinate data found")
            return None, None, None
        
        # Check if we have actual non-zero data
        coord_data = df[coord_columns]
        if coord_data.abs().sum().sum() == 0:
            st.warning("⚠️ Raw method: All coordinates are zero")
        else:
            st.success(f"✅ Raw method: Successfully extracted {len(coord_columns)} coordinate columns")
        
        return df, marker_labels, frame_rate
        
    except Exception as e:
        st.error(f"Raw access method failed: {str(e)}")
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
            # First, show detailed debugging information
            with st.expander("🔍 **Debug C3D File Structure** (Click to expand)", expanded=False):
                debug_reader = debug_c3d_structure(uploaded_file)
            
            # Now try the fixed loading method
            st.info("🚀 Attempting to load with fixed method...")
            df, markers, frame_rate = load_c3d_file_fixed(uploaded_file)
            
            # If fixed method fails, try the original methods
            if df is None:
                st.warning("Fixed method failed, trying original methods...")
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
