# C3D Biomechanical Data Viewer

A Streamlit application for visualizing and analyzing biomechanical motion capture data from C3D files (Qualisys format).

## Features

- 📊 **3D Visualization**: Interactive 3D plots of marker positions
- 📈 **Trajectory Analysis**: Track marker movement over time
- 📐 **Joint Angle Calculation**: Calculate angles between three selected markers
- 🎮 **Interactive Controls**: Frame navigation and marker selection
- 💾 **Data Export**: Download processed data as CSV

## Live Demo

🚀 **[Try the live app on Streamlit Cloud](your-app-url-here)**

## Local Installation

```bash
pip install streamlit plotly pandas numpy c3d
streamlit run app.py
```

## Usage

1. Upload a C3D file from your motion capture system
2. Use the frame slider to navigate through time
3. Select markers for trajectory analysis
4. Calculate joint angles by choosing three markers
5. Export your analysis as CSV

## Demo Data

The app includes demo data that simulates walking motion with 6 markers, so you can explore all features even without uploading a file.

## Deployment Notes

- The app gracefully handles cases where the C3D library isn't available
- All visualizations use Plotly for smooth web performance  
- File uploads are processed in memory for security
- Works on mobile devices with responsive design

## Technical Details

- Built with Streamlit for the web interface
- Uses Plotly for 3D visualizations and charts
- Implements proper 3D vector mathematics for angle calculations
- Handles C3D files using the `c3d` Python library
- Responsive design works on desktop and mobile

## Contributing

Feel free to submit issues and enhancement requests!
