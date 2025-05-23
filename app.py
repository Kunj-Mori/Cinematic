import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tempfile
import os
import time

# Add this function to check for cloud environment
def is_cloud_environment():
    return (
        os.getenv('STREAMLIT_RUNTIME_ENV') == 'cloud' or
        os.getenv('IS_CLOUD_ENVIRONMENT') == 'true' or
        not os.path.exists('/dev/video0')  # Check if webcam device exists
    )

# Set page config
st.set_page_config(
    page_title="Cinematic Filter App",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5757;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B4B4B;
        margin-bottom: 1rem;
    }
    .stSlider > div > div > div {
        background-color: #FF5757;
    }
    .download-btn {
        background-color: #FF5757;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        text-decoration: none;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>Cinematic Filter Studio</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Transform your media with professional cinematic filters</p>", unsafe_allow_html=True)

# Sidebar for filter controls
st.sidebar.markdown("## Filter Controls")

# Initialize session state for filter parameters
if 'contrast' not in st.session_state:
    st.session_state.contrast = 1.0
if 'brightness' not in st.session_state:
    st.session_state.brightness = 1.0
if 'saturation' not in st.session_state:
    st.session_state.saturation = 1.0
if 'tint_r' not in st.session_state:
    st.session_state.tint_r = 1.0
if 'tint_g' not in st.session_state:
    st.session_state.tint_g = 1.0
if 'tint_b' not in st.session_state:
    st.session_state.tint_b = 1.0
if 'vignette' not in st.session_state:
    st.session_state.vignette = 0.0
if 'grain' not in st.session_state:
    st.session_state.grain = 0.0
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'webcam_on' not in st.session_state:
    st.session_state.webcam_on = False

# Initialize session state
if 'is_cloud' not in st.session_state:
    st.session_state.is_cloud = is_cloud_environment()

# Preset filters
st.sidebar.markdown("## Preset Filters")
preset_options = {
    "None": {"contrast": 1.0, "brightness": 1.0, "saturation": 1.0, 
             "tint_r": 1.0, "tint_g": 1.0, "tint_b": 1.0, 
             "vignette": 0.0, "grain": 0.0},
    "Cinematic Pro": {"contrast": 1.3, "brightness": 0.95, "saturation": 0.85, 
                    "tint_r": 1.1, "tint_g": 0.95, "tint_b": 0.9, 
                    "vignette": 0.35, "grain": 0.15},
    "Hollywood": {"contrast": 1.25, "brightness": 0.9, "saturation": 0.8, 
                "tint_r": 1.15, "tint_g": 0.9, "tint_b": 0.85, 
                "vignette": 0.4, "grain": 0.2},
    "Vintage Film": {"contrast": 1.1, "brightness": 1.05, "saturation": 0.7, 
                   "tint_r": 1.2, "tint_g": 0.95, "tint_b": 0.8, 
                   "vignette": 0.45, "grain": 0.25},
    "Noir": {"contrast": 1.4, "brightness": 0.85, "saturation": 0.0, 
            "tint_r": 1.0, "tint_g": 1.0, "tint_b": 1.0, 
            "vignette": 0.6, "grain": 0.2},
    "Warm Sunset": {"contrast": 1.15, "brightness": 1.1, "saturation": 1.2, 
                  "tint_r": 1.25, "tint_g": 1.0, "tint_b": 0.8, 
                  "vignette": 0.25, "grain": 0.1},
    "Cool Morning": {"contrast": 1.1, "brightness": 1.05, "saturation": 0.9, 
                   "tint_r": 0.85, "tint_g": 1.0, "tint_b": 1.15, 
                   "vignette": 0.2, "grain": 0.1},
    "Blockbuster": {"contrast": 1.35, "brightness": 0.95, "saturation": 1.1, 
                  "tint_r": 1.05, "tint_g": 0.95, "tint_b": 1.1, 
                  "vignette": 0.3, "grain": 0.15},
    "Muted Tones": {"contrast": 1.1, "brightness": 0.95, "saturation": 0.75, 
                  "tint_r": 1.0, "tint_g": 1.0, "tint_b": 1.0, 
                  "vignette": 0.2, "grain": 0.1},
    "Custom": {"contrast": 1.0, "brightness": 1.0, "saturation": 1.0, 
             "tint_r": 1.0, "tint_g": 1.0, "tint_b": 1.0, 
             "vignette": 0.0, "grain": 0.0}
}

selected_preset = st.sidebar.selectbox("Choose a preset", list(preset_options.keys()))

# Show filter controls only when "Custom" is selected
if selected_preset == "Custom":
    st.sidebar.markdown("### Custom Filter Settings")
else:
    if st.sidebar.button("Apply Preset"):
        preset = preset_options[selected_preset]
        st.session_state.contrast = preset["contrast"]
        st.session_state.brightness = preset["brightness"]
        st.session_state.saturation = preset["saturation"]
        st.session_state.tint_r = preset["tint_r"]
        st.session_state.tint_g = preset["tint_g"]
        st.session_state.tint_b = preset["tint_b"]
        st.session_state.vignette = preset["vignette"]
        st.session_state.grain = preset["grain"]
        st.rerun()

# Filter parameters - only show when Custom is selected or after a preset is applied
if selected_preset == "Custom" or "contrast" in st.session_state:
    contrast = st.sidebar.slider("Contrast", 0.0, 2.0, st.session_state.contrast, 0.1)
    brightness = st.sidebar.slider("Brightness", 0.0, 2.0, st.session_state.brightness, 0.1)
    saturation = st.sidebar.slider("Saturation", 0.0, 2.0, st.session_state.saturation, 0.1)

    st.sidebar.markdown("### Tint")
    tint_r = st.sidebar.slider("Red Tint", 0.5, 1.5, st.session_state.tint_r, 0.05)
    tint_g = st.sidebar.slider("Green Tint", 0.5, 1.5, st.session_state.tint_g, 0.05)
    tint_b = st.sidebar.slider("Blue Tint", 0.5, 1.5, st.session_state.tint_b, 0.05)

    vignette = st.sidebar.slider("Vignette", 0.0, 1.0, st.session_state.vignette, 0.05)
    grain = st.sidebar.slider("Film Grain", 0.0, 1.0, st.session_state.grain, 0.05)

    # Update session state
    st.session_state.contrast = contrast
    st.session_state.brightness = brightness
    st.session_state.saturation = saturation
    st.session_state.tint_r = tint_r
    st.session_state.tint_g = tint_g
    st.session_state.tint_b = tint_b
    st.session_state.vignette = vignette
    st.session_state.grain = grain

# Add preset description
preset_descriptions = {
    "None": "No filters applied - original image",
    "Cinematic Pro": "Professional cinematic look with balanced contrast and subtle grain",
    "Hollywood": "Classic Hollywood style with warm tones and medium contrast",
    "Vintage Film": "Nostalgic film look with muted colors and authentic grain",
    "Noir": "Classic black and white with strong contrast and dramatic vignette",
    "Warm Sunset": "Warm, golden hour tones with enhanced reds",
    "Cool Morning": "Cool, crisp tones perfect for morning scenes",
    "Blockbuster": "High-impact look with strong contrast and vibrant colors",
    "Muted Tones": "Subtle, understated look with slightly desaturated colors",
    "Custom": "Create your own custom filter settings"
}

if selected_preset in preset_descriptions:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Preset Description")
    st.sidebar.markdown(f"*{preset_descriptions[selected_preset]}*")

# Function to apply vignette effect
def apply_vignette(image, amount):
    if amount <= 0:
        return image
    
    height, width = image.shape[:2]
    
    # Create a vignette mask
    X_resultant = np.abs(np.linspace(-1, 1, width))
    Y_resultant = np.abs(np.linspace(-1, 1, height))
    X_resultant, Y_resultant = np.meshgrid(X_resultant, Y_resultant)
    
    radius = np.sqrt(X_resultant**2 + Y_resultant**2)
    radius = radius / np.max(radius)
    
    # Apply vignette
    mask = 1 - amount * radius
    mask = np.clip(mask, 0, 1)
    mask = np.dstack([mask] * 3) if len(image.shape) == 3 else mask
    
    return (image * mask).astype(np.uint8)

# Function to apply film grain
def apply_grain(image, amount):
    if amount <= 0:
        return image
    
    # Reduce grain intensity and make it more film-like
    grain_intensity = amount * 25  # Reduced from 50 to 25 for subtler effect
    
    # Create more natural-looking noise
    noise = np.random.normal(0, 1, image.shape[:2])
    # Apply Gaussian blur to make grain more natural
    noise = cv2.GaussianBlur(noise, (3, 3), 0)
    # Normalize noise
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = noise * grain_intensity
    
    # Add noise with proper channel handling
    if len(image.shape) == 3:
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    
    # Apply noise more naturally
    grainy_image = image.astype(np.float32)
    grainy_image = grainy_image * (1 - amount) + (grainy_image + noise) * amount
    grainy_image = np.clip(grainy_image, 0, 255).astype(np.uint8)
    
    return grainy_image

# Function to apply tint
def apply_tint(image, r_scale, g_scale, b_scale):
    if r_scale == 1.0 and g_scale == 1.0 and b_scale == 1.0:
        return image
    
    # Split the image into channels
    b, g, r = cv2.split(image)
    
    # Apply tint by scaling each channel
    r = np.clip(r * r_scale, 0, 255).astype(np.uint8)
    g = np.clip(g * g_scale, 0, 255).astype(np.uint8)
    b = np.clip(b * b_scale, 0, 255).astype(np.uint8)
    
    # Merge channels back
    return cv2.merge([b, g, r])

# Function to apply all filters to an image
def apply_filters(image):
    # Convert PIL Image to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        # Convert RGB to BGR (OpenCV format)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Apply contrast and brightness
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=(brightness-1.0)*100)
    
    # Apply saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply tint
    image = apply_tint(image, tint_r, tint_g, tint_b)
    
    # Apply vignette
    image = apply_vignette(image, vignette)
    
    # Apply film grain
    image = apply_grain(image, grain)
    
    return image

# Function to convert OpenCV image to PIL Image
def cv2_to_pil(image):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

# Function to process image and return both original and processed
def process_image(uploaded_file):
    # Read image
    image = Image.open(uploaded_file)
    
    # Store original image
    st.session_state.original_image = image.copy()
    
    # Apply filters
    processed_cv = apply_filters(image)
    
    # Convert back to PIL for display
    processed_pil = cv2_to_pil(processed_cv)
    
    # Store processed image
    st.session_state.processed_image = processed_pil
    
    return st.session_state.original_image, st.session_state.processed_image

# Function to process video frame
def process_frame(frame):
    # Apply filters
    return apply_filters(frame)

# Create tabs for different media types
tab1, tab2, tab3 = st.tabs(["Image", "Video", "Webcam"])

with tab1:
    st.markdown("<h2 class='sub-header'>Image Filtering</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Process image
        original, processed = process_image(uploaded_file)
        
        # Display images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original")
            st.image(original, use_column_width=True)
        
        with col2:
            st.markdown("### Processed")
            st.image(processed, use_column_width=True)
        
        # Download button
        if st.session_state.processed_image is not None:
            # Save processed image to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            st.session_state.processed_image.save(temp_file.name)
            
            with open(temp_file.name, "rb") as file:
                btn = st.download_button(
                    label="Download Processed Image",
                    data=file,
                    file_name="processed_image.png",
                    mime="image/png"
                )

with tab2:
    st.markdown("<h2 class='sub-header'>Video Filtering</h2>", unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader("Upload a video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Save uploaded video to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_video.read())
        
        # Process video
        video_path = temp_file.name
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create a temporary file for the processed video
        processed_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
        
        # Progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process video frames
        frame_count = 0
        
        # Preview frames
        preview_col1, preview_col2 = st.columns(2)
        preview_original = preview_col1.empty()
        preview_processed = preview_col2.empty()
        
        preview_col1.markdown("### Original")
        preview_col2.markdown("### Processed")
        
        # Process button
        if st.button("Process Video"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = process_frame(frame)
                
                # Write processed frame
                out.write(processed_frame)
                
                # Update progress
                frame_count += 1
                progress = int(frame_count / total_frames * 100)
                progress_bar.progress(progress / 100)
                status_text.text(f"Processing: {progress}% complete")
                
                # Update preview every 10 frames
                if frame_count % 10 == 0:
                    preview_original.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                    preview_processed.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Release resources
            cap.release()
            out.release()
            
            # Show completion message
            status_text.text("Processing complete!")
            
            # Provide download link
            with open(processed_video_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
            
            # Clean up temporary files
            os.unlink(temp_file.name)

with tab3:
    st.markdown("<h2 class='sub-header'>Webcam Filtering</h2>", unsafe_allow_html=True)
    
    if not st.session_state.camera_found:
        st.error("❌ No camera found! Please check your camera connection.")
        st.info("💡 Make sure your camera is properly connected and not in use by another application.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.webcam_on:
                if st.button("Start Webcam"):
                    st.session_state.webcam_on = True
                    st.rerun()
        
        with col2:
            if st.session_state.webcam_on:
                if st.button("Stop Webcam"):
                    st.session_state.webcam_on = False
                    st.rerun()
        
        # Display webcam status
        st.write(f"Webcam is {'ON' if st.session_state.webcam_on else 'OFF'}")
        
        # Webcam frame placeholder
        webcam_frame = st.empty()
        
        # If webcam is on, capture and process frames
        if st.session_state.webcam_on:
            cap = None
            try:
                # Initialize camera with detected backend
                cap = cv2.VideoCapture(st.session_state.camera_index, st.session_state.camera_backend)
                
                if not cap.isOpened():
                    st.error("Failed to open camera. Please try again.")
                    st.session_state.webcam_on = False
                else:
                    # Set camera properties for better performance
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    frame_count = 0
                    max_frames = 50  # Limit frames before refresh to prevent infinite loop
                    
                    while st.session_state.webcam_on and frame_count < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to capture frame from webcam.")
                            break
                        
                        # Process frame
                        processed_frame = process_frame(frame)
                        
                        # Display side by side
                        combined_frame = np.hstack((frame, processed_frame))
                        
                        # Convert to RGB for display
                        combined_frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame
                        webcam_frame.image(combined_frame_rgb, caption="Original (Left) vs Processed (Right)", use_column_width=True)
                        
                        frame_count += 1
                        time.sleep(0.03)  # ~30 FPS
                    
                    # Auto-refresh for continuous streaming
                    if st.session_state.webcam_on:
                        time.sleep(0.1)
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
                st.info("Try refreshing the page or check if another app is using the camera.")
            finally:
                st.session_state.webcam_on = False
                if cap is not None:
                    cap.release()
        
        # Add static camera capture option
        st.write("---")
        st.write("📸 Or capture a single photo:")
        camera_file = st.camera_input("Take a picture")
        
        if camera_file is not None:
            # Process the captured image
            image = Image.open(camera_file)
            image_np = np.array(image.convert("RGB"))
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Process frame
            result = process_frame(image_bgr)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.header("Original")
                st.image(image_np)
            with col2:
                st.header("With Filter")
                st.image(result_rgb)
            
            # Download button
            processed_img = cv2.imencode('.jpg', result)[1].tobytes()
            st.download_button(
                label="Download Filtered Image",
                data=processed_img,
                file_name="filtered_photo.jpg",
                mime="image/jpeg"
            )

# Footer
st.markdown("---")
st.markdown("### How to Use")
st.markdown("""
1. Select the tab for your media type (Image, Video, or Webcam)
2. Upload your media or activate your webcam
3. Adjust the filter parameters in the sidebar
4. Download the processed media when satisfied
""")

st.markdown("### Supported Formats")
st.markdown("""
- Images: JPG, JPEG, PNG
- Videos: MP4, AVI, MOV
""")

# Add this function after imports
def find_working_camera():
    """Find the first working camera"""
    # Try common camera indices with different backends
    backends_to_try = []
    
    if os.name == 'nt':  # Windows
        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    else:  # Linux/Mac
        backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
    
    for backend in backends_to_try:
        for i in range(5):  # Try indices 0-4
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.release()
                        return i, backend
                cap.release()
            except:
                continue
    
    # Fallback - try default method
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    return i, cv2.CAP_ANY
            cap.release()
        except:
            continue
    
    return None, None

# Initialize session state
if 'webcam_on' not in st.session_state:
    st.session_state.webcam_on = False
if 'camera_found' not in st.session_state:
    st.session_state.camera_found = False
    # Try to find camera on startup
    cam_idx, backend = find_working_camera()
    if cam_idx is not None:
        st.session_state.camera_index = cam_idx
        st.session_state.camera_backend = backend
        st.session_state.camera_found = True