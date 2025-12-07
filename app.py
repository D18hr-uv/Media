import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import lime
import lime.lime_image
import shap
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import video_processor
import json
import uuid

# Set Matplotlib backend
plt.switch_backend('Agg')

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure video frames folder
VIDEO_FRAMES_FOLDER = os.path.join(UPLOAD_FOLDER, 'video_frames')
if not os.path.exists(VIDEO_FRAMES_FOLDER):
    os.makedirs(VIDEO_FRAMES_FOLDER)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Load the trained model
try:
    model = tf.keras.models.load_model('fake_face_detection_model.h5')
except:
    print("Could not load model. Please ensure 'fake_face_detection_model.h5' exists.")
    model = None

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to check if file is a video
def is_video(filename):
    video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

# Preprocess image for model input
def preprocess_image(file_path, target_size=(224, 224)): # EfficientNetB0 standard
    try:
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        # EfficientNetB0 expects 0-255 inputs. No scaling needed for B0-B7 if weights included
        # But let's verify if explicit call does anything.
        # tf.keras.applications.efficientnet.preprocess_input is essentially a pass-through for B0-B7
        return img_array
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None

# Home route to render the upload form
@app.route('/')
def index():
    return render_template('index.html')

# LIME explanation function
def generate_lime_explanation(image, model, save_path):
    explainer = lime.lime_image.LimeImageExplainer()
    # Image is [0, 255], LIME expects double/float.
    explanation = explainer.explain_instance(image[0].astype('double'), model.predict, top_labels=1, hide_color=0, num_samples=1000)

    # temp is in range [0, 255]
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)

    # Convert temp to [0, 1] for display (mark_boundaries expects float images in [0, 1] or uint8)
    temp_display = temp.astype(np.uint8)

    img_boundry = mark_boundaries(temp_display, mask)

    plt.figure(figsize=(12, 6))

    # Display original image
    plt.subplot(1, 2, 1)
    # EfficientNet inputs are [0, 255]. To display, normalize to [0, 1]
    plt.imshow(image[0] / 255.0)
    plt.title("Original Image")
    plt.axis('off')

    # Display LIME explanation
    plt.subplot(1, 2, 2)
    plt.imshow(img_boundry)
    plt.title("LIME Explanation")
    plt.axis('off')

    # Custom Legend
    green_patch = mpatches.Patch(color='green', label='Supports Prediction')
    red_patch = mpatches.Patch(color='red', label='Opposes Prediction')
    plt.legend(handles=[green_patch, red_patch], loc='lower right')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def analyze_quarters(shap_values_rescaled):
    """
    Analyzes which quarter of the image contributes most to the decision.
    shap_values_rescaled: 2D array of SHAP values (height, width)
    """
    h, w = shap_values_rescaled.shape
    mid_h, mid_w = h // 2, w // 2

    quarters = {
        "Top-Left": shap_values_rescaled[0:mid_h, 0:mid_w],
        "Top-Right": shap_values_rescaled[0:mid_h, mid_w:w],
        "Bottom-Left": shap_values_rescaled[mid_h:h, 0:mid_w],
        "Bottom-Right": shap_values_rescaled[mid_h:h, mid_w:w]
    }

    # Calculate sum of absolute contributions
    contributions = {k: np.sum(np.abs(v)) for k, v in quarters.items()}
    total_contribution = sum(contributions.values())

    max_quarter = max(contributions, key=contributions.get)
    max_val = contributions[max_quarter]

    if total_contribution > 0:
        percentage = (max_val / total_contribution) * 100
        return f"The {max_quarter} area contributed {percentage:.1f}% to the decision.", contributions

    return f"The {max_quarter} area shows the strongest features influencing the decision.", contributions

# Advanced SHAP explanation function using DeepExplainer with proper scaling and legend
def generate_shap_explanation(image, model, save_path):
    # Using SHAP's generic Explainer (PartitionExplainer for images) which is more robust
    # than DeepExplainer/GradientExplainer for TF 2.x + EfficientNet

    # Define a prediction function that wraps the model
    def f(X):
        return model.predict(X)

    # Use an Image masker
    # "inpaint_telea" fills masked regions by inpainting.
    # image[0].shape is (224, 224, 3)
    masker = shap.maskers.Image("inpaint_telea", image[0].shape)

    explainer = shap.Explainer(f, masker)

    # Compute SHAP values
    # max_evals controls the tradeoff between speed and accuracy
    shap_values_obj = explainer(image, max_evals=300, batch_size=50)

    # Extract values from the Explanation object
    # shap_values_obj.values shape is (1, 224, 224, 3, 1) for single output
    values = shap_values_obj.values[..., 0] # Now (1, 224, 224, 3)

    # Rescale SHAP values to fit within the range [-1, 1]
    shap_values_rescaled = np.sum(values, axis=-1) # shape (1, 224, 224)
    shap_values_rescaled = shap_values_rescaled[0] # shape (224, 224)

    max_val = np.max(np.abs(shap_values_rescaled))
    if max_val > 0:
        shap_values_rescaled = shap_values_rescaled / max_val  # Normalize between -1 and 1

    # Analyze quarters
    max_quarter, contributions = analyze_quarters(shap_values_rescaled)

    # Plot the original image with SHAP values as an overlay (heatmap)
    plt.figure(figsize=(12, 6))

    # Display original image
    plt.subplot(1, 2, 1)
    # EfficientNet inputs are [0, 255]. To display, normalize to [0, 1]
    display_img = image[0] / 255.0
    plt.imshow(display_img)
    plt.title("Original Image")
    plt.axis('off')

    # Overlay SHAP values
    plt.subplot(1, 2, 2)
    plt.imshow(display_img)
    # Use coolwarm: Blue=Negative (Real), Red=Positive (Fake) usually for sigmoid output 0-1
    # But check model output interpretation.
    im = plt.imshow(shap_values_rescaled, cmap='coolwarm', alpha=0.6, vmin=-1, vmax=1)

    # Custom Legend
    # Create patches for legend
    red_patch = mpatches.Patch(color='red', label='Contributions to Fake')
    blue_patch = mpatches.Patch(color='blue', label='Contributions to Real')
    plt.legend(handles=[red_patch, blue_patch], loc='lower right')

    cbar = plt.colorbar(im, label='SHAP Value')
    cbar.set_label('Feature Contribution (Blue=Real, Red=Fake)', rotation=270, labelpad=15)

    plt.title("SHAP Explanation")
    plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return max_quarter

# Serve uploaded files
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if not model:
                 return jsonify({'error': 'Model not loaded'}), 500

            # Check if the file is a video
            if is_video(filename):
                # Process video
                session_id = str(uuid.uuid4())[:8]
                output_dir = os.path.join(VIDEO_FRAMES_FOLDER, session_id)
                
                # Process the video and get results
                results = video_processor.process_video(
                    video_path=file_path,
                    model=model,
                    output_dir=output_dir,
                    max_frames=20
                )
                
                if "error" in results:
                    return jsonify({'error': results["error"]}), 500
                
                # Get the frame with highest fake probability for visualization
                max_prob_frame = results["max_prob_frame"]
                
                # Generate explanations for the frame with highest fake probability
                img_array = preprocess_image(max_prob_frame)
                
                # Generate LIME and SHAP explanations for the representative frame
                lime_path = os.path.join(output_dir, 'lime_explanation.png')
                shap_path = os.path.join(output_dir, 'shap_explanation.png')
                
                generate_lime_explanation(img_array, model, lime_path)
                max_quarter = generate_shap_explanation(img_array, model, shap_path)
                
                # Save detailed results to a JSON file
                results_path = os.path.join(output_dir, 'results.json')
                with open(results_path, 'w') as f:
                    # Convert frame_predictions to a simpler format for JSON
                    simplified_results = results.copy()
                    simplified_results['frame_predictions'] = [
                        {
                            'frame': os.path.basename(p['frame_path']),
                            'probability': p['probability'],
                            'prediction': p['prediction']
                        } for p in results['frame_predictions']
                    ]
                    json.dump(simplified_results, f, indent=2)
                
                # Return video analysis results
                return jsonify({
                    'media_type': 'video',
                    'prediction': results['overall_prediction'],
                    'probability': results['average_fake_probability'],
                    'frames_analyzed': results['frames_analyzed'],
                    'fake_frames_count': results['fake_frames_count'],
                    'fake_frames_percentage': results['fake_frames_percentage'],
                    'max_fake_probability': results['max_fake_probability'],
                    'representative_frame': os.path.relpath(max_prob_frame, app.config['UPLOAD_FOLDER']),
                    'lime_path': os.path.relpath(lime_path, app.config['UPLOAD_FOLDER']),
                    'shap_path': os.path.relpath(shap_path, app.config['UPLOAD_FOLDER']),
                    'session_id': session_id,
                    'quarter_analysis': max_quarter # It's now a full sentence
                })
            else:
                # Process image (existing code)
                # Preprocess the uploaded image
                img_array = preprocess_image(file_path)
                if img_array is None:
                    return jsonify({'error': 'Image preprocessing failed'}), 500

                # Make a prediction
                prediction = model.predict(img_array)
                pred_class = "fake" if prediction[0][0] > 0.4 else "real"
                prob = float(prediction[0][0])

                # Generate LIME and SHAP explanations
                lime_path = os.path.join(app.config['UPLOAD_FOLDER'], 'lime_explanation.png')
                shap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'shap_explanation.png')

                generate_lime_explanation(img_array, model, lime_path)
                max_quarter = generate_shap_explanation(img_array, model, shap_path)

                # Return result
                return jsonify({
                    'media_type': 'image',
                    'prediction': pred_class,
                    'probability': prob,
                    'lime_path': 'lime_explanation.png',
                    'shap_path': 'shap_explanation.png',
                    'quarter_analysis': max_quarter # It's now a full sentence
                })

        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Supported formats: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400

# Get video frames route
@app.route('/video_frames/<session_id>', methods=['GET'])
def get_video_frames(session_id):
    try:
        frames_dir = os.path.join(VIDEO_FRAMES_FOLDER, session_id)
        
        if not os.path.exists(frames_dir):
            return jsonify({'error': 'Session not found'}), 404
        
        # Get results.json
        results_path = os.path.join(frames_dir, 'results.json')
        if not os.path.exists(results_path):
            return jsonify({'error': 'Results not found'}), 404
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Get all frame paths
        frame_data = []
        for pred in results['frame_predictions']:
            frame_name = pred['frame']
            frame_path = os.path.join(session_id, frame_name)
            frame_data.append({
                'path': frame_path,
                'probability': pred['probability'],
                'prediction': pred['prediction']
            })
        
        return jsonify({
            'frames': frame_data,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error getting video frames: {e}")
        return jsonify({'error': f'Failed to get video frames: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
