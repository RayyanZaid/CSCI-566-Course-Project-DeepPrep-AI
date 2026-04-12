from flask import Flask, request, jsonify
from flask_cors import CORS  
import os

from startInterview import processVideo

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# Directory to save uploaded videos
UPLOAD_FOLDER = 'serverFiles/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']

    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(filepath)

    feedbackSummary = processVideo(filepath)

    print("Feedback Summary:", feedbackSummary)
    
    return jsonify({'result': feedbackSummary})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)