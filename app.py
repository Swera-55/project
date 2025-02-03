from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import nltk
import sys
from textblob import TextBlob

# Download all required NLTK data
print("Downloading required NLTK data...")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('brown')
nltk.download('wordnet')

try:
    # Import after NLTK downloads
    from Deception_Detection import POS_Tagging, count_sentiment_words
    print("Successfully imported Deception_Detection")
except Exception as e:
    print(f"Error importing Deception_Detection: {str(e)}")
    sys.exit(1)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def home():
    return "Server is running!"

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_review():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        print("Received analyze request")
        data = request.get_json()
        review_text = data.get('text', '')
        
        if not review_text:
            return jsonify({'status': 'error', 'error': 'No text provided'})
        
        print(f"Analyzing review: {review_text[:50]}...")
        
        # Write review to a temporary file for analysis
        temp_file = 'temp_review.txt'
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(review_text)
        
        try:
            # Perform analysis
            print("Starting deception detection...")
            deception_result = POS_Tagging(review_text)
            print(f"Deception result: {deception_result}")
            
            print("Starting sentiment analysis...")
            sentiment_score = count_sentiment_words(temp_file)
            print(f"Sentiment score: {sentiment_score}")
            
            response = jsonify({
                'status': 'success',
                'sentiment_score': float(sentiment_score),
                'deception_result': deception_result
            })
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        return response
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e)})

if __name__ == '__main__':
    print("Starting Flask server...")
    try:
        app.run(debug=True, port=5000, host='0.0.0.0')
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        sys.exit(1)
