from flask import Flask, request, jsonify
from flask_cors import CORS
from logistic_regression import detector

app = Flask(__name__)
CORS(app)

@app.route('/endpoint', methods=['POST'])
def handle_data():
    data = request.json  # Extract JSON data from the request
    received_text = data.get('text', '')  # Extract the 'text' field from the JSON data
    #return jsonify({'received_text': received_text})
    review = data["text"]
    print(f"{detector(review)}: {review}")
    return {"text": detector(review)}

if __name__ == '__main__':
    app.run(debug=True)
