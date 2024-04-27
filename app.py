from flask import Flask, request, jsonify
from similarity import find_similar_texts

app = Flask(__name__)

@app.route('/api/similarity', methods=['POST'])
def similarity():
    data = request.get_json()
    old_data = data['old_data']
    new_data = data['new_data']
    top_n = data.get('top_n', 5)  # Default to returning top 5 similar items
    results = find_similar_texts(old_data, new_data, top_n)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
