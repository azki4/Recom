from flask import Flask, request, jsonify
from surprise import Dataset, Reader, KNNBasic

app = Flask(__name__)

# Load the MovieLens dataset
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()

# Use KNNBasic algorithm
algo = KNNBasic()
algo.fit(trainset)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    item_id = int(request.args.get('item_id'))
    prediction = algo.predict(user_id, item_id)
    return jsonify({'rating': prediction.est})

if __name__ == '__main__':
    app.run(debug=True)
