import json
from flask import Flask, request, jsonify
import util

app = Flask(__name__)


# @app.route('/get_District_names', methods=['GET'])
# def get_District():
#     response = jsonify({
#         'District': util.get_District()
#     })
#     response.headers.add('Access-Control-Allow-Origin', '*')

#     return response


@app.route('/predict_Water_Quality', methods=['GET', 'POST'])
def get_estimated_Quality():
    if request.method == 'POST':
        request_data = request.data
        request_data = json.loads(request_data.decode('utf-8'))
        print(request_data)
        DO=request_data["DO"]
        BOD=request_data["BOD"]
        totalCaliform=request_data["totalCaliform"]
        fecalCaliform=request_data["fecalCaliform"]
        print(DO)
        x=util.get_estimated_Quality(float(DO), float(BOD), float(totalCaliform), float(fecalCaliform))
        print(x)
        response = jsonify({
            'estimated_Quality': int(x)
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    return  jsonify({
            'estimated_Quality': 0
        })


if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    app.run()
