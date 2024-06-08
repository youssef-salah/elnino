
from flask import Flask, request, jsonify, make_response
import pandas as pd
import numpy as np
from keras.models import load_model


app = Flask(__name__)



# Replace 'path_to_your_model.h5' with the actual path to your HDF5 file
elnino = load_model('.h5')
print("Top-level groups:", list(elnino.keys()))


@app.route('/elnino', methods=['POST', 'GET'])
def handle_data():
   try:

      if request.method == 'POST':
         jdata = request.json
         jdata_df = pd.DataFrame([jdata])
         
      elif request.method == 'GET':
         # Get parameters from query string
         params = ["previous_year" , "eleven_months_ago" , "ten_months_ago" ,
                   "nine_months_ago" , "eight_months_ago" , "seven_months_ago",
                    "six_months_ago" , "five_months_ago" , "four_months_ago",
                     "three_months_ago" , "two_months_ago" , "last_month" ]
         jdata = {param : float (request.args.get(param , 0)) for param in params}
         jdata_df = pd.DataFrame([jdata])

      # Call predict on the model, reshape for single sample
      elnino_forcasting = elnino.predict(jdata_df)
      elnino_forcasting = elnino_forcasting.tolist()

# Create response with CORS headers
      response = make_response(jsonify({'elnino forcasting': elnino_forcasting[0 , 1 , 2]}))
      response.headers['Access-Control-Allow-Origin'] = '*'
      response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
      response.headers['Access-Control-Allow-Methods'] = 'GET, POST'

      return response

   except Exception as e :
      response = make_response(jsonify({'error' : str(e)}), 400)

      return response

if __name__ == '__main__':
 app.run(debug=True, host='0.0.0.0', port=3000)


