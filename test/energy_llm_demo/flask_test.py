from flask import Flask
from flask import jsonify
from flask import request
from flask_executor import Executor
import requests
import json
import random
import traceback

app=Flask(__name__)
executor = Executor(app)

def test_opt():
    try:
        print("opt begin:")
        dd = {}
        dd["aa"] = dd["aa"] + 1
    except:
        print("error")
        traceback.print_exc()

@app.route("/test", methods=["POST"])
def request_process():
    executor.submit(test_opt)

    resp = {"code": 200, "msg": "请求已受理~"}

    return jsonify(resp)

if __name__ == "__main__":
    app.run(port=9555, host="0.0.0.0")