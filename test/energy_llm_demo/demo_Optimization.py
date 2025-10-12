from flask import Flask
from flask import jsonify
from flask import request
from flask_executor import Executor
import requests
import json
import random
import time
import traceback
from scipy.optimize import minimize
import numpy as np
import copy

app=Flask("Optimization")
executor = Executor(app)

bounds_dict = {"176c0991f24e30c2b25a9dbf1185b7b9": [7.0, 12.0],
               "5eb413037ba16ea6108c12e0d6353be3": [7.0, 12.0],
               "3da72e052a0b48759b0f4633df42235a": [7.0, 12.0]}

def dr_opt_run(weights, init_value, bounds, dr_value):
    w_v = [weights["ch_water_outlet_temperature"],
           weights["weather"],
           weights["bias"]]
    x_v = [init_value["ch_water_outlet_temperature"],
           init_value["weather"],
           1]
    x0 = [x_v[0]]

    hvac_load_ori = np.dot(w_v, x_v)
    
    def obj(x):
        x_v[0] = x[0]
        hvac_load_opt = np.dot(w_v, x_v)
        hvac_load_down = hvac_load_ori - hvac_load_opt
        
        return -hvac_load_down
    
    bds=[bounds]
    cons = [{'type': 'ineq', 'fun': lambda x: dr_value - (hvac_load_ori - np.dot([x[0], x_v[1], x_v[2]], w_v))}]
    
    res = minimize(obj,
                x0=x0,
                method="SLSQP",
                constraints=cons,
                bounds=bds,
                options={'maxiter': 1000, 'ftol': 0.1, 'iprint': 0, 'disp': False,
                     'eps': 100, 'finite_diff_rel_step': None})
    return res.success, -res.fun, res.x 


def dr_opt(args):
    try:
        print("opt begin:")

        url = "http://host.docker.internal:49090/AIEnergy/strategyAdd"

        init_value = json.loads(args["init_value"])
        print("init_value", init_value)
        opt_x = copy.deepcopy(init_value)
        
        weights = json.loads(args["weights"])
        print("weights", weights)

        bounds = bounds_dict.get(args["node_id"], [9, 20])
        
        is_success, y_opt, x_opt = dr_opt_run(weights, init_value, bounds, args["command_value"])

        print("the opt res is:", is_success, y_opt, x_opt)
        
        if is_success:
            opt_fx = y_opt
            opt_x["ch_water_outlet_temperature"] = x_opt[0]
        else:
            opt_fx = weights["ch_water_outlet_temperature"] * init_value["ch_water_outlet_temperature"] + \
                    weights["weather"] * init_value["weather"] + weights["bias"] + init_value["fixed_load"]

        strategy_res = {"resp_id": args["resp_id"],
                        "node_id": args["node_id"],
                        "command_value": args["command_value"],
                        "node_name": args["node_name"],
                        "opt_fx": opt_fx,
                        "opt_x": json.dumps(opt_x)}
        headers = {'Content-Type': 'application/json'}
        print("the result of strategy is:", strategy_res)

        response = requests.request("POST", url, headers=headers, data=json.dumps(strategy_res))
        print("the response of strategy upload is:", response.text)
    except:
        print("calculate error")
        traceback.print_exc()


@app.route("/Optimization", methods=["POST"])
def request_process():
    input_dict = request.get_json()

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("\ntask accept at " + current_time)
    
    opt_parm = {}

    assert "resp_id" in input_dict.keys()
    opt_parm["resp_id"] = input_dict["resp_id"]

    assert "node_id" in input_dict.keys()
    opt_parm["node_id"] = input_dict["node_id"]

    assert "node_name" in input_dict.keys()
    opt_parm["node_name"] = input_dict["node_name"]

    assert "command_value" in input_dict.keys()
    opt_parm["command_value"] = input_dict["command_value"]

    assert "weights" in input_dict.keys()
    opt_parm["weights"] = input_dict["weights"]

    assert "init_value" in input_dict.keys()
    opt_parm["init_value"] = input_dict["init_value"]

    assert "x" in input_dict.keys()
    opt_parm["x"] = input_dict["x"]

    print("opt_parm:", opt_parm)

    executor.submit(dr_opt, opt_parm)

    resp = {"code": 200, "msg": "请求已受理~"}

    return jsonify(resp)

if __name__ == "__main__":
    app.run(port=13350, host="0.0.0.0")