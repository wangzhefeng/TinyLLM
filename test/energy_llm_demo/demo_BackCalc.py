from flask import Flask
from flask import jsonify
from flask import request
from flask_executor import Executor
import requests
import json
import random
import time
import traceback
import copy
from scipy.optimize import minimize
import numpy as np

app=Flask("BackCalc")
executor = Executor(app)

bounds_dict = {"176c0991f24e30c2b25a9dbf1185b7b9": [7.0, 12.0],
               "5eb413037ba16ea6108c12e0d6353be3": [7.0, 12.0],
               "3da72e052a0b48759b0f4633df42235a": [7.0, 12.0]}

def dr_opt_run_fix(weights, init_value, bounds, dr_value, hvac_load_ori):
    w_v = [weights["ch_water_outlet_temperature"],
           weights["weather"],
           weights["bias"]]
    x_v = [init_value["ch_water_outlet_temperature"],
           init_value["weather"],
           1]
    x0 = [x_v[0]]
    
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

def dr_BackCalc(args):
    try:
        print("opt begin:")

        url = "http://host.docker.internal:49090/AIEnergy/strategyUpdateLoad"

        opt_x = json.loads(args["opt_x"])
        print("opt_x", opt_x)
        opt_x_modified = copy.deepcopy(opt_x)
        
        weights = json.loads(args["weights"])
        print("weights", weights)

        hvac_load_ori = weights["ch_water_outlet_temperature"] * opt_x["ch_water_outlet_temperature"] + \
                                weights["weather"] * opt_x["weather"] + weights["bias"] + args["opt_fx"]
        
        print("hvac load ori", hvac_load_ori)

        bounds = bounds_dict.get(args["node_id"], [9, 20])

        if args["alter_mode"] == "ch_water_outlet_temperature":
            ch_water_outlet_temperature_modified = max(bounds[0], min(args["alter_value"], bounds[1]))

            opt_fx_modified = hvac_load_ori - (weights["ch_water_outlet_temperature"] * ch_water_outlet_temperature_modified + \
                            weights["weather"] * opt_x["weather"] + weights["bias"])
            
            is_success = True
            
            print("alter mode:", args["alter_mode"], is_success, opt_fx_modified, ch_water_outlet_temperature_modified)
        else:
            is_success, y_opt, x_opt = dr_opt_run_fix(weights, opt_x, bounds, args["alter_value"], hvac_load_ori)
                    
            print("alter mode:", args["alter_mode"], is_success, y_opt, x_opt)

            if is_success:
                opt_fx_modified = y_opt
                ch_water_outlet_temperature_modified = x_opt[0]
            else:
                opt_fx_modified = hvac_load_ori - (weights["ch_water_outlet_temperature"] * opt_x["ch_water_outlet_temperature"] + \
                                weights["weather"] * opt_x["weather"] + weights["bias"])
                ch_water_outlet_temperature_modified = opt_x["ch_water_outlet_temperature"]
            
        opt_x_modified["ch_water_outlet_temperature"] = ch_water_outlet_temperature_modified


        strategy_res = {"resp_id": args["resp_id"],
                        "node_id": args["node_id"],
                        "node_name": args["node_name"],
                        "opt_fx": opt_fx_modified,
                        "opt_x": json.dumps(opt_x_modified)}
        headers = {'Content-Type': 'application/json'}
        print("the result of strategy is:", json.dumps(strategy_res))

        response = requests.request("POST", url, headers=headers, data=json.dumps(strategy_res))
        print("the response of strategy upload is:", response.text)
    except:
        print("calculate error")
        traceback.print_exc()


@app.route("/BackCalc", methods=["POST"])
def request_process():
    input_dict = request.get_json()

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("\ntask accept at " + current_time)

    print("input_json", input_dict)
    
    calc_parm = {}

    assert "resp_id" in input_dict.keys()
    calc_parm["resp_id"] = input_dict["resp_id"]

    assert "node_id" in input_dict.keys()
    calc_parm["node_id"] = input_dict["node_id"]

    assert "node_name" in input_dict.keys()
    calc_parm["node_name"] = input_dict["node_name"]

    assert "weights" in input_dict.keys()
    calc_parm["weights"] = input_dict["weights"]

    assert "opt_x" in input_dict.keys()
    calc_parm["opt_x"] = input_dict["opt_x"]

    assert "opt_fx" in input_dict.keys()
    calc_parm["opt_fx"] = input_dict["opt_fx"]

    assert "alter_mode" in input_dict.keys()
    calc_parm["alter_mode"] = input_dict["alter_mode"]

    assert "alter_value" in input_dict.keys()
    calc_parm["alter_value"] = input_dict["alter_value"]

    print(calc_parm)

    executor.submit(dr_BackCalc, calc_parm)

    resp = {"code": 200, "msg": "请求已受理~"}

    return jsonify(resp)

if __name__ == "__main__":
    app.run(port=13351, host="0.0.0.0")