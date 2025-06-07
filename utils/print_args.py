# -*- coding: utf-8 -*-

# ***************************************************
# * File        : print_args.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-03
# * Version     : 0.1.110322
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def print_args(args):
    # ------------------------------
    # Basic Config
    # ------------------------------
    logger.info(f'{100 * "-"}')
    logger.info(f'Args in experiment:')
    logger.info(f'{100 * "-"}')
    logger.info("\033[1m" + "Basic Config" + "\033[0m")
    logger.info(f'  {"Task Name:":<20}{args.task_name:<20}{"Des:":<20}{args.des:<20}')
    logger.info(f'  {"Is Training:":<20}{args.is_training:<20}{"Is Testing:":<20}{args.is_testing:<20}')
    logger.info(f'  {"Is Forecasting:":<20}{args.is_forecasting:<20}')
    logger.info(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    logger.info("")
    # ------------------------------
    # Data Loader
    # ------------------------------
    logger.info("\033[1m" + "Data Loader" + "\033[0m")
    logger.info(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    logger.info(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    logger.info(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    logger.info(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    logger.info(f'  {"Test results:":<20}{args.test_results:<20}')
    logger.info(f'  {"Predict results:":<20}{args.predict_results:<20}')
    logger.info("")
    # ------------------------------
    # task
    # ------------------------------
    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        logger.info("\033[1m" + "Forecasting Task" + "\033[0m")
        logger.info(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        logger.info(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
        logger.info(f'  {"Train ratio:":<20}{args.train_ratio:<20}{"Test ratio:":<20}{args.test_ratio:<20}')
        logger.info(f'  {"Inverse:":<20}{args.inverse:<20}{"Scale:":<20}{args.scale:<20}')
        logger.info(f'  {"Embed:":<20}{args.embed:<20}')
        logger.info("")
    elif args.task_name == 'imputation':
        logger.info("\033[1m" + "Imputation Task" + "\033[0m")
        logger.info(f'  {"Mask Rate:":<20}{args.mask_rate:<20}')
        logger.info("")
    elif args.task_name == 'anomaly_detection':
        logger.info("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        logger.info(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
        logger.info("")
    # ------------------------------
    # MOdel Parameters
    # ------------------------------
    logger.info("\033[1m" + "Model Parameters" + "\033[0m")
    logger.info(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    logger.info(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    logger.info(f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    logger.info(f'  {"n heads:":<20}{args.n_heads:<20}{"e layers:":<20}{args.e_layers:<20}')
    logger.info(f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{args.d_ff:<20}')
    logger.info(f'  {"Moving Avg:":<20}{args.moving_avg:<20}{"Factor:":<20}{args.factor:<20}')
    logger.info(f'  {"Distil:":<20}{args.distil:<20}{"Dropout:":<20}{args.dropout:<20}')
    logger.info(f'  {"Embed:":<20}{args.embed:<20}{"Activation:":<20}{args.activation:<20}')
    logger.info(f'  {"Output Attention:":<20}{args.output_attention:<20}')
    logger.info("")
    # ------------------------------
    # Run Parameters
    # ------------------------------
    logger.info("\033[1m" + "Run Parameters" + "\033[0m")
    logger.info(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.iters:<20}')
    logger.info(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    logger.info(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    logger.info(f'  {"Loss:":<20}{args.loss:<20}{"Lradj:":<20}{args.lradj:<20}')
    logger.info(f'  {"Use Amp:":<20}{args.use_amp:<20}{"Use DTW:":<20}{args.use_dtw:<20}')
    logger.info("")
    # ------------------------------
    # GPU
    # ------------------------------
    logger.info("\033[1m" + "GPU" + "\033[0m")
    logger.info(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU Type:":<20}{args.gpu_type:<20}')
    logger.info(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    logger.info("")
    # ------------------------------
    # De-stationary Projector Params
    # ------------------------------
    logger.info("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    logger.info(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}') 
    # logger.info("")
    logger.info(f'{100 * "-"}')




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
