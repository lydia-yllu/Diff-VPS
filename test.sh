#!/bin/bash
# The candidate competitors are listed here:
# MODEL_NAMES=('2022-MIR-PNSPlus' '2021-NIPS-AMD' '2021-MICCAI-PNSNet' '2021-ICCV-FSNet' '2021-ICCV-DCFNet' '2020-TIP-MATNet' '2020-MICCAI-23DCNN' '2020-AAAI-PCSA' '2019-TPAMI-COSNet' '2021-TPAMI-SINetV2' '2021-MICCAI-SANet' '2020-MICCAI-PraNet' '2020-MICCAI-ACSNet' '2018-TMI-UNet++' '2015-MICCAI-UNet')

# MODEL_NAMES=('DDP')

# nohup: 后台运行，&符号表示在后台运行该命令，>>指定保留原有内容输出文件的名称
# 2>&1 表示将标准错误（stderr）重定向到标准输出（stdout），这样可以确保将所有输出都保存到日志文件中，而不会在终端中显示
# for MODEL_NAME in ${MODEL_NAMES[*]}
# do
#   nohup python -u ./Diff_VPS/evaluation/vps_evaluator.py --data_lst TestEasyDataset/Seen --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> snapshot/eval_logs/$MODEL_NAME-Easy-Seen.log 2>&1 &
#   nohup python -u ./Diff_VPS/evaluation/vps_evaluator.py --data_lst TestEasyDataset/Unseen --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> snapshot/eval_logs/$MODEL_NAME-Easy-Unseen.log 2>&1 &
#   nohup python -u ./Diff_VPS/evaluation/vps_evaluator.py --data_lst TestHardDataset/Seen --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> snapshot/eval_logs/$MODEL_NAME-Hard-Seen.log 2>&1 &
#   nohup python -u ./Diff_VPS/evaluation/vps_evaluator.py --data_lst TestHardDataset/Unseen --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> snapshot/eval_logs/$MODEL_NAME-Hard-Unseen.log 2>&1 &
# done

# python -u ./Diff_VPS/test_diff_vps.py --data_lst TestEasyDataset/Seen  --txt_name '_test_'
python -u ./Diff_VPS/test_diff_vps.py --data_lst TestEasyDataset/Unseen  --txt_name '_test_'
python -u ./Diff_VPS/test_diff_vps.py --data_lst TestHardDataset/Seen  --txt_name '_test_'
python -u ./Diff_VPS/test_diff_vps.py --data_lst TestHardDataset/Unseen  --txt_name '_test_'