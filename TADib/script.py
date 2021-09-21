from subprocess import call
import sys

lr = [3]
hid = [2]
bt = [16]
pred = [32]
# groups = [1,2]
point = [4,6,8]
for l in lr:
       for hd in hid:
              # for pr in pred:
              for b in bt:
                     for p in point:
                            cmd = "python -u run_credit.py --hidden-size {} --batch {} --lradj {} --point_part {} > Credit_Results/NEW920_20/BLC_credit_BiPointMask_nhid{}_bt{}_lrtype{}_point{}.log 2>&1".format(hd,b,l, p, hd,b,l,p)
                            print(cmd)
                            call(cmd, shell=True)


#python -u run_SelfNet.py --hidden-size 16 --batch 32 --learning_rate 0.001 --pred_len 48 --seq_mask_range_high 1 > Log/swat/NewEnsembleSeqMask1Stack_nhid16_bt32_lr0.001_pred48_smhigh1.log 2>&1
