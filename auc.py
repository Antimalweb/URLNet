import numpy as np
import argparse
import pdb

parser = argparse.ArgumentParser(description='Nill')
parser.add_argument('--input_path', default="results/svm_bow_lexical/baseline1/", type=str)
parser.add_argument('--input_file', type=str)
parser.add_argument('--threshold', default=0, type=float)
args = parser.parse_args()
print(args)

threshold = args.threshold
frequence = 0.05
with open(args.input_path + args.input_file) as f:
    data = f.readlines()
out_file = args.input_file.split(".")[0] + '.auc'
f =  open(args.input_path + out_file, 'w')

total = 0.
negative = 0.
positive = 0.
predictions = []
for line in data[1:]:
    arr = line[:-1].split('\t')
    idx = int(arr[0])
    if idx == 1:
        positive += 1
        predictions.append([1, float(arr[-1])])
    else:
        negative += 1
        predictions.append([-1, float(arr[-1])])

total = negative + positive
f.write('Total number of instances: ' + str(total) + '\n')
f.write('P: ' + str(positive) + '\n')
f.write('N: ' + str(negative) + '\n')
f.write('-'*30 + '\n')
f.write('Figure\n')
f.write('-'*30 + '\n')
f.write('decision_boundary\tTP\tFP\tTPR\tFPR\n')
TP = 0
FP = 0
TP_0 = 0
FP_0 = 0
AUC = 0.

target_TP = 1
target_FPR = 1e-5
table=[]
table_header = []
predictions.sort(key= lambda x: x[-1])
predictions = predictions[::-1]
for i, pred in enumerate(predictions):
    if pred[0] > 0:
        TP+= 1
        if pred[1] > threshold:
            TP_0+= 1
        AUC+= FP
    else:
        FP+= 1
        if pred[1] > threshold:
            FP_0+= 1

    if ((TP > target_TP) or (i == total - 1)):
        target_TP = target_TP + frequence*positive
        f.write(str(pred[1]) + '\t' + str(TP) + '\t' + str(FP) + '\t')
        f.write(str(float(TP)/positive) + '\t' + str(float(FP)/negative) + '\n')

    if ((FP > target_FPR*negative) or (i==total-1)):
        table_header.append(target_FPR)
        table.append(float(TP)/positive)
        target_FPR = target_FPR * 10


f.write('-'*30 + '\n')
f.write('Table\n')
f.write('-'*30 + '\n')
f.write('FPR\tTPR\n')
for i,j in zip(table,table_header):
    f.write(str(j) + '\t' + str(i) + '\n')
f.write('-'*30 + '\n')
f.write('AUC:\t' + str(1. - (float(AUC)/positive)/negative) + '\n')
f.write('When the decision boundary is set to be 0\n')
f.write('TP:\t' + str(TP_0) + '\n')
f.write('FN:\t' + str(positive - TP_0) + '\n')
f.write('FP:\t' + str(FP_0) + '\n')
f.write('TN:\t' + str(negative - FP_0) + '\n')
f.close()
