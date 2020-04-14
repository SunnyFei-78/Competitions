nn_generator.py: 依据batchsize生成数据给模型训练和预测

NN_Training_Validation.py：验证集上训练模型，得到n组参数和n个模型。

NN_Training_Test.py：读取验证集上的n组参数训练模型，保存模型并且预测测试集结果。

train.sh: 脚本文件，在后台运行训练代码并将信息输出到.log文件中

predict.py: 读取n个模型的预测结果，并对结果做加权平均。

submission_for_8_nn.csv: 8个神经网络模型预测结果融合后得到的最终提交结果



