# -*- coding: utf-8 -*-
# @Time    : 2018/4/14 11:42
# @Author  : David
# @Email   : mingren4792@126.com
# @File    : test.py
# @Software: PyCharm
import numpy as np
import pandas as pd

positive = 0
negtive = 0
trial_y = pd.read_csv('./wassa2018_data/trial-v3.labels', header=None)
# trial_y = pd.read_csv('./result/cnn-lstm.csv', header=None)
trial_y[0] = trial_y[0] .replace({'sad': 0,
                                  'joy': 1, 'disgust': 2,
                                  'surprise': 3, 'anger': 4,
                                  'fear': 5
                                  })

result_output = pd.DataFrame(data={"sentiment": trial_y[0]})


trial_pre_labels = pd.read_csv('./result/train_trial_test_28epoch.csv')
trial_pre_labels["sentiment"] = trial_pre_labels["sentiment"] .replace({0: 'sad',
                                                                       1: 'joy', 2: 'disgust',
                                                                       3: 'surprise', 4: 'anger',
                                                                       5: 'fear'
                                                                        })


# Use pandas to write the comma-separated output file
# result_output.to_csv("./result/cnnCH.csv", index=False, quoting=3)


# #计算六类填充
# 计算皮尔斯系数
# trial_pre = pd.read_csv('./wassa2018_data/six_lable.csv')
# enmotoion_dict = ['anger', 'disgust', 'fear', 'sad', 'surprise', 'joy']
#
# trial_pre["sentiment"] = trial_pre["sentiment"] .replace({0: 'anger',
#                                                           1: 'disgust', 2: 'fear',
#                                                           3: 'sad', 4: 'surprise',
#                                                           5: 'joy'
#                                                           })
# trial_pre_labels = trial_pre.values
# print(trial_pre_labels)

# # 随机森林生成label
# pre = pd.read_csv('./wassa2018_data/Bag_of_Words_model_logistic.csv')
# trial_pre_labels = pre['sentiment']
# trial_pre_labels = trial_pre_labels.values

filename = './wassa2018_data/trial_pre.labels'
with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
    for i in range(len(trial_pre_labels)):
        # f.write(trial_pre_labels[i][0] + '\n')
        f.write(trial_pre_labels.values[i][0] + '\n')

#
# # 计算准确率
# for i in range(len(trial_y[0])):
#     if trial_y[0][i] == trial_pre_labels['sentiment'][i]:
#         positive += 1
#     else:
#         negtive += 1
#
# # print(positive+negtive)
# print(positive/(positive + negtive))





