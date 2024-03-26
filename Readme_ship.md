240325:get_datasets/get_label_for_yolo.py   针对船舶样本数据集，结合AIS数据生成label数据集，附加检查图片标注的功能
yolov5/compare_pred_with_labels.py   利用预训练模型实现船舶通行时间以及位置的预测，并与AIS数据进行比较，重点分析预测的精度和遗漏程度
yolov5/data/ship.yaml  指定训练集和验证集的图片位置和label位置
