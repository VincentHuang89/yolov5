240325:get_datasets/get_label_for_yolo.py   针对船舶样本数据集，结合AIS数据生成label数据集，附加检查图片标注的功能
yolov5/compare_pred_with_labels.py   利用预训练模型实现船舶通行时间以及位置的预测，并与AIS数据进行比较，重点分析预测的精度和遗漏程度
yolov5/data/ship.yaml  指定训练集和验证集的图片位置和label位置

240420: 
ship_online_detecter/ship_detect_experiment.py: 用以遍历指定时间内所有数据，依照所设置的时间步长来生成图片样本，并调用yolov5/detect_func.py（需依照yolov5的检测目标，选择合适的预训练参数模型/sda1/huangwj/DAS/DataSet/local_samples/model/ 以及对应的.yaml文件）依次输出检测结果，并汇总到pred.csv文件中。
yolov5/toReducePred.py：Pred.csv文件保存每个数据样本的检测结果，需要依照一定规则将不同样本的检测结果合并，用以和AIS记录进行对比（注意生成记录时候对船舶ais发送间隔的筛选）
ship_online_detecter/show_samples.py：在获得疑似没有上报船舶记录后，用以生成对应时间的图片样本。

