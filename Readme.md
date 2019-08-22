## 本库用于存放推荐相关算法代码和文档
#### tfrecords/
存放tfrecords文件的生成和使用代码
利用spark读取数据保存成tfrecods这里提供两个思路。  
1.如果集群上已经安装了tensorflow，可以用本文件下的代码进行处理。  
2.对于没有安装tensorflow的集群，可以参考:[spark-tensorflow-connector](https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector) 来将数据转换成tfrecords。这个更快。  
#### wd_model/
使用Dataset,Feature_columns ,Estimator这三个tensorflow API可以快速搭建99%的CTR预估模型。  
  
但是离线测试发现 Feature_columns存在很大rt问题，差不多消耗了整个前向过程的1/3。对于小企业来说，serving性能没有做到最优，只能从模型端减少性能上的消耗。  
  
所以这里我自己实现了一个wide&deep模型。支持连续特征，类别特征，embedding特征，特征交叉等常见操作，也支持list特征以及各种数据类型。能很方便的将生成的模型export，以及支持分布式训练。  
修改模型的核心代码可以很方便的扩展到其他DNN模型，现在暂时只支持Wide，DNN，WD三类。   

#### essm_model/
实现essm模型，参考了实现代码[essm](https://github.com/yangxudong/deeplearning/tree/master/esmm)  

###### 运行方式  
- 配置config/inputs.json，配置好输入的字段，以及需要做的操作，以及相对应的操作  
- 配置好对应的路径  
- 运行train_wd_model.py即可。    

##### config/  
存放模型配置文件  

#### utils/

