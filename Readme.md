## 本库用于存放推荐相关算法代码和文档
#### DeepRank/   
这里写了一套从数据生成到模型生成的完整框架。对输入特征，特征工程，核心模型，模型输出分别进行了封装。  
  模型更新迭代的时候可以针对性的修改指定模块，也方便后续模型的切换，特征的迭代，以及输入数据在输入，特征工程，保存模型，线上服务各个模块的统一配置。    
  模型的实现：对一些经典的网络会逐步实现，如Wide_and_Deep,DIN,ESMM,DeepFM,DCN等。  
    
  部分实现参考了:   
  [x-deeplearning的xdl-algorithm-solution](https://github.com/alibaba/x-deeplearning/tree/master/xdl-algorithm-solution)   
  [脚本配置方式](https://github.com/zhaoxin4data/atlas/tree/master/deeplearning/uciflowwd_train/config)  
  [esmm实现](https://github.com/yangxudong/deeplearning/tree/master/esmm)   
  [tensorflow自带的分类器](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/estimator/canned)
    

#### tfrecords/
存放tfrecords文件的生成和使用代码
利用spark读取数据保存成tfrecods这里提供两个思路。  
1.如果集群上已经安装了tensorflow，可以用本文件下的代码进行处理。  
2.对于没有安装tensorflow的集群，可以参考:[spark-tensorflow-connector](https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector) 来将数据转换成tfrecords。这个更快。  
  
                            -------------------------弃用--------------------------------
#### wd_model/(已删除，合并到DeepRank)  
所以这里我自己实现了一个wide&deep模型。支持连续特征，类别特征，embedding特征，特征交叉等常见操作，也支持list特征以及各种数据类型。能很方便的将生成的模型export，以及支持分布式训练。  
修改模型的核心代码可以很方便的扩展到其他DNN模型，现在暂时只支持Wide，DNN，WD三类。   

#### esmm_model/(已删除，合并到DeepRank)  
实现esmm模型
#### esmm_model/esmm_v2(已删除，合并到DeepRank)  
  便于迭代模型，以及之前脚本代码写的有点乱，部分同学反馈代码不是很优雅。所以针对性的进行了优化。  
  
##### 运行方式(弃用) 
- 配置config/inputs.json，配置好输入的字段，以及需要做的操作，以及相对应的操作  
- 配置好对应的路径  
- 运行train_wd_model.py即可。    

##### config/(弃用)   
存放模型配置文件  

#### utils/



