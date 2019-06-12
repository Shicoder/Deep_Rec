## 本库用于存放推荐相关算法代码和文档
#### tfrecords/
存放tfrecords文件的生成和使用代码
利用spark读取数据保存成tfrecods这里提供两个思路。  
1.如果集群上已经安装了tensorflow，可以用本文件下的代码进行处理。  
2.对于没有安装tensorflow的集群，可以参考https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector来将数据转换成tfrecords。这个更快。
