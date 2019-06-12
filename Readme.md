## 本库用于存放推荐相关算法代码和文档
#### tfrecords/
存放tfrecords文件的生成和使用代码
从hive读取数据，利用spark保存成tfrecords格式数据，保存在HDFS上。对于没有安装https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector 这个库的可以试试我写的这个方便，但是会比较慢，最好安装这个库来导数据。
