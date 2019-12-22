# Deep_Rank

## Overview  

**Deep_Rank是一套从数据生成到模型生成的完整框架。对输入特征，特征工程，核心模型，模型输出分别进行了封装。
模型更新迭代的时候可以针对性的修改指定模块，也方便后续模型的切换，特征的迭代，以及输入数据在输入，特征工程，保存模型，线上服务各个模块的统一配置。本库对一些经典的ctr预估模型进行了复现，方便自己在工作中迭代优化模型。**  

## Data Read
利用 `tfrecords/` 下的代码可以很方便的利用spark集群将数仓经过ETL的hive表训练数据转化成`frecords`格式，并存储HDFS上。
      
## Config Setup
### model_schema.json   
配置训练数据中的字段，可以只选择模型需要的字段；  
*格式：*
  
    "user_id":     
    {  
            "feature_type": "fixed_len",  
            "value_type": "int",  
            "default_value": 0  
    }  
    
    字段名为key;  
    feature_type：tensorflow特征类型，当前所有模型使用的是fixed_len;  
    value_type：特征值的类型，当前支持{'string', 'int', 'double'},  
                可以在train_model.py#19 函数get_input_schema_spec中添加/修改;    
    default_value: 该特征值的默认填充值，类型需要和value_type一致；          

### model_feature.json   
 
    配置模型中特征的形式，使用tensorflow FeatureColumns API实现；    
    具体支持的FeatureColumns包括：  
    "categorical_column_with_hash_bucket"  
    "numeric_column"  
    "embedding_column"  
    "categorical_column_with_vocabulary_list"  
    "categorical_column_with_identity"  
    "indicator_column"  
    "bucketized_column"  
    "shared_embedding_columns"  
    "crossed_column"  
    "NumericColumnV2" (这个是修改的原始numeric_column，用来自适应学习缺失值，具体思路可以看源码)  
    配置格式：  
    {  
      "name": "IndicatorColumn",  
      "parameters":   
      {  
        "input_tensor": "u_sc_gender_hashbucket",  
        "output_tensor": "u_sc_gender_indicator",  
        "wide_or_deep": "deep"  
      }  
    }  
    input_tensor：定义输入名  
    output_tensor：定义输出名  
    wide_or_deep：用于wd，deepfm的wide侧和deep侧区分;    
                  如果只是中间特征，不作为最后的输出则不填;   
                  其他模型默认只取"deep"值特征;  
                  din，dien等模型，用于attention的sequence特征和目标id不需要在model_feature.json进行配置;    
        
## Model Implementation

### model_brain.py  
*本文件存放模型核心代码，所有代码继承`BaseModel`类，底层使用tensorflow的FeatureColumns实现统一的embedding层。上层利用tensorlfow的Layers实现核心模型结构，最后把模型封装成Esitmator，输出统一的estimator_spec接口。*  

#### model brain
*已经实现好的模型有以下几个：*  
  
  
[`DNN`](https://github.com/Shicoder/Deep_Rec/blob/master/Deep_Rank/dnn_demo.py) 实现的是一个简单的embedding+MLP，方便调试整体代码，在`model_feature.json`中配置`wide_or_deep`参数值 `"deep"`;  
  
[`Deep Cross Network(DCN)` ](https://github.com/Shicoder/Deep_Rec/blob/master/Deep_Rank/dcn.py)在`model_feature.json`中只需配置`deep特征`，croos层和deep层同用统一的embeddig层。    
对应算法论文[[click here]](https://arxiv.org/abs/1708.05123)  
   
[`Wide and Deep Network(WD)`](https://github.com/Shicoder/Deep_Rec/blob/master/Deep_Rank/wide_deep.py)需要在`model_feature.json`配置wide侧和deep侧对应的特征。    
对应算法论文[[click here]](https://arxiv.org/abs/1606.07792)   

[`Deep Interest Network(DIN)`](https://github.com/Shicoder/Deep_Rec/blob/master/Deep_Rank/din.py)的上下文特征和基础画像特征需要配置`model_feature.json`中，统一使用`"deep"`特征,其中需要做`attention`的`sequence`和`sequence对应的目标id`不需要配置在`model_feature.json`中，直接配置在模型参数中。  
在对应算法论文[[click here]](https://arxiv.org/abs/1706.06978)  

[`Entire Space Multi-Task Model(ESMM)`](https://github.com/Shicoder/Deep_Rec/blob/master/Deep_Rank/esmm.py)也一样，默认只使用`"deep"`特征。 
对应算法论文[[click here]](https://arxiv.org/abs/1804.07931)  
  
[`Deep Interest Evolution Network(DIEN)`](https://github.com/Shicoder/Deep_Rec/blob/master/Deep_Rank/dien.py)和DIN一样，`序列特征`和`序列对应目标id`需要在模型中写。  
这里为了实现方便，序列的`负采样`部分没有按照原始论文的方式单独使用一份负采样的item数据集，而是直接使用`同一个batch`中的`其他sequence`作为当前的`负采样序列`。  
对应算法论文[[click here]](https://arxiv.org/abs/1809.03672)   
 
[`DeepFM`](https://github.com/Shicoder/Deep_Rec/blob/master/Deep_Rank/deepFM.py)的wide侧放线性特征，deep侧放的是`统一的MLP和FM的特征`，所有这里必须保证deep特征全部是embedding特征，并且embedding维度`全部一致`。这里后续会优化一下，对一些数值特征做优化，方便扩展。  
对应算法论文[[click here]](https://arxiv.org/abs/1703.04247)   

[`xDeepFM`](https://github.com/Shicoder/Deep_Rec/blob/master/Deep_Rank/xDeepFM.py)的参数配置和deepFM类似，wide侧放线性特征，deep侧放的是`统一的MLP和CIN特征`，所有这里必须保证deep特征全部是embedding特征，并且embedding维度`全部一致`。  
对应算法论文[[click here]](https://arxiv.org/pdf/1803.05170.pdf)  

[`DSSM`](https://github.com/Shicoder/Deep_Rec/blob/master/Deep_Rank/dssm.py)需要在wide和deep侧分别放置user info和item info。  
对应算法论文[[click here]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)   

[`youtube_net`](https://github.com/Shicoder/Deep_Rec/blob/master/Deep_Rank/dssm.py)本文是youtube提出的排序模型，不知道叫啥，所以直接叫youtube_net了。  
对应算法论文[[click here]](https://dl.acm.org/citation.cfm?id=3346997)    
    
*后续利用空闲时间和节假日会持续添加新算法*

## Training

直接运行主函数 
  
    `python train_model.py`


### 备注

*部分实现参考了:*   
  x-deeplearning的xdl-algorithm-solution: [[click here]](https://github.com/alibaba/x-deeplearning/tree/master/xdl-algorithm-solution)   
  脚本配置方式：[[click here]](https://github.com/zhaoxin4data/atlas/tree/master/deeplearning/uciflowwd_train/config)  
  esmm实现:[[click here]](https://github.com/yangxudong/deeplearning/tree/master/esmm)   
  tensorflow自带的分类器: [[click here]](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/estimator/canned)  
  xDeepFM实现:[click_here](https://github.com/Leavingseason/xDeepFM)  
  MMoE实现：[click_here](https://github.com/drawbridge/keras-mmoe)  
  
  *代码实现的问题：* 
    
  *`代码实现为了方便调试，DIN，DIEN中的所有的序列特征都只用了id，可自行添加类目序列，产品词序列等边界信息`*