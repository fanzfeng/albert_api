# 基于[albert](https://github.com/brightmart/albert_zh)的应用
之前几个项目中用到了albert，主要是文本检索和分类这类，简单总结下，希望有点用处~
## 文本检索引擎
1. 文本向量化text2vector，三者选其一：
   - albert-as-service
     - 根据项目[bert-as-service](https://github.com/hanxiao/bert-as-service)简单修改，支持albert model
     - 使用方法：
       - 安装服务端和客服端（须保证版本一致）
        ```
          cd albert-as-service/server
          python setup.py install
          cd ../client
          python setup.py install
        ```
       - 启动服务端
        ```
          cd albert-as-service
          sh albert_server.sh
        ```
     - 客户端api脚本是albert_client.py
   - 基于TensorFlow高级接口(使用方法：脚本bert_api1.py调用)
   - 基于TensorFlow静态图（使用方法：脚本bert_api2.py调用）
1. 文本检索引擎（apiRetrieval.py）
  - 选择文本向量化方式（albert_client.py | bert_api1.py | bert_api2.py）
  - 添加文本索引
  - 文本检索

## 文本分类
1. api选择：
  - 原始bert-as-service仅支持文本向量化，有高人在此基础上修改以支持文本分类、序列标注（用法基本相同），见https://github.com/macanv/BERT-BiLSTM-CRF-NER
  - bert_api1.py
  - bert_api2.py
2. 之前有做个文本审核项目，其中的文本分类就是用的albert，使用gpu服务器部署bert-as-service，平均延时小于5ms

# 动手
1. 下载官方model（我用的[albert-tiny](https://storage.googleapis.com/albert_zh/albert_tiny.zip)）
1. 将词表voca.txt放入模型文件夹中
1. 在自己数据上finetune，调优
1. 服务部署与测试

# 存在的问题分析
1. 这里只给出了api，每个api的效率不同，需要根据场景要求选择并做并发支持
   - bert-as-server属于成品类型，调整下workers数量就可以尝试生产使用了
   - 基于tf静态图的api貌似无法支持多进程（failed to get device properties error code 3 multiprocessing）
   - 可尝试多个docker示例+nginx代理的思路
2. bert-as-server这个框架有个问题(服务hang，客服端无响应，服务端无日志)，之前一直没解决，不知现在情况如何
   - 基于gpu和cpu分别部署了一个版本的服务，gpu版本服务okay，cpu版本遇到这个问题
