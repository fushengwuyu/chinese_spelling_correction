### 中文文本纠错模型 
这里提供三种文本纠错模型的实现  
    
1. bert语言模型+字音字形相似度  
    * correction_basic.py
    * 缺点: 
       1. 不能解决多字,少字问题
2. MLM  
    correction_mlm.py
    利用bert的MLM训练机制实现纠错功能  
    输入: [CLS]错误句子[SEP][MASK][MASK]...[MASK][SEP]  
    输出: 正确句子  
3. seq2seq    
    correction_seq2seq.py
    使用文本生成的方式生成正确句子  
    输入: [CLS]错误句子[SEP][MASK][MASK]...[MASK][SEP  
    输出: 正确句子  
    缺点：推断速度比较慢
        
exampe:
> wrong: 专家公步虎门大桥涡振原因  
> right: 专家公步虎门大桥涡振原因
#### 数据
1. 引用自https://github.com/iqiyi/FASPell里面的数据,所有数据打包在data/origin_data.zip  
2. 可以尝试自己构建纠错数据集,data/char_meta.txt提供了汉字的字音和字形数据.  
