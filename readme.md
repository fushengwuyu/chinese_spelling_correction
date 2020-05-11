### 中文文本纠错模型 
这里提供三种文本纠错模型的实现  
    
1. bert语言模型+字音字形相似度  
    * correction_basic.py
    * 缺点: 
       1. 不能解决多字,少字问题
       2. 容易导致错误积累
       3. 最好配合错误检测算法,语言模型完成错误纠正功能 
2. MLM(todo)  
    利用bert的MLM训练机制实现纠错功能  
    输入: [CLS]错误句子[SEP][MASK][MASK]...[MASK][SEP]  
    输出: 正确句子  
3. seq2seq(todo)  
    使用文本生成的方式生成正确句子  
    输入: [CLS]错误句子[SEP][MASK][MASK]...[MASK][SEP  
    输出: 正确句子  