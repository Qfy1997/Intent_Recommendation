

if __name__=='__main__':
    with open('./train_data.txt','r') as f:
        data = f.readlines()
    a=data[0]
    # print(a)
    label = a.split('\t')[1]
    data = a.split('\t')[0].split(',')
    # print(data)
    # print(label)
    # 每行数据由9种数据类型的表示构成，
    # 分别为wide_fear、user_item_seq、query_feat、user_query_seq、query_item_query、
    # user_query_item、user_item_query、query_user_item、label
    # 模型接受前8种数据类型的表示并输出第9种数据类型的表示去做训练。
    wide_feat = data[:81] 
    user_item_seq = data[81:276] 
    query_feat = data[276:292]
    user_query_seq = data[292:462] 
    query_item_query = data[462:562] 
    user_query_item = data[562:662]
    user_item_query = data[662:812]
    query_user_item = data[812:]
    print("wide feat:")
    print(wide_feat)
    print("========")
    print("user_item_seq:")
    print(user_item_seq)
    print(len(user_item_seq))
    print("======")
    print("query_feat:")
    print(query_feat)
    print("========")
    print("user_query_seq:")
    print(user_query_seq)
    print("========")
    print("query_item_query:")
    print(query_item_query)
    print("==========")
    print("user_query_item:")
    print(user_query_item)
    print("=============")
    print("user_item_query:")
    print(user_item_query)
    print("============")
    print("query_user_item:")
    print(query_user_item)
    # 首先模型会初始化一个280000*64维度的嵌入层，来存放数据集中对应序列索引的表示。
