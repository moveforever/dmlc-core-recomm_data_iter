dense_num = 9
cate_num = 36
multi_cate_num = 5
#dims1 = [10000011, 10000091, 8, 100021, 104001, 4, 110001, 130001, 131, 141, 151]
dims1 = [ 10000000 ] + [ 10000000 ] + [ 100000 ] * (cate_num - 2)
dims2 = [ 100000 ] * multi_cate_num
CONFIG = {
    'train': 'train_one_day',
    'test': 'test',
    'dense_num': dense_num,
    'sparse_dim': 10000000,
    'embed_dim': 32,
    'cate_num': cate_num,
    'cate_input_dims': dims1,
    'multi_cate_num': multi_cate_num,
    'multi_input_dims': dims2,
    'hidden_units': [1024, 1024, 512],
    'batch_size': 8192,
    'lr': 0.00002,
    'num_epoch': 2,
    'wd': 0.000001
}
import mxnet as mx
batch_size = 4
train_path = '/install_dir/package/part?multi_field_num=%d&label_width=2' % CONFIG['multi_cate_num']

multi_str = ','.join([('%d' % v) for v in CONFIG['multi_input_dims']])
data_names = ['dense', 'cate', 'sparse'] + ['field_%d' % i for i in range(CONFIG['multi_cate_num'])]
data_size = 3+CONFIG['multi_cate_num']
train_data = mx.io.RMFIter(data=train_path, dense_shape = (CONFIG['dense_num'], ), cate_shape = (CONFIG['cate_num'], ), sparse_shape = (CONFIG['sparse_dim'], ),
     multi_field_shape_str = multi_str, multi_field_num = CONFIG['multi_cate_num'], data_size = data_size, label_shape=(1, ), batch_size=batch_size, data_name=data_names, label_width=2)
for data in train_data:
    print("label_out")
    print(data.label)
    print(data.data)
    #arr = mx.nd.split(data = data.label[0], axis = 1, num_outputs = 2, squeeze_axis = True)
    #print(arr)
