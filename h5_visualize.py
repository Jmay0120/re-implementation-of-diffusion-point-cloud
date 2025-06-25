import h5py

file_path = 'data/shapenet_part_seg_hdf5_data/hdf5_data/ply_data_val0.h5'

with h5py.File(file_path, 'r') as f:
    # 打印根目录下所有键（可以理解为顶层文件夹）
    print("根目录下的键：")
    for key in f.keys():
        print(key)

    # 如果你知道某个键，例如 'pointclouds'
    if 'data' in f:
        dataset = f['data']
        print("数据集形状：", dataset.shape)
        print("数据类型：", dataset.dtype)
        print("部分数据内容：", dataset[0])  # 打印第一个样本
