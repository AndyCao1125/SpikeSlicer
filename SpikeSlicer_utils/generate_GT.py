import numpy as np
import os


def GT_generator_repeat(list_dir, root_dir, groupnum, datatype='test'):
    '''
    将eventdata细分后,构建新的groundtruth文件
    '''
    with open(list_dir, 'r') as f:
        seq_names = f.read().strip().split('\n')
    seq = seq_names[0:]
    # gt_dir = [os.path.join(root_dir, 'train', s, 'groundtruth_rect.txt') for s in seq]
    # new_gt_dir = [os.path.join(root_dir, 'train', s, 'groundtruth_rect_%d_repeat.txt' %groupnum) for s in seq]
    gt_dir = [os.path.join(root_dir, datatype, s, 'groundtruth_rect.txt') for s in seq]
    new_gt_dir = [os.path.join(root_dir, datatype, s, 'groundtruth_rect_%d_repeat.txt' %groupnum) for s in seq]
    print(new_gt_dir)

    for s in range(len(seq)):
        origin_anno = np.loadtxt(gt_dir[s],delimiter=',')
        anno_len = len(origin_anno)
        assert origin_anno.shape == (anno_len,4)
        
        repeated_anno = np.repeat(origin_anno, groupnum, axis=0)

        assert len(repeated_anno) == groupnum * anno_len
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=14)
        np.savetxt(new_gt_dir[s],repeated_anno,delimiter=',',fmt='%.014f')
        print(f' New groundtruth divided by {groupnum} is saved to {new_gt_dir[s]} with len:{len(repeated_anno)}, original len:{anno_len}')

def GT_generator_linear(list_dir, root_dir, groupnum, datatype='test'):
    '''
    将eventdata细分后,构建新的groundtruth文件
    '''
    with open(list_dir, 'r') as f:
        seq_names = f.read().strip().split('\n')
    seq = seq_names[0:]

    gt_dir = [os.path.join(root_dir, datatype, s, 'groundtruth_rect.txt') for s in seq]
    new_gt_dir = [os.path.join(root_dir, datatype, s, 'groundtruth_rect_%d_linear.txt' %groupnum) for s in seq]
    print(new_gt_dir)

    for s in range(len(seq)):
        origin_anno = np.loadtxt(gt_dir[s],delimiter=',')
        anno_len = len(origin_anno)
        assert origin_anno.shape == (anno_len,4)
        
        output_array = np.zeros((groupnum * anno_len, 4))

        # Perform linear interpolation for each column in the input_array
        for i in range(4):
            output_array[:, i] = np.interp(
                np.linspace(0, anno_len, groupnum * anno_len, endpoint=False),
                np.arange(anno_len),
                origin_anno[:, i]
            )
        assert len(output_array) == groupnum * anno_len
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=14)
        np.savetxt(new_gt_dir[s],output_array,delimiter=',',fmt='%.014f')
        print(f' New groundtruth divided by {groupnum} is saved to {new_gt_dir[s]} with len:{len(output_array)}, original len:{anno_len}')

if __name__ == '__main__':
    list_dir = '/home/lsf_node01/dataset/FE108/train.txt'
    # list_dir = '/home/lsf_node01/dataset/FE108/test_all.txt'
    root_dir = '/home/lsf_node01/dataset/FE108'
    groupnum = 5
    # GT_generator_repeat(list_dir, root_dir, groupnum, datatype='test')
    GT_generator_linear(list_dir, root_dir, groupnum, datatype='train')
