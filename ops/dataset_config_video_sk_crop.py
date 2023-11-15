import os

ROOT_DATASET = '/home/chengqin/TSM_34' # '/data/jilin/'

def return_ntu_xsub(modality):
    filename_categories = 60
    if modality in ['RGB', 'motion', 'dense']:
        root_data = '/data/cq/ntu_video/nturgb_sk_guide_video'
        filename_imglist_train = '/data/cq/exp/DSCNet_34/data/ntu60_xsub/video_train.txt'
        filename_imglist_val = '/data/cq/exp/DSCNet_34/data/ntu60_xsub/video_val.txt'
        prefix = 'img_{:05d}.jpg'

    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_ntu_xview(modality):
    filename_categories = 60
    if modality in ['RGB', 'motion', 'dense']:
        root_data = '/data/cq/ntu_video/nturgb_sk_guide_video'
        filename_imglist_train = '/data/cq/exp/DSCNet_34/data/ntu60_xview/video_train.txt'
        filename_imglist_val = '/data/cq/exp/DSCNet_34/data/ntu60_xview/video_val.txt'
        # filename_imglist_val = '/home/chengqin/TSM/data/ntu_xview/new_train.txt'
        prefix = 'img_{:05d}.jpg'

    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_ntu120_xsub(modality):
    filename_categories = 120
    if modality in ['RGB', 'motion', 'dense']:
        root_data = '/data/cq/ntu_video/nturgb_sk_guide_video'
        filename_imglist_train = '/data/cq/exp/DSCNet_34/data/ntu120_xsub/video_train.txt'
        filename_imglist_val = '/data/cq/exp/DSCNet_34/data/ntu120_xsub/video_val.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ntu120_xset(modality):
    print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm', modality)
    filename_categories = 120
    if modality in ['RGB', 'motion', 'dense']:
        root_data = '/data/cq/ntu_video/nturgb_sk_guide_video'
        filename_imglist_train = '/data/cq/exp/DSCNet_34/data/ntu120_xset/video_train.txt'
        filename_imglist_val = '/data/cq/exp/DSCNet_34/data/ntu120_xset/video_val.txt'
        prefix = 'img_{:05d}.jpg'

    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_pku_xsub(modality):
    print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm', modality)
    filename_categories = 51
    if modality in ['RGB', 'motion', 'dense']:
        root_data = '/data/cq/pku-rgb-actions-hrnet-sk-guide-pad2030'
        filename_imglist_train = '/data/cq/exp/DSCNet_34/data/pku_video_list_for_3modal_hrnet_sk/xsub_train.txt'
        filename_imglist_val = '/data/cq/exp/DSCNet_34/data/pku_video_list_for_3modal_hrnet_sk/xsub_val.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_pku_xview(modality):
    print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm', modality)
    filename_categories = 51
    if modality in ['RGB', 'motion', 'dense']:
        root_data = '/data/cq/pku-rgb-actions-hrnet-sk-guide-pad2030'
        filename_imglist_train = '/data/cq/exp/DSCNet_34/data/pku_video_list_for_3modal_hrnet_sk/xview_train.txt'
        filename_imglist_val = '/data/cq/exp/DSCNet_34/data/pku_video_list_for_3modal_hrnet_sk/xview_val.txt'
        prefix = 'img_{:05d}.jpg'

    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_uav_xsub1(modality):
    print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm', modality)
    filename_categories = 155
    if modality in ['RGB', 'motion', 'dense']:
        root_data = '/data/cq/UAV/rgb/rgb_skguide'
        filename_imglist_train = '/data/cq/exp/DSCNet_34/data/uav_videolist/xsub1-train.txt'
        filename_imglist_val = '/data/cq/exp/DSCNet_34/data/uav_videolist/xsub1-val.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_uav_xsub2(modality):
    print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm', modality)
    filename_categories = 155
    if modality in ['RGB', 'motion', 'dense']:
        root_data = '/data/cq/UAV/rgb/rgb_skguide'
        filename_imglist_train = '/data/cq/exp/DSCNet_34/data/uav_videolist/xsub2-train.txt'
        filename_imglist_val = '/data/cq/exp/DSCNet_34/data/uav_videolist/xsub2-val.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_ikea(modality):
    print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm', modality)
    filename_categories = 33
    if modality in ['RGB', 'motion', 'dense']:
        root_data = '/data/cq/ikea/ikea_action_clip_sk_crop_33'
        filename_imglist_train = '/data/cq/exp/DSCNet_34/data/ikea/train_list.txt'
        filename_imglist_val = '/data/cq/exp/DSCNet_34/data/ikea/val_list.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset(dataset, modality):
    dict_single = {'ntu_xsub': return_ntu_xsub, 'ntu_xview': return_ntu_xview,'ntu120_xsub': return_ntu120_xsub, 'ntu120_xset': return_ntu120_xset,
                   'pku_xsub': return_pku_xsub, 'pku_xview': return_pku_xview,'uav_xsub1':return_uav_xsub1, 'uav_xsub2':return_uav_xsub2,
                    'ikea':return_ikea}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
