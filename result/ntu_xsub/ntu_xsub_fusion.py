import numpy as np
from sklearn.metrics import confusion_matrix
video_labels = np.load('ntu60_xsub_msst_seg2_joint.npz', allow_pickle=True)['labels']#

####################### imagenet -pretrained ################################
joint = np.load('ntu60_xsub_msst_seg8_joint.npz', allow_pickle=True)['scores']#
bone = np.load('ntu60_xsub_msst_seg8_bone.npz', allow_pickle=True)['scores']#
# #
rgb = np.load('model2_TSM_stmem_ntu_xsub_sk_guide_croped_rgb_seg8_34_60epoch.npz')['scores']#
#     #joint bone rgb  0.6;0.6;2

print(joint.shape)



video_pred = []
for i in range(len(joint)):
    pre_1 = joint[i] * 0.3 + bone[i] * 0.3
    pre_2 = rgb[i] * 1
    pre = pre_1*1 + pre_2*1
    video_pred.extend([np.argmax(pre)])

# video_labels = [x[1] for x in joint]
# print(video_labels[0:100])
cf = confusion_matrix(video_labels, video_pred).astype(float)
print('ccccccccccccccccc', cf.shape)
cls_cnt = cf.sum(axis=1)  # 得到是每一类各自总评估次数.
cls_hit = np.diag(cf)  # 每一类总的评估对的次数.
cls_acc = cls_hit / cls_cnt
# print(video_labels)
print('各类正确率：\n', cls_acc)
print('各类平均精度 {:.04f}%，总数累加正确率{:.04f}%'.format(np.mean(cls_acc) * 100, (np.sum(cls_hit)) / (np.sum(cls_cnt)) * 100))



