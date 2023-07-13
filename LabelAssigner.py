import torch
from utils import xywh2xyxy, bbox_iou
import numpy as np
from loguru import logger

# LOGGER
def LOGGER(*namelist):
    def log_format(func):
        def inner_func(*args, **kwargs):
            res = func(*args, **kwargs)
            show_res = res if isinstance(res, tuple) else [res]
            for name, return_res in zip(namelist, show_res):
                if isinstance(return_res, torch.Tensor) or isinstance(return_res, np.ndarray):
                    logger.info(f"({func.__name__}) {name}: {return_res.shape}")
                logger.info(f"({func.__name__}) {name}: {return_res}")
            return res
        return inner_func
    return log_format

class LabelAssigner(object):
    def __init__(self, inputs, annotations) -> None:
        logger.add('run.log')
        self.reg_max = 16
        self.class_num = 3
        self.input_h, self.input_w = 640, 640
        self.use_dfl = True
        self.proj = torch.arange(self.reg_max, dtype=torch.float)
        self.nc = 3
            # ---------- 预测结果预处理 ---------- #
        # 将多尺度输出整合为一个Tensor,便于整体进展矩阵运算
        #  pred_regs (b, 8400, 64)
        pred_scores, pred_regs, strides = self.pred_process(inputs)
        # --------- 生成anchors锚点 ---------#
        # 各尺度特征图每个位置一个锚点Anchors(与yolov5中的anchors不同,此处不是先验框)
        # 表示每个像素点只有一个预测结果
        self.anc_points, self.stride_scales = self.make_anchors(strides)
        
        # -------------     解码 ------------- #
        # 预测回归结果解码到bbox xmin, ymin, xmax, ymax格式
        # (b, 8400, 4) 
        pred_bboxes = self.decode(pred_regs)

        # ---------- 标注数据预处理 ----------- #
        gt_bboxes, gt_labels, gt_mask = self.ann_process(annotations)
        # ---------- 正负样本分配 ----------- #
        # 每个gt box 最多选择topk个候选框作为正样本
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        # ----------- 正负样本筛选 ------------ #
        target_bboxes, target_scores, fg_mask= self.assigner(pred_scores.detach().sigmoid(),
                                                             pred_bboxes.detach() * self.stride_scales,
                                                             self.anc_points * self.stride_scales,
                                                             gt_labels,
                                                             gt_bboxes,
                                                             gt_mask)
        

        
    @LOGGER('pred_scores', 'pred_regs', 'strides')
    def pred_process(self, inputs):
        '''     
        L = class_num + 4 * self.reg_max = class_num + 64
        多尺度结果(b, L, 80, 80), (b, L, 40, 40), (b, L, 20, 20)整合到一起为 (b, 8400, L) 
        按照cls 与 box 拆分为 (b, 8400, 2), (b, 8400, 64)
        '''
        predictions = [] # 记录每个尺度的转换结果 
        strides = [] # 记录每个尺度的缩放倍数
        for input in inputs:
            self.bs, cs, in_h, in_w = input.shape 
            # 计算该尺度特征图相对于网络输入的缩放倍数
            stride = self.input_h // in_h 
            strides.append(stride)
            # shape 转换 如 (b, 80, 80, 64+cls_num) -> (b, 6400, 64+cls_num)
            prediction = input.view(self.bs, 4 * self.reg_max + self.class_num, -1).permute(0, 2, 1).contiguous()
            predictions.append(prediction)
        # (b, 6400+1600+400, cls_num+64) = (b, 8400, 64+cls_num) = (b, 8400, 67)
        predictions = torch.cat(predictions, dim=1)
        # 按照cls 与 reg 进行拆分
        # (b, 8400, cls_num) = (b, 8400, 3)
        pred_scores = predictions[..., 4 * self.reg_max:]
        # (b, 8400, 64)
        pred_regs = predictions[..., :4 * self.reg_max]
        return pred_scores, pred_regs, strides 
    
    @LOGGER('anc_points', 'strides_tensor')
    def make_anchors(self, strides, grid_cell_offset=0.5):
        '''
        各特征图每个像素点一个锚点即Anchors, 即每个像素点只预测一个box
        故共有 80x80 + 40x40 + 20x20 = 8400个anchors
        '''
        # anc_points : (8400, 2) ，每个像素中心点坐标
        # strides_tensor: (8400, 1) ，每个像素的缩放倍数
        anc_points, strides_tensor = [], []
        for i , stride in enumerate(strides):
            in_h = self.input_h // stride 
            in_w = self.input_w // stride 
            
            # anchor坐标取特征图每个特征点的中心点
            sx = torch.arange(0, in_w).type(torch.float32) + grid_cell_offset
            sy = torch.arange(0, in_h).type(torch.float32) + grid_cell_offset
            # (in_h, in_w) 
            grid_y, grid_x = torch.meshgrid(sy, sx)
            # (in_h, in_w, 2) -> (N, 2)
            anc_points.append(torch.stack((grid_x, grid_y), -1).view(-1, 2).type(torch.float32))
            strides_tensor.append(torch.full((in_h * in_w, 1), stride).type(torch.float32))
        
        return torch.cat(anc_points, dim=0), torch.cat(strides_tensor, dim=0)
    
    @LOGGER('pred_bboxes')
    def decode(self, pred_regs):
        '''
            预测结果解码
            1. 对bbox预测回归的分布进行积分
            2. 结合anc_points，得到所有8400个像素点的预测结果
        '''
        if self.use_dfl:
            b, a, c = pred_regs.shape # (b, 8400, 64) 
            # 分布通过 softmax 进行离散化处理
            # (b, 8400, 64) -> (b, 8400, 4, 16) -> softmax处理 
            # l, t, r, b其中每个坐标值对应16个位置(0-15)的概率值
            # 概率表示每个位置对于最终坐标值的重要程度 
            pred_regs = pred_regs.view(b, a, 4, c//4).softmax(3)
            # 积分，相当于对16个分布值进行加权求和，最终的结果是所有位置的加权求和
            # (b, 8400, 4)
            pred_regs = pred_regs.matmul(self.proj.type(torch.float32))

        # 此时的regs, shape-> (b，8400, 4),其中4表示 anc_point中心点分别距离预测box的左上边与右下边的距离
        lt = pred_regs[..., :2]
        rb = pred_regs[..., 2:]
        # xmin ymin 
        x1y1 = self.anc_points - lt 
        # xmax ymax
        x2y2 = self.anc_points + rb 
        # (b, 8400, 4)        
        pred_bboxes = torch.cat([x1y1, x2y2], dim=-1)
        return pred_bboxes

    @LOGGER('gt_bboxes', 'gt_labels', 'gt_mask')
    def ann_process(self, annotations):
        '''
            batch内不同图像标注box个数可能不同，故进行对齐处理
            1. 按照batch内的最大box数目M,新建全0tensor
            2. 然后将实际标注数据填充与前面，如后面为0，则说明不足M，用0补齐
        '''
        # 获取batch内每张图像标注box的bacth_idx
        batch_idx = annotations[:, 0]
        # 计算每张图像中标注框的个数
        # 原理对tensor内相同值进行汇总
        _, counts = batch_idx.unique(return_counts=True)
        counts = counts.type(torch.int32)
        # 按照batch内最大M个GT创新全0的tensor (b, M, 5), 其中5 = (cls, cx, cy, width, height)
        res = torch.zeros(self.bs, counts.max(), 5).type(torch.float32)
        for j in range(self.bs):
            matches = batch_idx == j 
            n = matches.sum()
            if n: 
                res[j, :n] = annotations[matches, 1:]
        # res 为归一化之后的结果, 需通过scales映射回输入尺度
        scales = [self.input_w, self.input_h, self.input_w, self.input_h]
        scales = torch.tensor(scales).type(torch.float32)
        res[..., 1:5] = xywh2xyxy(res[..., 1:5]).mul_(scales)
        # gt_labels (b, M, 1)
        # gt_bboxes （b, M, 4）
        gt_labels, gt_bboxes = res[..., :1], res[..., 1:]
        # gt_mask (b, M, 1)
        # 通过对四个坐标值相加，如果为0，则说明该gt信息为填充信息，在mask中为False，
        # 后期计算过程中会进行过滤
        gt_mask = gt_bboxes.sum(2, keepdim=True).gt_(0)
        return gt_bboxes, gt_labels, gt_mask

class TaskAlignedAssigner(object):
    def __init__(self, topk=10, num_classes=3, alpha=0.5, beta=6.0, eps=1e-9) -> None:
       self.eps = eps
       self.nc = num_classes
       self.n_max_boxes = 3
       self.na = 84
       self.alpha = alpha
       self.beta = beta
       self.topk = topk

    @LOGGER('b_target_boxes', 'b_target_labels', 'b_fg_masks')
    def __call__(self,
                 pb_scores,
                 pb_bboxes,
                 anc_points,
                 gt_labels,
                 gt_bboxes,
                 gt_mask
                 ):
        batch_size = pb_scores.shape[0]
        b_target_labels = [torch.zeros((self.na, self.nc)) for _ in range(batch_size)]
        b_target_boxes = [torch.zeros((self.na, 4)) for _ in range(batch_size)]
        b_fg_masks = [torch.zeros((self.na,)) for _ in range(batch_size)]
        for b in range(batch_size):
            # ---------------------- 初筛正样本 ------------------------- #
            # -------------- 判断anchor锚点是否在gtbox内部 --------------- #
            # （M, 8400）
            in_gts_mask = self.__get_in_gts_mask(gt_bboxes[b], anc_points)
            # ---------------------- 精细筛选 ---------------------- #
            # 按照公式获取计算结果
            # pb_scores (4, 8400, cls_num)
            # pb_bboxes (4, 8400, 4)
            # gt_labels (4, M, 1)
            # gt_bboxes (4, M, 4)
            # gt_mask (4, M, 1)
            align_metrics, overlaps = self.__refine_select(pb_scores[b], 
                                                           pb_bboxes[b], 
                                                           gt_labels[b], 
                                                           gt_bboxes[b], 
                                                           in_gts_mask * gt_mask[b])
            # 根据计算结果,排序并选择top10
            # (M, 8400) 
            topk_mask = self.__select_topk_candidates(align_metrics, gt_mask[b].repeat(1, self.na))
            # ------------------ 排除某个anchor被重复分配的问题 ---------------- #
            # target_gt_idx : 8400
            # fg_mask : 8400
            # pos_mask: (M, 8400)
            target_gt_idx, fg_mask, pos_mask = self.__filter_repeat_assign_candidates(topk_mask, overlaps)
            # ------------------ 根据Mask设置训练标签 ------------------ #
            # target_labels : 8400 x cls_num
            # target_bboxes : 8400 x 4
            target_labels, target_bboxes = self.__get_train_targets(gt_labels[b], 
                                                                    gt_bboxes[b], 
                                                                    target_gt_idx, 
                                                                    fg_mask)
            
            # align_metric, overlaps均需要进行过滤
            align_metrics *= pos_mask # M x 8400 
            overlaps *= pos_mask # M x 8400
                    
            # 找个每个GT的最大匹配值 M x 1
            gt_max_metrics = align_metrics.amax(axis=-1, keepdim=True)
            # 找到每个GT的最大CIOU值 M x 1
            gt_max_overlaps = overlaps.amax(axis=-1, keepdim=True)
            # 为类别one-hot标签添加惩罚项 M x 8400 -> 8400 -> 8400 x 1
            # 通过M个GT与所有anchor的匹配值 x 每个GT与所有anchor最大IOU / 每个类别与所有anchor最大的匹配值
            norm_align_metric = (align_metrics * gt_max_overlaps / (gt_max_metrics + self.eps)).amax(-2).unsqueeze(-1)
            # 8400 x cls_num，为类别添加惩罚项
            target_labels = target_labels * norm_align_metric
            b_target_labels[b] = target_labels
            b_target_boxes[b] = target_bboxes
            b_fg_masks[b] = fg_mask
            # break
        
        b_target_boxes = torch.stack(b_target_boxes, 0).float()
        b_target_labels = torch.stack(b_target_labels, 0).long()
        b_fg_masks = torch.stack(b_fg_masks, 0).bool()
        
        return b_target_boxes, b_target_labels, b_fg_masks

    @LOGGER('in_gts_mask')
    def __get_in_gts_mask(self, gt_bboxes, anc_points):
        # 找到M个GTBox的左上与右下坐标 (M, 1, 2)
        gt_bboxes = gt_bboxes.view(-1, 1, 4)
        lt,rb = gt_bboxes[..., :2], gt_bboxes[..., 2:]
        # anc_points 增加一个维度 (1, 8400, 2)
        anc_points = anc_points.view(1, -1, 2)
        # 差值结果 (M, 8400, 4) 
        bbox_detals = torch.cat([anc_points - lt, rb - anc_points], dim=-1)
        # 第三个维度均大于0才说明在gt内部
        # (M, 8400)
        in_gts_mask = bbox_detals.amin(2).gt_(self.eps)
        return in_gts_mask 
    
    @LOGGER('align_metric', 'gt_pb_cious')
    def __refine_select(self, pb_scores, pb_bboxes, gt_labels, gt_bboxes, gt_mask):
        # pb_scores (8400, cls_num)
        # pb_bboxes (8400, 4)
        # gt_labels (M, 1)
        # gt_bboxes (M, 4)
        # gt_mask（M, 8400) = gt_mask(M, 1) * in_gts_mask(M, 8400)
        # 根据论文公式进行计算得到对应的计算结果
        # reshape (M, 4) -> (M, 1, 4) -> (M, 8400, 4) 
        gt_bboxes = gt_bboxes.unsqueeze(1).repeat(1, self.na, 1)
        # reshape (8400, 4) -> (1, 8400, 4) -> (M, 8400, 4) 
        pb_bboxes = pb_bboxes.unsqueeze(0).repeat(self.n_max_boxes, 1, 1)
        # 计算所有预测box与所有gtbox的ciou，相当于公式中的U
        gt_pb_cious = bbox_iou(gt_bboxes, pb_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp(0)
        # 过滤填充的GT以及不在GTbox范围内的部分
        # (M, 8400)
        gt_pb_cious = gt_pb_cious * gt_mask 
 
        # 获取与GT同类别的预测结果的scores 
        # (8400, cls_num) -> (1, 8400, cls_num) -> (M, 8400, cls_num)
        pb_scores = pb_scores.unsqueeze(0).repeat(self.n_max_boxes, 1, 1)
        # (M, 1) -> M 
        gt_labels = gt_labels.long().squeeze(-1)
        # 针对每个GTBOX从预测值(M, 8400, cls_num)中筛选出对应自己类别Cls的结果, 每个结果shape (1, 8400)
        # (M, 8400) 
        scores  = pb_scores[torch.arange(self.n_max_boxes), :, gt_labels]

        # 根据公式进行计算 (M, 8400)
        align_metric = scores.pow(self.alpha) * gt_pb_cious.pow(self.beta)
        # 过滤填充的GT以及不在GTbox范围内的部分
        align_metric = align_metric * gt_mask
        return align_metric, gt_pb_cious
    
    @LOGGER('topk_mask')
    def __select_topk_candidates(self, align_metric, gt_mask):
        # 从大到小排序,每个GT的从8400个结果中取前 topk个值，以及其中的对应索引
        # top_metrics :(M, topk)
        # top_idx : (M, topk)
        topk_metrics, topk_idx = torch.topk(align_metric, self.topk, dim=-1, largest=True)
        # 生成一个全0矩阵用于记录每个GT的topk的mask
        topk_mask = torch.zeros_like(align_metric, dtype=gt_mask.dtype, device=align_metric.device)
        for i in range(self.topk):
            top_i = topk_idx[:, i]
            # 对应的top_i位置值为1
            topk_mask[torch.arange(self.n_max_boxes), top_i] = 1
        topk_mask = topk_mask * gt_mask 
        # (M, 8400)
        return topk_mask 
    
    @LOGGER('target_gt_idx', 'fg_mask', 'pos_mask')
    def __filter_repeat_assign_candidates(self, pos_mask, overlaps):
        '''
            pos_mask : (M, 8400)
            overlaps: (M, 8400)
            过滤原则:如某anchor被重复分配,则保留与anchor的ciou值最大的GT
        '''
        # 对列求和,即每个anchor对应的M个GT的mask值求和，如果大于1，则说明该anchor被多次分配给多个GT
        # 8400
        fg_mask = pos_mask.sum(0)
        if fg_mask.max() > 1:#某个anchor被重复分配
            # 找到被重复分配的anchor，mask位置设为True,复制M个，为了后面与overlaps shape匹配
            # 8400 -> (1, 8400) -> (M, 8400) 
            mask_multi_gts = (fg_mask.unsqueeze(0) > 1).repeat([self.n_max_boxes, 1])
            # 每个anchor找到CIOU值最大的GT 索引  
            # 8400 
            max_overlaps_idx = overlaps.argmax(0)
            # 用于记录重复分配的anchor的与所有GTbox的CIOU最大的位置mask
            # (M, 8400)
            is_max_overlaps = torch.zeros(overlaps.shape, dtype=pos_mask.dtype, device=overlaps.device)
            # 每个anchor只保留ciou值最大的GT，对应位置设置为1
            is_max_overlaps.scatter_(0, max_overlaps_idx.unsqueeze(0), 1)
            # 过滤掉重复匹配的情况
            pos_mask = torch.where(mask_multi_gts, is_max_overlaps, pos_mask).float()
            # 得到更新后的每个anchor的mask 8400
            fg_mask = pos_mask.sum(0)
        # 找到每个anchor最匹配的GT 8400
        target_gt_idx = pos_mask.argmax(0)
        '''
            target_gt_idx: 8400 为每个anchor最匹配的GT索引(包含了正负样本)
            fg_mask: 8400 为每个anchor设置mask,用于区分正负样本
            pos_mask: (M, 8400)  每张图像中每个GT设置正负样本的mask
        '''
        return target_gt_idx, fg_mask, pos_mask
    
    @LOGGER('target_one_hot_labels', 'target_bboxes')
    def __get_train_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        '''
            gt_labels: (M, 1) 
            gt_bboxes: (M, 4) 
            fg_mask  : 8400 每个anchor为正负样本0或1
            target_gt_idx: 8400 每个anchor最匹配的GT索引(0~M)
        '''
        # gt_labels 拉直
        gt_labels = gt_labels.long().flatten()
        # 根据索引矩阵,获得cls  (8400, )
        target_labels = gt_labels[target_gt_idx]
        # 同理bbox同样操作，
        # 根据索引矩阵，获得bbox (8400, 4) 
        target_bboxes = gt_bboxes[target_gt_idx]
        
        # 类别转换为one-hot形式，(8400, cls_num)
        target_one_hot_labels = torch.zeros((target_labels.shape[0], self.nc),
                                           dtype=torch.int64,
                                           device=target_labels.device)
        # 赋值，对应的类别位置置为1， 即one-hot形式
        target_one_hot_labels.scatter_(1, target_labels.unsqueeze(-1), 1)
        
        # 生成对应的mask，用于过滤负样本 (8400, ) -> (8400, 1) -> （8400， cls_num）
        fg_labels_mask = fg_mask.unsqueeze(-1).repeat(1, self.nc)
        
        # 正负样本过滤
        target_one_hot_labels = torch.where(fg_labels_mask>0, target_one_hot_labels, 0)
        
        return target_one_hot_labels, target_bboxes
    

if __name__ == '__main__':
    # inputs = [torch.randn(4, 67, 80, 80),
    #           torch.randn(4, 67, 40, 40),
    #           torch.randn(4, 67, 20, 20)
    #           ]  # b=4
    # fake input na = 84
    inputs = [torch.randn(4, 67, 8, 8),
              torch.randn(4, 67, 4, 4),
              torch.randn(4, 67, 2, 2)
              ]  # b=4
    labels = torch.tensor(np.array([[0, 1.0, 0.612, 0.334, 0.666, 0.378],
                                    [0, 0.0, 0.553, 0.054, 0.426, 0.109],
                                    [1, 1.0, 0.457, 0.324, 0.747, 0.359],
                                    [2, 2.0, 0.875 , 0.484, 0.25, 0.315],
                                    [2, 1.0, 0.45, 0.36, 0.72, 0.411],
                                    [2, 0.0, 0.27, 0.064, 0.46, 0.12],
                                    [3, 2.0, 0.348, 0.521, 0.151, 0.237]]), dtype=torch.float32)
    print(labels.shape)
    print(labels)
    labelassigner = LabelAssigner(inputs, labels)
