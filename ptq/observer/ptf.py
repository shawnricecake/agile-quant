# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch

from .base import BaseObserver
from .utils import lp_loss


class PtfObserver_2_0to1(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to1, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale2 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale2)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score = [score1, score2]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to2(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to2, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale4 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale4)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score = [score1, score2, score4]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to3(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to3, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale8 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale8.clamp_(self.eps)
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale8)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score = [score1, score2, score4, score8]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to4(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to4, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale16 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale16)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to5(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to5, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale32 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale32)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to6(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to6, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale64 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale64.clamp_(self.eps)
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale64)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32, score64]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to7(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to7, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale128 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale64 = scale128 / 2
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale128)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32, score64, score128]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to8(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to8, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale256 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale256.clamp_(self.eps)
        scale128 = scale256 / 2
        scale64 = scale128 / 2
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale256)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128
            data_q256 = ((data / scale256 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale256

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score256 = lp_loss(data, data_q256, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32, score64, score128, score256]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0246(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0246, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale64 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale64.clamp_(self.eps)
        scale16 = scale64 / 4
        scale4 = scale16 / 4
        scale1 = scale4 / 4
        zero_point = qmin - torch.round(min_val_t / scale64)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score = [score1, score4, score16, score64]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_01236(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_01236, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale64 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale64.clamp_(self.eps)
        scale8 = scale64 / 8
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale64)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale8
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score64]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0126(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0126, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale64 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale64.clamp_(self.eps)
        scale4 = scale64 / 16
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale64)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score = [score1, score2, score4, score64]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0136(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0136, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale64 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale64.clamp_(self.eps)
        scale8 = scale64 / 8
        scale2 = scale8 / 4
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale64)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score = [score1, score2, score8, score64]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0236(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0236, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale64 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale64.clamp_(self.eps)
        scale8 = scale64 / 8
        scale4 = scale8 / 2
        scale1 = scale4 / 1
        zero_point = qmin - torch.round(min_val_t / scale64)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score = [score1, score4, score8, score64]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_01237(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_01237, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale128 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale128.clamp_(self.eps)
        scale8 = scale128 / 16
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale128)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale8
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale128

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score128]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to9(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to9, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale512 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale512.clamp_(self.eps)
        scale256 = scale512 / 2
        scale128 = scale256 / 2
        scale64 = scale128 / 2
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale512)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128
            data_q256 = ((data / scale256 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale256
            data_q512 = ((data / scale512 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale512

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score256 = lp_loss(data, data_q256, p=2.0, reduction='all')
            score512 = lp_loss(data, data_q512, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32, score64, score128, score256, score512]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to10(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to10, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale1024 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale1024.clamp_(self.eps)
        scale512 = scale1024 / 2
        scale256 = scale512 / 2
        scale128 = scale256 / 2
        scale64 = scale128 / 2
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale1024)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128
            data_q256 = ((data / scale256 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale256
            data_q512 = ((data / scale512 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale512
            data_q1024 = ((data / scale1024 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale1024

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score256 = lp_loss(data, data_q256, p=2.0, reduction='all')
            score512 = lp_loss(data, data_q512, p=2.0, reduction='all')
            score1024 = lp_loss(data, data_q1024, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32, score64, score128, score256, score512, score1024]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_3_0to10(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_3_0to10, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale1024 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale1024.clamp_(self.eps)
        scale512 = scale1024 / 3
        scale256 = scale512 / 3
        scale128 = scale256 / 3
        scale64 = scale128 / 3
        scale32 = scale64 / 3
        scale16 = scale32 / 3
        scale8 = scale16 / 3
        scale4 = scale8 / 3
        scale2 = scale4 / 3
        scale1 = scale2 / 3
        zero_point = qmin - torch.round(min_val_t / scale1024)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128
            data_q256 = ((data / scale256 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale256
            data_q512 = ((data / scale512 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale512
            data_q1024 = ((data / scale1024 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale1024

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score256 = lp_loss(data, data_q256, p=2.0, reduction='all')
            score512 = lp_loss(data, data_q512, p=2.0, reduction='all')
            score1024 = lp_loss(data, data_q1024, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32, score64, score128, score256, score512, score1024]

            scale_mask[j] *= 3 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to11(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to11, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale2048 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale2048.clamp_(self.eps)
        scale1024 = scale2048 / 2
        scale512 = scale1024 / 2
        scale256 = scale512 / 2
        scale128 = scale256 / 2
        scale64 = scale128 / 2
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale2048)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128
            data_q256 = ((data / scale256 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale256
            data_q512 = ((data / scale512 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale512
            data_q1024 = ((data / scale1024 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale1024
            data_q2048 = ((data / scale2048 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale2048

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score256 = lp_loss(data, data_q256, p=2.0, reduction='all')
            score512 = lp_loss(data, data_q512, p=2.0, reduction='all')
            score1024 = lp_loss(data, data_q1024, p=2.0, reduction='all')
            score2048 = lp_loss(data, data_q2048, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32,
                     score64, score128, score256, score512, score1024, score2048]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to12(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to12, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale4096 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale4096.clamp_(self.eps)
        scale2048 = scale4096 / 2
        scale1024 = scale2048 / 2
        scale512 = scale1024 / 2
        scale256 = scale512 / 2
        scale128 = scale256 / 2
        scale64 = scale128 / 2
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale4096)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128
            data_q256 = ((data / scale256 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale256
            data_q512 = ((data / scale512 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale512
            data_q1024 = ((data / scale1024 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale1024
            data_q2048 = ((data / scale2048 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale2048
            data_q4096 = ((data / scale4096 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale4096

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score256 = lp_loss(data, data_q256, p=2.0, reduction='all')
            score512 = lp_loss(data, data_q512, p=2.0, reduction='all')
            score1024 = lp_loss(data, data_q1024, p=2.0, reduction='all')
            score2048 = lp_loss(data, data_q2048, p=2.0, reduction='all')
            score4096 = lp_loss(data, data_q4096, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32,
                     score64, score128, score256, score512, score1024,
                     score2048, score4096]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to13(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to13, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale8192 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale8192.clamp_(self.eps)
        scale4096 = scale8192 / 2
        scale2048 = scale4096 / 2
        scale1024 = scale2048 / 2
        scale512 = scale1024 / 2
        scale256 = scale512 / 2
        scale128 = scale256 / 2
        scale64 = scale128 / 2
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale8192)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128
            data_q256 = ((data / scale256 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale256
            data_q512 = ((data / scale512 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale512
            data_q1024 = ((data / scale1024 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale1024
            data_q2048 = ((data / scale2048 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale2048
            data_q4096 = ((data / scale4096 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale4096
            data_q8192 = ((data / scale8192 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale8192

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score256 = lp_loss(data, data_q256, p=2.0, reduction='all')
            score512 = lp_loss(data, data_q512, p=2.0, reduction='all')
            score1024 = lp_loss(data, data_q1024, p=2.0, reduction='all')
            score2048 = lp_loss(data, data_q2048, p=2.0, reduction='all')
            score4096 = lp_loss(data, data_q4096, p=2.0, reduction='all')
            score8192 = lp_loss(data, data_q8192, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32,
                     score64, score128, score256, score512, score1024,
                     score2048, score4096, score8192]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to14(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to14, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale16384 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale16384.clamp_(self.eps)
        scale8192 = scale16384 / 2
        scale4096 = scale8192 / 2
        scale2048 = scale4096 / 2
        scale1024 = scale2048 / 2
        scale512 = scale1024 / 2
        scale256 = scale512 / 2
        scale128 = scale256 / 2
        scale64 = scale128 / 2
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale16384)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128
            data_q256 = ((data / scale256 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale256
            data_q512 = ((data / scale512 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale512
            data_q1024 = ((data / scale1024 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale1024
            data_q2048 = ((data / scale2048 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale2048
            data_q4096 = ((data / scale4096 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale4096
            data_q8192 = ((data / scale8192 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale8192
            data_q16384 = ((data / scale16384 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale16384

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score256 = lp_loss(data, data_q256, p=2.0, reduction='all')
            score512 = lp_loss(data, data_q512, p=2.0, reduction='all')
            score1024 = lp_loss(data, data_q1024, p=2.0, reduction='all')
            score2048 = lp_loss(data, data_q2048, p=2.0, reduction='all')
            score4096 = lp_loss(data, data_q4096, p=2.0, reduction='all')
            score8192 = lp_loss(data, data_q8192, p=2.0, reduction='all')
            score16384 = lp_loss(data, data_q16384, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32,
                     score64, score128, score256, score512, score1024,
                     score2048, score4096, score8192, score16384]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to15(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to15, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale32768 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale32768.clamp_(self.eps)
        scale16384 = scale32768 / 2
        scale8192 = scale16384 / 2
        scale4096 = scale8192 / 2
        scale2048 = scale4096 / 2
        scale1024 = scale2048 / 2
        scale512 = scale1024 / 2
        scale256 = scale512 / 2
        scale128 = scale256 / 2
        scale64 = scale128 / 2
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale32768)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128
            data_q256 = ((data / scale256 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale256
            data_q512 = ((data / scale512 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale512
            data_q1024 = ((data / scale1024 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale1024
            data_q2048 = ((data / scale2048 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale2048
            data_q4096 = ((data / scale4096 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale4096
            data_q8192 = ((data / scale8192 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale8192
            data_q16384 = ((data / scale16384 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale16384
            data_q32768 = ((data / scale32768 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale32768

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score256 = lp_loss(data, data_q256, p=2.0, reduction='all')
            score512 = lp_loss(data, data_q512, p=2.0, reduction='all')
            score1024 = lp_loss(data, data_q1024, p=2.0, reduction='all')
            score2048 = lp_loss(data, data_q2048, p=2.0, reduction='all')
            score4096 = lp_loss(data, data_q4096, p=2.0, reduction='all')
            score8192 = lp_loss(data, data_q8192, p=2.0, reduction='all')
            score16384 = lp_loss(data, data_q16384, p=2.0, reduction='all')
            score32768 = lp_loss(data, data_q32768, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32,
                     score64, score128, score256, score512, score1024,
                     score2048, score4096, score8192, score16384, score32768]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to16(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to16, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale65536 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale65536.clamp_(self.eps)
        scale32768 = scale65536 / 2
        scale16384 = scale32768 / 2
        scale8192 = scale16384 / 2
        scale4096 = scale8192 / 2
        scale2048 = scale4096 / 2
        scale1024 = scale2048 / 2
        scale512 = scale1024 / 2
        scale256 = scale512 / 2
        scale128 = scale256 / 2
        scale64 = scale128 / 2
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale65536)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128
            data_q256 = ((data / scale256 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale256
            data_q512 = ((data / scale512 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale512
            data_q1024 = ((data / scale1024 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale1024
            data_q2048 = ((data / scale2048 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale2048
            data_q4096 = ((data / scale4096 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale4096
            data_q8192 = ((data / scale8192 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale8192
            data_q16384 = ((data / scale16384 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale16384
            data_q32768 = ((data / scale32768 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale32768
            data_q65536 = ((data / scale65536 + zero_point).round().clamp(qmin, qmax) -
                           zero_point) * scale65536

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score256 = lp_loss(data, data_q256, p=2.0, reduction='all')
            score512 = lp_loss(data, data_q512, p=2.0, reduction='all')
            score1024 = lp_loss(data, data_q1024, p=2.0, reduction='all')
            score2048 = lp_loss(data, data_q2048, p=2.0, reduction='all')
            score4096 = lp_loss(data, data_q4096, p=2.0, reduction='all')
            score8192 = lp_loss(data, data_q8192, p=2.0, reduction='all')
            score16384 = lp_loss(data, data_q16384, p=2.0, reduction='all')
            score32768 = lp_loss(data, data_q32768, p=2.0, reduction='all')
            score65536 = lp_loss(data, data_q65536, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32,
                     score64, score128, score256, score512, score1024,
                     score2048, score4096, score8192, score16384, score32768, score65536]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to17(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to17, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale_2_17 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale_2_17.clamp_(self.eps)
        scale65536 = scale_2_17 / 2
        scale32768 = scale65536 / 2
        scale16384 = scale32768 / 2
        scale8192 = scale16384 / 2
        scale4096 = scale8192 / 2
        scale2048 = scale4096 / 2
        scale1024 = scale2048 / 2
        scale512 = scale1024 / 2
        scale256 = scale512 / 2
        scale128 = scale256 / 2
        scale64 = scale128 / 2
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale_2_17)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128
            data_q256 = ((data / scale256 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale256
            data_q512 = ((data / scale512 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale512
            data_q1024 = ((data / scale1024 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale1024
            data_q2048 = ((data / scale2048 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale2048
            data_q4096 = ((data / scale4096 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale4096
            data_q8192 = ((data / scale8192 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale8192
            data_q16384 = ((data / scale16384 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale16384
            data_q32768 = ((data / scale32768 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale32768
            data_q65536 = ((data / scale65536 + zero_point).round().clamp(qmin, qmax) -
                           zero_point) * scale65536
            data_q2_17 = ((data / scale_2_17 + zero_point).round().clamp(qmin, qmax) -
                           zero_point) * scale_2_17

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score256 = lp_loss(data, data_q256, p=2.0, reduction='all')
            score512 = lp_loss(data, data_q512, p=2.0, reduction='all')
            score1024 = lp_loss(data, data_q1024, p=2.0, reduction='all')
            score2048 = lp_loss(data, data_q2048, p=2.0, reduction='all')
            score4096 = lp_loss(data, data_q4096, p=2.0, reduction='all')
            score8192 = lp_loss(data, data_q8192, p=2.0, reduction='all')
            score16384 = lp_loss(data, data_q16384, p=2.0, reduction='all')
            score32768 = lp_loss(data, data_q32768, p=2.0, reduction='all')
            score65536 = lp_loss(data, data_q65536, p=2.0, reduction='all')
            score2_17 = lp_loss(data, data_q2_17, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32,
                     score64, score128, score256, score512, score1024,
                     score2048, score4096, score8192, score16384, score32768, score65536,
                     score2_17]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to18(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to18, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale_2_18 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale_2_18.clamp_(self.eps)
        scale_2_17 = scale_2_18 / 2
        scale65536 = scale_2_17 / 2
        scale32768 = scale65536 / 2
        scale16384 = scale32768 / 2
        scale8192 = scale16384 / 2
        scale4096 = scale8192 / 2
        scale2048 = scale4096 / 2
        scale1024 = scale2048 / 2
        scale512 = scale1024 / 2
        scale256 = scale512 / 2
        scale128 = scale256 / 2
        scale64 = scale128 / 2
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale_2_18)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128
            data_q256 = ((data / scale256 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale256
            data_q512 = ((data / scale512 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale512
            data_q1024 = ((data / scale1024 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale1024
            data_q2048 = ((data / scale2048 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale2048
            data_q4096 = ((data / scale4096 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale4096
            data_q8192 = ((data / scale8192 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale8192
            data_q16384 = ((data / scale16384 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale16384
            data_q32768 = ((data / scale32768 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale32768
            data_q65536 = ((data / scale65536 + zero_point).round().clamp(qmin, qmax) -
                           zero_point) * scale65536
            data_q2_17 = ((data / scale_2_17 + zero_point).round().clamp(qmin, qmax) -
                           zero_point) * scale_2_17
            data_q2_18 = ((data / scale_2_18 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale_2_18

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score256 = lp_loss(data, data_q256, p=2.0, reduction='all')
            score512 = lp_loss(data, data_q512, p=2.0, reduction='all')
            score1024 = lp_loss(data, data_q1024, p=2.0, reduction='all')
            score2048 = lp_loss(data, data_q2048, p=2.0, reduction='all')
            score4096 = lp_loss(data, data_q4096, p=2.0, reduction='all')
            score8192 = lp_loss(data, data_q8192, p=2.0, reduction='all')
            score16384 = lp_loss(data, data_q16384, p=2.0, reduction='all')
            score32768 = lp_loss(data, data_q32768, p=2.0, reduction='all')
            score65536 = lp_loss(data, data_q65536, p=2.0, reduction='all')
            score2_17 = lp_loss(data, data_q2_17, p=2.0, reduction='all')
            score2_18 = lp_loss(data, data_q2_18, p=2.0, reduction='all')
            score = [score1, score2, score4, score8, score16, score32,
                     score64, score128, score256, score512, score1024,
                     score2048, score4096, score8192, score16384, score32768, score65536,
                     score2_17, score2_18]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to19(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to19, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale_2_19 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale_2_19.clamp_(self.eps)
        scale_2_18 = scale_2_19 / 2
        scale_2_17 = scale_2_18 / 2
        scale65536 = scale_2_17 / 2
        scale32768 = scale65536 / 2
        scale16384 = scale32768 / 2
        scale8192 = scale16384 / 2
        scale4096 = scale8192 / 2
        scale2048 = scale4096 / 2
        scale1024 = scale2048 / 2
        scale512 = scale1024 / 2
        scale256 = scale512 / 2
        scale128 = scale256 / 2
        scale64 = scale128 / 2
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale_2_19)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128
            data_q256 = ((data / scale256 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale256
            data_q512 = ((data / scale512 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale512
            data_q1024 = ((data / scale1024 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale1024
            data_q2048 = ((data / scale2048 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale2048
            data_q4096 = ((data / scale4096 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale4096
            data_q8192 = ((data / scale8192 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale8192
            data_q16384 = ((data / scale16384 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale16384
            data_q32768 = ((data / scale32768 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale32768
            data_q65536 = ((data / scale65536 + zero_point).round().clamp(qmin, qmax) -
                           zero_point) * scale65536
            data_q2_17 = ((data / scale_2_17 + zero_point).round().clamp(qmin, qmax) -
                           zero_point) * scale_2_17
            data_q2_18 = ((data / scale_2_18 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale_2_18
            data_q2_19 = ((data / scale_2_19 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale_2_19

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score256 = lp_loss(data, data_q256, p=2.0, reduction='all')
            score512 = lp_loss(data, data_q512, p=2.0, reduction='all')
            score1024 = lp_loss(data, data_q1024, p=2.0, reduction='all')
            score2048 = lp_loss(data, data_q2048, p=2.0, reduction='all')
            score4096 = lp_loss(data, data_q4096, p=2.0, reduction='all')
            score8192 = lp_loss(data, data_q8192, p=2.0, reduction='all')
            score16384 = lp_loss(data, data_q16384, p=2.0, reduction='all')
            score32768 = lp_loss(data, data_q32768, p=2.0, reduction='all')
            score65536 = lp_loss(data, data_q65536, p=2.0, reduction='all')
            score2_17 = lp_loss(data, data_q2_17, p=2.0, reduction='all')
            score2_18 = lp_loss(data, data_q2_18, p=2.0, reduction='all')
            score2_19 = lp_loss(data, data_q2_19, p=2.0, reduction='all')

            score = [score1, score2, score4, score8, score16, score32,
                     score64, score128, score256, score512, score1024,
                     score2048, score4096, score8192, score16384, score32768, score65536,
                     score2_17, score2_18, score2_19]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point


class PtfObserver_2_0to20(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver_2_0to20, self).__init__(module_type, bit_type, calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()

        scale_2_20 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale_2_20.clamp_(self.eps)
        scale_2_19 = scale_2_20 / 2
        scale_2_18 = scale_2_19 / 2
        scale_2_17 = scale_2_18 / 2
        scale65536 = scale_2_17 / 2
        scale32768 = scale65536 / 2
        scale16384 = scale32768 / 2
        scale8192 = scale16384 / 2
        scale4096 = scale8192 / 2
        scale2048 = scale4096 / 2
        scale1024 = scale2048 / 2
        scale512 = scale1024 / 2
        scale256 = scale512 / 2
        scale128 = scale256 / 2
        scale64 = scale128 / 2
        scale32 = scale64 / 2
        scale16 = scale32 / 2
        scale8 = scale16 / 2
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale_2_20)

        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        num_channel = inputs.shape[2] if len(inputs.shape) == 3 else inputs.shape[1]
        for j in range(num_channel):
            data = inputs[..., j].unsqueeze(-1)

            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            data_q16 = ((data / scale16 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale16
            data_q32 = ((data / scale32 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale32
            data_q64 = ((data / scale64 + zero_point).round().clamp(qmin, qmax) -
                        zero_point) * scale64
            data_q128 = ((data / scale128 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale128
            data_q256 = ((data / scale256 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale256
            data_q512 = ((data / scale512 + zero_point).round().clamp(qmin, qmax) -
                         zero_point) * scale512
            data_q1024 = ((data / scale1024 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale1024
            data_q2048 = ((data / scale2048 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale2048
            data_q4096 = ((data / scale4096 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale4096
            data_q8192 = ((data / scale8192 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale8192
            data_q16384 = ((data / scale16384 + zero_point).round().clamp(qmin, qmax) -
                           zero_point) * scale16384
            data_q32768 = ((data / scale32768 + zero_point).round().clamp(qmin, qmax) -
                           zero_point) * scale32768
            data_q65536 = ((data / scale65536 + zero_point).round().clamp(qmin, qmax) -
                           zero_point) * scale65536
            data_q2_17 = ((data / scale_2_17 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale_2_17
            data_q2_18 = ((data / scale_2_18 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale_2_18
            data_q2_19 = ((data / scale_2_19 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale_2_19
            data_q2_20 = ((data / scale_2_20 + zero_point).round().clamp(qmin, qmax) -
                          zero_point) * scale_2_20

            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score16 = lp_loss(data, data_q16, p=2.0, reduction='all')
            score32 = lp_loss(data, data_q32, p=2.0, reduction='all')
            score64 = lp_loss(data, data_q64, p=2.0, reduction='all')
            score128 = lp_loss(data, data_q128, p=2.0, reduction='all')
            score256 = lp_loss(data, data_q256, p=2.0, reduction='all')
            score512 = lp_loss(data, data_q512, p=2.0, reduction='all')
            score1024 = lp_loss(data, data_q1024, p=2.0, reduction='all')
            score2048 = lp_loss(data, data_q2048, p=2.0, reduction='all')
            score4096 = lp_loss(data, data_q4096, p=2.0, reduction='all')
            score8192 = lp_loss(data, data_q8192, p=2.0, reduction='all')
            score16384 = lp_loss(data, data_q16384, p=2.0, reduction='all')
            score32768 = lp_loss(data, data_q32768, p=2.0, reduction='all')
            score65536 = lp_loss(data, data_q65536, p=2.0, reduction='all')
            score2_17 = lp_loss(data, data_q2_17, p=2.0, reduction='all')
            score2_18 = lp_loss(data, data_q2_18, p=2.0, reduction='all')
            score2_19 = lp_loss(data, data_q2_19, p=2.0, reduction='all')
            score2_20 = lp_loss(data, data_q2_20, p=2.0, reduction='all')

            score = [score1, score2, score4, score8, score16, score32,
                     score64, score128, score256, score512, score1024,
                     score2048, score4096, score8192, score16384, score32768, score65536,
                     score2_17, score2_18, score2_19, score2_20]

            scale_mask[j] *= 2 ** score.index(min(score))

        scale = scale1 * scale_mask

        return scale, zero_point