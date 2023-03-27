## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##
import torch
import numpy as np
from torch.nn import DataParallel
from torch.nn import Sequential, Conv2d, Linear, ReLU, Tanh, Softplus
from crown_ibp.model_defs import Flatten, model_mlp_any
import torch.nn.functional as F
from itertools import chain
import logging
from itertools import product

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BoundFlatten(torch.nn.Module):
    def __init__(self, bound_opts=None):
        super(BoundFlatten, self).__init__()
        self.bound_opts = bound_opts

    def forward(self, x):
        self.shape = x.size()[1:]
        return x.view(x.size(0), -1)

    def interval_propagate(self, norm, h_U, h_L, eps):
        return norm, h_U.view(h_U.size(0), -1), h_L.view(h_L.size(0), -1), 0, 0, 0, 0

    def bound_backward(self, last_uA, last_lA):
        def _bound_oneside(A):
            if A is None:
                return None
            return A.view(A.size(0), A.size(1), *self.shape)
        if self.bound_opts.get("same-slope", False) and (last_uA is not None) and (last_lA is not None):
            new_bound = _bound_oneside(last_uA)
            return new_bound, 0, new_bound, 0
        else:
            return _bound_oneside(last_uA), 0, _bound_oneside(last_lA), 0

class BoundLinear(Linear):
    def __init__(self, in_features, out_features, bias=True, bound_opts=None):
        super(BoundLinear, self).__init__(in_features, out_features, bias)
        self.bound_opts = bound_opts

    @staticmethod
    def convert(linear_layer, bound_opts=None):
        l = BoundLinear(linear_layer.in_features, linear_layer.out_features, linear_layer.bias is not None, bound_opts)
        l.weight.data.copy_(linear_layer.weight.data)
        l.bias.data.copy_(linear_layer.bias.data)
        return l

    def bound_backward(self, last_uA, last_lA):
        def _bound_oneside(last_A, compute_A=True):
            if last_A is None:
                return None, 0
            logger.debug('last_A %s', last_A.size())
            # propagate A to the next layer
            if compute_A:
                next_A = last_A.matmul(self.weight)
                logger.debug('next_A %s', next_A.size())
            else:
                next_A = None
            # compute the bias of this layer
            sum_bias = last_A.matmul(self.bias)
            logger.debug('sum_bias %s', sum_bias.size())
            return next_A, sum_bias
        if self.bound_opts.get("same-slope", False) and (last_uA is not None) and (last_lA is not None):
            uA, ubias = _bound_oneside(last_uA, True)
            _, lbias = _bound_oneside(last_lA, False)
            lA = uA
        else:
            uA, ubias = _bound_oneside(last_uA)
            lA, lbias = _bound_oneside(last_lA)
        return uA, ubias, lA, lbias

    def interval_propagate(self, norm, h_U, h_L, eps, C = None):
        # merge the specification
        if C is not None:
            # after multiplication with C, we have (batch, output_shape, prev_layer_shape)
            # we have batch dimension here because of each example has different C
            weight = C.matmul(self.weight)
            bias = C.matmul(self.bias)
        else:
            # weight dimension (this_layer_shape, prev_layer_shape)
            weight = self.weight
            bias = self.bias

        if norm == np.inf:
            # Linf norm
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = weight.abs()
            if C is not None:
                center = weight.matmul(mid.unsqueeze(-1)) + bias.unsqueeze(-1)
                deviation = weight_abs.matmul(diff.unsqueeze(-1))
                # these have an extra (1,) dimension as the last dimension
                center = center.squeeze(-1)
                deviation = deviation.squeeze(-1)
            else:
                # fused multiply-add
                center = torch.addmm(bias, mid, weight.t())
                deviation = diff.matmul(weight_abs.t())
        else:
            # L2 norm
            h = h_U # h_U = h_L, and eps is used
            dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
            if C is not None:
                center = weight.matmul(h.unsqueeze(-1)) + bias.unsqueeze(-1)
                center = center.squeeze(-1)
            else:
                center = torch.addmm(bias, h, weight.t())
            deviation = weight.norm(dual_norm, -1) * eps

        upper = center + deviation
        lower = center - deviation
        # output 
        return np.inf, upper, lower, 0, 0, 0, 0
            


class BoundConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bound_opts=None):
        super(BoundConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bound_opts = bound_opts

    @staticmethod
    def convert(l, bound_opts=None):
        nl = BoundConv2d(l.in_channels, l.out_channels, l.kernel_size, l.stride, l.padding, l.dilation, l.groups, l.bias is not None, bound_opts)
        nl.weight.data.copy_(l.weight.data)
        nl.bias.data.copy_(l.bias.data)
        logger.debug(nl.bias.size())
        logger.debug(nl.weight.size())
        return nl

    def forward(self, input):
        output = super(BoundConv2d, self).forward(input)
        self.output_shape = output.size()[1:]
        self.input_shape = input.size()[1:]
        return output

    def bound_backward(self, last_uA, last_lA):
        def _bound_oneside(last_A, compute_A=True):
            if last_A is None:
                return None, 0
            logger.debug('last_A %s', last_A.size())
            shape = last_A.size()
            # propagate A to the next layer, with batch concatenated together
            if compute_A:
                
                output_padding0 = int(self.input_shape[1]) - (int(self.output_shape[1]) - 1) * self.stride[0] + 2 * self.padding[0] - int(self.weight.size()[2])
                output_padding1 = int(self.input_shape[2]) - (int(self.output_shape[2]) - 1) * self.stride[1] + 2 * self.padding[1] - int(self.weight.size()[3]) 
                next_A = F.conv_transpose2d(last_A.view(shape[0] * shape[1], *shape[2:]), self.weight, None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, output_padding=(output_padding0, output_padding1))
                next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
                logger.debug('next_A %s', next_A.size())
            else:
                next_A = False
            logger.debug('bias %s', self.bias.size())
            # dot product, compute the bias of this layer, do a dot product
            sum_bias = (last_A.sum((3,4)) * self.bias).sum(2)
            logger.debug('sum_bias %s', sum_bias.size()) 
            return next_A, sum_bias
        # if the slope is the same (Fast-Lin) and both matrices are given, only need to compute one of them
        if self.bound_opts.get("same-slope", False) and (last_uA is not None) and (last_lA is not None):
            uA, ubias = _bound_oneside(last_uA, True)
            _, lbias = _bound_oneside(last_lA, False)
            lA = uA
        else:
            uA, ubias = _bound_oneside(last_uA)
            lA, lbias = _bound_oneside(last_lA)
        return uA, ubias, lA, lbias

    def interval_propagate(self, norm, h_U, h_L, eps):
        if norm == np.inf:
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = self.weight.abs()
            deviation = F.conv2d(diff, weight_abs, None, self.stride, self.padding, self.dilation, self.groups)
        else:
            # L2 norm
            mid = h_U
            logger.debug('mid %s', mid.size())
            # TODO: consider padding here?
            deviation = torch.mul(self.weight, self.weight).sum((1,2,3)).sqrt() * eps
            logger.debug('weight %s', self.weight.size())
            logger.debug('deviation %s', deviation.size())
            deviation = deviation.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            logger.debug('unsqueezed deviation %s', deviation.size())
        center = F.conv2d(mid, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        logger.debug('center %s', center.size())
        upper = center + deviation
        lower = center - deviation
        return np.inf, upper, lower, 0, 0, 0, 0

class BoundTanh(Tanh):
    def __init__(self, prev_layer, inplace=False, bound_opts=None):
        super(BoundTanh, self).__init__()
        self.bound_opts = bound_opts
    
    ## Convert a Tanh layer to BoundTanh layer
    # @param act_layer ReLU layer object
    # @param prev_layer Pre-activation layer, used for get preactivation bounds
    @staticmethod
    def convert(act_layer, prev_layer, bound_opts=None):
        l = BoundTanh(prev_layer, inplace=act_layer, bound_opts=bound_opts)
        return l

    def interval_propagate(self, norm, h_U, h_L, eps):
        assert norm == np.inf
        guard_eps = 1e-5
        self.unstab = ((h_L < -guard_eps) & (h_U > guard_eps))
        # stored upper and lower bounds will be used for backward bound propagation
        self.upper_u = h_U
        self.lower_l = h_L 
        tightness_loss = self.unstab.sum()
        # tightness_loss = torch.min(h_U_unstab * h_U_unstab, h_L_unstab * h_L_unstab).sum()
        return norm, F.tanh(h_U), F.tanh(h_L), tightness_loss, tightness_loss, \
               (h_U < 0).sum(), (h_L > 0).sum()

    def bound_backward(self, last_uA, last_lA):
        print("[WARNING] BoundTanh bound_backward was copied from BoundReLU! Not fully implemented yet.")
        assert(0)
        lb_r = self.lower_l.clamp(max=0)
        ub_r = self.upper_u.clamp(min=0)
        # avoid division by 0 when both lb_r and ub_r are 0
        ub_r = torch.max(ub_r, lb_r + 1e-8)
        # CROWN upper and lower linear bounds
        upper_d = ub_r / (ub_r - lb_r)
        upper_b = - lb_r * upper_d
        upper_d = upper_d.unsqueeze(1)
        if self.bound_opts.get("same-slope", False):
            # the same slope for upper and lower
            lower_d = upper_d
        elif self.bound_opts.get("zero-lb", False):
            # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
            lower_d = (upper_d >= 1.0).float()
        elif self.bound_opts.get("one-lb", False):
            # Always use slope 1 as lower bound
            lower_d = (upper_d > 0.0).float()
        else:
            lower_d = (upper_d > 0.5).float()
        uA = lA = None
        ubias = lbias = 0
        # Choose upper or lower bounds based on the sign of last_A
        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0)
            if self.bound_opts.get("same-slope", False):
                # same upper_d and lower_d, no need to check the sign
                uA = upper_d * last_uA
            else:
                neg_uA = last_uA.clamp(max=0)
                uA = upper_d * pos_uA + lower_d * neg_uA
            mult_uA = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
            ubias = mult_uA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            if self.bound_opts.get("same-slope", False):
                lA = uA if uA is not None else lower_d * last_lA
            else:
                pos_lA = last_lA.clamp(min=0) 
                lA = upper_d * neg_lA + lower_d * pos_lA
            mult_lA = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = mult_lA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
        return uA, ubias, lA, lbias
    
class BoundReLU(ReLU):
    def __init__(self, prev_layer, inplace=False, bound_opts=None):
        super(BoundReLU, self).__init__(inplace)
        # ReLU needs the previous layer's bounds
        # self.prev_layer = prev_layer
        self.bound_opts = bound_opts
    
    ## Convert a ReLU layer to BoundReLU layer
    # @param act_layer ReLU layer object
    # @param prev_layer Pre-activation layer, used for get preactivation bounds
    @staticmethod
    def convert(act_layer, prev_layer, bound_opts=None):
        l = BoundReLU(prev_layer, act_layer.inplace, bound_opts)
        return l

    def interval_propagate(self, norm, h_U, h_L, eps):
        assert norm == np.inf
        guard_eps = 1e-5
        self.unstab = ((h_L < -guard_eps) & (h_U > guard_eps))
        # stored upper and lower bounds will be used for backward bound propagation
        self.upper_u = h_U
        self.lower_l = h_L 
        tightness_loss = self.unstab.sum()
        # tightness_loss = torch.min(h_U_unstab * h_U_unstab, h_L_unstab * h_L_unstab).sum()
        return norm, F.relu(h_U), F.relu(h_L), tightness_loss, tightness_loss, \
               (h_U < 0).sum(), (h_L > 0).sum()

    def bound_backward(self, last_uA, last_lA): 
        lb_r = self.lower_l.clamp(max=0)
        ub_r = self.upper_u.clamp(min=0)
        # avoid division by 0 when both lb_r and ub_r are 0
        ub_r = torch.max(ub_r, lb_r + 1e-8)
        # CROWN upper and lower linear bounds
        upper_d = ub_r / (ub_r - lb_r)
        upper_b = - lb_r * upper_d
        upper_d = upper_d.unsqueeze(1)
        if self.bound_opts.get("same-slope", False):
            # the same slope for upper and lower
            lower_d = upper_d
        elif self.bound_opts.get("zero-lb", False):
            # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
            lower_d = (upper_d >= 1.0).float()
        elif self.bound_opts.get("one-lb", False):
            # Always use slope 1 as lower bound
            lower_d = (upper_d > 0.0).float()
        else:
            lower_d = (upper_d > 0.5).float()
        uA = lA = None
        ubias = lbias = 0
        # Choose upper or lower bounds based on the sign of last_A
        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0)
            if self.bound_opts.get("same-slope", False):
                # same upper_d and lower_d, no need to check the sign
                uA = upper_d * last_uA
            else:
                neg_uA = last_uA.clamp(max=0)
                uA = upper_d * pos_uA + lower_d * neg_uA
            mult_uA = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
            ubias = mult_uA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            if self.bound_opts.get("same-slope", False):
                lA = uA if uA is not None else lower_d * last_lA
            else:
                pos_lA = last_lA.clamp(min=0) 
                lA = upper_d * neg_lA + lower_d * pos_lA
            mult_lA = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = mult_lA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
        return uA, ubias, lA, lbias
    

class BoundSoftplus(Softplus):
    def __init__(self, prev_layer, inplace=False, bound_opts=None):
        self.beta = 100
        super(BoundSoftplus, self).__init__(beta=self.beta)
        # ReLU needs the previous layer's bounds
        # self.prev_layer = prev_layer
        self.bound_opts = bound_opts
    
    ## Convert a ReLU layer to BoundReLU layer
    # @param act_layer ReLU layer object
    # @param prev_layer Pre-activation layer, used for get preactivation bounds
    @staticmethod
    def convert(act_layer, prev_layer, bound_opts=None):
        l = BoundSoftplus(prev_layer, None, bound_opts)
        return l

    def interval_propagate(self, norm, h_U, h_L, eps):
        assert norm == np.inf
        guard_eps = 1e-5
        self.unstab = ((h_L < -guard_eps) & (h_U > guard_eps))
        # stored upper and lower bounds will be used for backward bound propagation
        self.upper_u = h_U
        self.lower_l = h_L 
        tightness_loss = self.unstab.sum()
        # tightness_loss = torch.min(h_U_unstab * h_U_unstab, h_L_unstab * h_L_unstab).sum()
        return norm, F.softplus(h_U, beta=self.beta), F.softplus(h_L, beta=self.beta), tightness_loss, tightness_loss, \
               (h_U < 0).sum(), (h_L > 0).sum()

    def bound_backward(self, last_uA, last_lA):
        # Note: BoundSoftplus bound_backward was copied from BoundReLU.
        # With this, ReachLP is slightly less conservative than it should be.
        # Indeed, softplus(x) >= relu(x). Since CROWN uses linear approximations
        # to compute outer-bounds (see https://arxiv.org/pdf/1811.00866.pdf),
        # we will obtain under-approximations with this implementation.
        # Thus, our comparison is fair, since ReachLP is already very conservative
        # compared to our proposed algorithm (see Algorithm 1 and results).
        # Note: as beta increases, the softplus function becomes closer to relu. This
        # justifies the approximation, whose error is arbitrarily small by increasing beta.
        lb_r = self.lower_l.clamp(max=0)
        ub_r = self.upper_u.clamp(min=0)
        # avoid division by 0 when both lb_r and ub_r are 0
        ub_r = torch.max(ub_r, lb_r + 1e-8)
        # CROWN upper and lower linear bounds
        upper_d = ub_r / (ub_r - lb_r)
        upper_b = - lb_r * upper_d
        upper_d = upper_d.unsqueeze(1)
        if self.bound_opts.get("same-slope", False):
            # the same slope for upper and lower
            lower_d = upper_d
        elif self.bound_opts.get("zero-lb", False):
            # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
            lower_d = (upper_d >= 1.0).float()
        elif self.bound_opts.get("one-lb", False):
            # Always use slope 1 as lower bound
            lower_d = (upper_d > 0.0).float()
        else:
            lower_d = (upper_d > 0.5).float()
        uA = lA = None
        ubias = lbias = 0
        # Choose upper or lower bounds based on the sign of last_A
        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0)
            if self.bound_opts.get("same-slope", False):
                # same upper_d and lower_d, no need to check the sign
                uA = upper_d * last_uA
            else:
                neg_uA = last_uA.clamp(max=0)
                uA = upper_d * pos_uA + lower_d * neg_uA
            mult_uA = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
            ubias = mult_uA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            if self.bound_opts.get("same-slope", False):
                lA = uA if uA is not None else lower_d * last_lA
            else:
                pos_lA = last_lA.clamp(min=0) 
                lA = upper_d * neg_lA + lower_d * pos_lA
            mult_lA = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = mult_lA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
        return uA, ubias, lA, lbias


class BoundSequential(Sequential):
    def __init__(self, *args):
        super(BoundSequential, self).__init__(*args) 

    ## Convert a Pytorch model to a model with bounds
    # @param sequential_model Input pytorch model
    # @return Converted model
    @staticmethod
    def convert(sequential_model, bound_opts=None):
        layers = BoundSequential.sequential_model_to_layers(sequential_model, bound_opts=bound_opts)
        return BoundSequential(*layers)

    @staticmethod
    def sequential_model_to_layers(sequential_model, bound_opts=None):
        layers = []
        if isinstance(sequential_model, Sequential):
            seq_model = sequential_model
        else:
            seq_model = sequential_model.module
        for l in seq_model:
            if isinstance(l, Linear):
                layers.append(BoundLinear.convert(l, bound_opts))
            if isinstance(l, Conv2d):
                layers.append(BoundConv2d.convert(l, bound_opts))
            if isinstance(l, ReLU):
                layers.append(BoundReLU.convert(l, layers[-1], bound_opts))
            if isinstance(l, Softplus):
                # layers.append(BoundReLU.convert(l, layers[-1], bound_opts))
                layers.append(BoundSoftplus.convert(l, layers[-1], bound_opts))
            if isinstance(l, Tanh):
                layers.append(BoundTanh.convert(l, layers[-1], bound_opts))
            if isinstance(l, Flatten):
                layers.append(BoundFlatten(bound_opts))
        return layers

    ## The __call__ function is overwritten for DataParallel
    def __call__(self, *input, **kwargs):
        
        if "method_opt" in kwargs:
            opt = kwargs["method_opt"]
            kwargs.pop("method_opt")
        else:
            raise ValueError("Please specify the 'method_opt' as the last argument.")
        if "disable_multi_gpu" in kwargs:
            kwargs.pop("disable_multi_gpu")
        if opt == "full_backward_range":
            return self.full_backward_range(*input, **kwargs)
        elif opt == "backward_range":
            return self.backward_range(*input, **kwargs)
        elif opt == "interval_range":
            return self.interval_range(*input, **kwargs)
        elif opt == "interval_range-mfe":
            return self.mfe_range2(*input, method='interval_range', **kwargs)
        elif opt == "full_backward_range-mfe":
            return self.mfe_range2(*input, method='full_backward_range', **kwargs)
        else:
            return super(BoundSequential, self).__call__(*input, **kwargs)

    ## Full CROWN bounds with all intermediate layer bounds computed by CROWN
    ## This can be slow for training, and it is recommend to use it for verification only
    # @param norm perturbation norm (np.inf, 2)
    # @param x_L lower bound of input, shape (batch, *image_shape)
    # @param x_U upper bound of input, shape (batch, *image_shape)
    # @param eps perturbation epsilon (not used for Linf)
    # @param C vector of specification, shape (batch, specification_size, output_size)
    # @param upper compute CROWN upper bound
    # @param lower compute CROWN lower bound
    # @param return_matrices don't actually compute the CROWN bounds, stop when you have the matrices
    def full_backward_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None, upper=True, lower=True, return_matrices=False):
        h_U = x_U
        h_L = x_L
        modules = list(self._modules.values())
        # IBP through the first weight (it is the same bound as CROWN for 1st layer, and IBP can be faster)
        for i, module in enumerate(modules):
            norm, h_U, h_L, _, _, _, _ = module.interval_propagate(norm, h_U, h_L, eps)
            # skip the first flatten and linear layer, until we reach the first ReLU layer
            if isinstance(module, BoundReLU) or isinstance(module, BoundTanh):
                # now the upper and lower bound of this ReLU layer has been set in interval_propagate()
                last_module = i
                break
        # CROWN propagation for all rest layers
        # outer loop, starting from the 2nd layer until we reach the output layer
        for i in range(last_module + 1, len(modules)):
            # we do not need bounds after ReLU/flatten layers; we only need the bounds
            # before a ReLU layer
            if isinstance(modules[i], BoundReLU):
                # we set C as the weight of previous layer
                if isinstance(modules[i-1], BoundLinear):
                    # add a batch dimension; all images have the same C in this case
                    newC = modules[i-1].weight.unsqueeze(0)
                    # we skip the layer i, and use CROWN to compute pre-activation bounds
                    # starting from layer i-2 (layer i-1 passed as specification)
                    ub, lb, _, _ = self.backward_range(norm = norm, x_U = x_U, x_L = x_L, eps = eps, C = newC, upper = True, lower = True, modules = modules[:i-1])
                    # add the missing bias term (we propagate newC which do not have bias)
                    ub += modules[i-1].bias
                    lb += modules[i-1].bias
                elif isinstance(modules[i-1], BoundConv2d):
                    # we need to unroll the convolutional layer here
                    c, h, w = modules[i-1].output_shape
                    newC = torch.eye(c*h*w, device = x_U.device, dtype = x_U.dtype)
                    newC = newC.view(1, c*h*w, c, h, w)
                    # use CROWN to compute pre-actiation bounds starting from layer i-1
                    ub, lb, _, _ = self.backward_range(norm = norm, x_U = x_U, x_L = x_L, eps = eps, C = newC, upper = True, lower = True, modules = modules[:i])
                    # reshape to conv output shape; these are pre-activation bounds
                    ub = ub.view(ub.size(0), c, h, w)
                    lb = lb.view(lb.size(0), c, h, w)
                else:
                    raise RuntimeError("Unsupported network structure")
                # set pre-activation bounds for layer i (the ReLU layer)
                modules[i].upper_u = ub
                modules[i].lower_l = lb
        
        # Either (depending on return_matrices):
        # - False: get the final layer bound with spec C
        # - True: get the matrices that can be used to compute the final layer bound
        return self.backward_range(norm = norm, x_U = x_U, x_L = x_L, eps = eps, C = C, upper = upper, lower = lower, return_matrices=return_matrices, final_layer=True)

    ## High level function, will be called outside
    # @param norm perturbation norm (np.inf, 2)
    # @param x_L lower bound of input, shape (batch, *image_shape)
    # @param x_U upper bound of input, shape (batch, *image_shape)
    # @param eps perturbation epsilon (not used for Linf)
    # @param C vector of specification, shape (batch, specification_size, output_size)
    # @param upper compute CROWN upper bound
    # @param lower compute CROWN lower bound
    def backward_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None, upper=False, lower=True, modules=None, return_matrices=False, final_layer=False):
        lower_A, upper_A, lower_sum_b, upper_sum_b = self._prop_from_last_layer(C=C, x_U=x_U, modules=modules, upper=upper, lower=lower)
        if return_matrices:
            # The caller doesn't care about the actual bounds, just the matrices used to compute them
            return lower_A, upper_A, lower_sum_b, upper_sum_b
        lb = self._get_concrete_bound(lower_A, lower_sum_b, sign = -1, x_U=x_U, x_L=x_L, norm=norm)
        ub = self._get_concrete_bound(upper_A, upper_sum_b, sign = +1, x_U=x_U, x_L=x_L, norm=norm)
        ub, lb = self._check_if_bnds_exist(ub=ub, lb=lb, x_U=x_U, x_L=x_L)
        return ub, lb, upper_sum_b, lower_sum_b

    def _check_if_bnds_exist(self, ub=None, lb=None, x_U=None, x_L=None):
        if ub is None:
            ub = x_U.new([np.inf])
        if lb is None:
            lb = x_L.new([-np.inf])
        return ub, lb

    def _prop_from_last_layer(self, C=None, x_U=None, modules=None, upper=False, lower=True):
        # start propagation from the last layer
        modules = list(self._modules.values()) if modules is None else modules
        upper_A = C if upper else None
        lower_A = C if lower else None
        upper_sum_b = lower_sum_b = x_U.new([0])
        for i, module in enumerate(reversed(modules)):
            upper_A, upper_b, lower_A, lower_b = module.bound_backward(upper_A, lower_A)
            # squeeze is for using broadcasting in the cast that all examples use the same spec
            upper_sum_b = upper_b + upper_sum_b
            lower_sum_b = lower_b + lower_sum_b
        return lower_A, upper_A, lower_sum_b, upper_sum_b 

    # sign = +1: upper bound, sign = -1: lower bound
    def _get_concrete_bound(self, A, sum_b, x_U=None, x_L=None, norm=np.inf, sign = -1):
        if A is None:
            return None
        A = A.view(A.size(0), A.size(1), -1)
        # A has shape (batch, specification_size, flattened_input_size)
        logger.debug('Final A: %s', A.size())
        if norm == np.inf:
            x_ub = x_U.view(x_U.size(0), -1, 1)
            x_lb = x_L.view(x_L.size(0), -1, 1)
            center = (x_ub + x_lb) / 2.0
            diff = (x_ub - x_lb) / 2.0
            logger.debug('A_0 shape: %s', A.size())
            logger.debug('sum_b shape: %s', sum_b.size())
            # we only need the lower bound

            # First 2 terms in Eq. 20 of: https://arxiv.org/pdf/1811.00866.pdf
            # eps*q_norm(Lamb) + Lamb*x0 <==> x0=center, eps=diff, Lamb=A
            bound = A.bmm(center) + sign * A.abs().bmm(diff)
            logger.debug('bound shape: %s', bound.size())
        else:
            x = x_U.view(x_U.size(0), -1, 1)
            dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
            deviation = A.norm(dual_norm, -1) * eps
            bound = A.bmm(x) + sign * deviation.unsqueeze(-1)
        # Add big summation term from Eq. 20 of: https://arxiv.org/pdf/1811.00866.pdf

        if sum_b.ndim == 3:
            sum_b = sum_b.squeeze(-1)
        bound = bound.squeeze(-1) + sum_b
        # print("bound.size(): {}".format(bound.size()))
        return bound

    def interval_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None):
        losses = 0
        unstable = 0
        dead = 0
        alive = 0
        h_U = x_U
        h_L = x_L
        for i, module in enumerate(list(self._modules.values())[:-1]):
            # all internal layers should have Linf norm, except for the first layer
            norm, h_U, h_L, loss, uns, d, a = module.interval_propagate(norm, h_U, h_L, eps)
            # this is some stability loss used for initial experiments, not used in CROWN-IBP as it is not very effective
            losses += loss
            unstable += uns
            dead += d
            alive += a
        # last layer has C to merge
        norm, h_U, h_L, loss, uns, d, a = list(self._modules.values())[-1].interval_propagate(norm, h_U, h_L, eps, C)
        losses += loss
        unstable += uns
        dead += d
        alive += a
        return h_U, h_L, losses, unstable, dead, alive

    def interval_range_mfe_orig(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None):
        ### MFE: TODO: split up the x_U and x_L into groups
        splits = []
        for i in range(11):
            x,y = np.random.randint(0,28,2)
            splits.append([Ellipsis,x,y])

        x_Us, x_Ls = self.split(x_U, x_L)

        # import matplotlib.pyplot as plt
        # plt.imshow(x_Ls[1][0,0]); plt.colorbar(); plt.show()

        # Original call
        h_U, h_L, losses, unstable, dead, alive = self.interval_range(norm=norm, x_U=x_U, x_L=x_L, eps=eps, C=C)
        orig_h_U = h_U
        orig_h_L = h_L

        h_U = torch.ones_like(h_U)*(-float('inf'))
        h_L = torch.ones_like(h_L)*(float('inf'))

        num_entries = len(x_Us)
        for i in range(num_entries):
            print(i/float(num_entries))
            x_U_ = x_Us[i]
            x_L_ = x_Ls[i]
            h_U_, h_L_, losses_, unstable_, dead_, alive_ = self.interval_range(norm=norm, x_U=x_U_, x_L=x_L_, eps=eps, C=C)
            h_U = torch.max(h_U, h_U_)
            h_L = torch.min(h_L, h_L_)

        print("Orig:")
        print("h_U: {}".format(orig_h_U[0]))
        print("h_L: {}".format(orig_h_L[0]))
        print('---')

        print("New:")
        print("h_U: {}".format(h_U[0]))
        print("h_L: {}".format(h_L[0]))

        import pdb; pdb.set_trace()
        assert(0)

        ### MFE: TODO: update ((h_U, h_L, losses, unstable, dead, alive)) using worst-case of groups
        return h_U, h_L, losses, unstable, dead, alive


    def split(self, x_U, x_L, splits):

        x_Us = [x_U]
        x_Ls = [x_L]

        # splits = [[Ellipsis,26,11], [Ellipsis,26,12]]
        for split in splits:
            num_entries = len(x_Us)
            for i in range(num_entries):
                prev_low = x_Ls[2*i].detach().clone()
                prev_high = x_Us[2*i].detach().clone()
                midpt = (prev_high[split] - prev_low[split])/2
                new_x_L = prev_low
                new_x_U = prev_high
                new_x_L[split] += midpt
                new_x_U[split] -= midpt
                x_Ls.insert(2*i+1, new_x_L)
                x_Us.insert(2*i, new_x_U)

    def split2(self, x_U, x_L, splits):
        # Don't fully split all first, split one-by-one (less memory)

        inds = []
        counts = []
        for split in splits:
            if split not in inds:
                inds.append(split)
                counts.append(1)
            else:
                counts[inds.index(split)] += 1

        vals = []
        val_inds = []
        for i in range(len(inds)):
            split = inds[i]
            num_splits = counts[i]
            vals.append(torch.Tensor(np.linspace(x_L[split], x_U[split], 2*num_splits+1)))
            val_inds.append(np.arange(2*num_splits))
        return inds, counts, vals, val_inds

    def mfe_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None, method=interval_range, upper=True, lower=True, return_matrices=False):
        # MFE: TODO: pass upper/lower/return_matrices args to methods!!!

        # Original call
        h_U, h_L = getattr(self, method)(norm=norm, x_U=x_U, x_L=x_L, eps=eps, C=C)[:2]
        orig_h_U = h_U
        orig_h_L = h_L
        h_U = torch.ones_like(h_U)*(-float('inf'))
        h_L = torch.ones_like(h_L)*(float('inf'))

        # splits = [[Ellipsis,0,0], [Ellipsis,0,1], [Ellipsis,0,0]]
        # splits = [[Ellipsis,26,11], [Ellipsis,26,12]]
        splits = []
        for i in range(3):
            x,y = np.random.randint(0,28,2)
            splits.append([Ellipsis,x,y])
        inds, counts, vals, val_inds = self.split2(x_U, x_L, splits)

        i = 0
        for element in product(*val_inds):
            x_L_ = x_L.detach().clone()
            x_U_ = x_U.detach().clone()
            for split in range(len(inds)):
                state = inds[split]
                x_L_[state] = vals[split][element[split]]
                x_U_[state] = vals[split][element[split]+1]
            print(i/(2**len(splits)))
            h_U_, h_L_ = getattr(self, method)(norm=norm, x_U=x_U_, x_L=x_L_, eps=eps, C=C)[:2]
            h_U = torch.max(h_U, h_U_)
            h_L = torch.min(h_L, h_L_)
            i += 1

        print("Orig:")
        print("h_U: {}".format(orig_h_U[0]))
        print("h_L: {}".format(orig_h_L[0]))
        print('---')

        print("New:")
        print("h_U: {}".format(h_U[0]))
        print("h_L: {}".format(h_L[0]))

        import pdb; pdb.set_trace()

        ### MFE: TODO: update ((h_U, h_L, losses, unstable, dead, alive)) using worst-case of groups
        return h_U, h_L

    def mfe_range2(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None, method=interval_range, upper=True, lower=True, return_matrices=False):
        # MFE: TODO: pass upper/lower/return_matrices args to methods!!!

        # Original call
        h_U, h_L = getattr(self, method)(norm=norm, x_U=x_U, x_L=x_L, eps=eps, C=C)[:2]
        orig_h_U = h_U
        orig_h_L = h_L
        h_U = torch.ones_like(h_U)*(-float('inf'))
        h_L = torch.ones_like(h_L)*(float('inf'))
        best_h_U = torch.ones_like(h_U)*(float('inf'))
        best_h_L = torch.ones_like(h_L)*(-float('inf'))

        import matplotlib.pyplot as plt


        # splits = [[Ellipsis,0,0], [Ellipsis,0,1], [Ellipsis,0,0]]
        # splits = [[Ellipsis,26,11], [Ellipsis,26,12]]
        num_states = 784
        for i in range(num_states):
            print(i/float(num_states))
            splits = []
            x = int(i/28)
            y = int(i%28)
            # x,y = np.random.randint(0,28,2)
            splits.append([Ellipsis,x,y])
            inds, counts, vals, val_inds = self.split2(x_U, x_L, splits)
            for element in product(*val_inds):
                x_L_ = x_L.detach().clone()
                x_U_ = x_U.detach().clone()
                for split in range(len(inds)):
                    state = inds[split]
                    x_L_[state] = vals[split][element[split]]
                    x_U_[state] = vals[split][element[split]+1]
                # plt.imshow(x_L_[0,0]); plt.colorbar(); plt.show()
                h_U_, h_L_ = getattr(self, method)(norm=norm, x_U=x_U_, x_L=x_L_, eps=eps, C=C)[:2]
                print(h_U_[0])
                print(h_L_[0])
                h_U = torch.max(h_U, h_U_)
                h_L = torch.min(h_L, h_L_)
                # print("h_U: {}".format(h_U[0]))
                # print("h_L: {}".format(h_L[0]))
            print('--')
            assert(0)
            best_h_U = torch.min(h_U, best_h_U)
            best_h_L = torch.max(h_L, best_h_L)

        print("Orig:")
        print("h_U: {}".format(orig_h_U[0]))
        print("h_L: {}".format(orig_h_L[0]))
        print('---')

        print("New:")
        print("h_U: {}".format(best_h_U[0]))
        print("h_L: {}".format(best_h_L[0]))

        import pdb; pdb.set_trace()

        ### MFE: TODO: update ((h_U, h_L, losses, unstable, dead, alive)) using worst-case of groups
        return h_U, h_L

class BoundDataParallel(DataParallel):
    # This is a customized DataParallel class for our project
    def __init__(self, *inputs, **kwargs):
        super(BoundDataParallel, self).__init__(*inputs, **kwargs)
        self._replicas = None
    # Overide the forward method
    def forward(self, *inputs, **kwargs):
        disable_multi_gpu = False
        if "disable_multi_gpu" in kwargs:
            disable_multi_gpu = kwargs["disable_multi_gpu"]
            kwargs.pop("disable_multi_gpu")
         
        if not self.device_ids or disable_multi_gpu: 
            return self.module(*inputs, **kwargs)
       
        # Only replicate during forwarding propagation. Not during interval bounds
        # and CROWN-IBP bounds, since weights have not been updated. This saves 2/3
        # of communication cost.
        if self._replicas is None or kwargs.get("method_opt", "forward") == "forward":
            self._replicas = self.replicate(self.module, self.device_ids)  

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids) 
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        outputs = self.parallel_apply(self._replicas[:len(inputs)], inputs, kwargs)
        return self.gather(outputs, self.output_device)

if __name__ == '__main__':
    from tensorflow.keras.models import model_from_json
    from crown_ibp.conversions.keras2torch import keras2torch, get_keras_model
    import matplotlib.pyplot as plt

    # load json and create model
    json_file = open('/Users/mfe/Downloads/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("/Users/mfe/Downloads/model.h5")
    print("Loaded model from disk")

    torch_model = keras2torch(model, "torch_model")
    
    ###########################
    # To get NN nominal prediction:
    print('---')
    print("Example of a simple forward pass for a single point input.")
    x = [2.5, 0.2]
    out = torch_model.forward(torch.Tensor([[x]]))
    print("For x={}, NN output={}.".format(x,out))
    print('---')
    #
    ###########################


    ###########################
    # To get NN output bounds:
    print('---')
    print("Example of bounding the NN output associated with an input set.")
    torch_model_ = BoundSequential.convert(torch_model, {"same-slope": True})
    x0_min, x0_max, x1_min, x1_max = [2.5, 3.0, -0.25, 0.25]

    # Evaluate CROWN bounds
    out_max_crown, out_min_crown, _, _ = torch_model_.full_backward_range(norm=np.inf,
                                x_U=torch.Tensor([[x0_max, x1_max]]),
                                x_L=torch.Tensor([[x0_min, x1_min]]),
                                upper=True, lower=True,
                                C=torch.Tensor([[[1]]]),
                                )

    # Sample a grid of pts from the input set, to get exact NN output polytope
    x0 = np.linspace(x0_min, x0_max, num=10)
    x1 = np.linspace(x1_min, x1_max, num=10)
    xx,yy = np.meshgrid(x0, x1)
    pts = np.reshape(np.dstack([xx,yy]), (-1,2))
    sampled_outputs = torch_model.forward(torch.Tensor(pts))

    # Print and compare the two bounds numerically    
    sampled_output_min = np.min(sampled_outputs.data.numpy())
    sampled_output_max = np.max(sampled_outputs.data.numpy())
    crown_min = out_min_crown.data.numpy()[0,0]
    crown_max = out_max_crown.data.numpy()[0,0]
    print("The sampled outputs lie between: [{},{}]".format(
        sampled_output_min, sampled_output_max))
    print("CROWN bounds are: [{},{}]".format(
        crown_min, crown_max))
    conservatism_above = crown_max - sampled_output_max
    conservatism_below = sampled_output_min - crown_min
    print("Over-conservatism: [{},{}]".format(
        conservatism_below, conservatism_above))
    print("^ These should both be positive! {}".format(
        "They are :)" if conservatism_above>=0 and conservatism_below>=0 else "*** THEY AREN'T ***"))

    # Plot vertical lines for CROWN bounds, x's for sampled outputs
    plt.axvline(out_min_crown.data.numpy()[0,0], ls='--', label='CROWN Bounds')
    plt.axvline(out_max_crown.data.numpy()[0,0], ls='--')
    plt.scatter(sampled_outputs.data.numpy(), np.zeros(pts.shape[0]), marker='x', label="Samples")

    plt.legend()
    print("Showing plot...")
    plt.show()
    print('---')

    #
    ###########################




