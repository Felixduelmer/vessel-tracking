import torch
from torch.autograd import Variable
import torch.optim as optim

from collections import OrderedDict
import utils.util as util
from .base_model import BaseModel
from .networks import get_network
from .layers.loss import *
from .networks_other import get_scheduler, print_network, benchmark_fp_bp_time
from .utils import segmentation_stats, get_optimizer, get_criterion
from .networks.utils import HookBasedFeatureExtractor
from models.networks.VesNet import VesNet


class FeedForwardSegmentation(BaseModel):

    def name(self):
        return 'FeedForwardSegmentation'

    def initialize(self, opts, **kwargs):
        BaseModel.initialize(self, opts, **kwargs)
        self.isTrain = opts.isTrain
        self.isRNN = opts.isRNN
        self.inChannels = opts.input_nc

        # define network input and output pars
        self.input = None
        self.target = None

        self.outputs = []
        self.targets = []
        self.states = None

        self.scaler = torch.cuda.amp.GradScaler()

        # load/define networks
        self.net = get_network(opts.model_type, in_channels=len(opts.input_nc),
                               nonlocal_mode=opts.nonlocal_mode, feature_scale=opts.feature_scale,
                               attention_dsample=opts.attention_dsample)
        if self.use_cuda:
            self.net = self.net.cuda()

        # load the model if a path is specified or it is in inference mode
        if not self.isTrain or opts.continue_train:
            self.path_pre_trained_model = opts.path_pre_trained_model
            if self.path_pre_trained_model:
                self.load_network_from_path(
                    self.net, self.path_pre_trained_model, strict=False)
                self.which_epoch = int(0)
            else:
                self.which_epoch = opts.which_epoch
                self.load_network(self.net, 'S', self.which_epoch)

        # training objective
        if self.isTrain:
            self.criterion = get_criterion(opts)
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_S = get_optimizer(opts, self.net.parameters())
            self.optimizers.append(self.optimizer_S)

            # print the network details
            # print the network details
            if kwargs.get('verbose', True):
                print('Network is initialized')
                print_network(self.net)

    def set_scheduler(self, train_opt):
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, train_opt))
            print('Scheduler is added for optimiser {0}'.format(optimizer))

    def init_hidden(self, bs, input_size):
        if self.isRNN:
            # if isinstance(self.net, VesNet):
            #     self.states = [(None, self.net.init_hidden(bs, input_size))]
            # else:
            self.states = self.net.init_hidden(bs, input_size)

    def set_input(self, *inputs):
        for idx, _input in enumerate(inputs):
            # Define that it's a cuda array
            if idx == 0:
                self.input = _input.cuda().float() if self.use_cuda else _input.float()
            elif idx == 1:
                self.target = Variable(
                    _input.cuda()).float() if self.use_cuda else Variable(_input).float()
                assert len(self.input) == len(self.target)

    def forward(self, split):
        if split == 'train':
            self.net.train()
            if not self.isRNN:
                with torch.cuda.amp.autocast():
                    self.prediction = self.net(Variable(self.input[:, self.inChannels, :, :]))
            else:
                with torch.cuda.amp.autocast():
                    self.prediction, self.states = self.net(Variable(self.input[:, self.inChannels, :, :]), self.states)
                self.outputs.append(self.prediction)
                self.targets.append(self.target)
        elif split == 'test':
            self.net.eval()
            if not self.isRNN:
                with torch.no_grad():
                    self.prediction = self.net(
                        Variable(self.input[:, self.inChannels, :, :]))
            else:
                with torch.no_grad():
                    self.prediction, self.states = self.net(
                        Variable(self.input[:, self.inChannels, :, :]), self.states)
            self.prediction = self.net.apply_sigmoid(self.prediction)
            # Apply a softmax and return a segmentation map
            self.pred_seg = torch.round(self.prediction.data) * 255

    def forward_state_aware(self, split, bptt_steps=1):
        state = [i.detach() for i in self.states[-1][1]]
        if split == 'train':
            for item in state:
                item.requires_grad = True
            self.prediction, new_state = self.net(
                Variable(self.input), state)
            for state in new_state:
                state.retain_grad()
            self.outputs.append(self.prediction)
            self.targets.append(self.target)

            while len(self.outputs) > bptt_steps:
                del self.outputs[0]
                del self.targets[0]

        elif split == 'test':
            for item in state:
                item.requires_grad = False
            self.prediction, new_state = self.net(
                Variable(self.input, requires_grad=False), state)
            self.prediction = self.net.apply_sigmoid(self.prediction)
            # Apply a softmax and return a segmentation map
            self.pred_seg = torch.round(self.prediction.data) * 255
        self.states.append((state, new_state))

        while len(self.states) > bptt_steps:
            # Delete stuff that is too old
            del self.states[0]

    def backward(self):
        if self.isRNN:
            self.loss_S = self.criterion(torch.cat(self.outputs), torch.cat(self.targets))
        else:
            self.loss_S = self.criterion(self.prediction, self.target)
        self.scaler.scale(self.loss_S).backward()

    def optimize_parameters(self):
        self.optimizer_S.zero_grad()
        self.backward()
        self.scaler.step(self.optimizer_S)
        self.scaler.update()
        if self.isRNN:
            self.states = [Variable(i.data) for i in self.states]
            self.outputs = []
            self.targets = []

    # for reference check this post:
    # https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/3
    def optimize_parameters_state_aware(self, iteration, bptt_step):
        with torch.autograd.set_detect_anomaly(True):
            self.net.train()
            self.forward_state_aware(split='train', bptt_steps=bptt_step)
            backprop_through_time = (iteration + 1) % bptt_step == 0
            if backprop_through_time:
                self.optimizer_S.zero_grad()
                for i in range(bptt_step - 1):
                    self.loss_S = self.criterion(self.outputs[-i - 1], self.targets[-i - 1])
                    self.loss_S.backward(retain_graph=True)
                    if self.states[-i - 2][0] == None:
                        break
                    for state_1, state_2 in zip(self.states[-i - 1][1], self.states[-i - 2][1]):
                        curr_grad = state_1.grad
                        state_2.backward(curr_grad, retain_graph=True)
                self.optimizer_S.step()

    # This function updates the network parameters every "accumulate_iters"
    def optimize_parameters_accumulate_grd(self, iteration):
        accumulate_iters = int(2)
        if iteration == 0:
            self.optimizer_S.zero_grad()
        self.net.train()
        self.forward(split='train')
        self.backward()

        if iteration % accumulate_iters == 0:
            self.optimizer_S.step()
            self.optimizer_S.zero_grad()

    def test(self):
        self.net.eval()
        self.forward(split='test')

    def inference(self):
        with torch.no_grad():
            self.net.eval()
            self.forward(split="test")
            return self.pred_seg

    def validate(self):
        self.net.eval()
        self.forward(split='test')
        self.loss_S = self.criterion(self.prediction, self.target)


    def validate_state_aware(self):
        self.net.eval()
        self.forward_state_aware(split='test')
        self.loss_S = self.criterion(self.prediction, self.target)

    def get_segmentation_stats(self):
        self.seg_scores, self.dice_score = segmentation_stats(
            self.prediction, self.target)
        seg_stats = [('Overall_Acc', self.seg_scores['overall_acc']),
                     ('Mean_IOU', self.seg_scores['mean_iou'])]
        for class_id in range(self.dice_score.size):
            seg_stats.append(('Class_{}'.format(class_id),
                              self.dice_score[class_id]))
        return OrderedDict(seg_stats)

    def get_current_errors(self):
        return OrderedDict([('Seg_Loss', self.loss_S.item())
                            ])

    def get_current_visuals(self, labels):
        inp_img = util.tensor2im(self.input[:, [0], :, :], 'bmode')
        inp_doppler = util.tensor2im(self.input[:, [1], :, :], 'doppler')
        seg_img = util.tensor2im(self.pred_seg, 'lbl')
        ground_truth = util.tensor2im(labels, 'ground_truth')
        res = OrderedDict([('out_S', seg_img), ('ground_truth', ground_truth)])
        if 0 in self.inChannels:
            res['inp_Bmode'] = inp_img
        if 1 in self.inChannels:
            res['inp_Doppler'] = inp_doppler
        return res

    def get_feature_maps(self, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(
            self.net, layer_name, upscale)
        return feature_extractor.forward(Variable(self.input))

    # returns the fp/bp times of the model
    def get_fp_bp_time(self, size=None):
        if size is None:
            size = (1, 1, 160, 160, 96)

        inp_array = Variable(torch.zeros(*size)).cuda()
        out_array = Variable(torch.zeros(*size)).cuda()
        fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array)

        bsize = size[0]
        return fp / float(bsize), bp / float(bsize)

    def save(self, epoch_label):
        self.save_network(self.net, 'S', epoch_label, self.gpu_ids)

    def save_fold(self, epoch_label, fold):
        self.save_network(self.net, 'S_fold_' + str(fold), epoch_label, self.gpu_ids)
