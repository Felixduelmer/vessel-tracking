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


class FeedForwardSegmentation(BaseModel):

    def name(self):
        return 'FeedForwardSegmentation'

    def initialize(self, opts, **kwargs):
        BaseModel.initialize(self, opts, **kwargs)
        self.isTrain = opts.isTrain

        # define network input and output pars
        self.input = None
        self.target = None
        self.hidden = None
        self.tensor_dim = opts.tensor_dim
        self.bptt = opts.bptt_step

        # load/define networks
        self.net = get_network(opts.model_type, in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                               tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
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

    def set_input(self, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
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
            self.prediction, self.hidden = self.net(
                Variable(self.input), self.bptt, self.hidden)
        elif split == 'test':
            self.prediction, _ = self.net(
                Variable(self.input, requires_grad=False), self.bptt, self.hidden)
            self.prediction = self.net.apply_sigmoid(self.prediction)
            # Apply a softmax and return a segmentation map
            self.pred_seg = torch.round(self.prediction.data)*255

    def backward(self):
        self.loss_S = self.criterion(self.prediction, self.target)
        self.loss_S.backward()
        self.hidden = [h.detach() for h in self.hidden]

    def optimize_parameters(self):
        self.net.train()
        self.forward(split='train')

        self.optimizer_S.zero_grad()
        self.backward()
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

    def validate(self):
        self.net.eval()
        self.forward(split='test')
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
        return OrderedDict([('out_S', seg_img), ('ground_truth', ground_truth), ('inp_S', inp_img), ('inp_doppler', inp_doppler)])

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
        return fp/float(bsize), bp/float(bsize)

    def save(self, epoch_label):
        self.save_network(self.net, 'S', epoch_label, self.gpu_ids)
