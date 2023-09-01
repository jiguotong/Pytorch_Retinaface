from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from configs import cfg_mnet, cfg_re50
from utils.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from utils.utils import show_config

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()

class Prediction(object):
    _defaults={
        "model_path": './checkpoints/mobilenet0.25_Final.pth',
        "backbone_name": 'mobile0.25',
        "save_txt": False,
        "save_image": True,
        "cuda": False,
    }

    def __init__(self, **kwargs):
        self._defaults.update(kwargs)               ## 更新传进来的参数到_defaults
        self.__dict__.update(self._defaults)        ## 更新_defaults到self属性
        self.load()
        
        show_config(**self._defaults)

    def load(self):
        cfg = None
        if self.backbone_name == "mobile0.25":
            self.cfg = cfg_mnet
        elif self.backbone_name == "resnet50":
            self.cfg = cfg_re50
        # 加载网络模型
        net = RetinaFace(cfg=self.cfg, phase = 'test')
        net.load_state_dict(torch.load(self.model_path))
        net.eval()
        cudnn.benchmark = True
        self.net=net

    def detect(self, input_path, output_path):
        input_dir = os.path.dirname(input_path)
        input_filename = os.path.basename(input_path)

        device = torch.device("cuda" if self.cuda else "cpu")
        self.net.to(device)

        img_raw = cv2.imread(input_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)    #(1,3,768,1024)
        img = img.to(device)
        scale = scale.to(device)

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]   #[:, 1]只取了较大的一个anchor的置信度
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS  使用极大值抑制去除重叠过高的候选框，len(keep)即为最终的人脸数目，在此例中，从396中选择出了55个
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        
        if self.save_txt:
            # 获取txt生成目录
            output_dir = os.path.dirname(output_path)
            output_imagename = os.path.basename(output_path)
            filename, ext = os.path.splitext(output_imagename)
            output_txtname = filename + '.txt'

            save_name = os.path.join(output_dir, output_txtname)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            with open(save_name, "w") as fd:
                bboxs = dets
                file_name = os.path.basename(save_name)[:-4] + "\n"
                bboxs_num = str(len(bboxs)) + "\n"
                fd.write(file_name)
                fd.write(bboxs_num)
                for box in bboxs:
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2]) - int(box[0])
                    h = int(box[3]) - int(box[1])
                    confidence = str(box[4])
                    if float(confidence) < args.vis_thres:
                        continue
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                    fd.write(line)
            print("Save txt file {} done.\n".format(output_txtname))
        
        if self.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            cv2.imwrite(output_path, img_raw)
            print("Save image file done.\n")

if __name__ == '__main__':
    image_path = 'data/inputs/00002.jpg'
    result_path = 'data/outputs/00002.jpg'
    predict = Prediction(cuda=True, save_txt=True)
    predict.detect(image_path, result_path)