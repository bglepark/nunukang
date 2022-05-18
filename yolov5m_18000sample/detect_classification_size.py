# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""
import matplotlib.pyplot as plt  # 폰트설정

plt.rcParams['font.family'] = 'Malgun Gothic'
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # (save_dir / 'labels').mkdir(parents=True, exist_ok=True) -> 나중에 이걸로 변경
    # (save_dir / 'images').mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    fish_size = []  # 어종별 class와 길이정보를 담을 빈 리스트 생성

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)
            # save_path = str(save_dir / 'images' / p.stem) + f'_{frame}' + '.jpg'   # im.jpg -> 나중에 이걸로 변경
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                fish_name = None
                fish_name_number=None
                of_width = None
                of_height = None
                kr_width = None
                kr_height = None
                cp_width = None
                cp_height = None
                rs_width = None
                rs_height = None
                bp_width = None
                bp_height = None
                rb_width = None
                rb_height = None
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        conf = conf * 100

                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} : {conf:.2f} %')
                        if 'cigarette pack' in label:
                            with open(txt_path + '.txt', 'w') as f: #담뱃갑만 있거나 담뱃갑만 인식될 경우 5만 적힘
                                f.write('5')
                            # fish_size.extend([5])

                            cp_width = (int(xyxy[2]) - int(xyxy[0]))  # 담뱃갑 너비
                            if type(cp_width) == 'tuple':
                                cp_width = cp_width[0]
                            cp_height = (int(xyxy[3]) - int(xyxy[1]))  # 담뱃갑 높이
                            if type(cp_height) == 'tuple':
                                cp_height = cp_height[0]

                            annotator.box_label(xyxy, label=None, color=colors(c, True))  # 라벨은 하지 말고 bbox만 그림


                            if cp_width < cp_height:  # 담뱃갑 너비가 담뱃갑 높이보다 클때 -> 가로로 배치된 경우
                                cv2.putText(im0, '5.5cm',
                                            (int(xyxy[0] + 1), int(xyxy[1]) - 6),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6,
                                            (255, 0, 255), 2)
                            else:  # 담뱃갑 높이가 담뱃갑 너비보다 클때 -> 세로로 배치된 경우
                                cv2.putText(im0, '5.5cm',
                                            (int(xyxy[2] + 5), int((xyxy[1] + xyxy[3]) / 2)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6,
                                            (255, 0, 255), 2)



                        if 'Red seabream' in label:  # 참돔
                            annotator.box_label(xyxy, label=None, color=colors(c, True))  # 라벨은 하지 말고 bbox만 그림
                            fish_name = names[c]
                            fish_name_number = c
                            rs_width = (int(xyxy[2]) - int(xyxy[0]))
                            if type(rs_width) == 'tuple':
                                rs_width = rs_width[0]

                            rs_height = (int(xyxy[3]) - int(xyxy[1]))  # 생선 높이
                            if type(rs_height) == 'tuple':
                                rs_height = rs_height[0]



                            if rs_width > rs_height:
                                rs_x_position_w = int(xyxy[0] + 1)
                                rs_y_position_w = int(xyxy[1] - 6)
                            else:
                                rs_x_position_h = int(xyxy[2] + 5)
                                rs_y_position_h = int((xyxy[1] + xyxy[3]) / 2)

                        if 'Black porgy' in label:  # 감성돔
                            annotator.box_label(xyxy, label=None, color=colors(c, True))  # 라벨은 하지 말고 bbox만 그림
                            fish_name = names[c]
                            fish_name_number = c
                            bp_width = (int(xyxy[2]) - int(xyxy[0])) # 생선 너비
                            if type(bp_width) == 'tuple':
                                bp_width = bp_width[0]

                            bp_height = (int(xyxy[3]) - int(xyxy[1]))  # 생선 높이
                            if type(bp_height) == 'tuple':
                                bp_height = bp_height[0]

                            if bp_width > bp_height:
                                bp_x_position_w = int(xyxy[0] + 1)
                                bp_y_position_w = int(xyxy[1] - 6)
                            else:
                                bp_x_position_h = int(xyxy[2] + 5)
                                bp_y_position_h = int((xyxy[1] + xyxy[3]) / 2)

                        if 'Olive flounder' in label:  # 광어(넙치)
                            annotator.box_label(xyxy, label=None, color=colors(c, True))  # 라벨은 하지 말고 bbox만 그림
                            fish_name = names[c]
                            fish_name_number = c
                            of_width = (int(xyxy[2]) - int(xyxy[0])) # 생선 너비
                            if type(of_width) == 'tuple':
                                of_width = of_width[0]

                            of_height = (int(xyxy[3]) - int(xyxy[1]))  # 생선 높이
                            if type(bp_height) == 'tuple':
                                of_height = of_height[0]

                            if of_width > of_height:
                                of_x_position_w = int(xyxy[0] + 1)
                                of_y_position_w = int(xyxy[1] - 6)
                            else:
                                of_x_position_h = int(xyxy[2] + 5)
                                of_y_position_h = int((xyxy[1] + xyxy[3]) / 2)

                        if 'Korea rockfish' in label:  # 우럭(조피볼락)
                            annotator.box_label(xyxy, label=None, color=colors(c, True))  # 라벨은 하지 말고 bbox만 그림
                            fish_name = names[c]
                            fish_name_number = c
                            kr_width = (int(xyxy[2]) - int(xyxy[0]))  # 생선 너비
                            if type(kr_width) == 'tuple':
                                kr_width = kr_width[0]

                            kr_height = (int(xyxy[3]) - int(xyxy[1]))  # 생선 높이
                            if type(kr_height) == 'tuple':
                                kr_height = kr_height[0]

                            if kr_width > kr_height:
                                kr_x_position_w = int(xyxy[0] + 1)
                                kr_y_position_w = int(xyxy[1] - 6)
                            else:
                                kr_x_position_h = int(xyxy[2] + 5)
                                kr_y_position_h = int((xyxy[1] + xyxy[3]) / 2)

                        if 'Rock bream' in label:  # 돌돔
                            annotator.box_label(xyxy, label=None, color=colors(c, True))  # 라벨은 하지 말고 bbox만 그림
                            fish_name = names[c]
                            fish_name_number = c
                            rb_width = (int(xyxy[2]) - int(xyxy[0]))  # 생선 너비
                            if type(rb_width) == 'tuple':
                                rb_width = rb_width[0]

                            rb_height = (int(xyxy[3]) - int(xyxy[1]))  # 생선 높이
                            if type(rb_height) == 'tuple':
                                rb_height = rb_height[0]

                            if rb_width > rb_height:
                                rb_x_position_w = int(xyxy[0] + 1)
                                rb_y_position_w = int(xyxy[1] - 6)
                            else:
                                rb_x_position_h = int(xyxy[2] + 5)
                                rb_y_position_h = int((xyxy[1] + xyxy[3]) / 2)


                    if fish_name == 'Red seabream':
                        try:
                            cp_width = min(cp_width , cp_height)
                            if rs_width>rs_height:
                                rs_actual_width = round(float(rs_width / cp_width * 5.5),2)
                                cv2.putText(im0, fish_name+' : '+str(rs_actual_width)+'cm' , (rs_x_position_w, rs_y_position_w),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)



                                with open(txt_path + '.txt', 'w') as f:
                                    f.write(str(fish_name_number)+' '+str(round(rs_actual_width,2)))
                            else:
                                rs_actual_width = round(float(rs_height / cp_width * 5.5), 2)
                                cv2.putText(im0, fish_name+' : ' + str(rs_actual_width) + 'cm',
                                            (rs_x_position_h, rs_y_position_h),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6,
                                            (0, 0, 255), 2)


                                with open(txt_path + '.txt', 'w') as f:
                                    f.write(str(fish_name_number)+' '+str(round(rs_actual_width,2)))

                        except:
                            # 물고기만 있을 때(담뱃갑 x)

                            with open(txt_path + '.txt', 'w') as f:
                                f.write(str(fish_name_number))

                            if rs_width>rs_height: #물고기가 가로로 배치
                                cv2.putText(im0, fish_name,
                                        (rs_x_position_w, rs_y_position_w),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 0, 255), 2)
                            else: #물고기가 세로로 배치
                                cv2.putText(im0, fish_name,
                                            (rs_x_position_h, rs_y_position_h),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6,
                                            (0, 0, 255), 2)

                    if fish_name == 'Black porgy':
                        try:
                            cp_width = min(cp_width, cp_height)
                            if bp_width > bp_height:
                                bp_actual_width = round(float(bp_width / cp_width * 5.5), 2)
                                cv2.putText(im0, fish_name + ' : ' + str(bp_actual_width) + 'cm',
                                                (bp_x_position_w, bp_y_position_w),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)


                                with open(txt_path + '.txt', 'w') as f:
                                    f.write(str(fish_name_number) + ' ' + str(round(bp_actual_width, 2)))
                            else:
                                bp_actual_width = round(float(bp_height / cp_width * 5.5), 2)
                                cv2.putText(im0, fish_name + ' : ' + str(bp_actual_width) + 'cm',
                                                (bp_x_position_h, bp_y_position_h),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)


                                with open(txt_path + '.txt', 'w') as f:
                                    f.write(str(fish_name_number) + ' ' + str(round(bp_actual_width, 2)))

                        except:
                            # 물고기만 있을 때(담뱃갑 x)
                            # fish_size.extend([fish_name_number])
                            with open(txt_path + '.txt', 'w') as f:
                                f.write(str(fish_name_number))

                            if bp_width > bp_height:  # 물고기가 가로로 배치
                                cv2.putText(im0, fish_name,
                                                (bp_x_position_w, bp_y_position_w),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)
                            else:  # 물고기가 세로로 배치
                                cv2.putText(im0, fish_name,
                                                (bp_x_position_h, bp_y_position_h),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)

                    if fish_name == 'Olive flounder':
                        try:
                            cp_width = min(cp_width, cp_height)
                            if of_width > of_height:
                                of_actual_width = round(float(of_width / cp_width * 5.5), 2)
                                cv2.putText(im0, fish_name + ' : ' + str(of_actual_width) + 'cm',
                                                (of_x_position_w, of_y_position_w),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)


                                with open(txt_path + '.txt', 'w') as f:
                                    f.write(str(fish_name_number) + ' ' + str(round(of_actual_width, 2)))
                            else:
                                of_actual_width = round(float(of_height / cp_width * 5.5), 2)
                                cv2.putText(im0, fish_name + ' : ' + str(of_actual_width) + 'cm',
                                                (of_x_position_h, of_y_position_h),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)


                                with open(txt_path + '.txt', 'w') as f:
                                    f.write(str(fish_name_number) + ' ' + str(round(of_actual_width, 2)))

                        except:
                            # 물고기만 있을 때(담뱃갑 x)
                            # fish_size.extend([fish_name_number])
                            with open(txt_path + '.txt', 'w') as f:
                                f.write(str(fish_name_number))

                            if of_width > of_height:  # 물고기가 가로로 배치
                                cv2.putText(im0, fish_name,
                                                (of_x_position_w, of_y_position_w),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)
                            else:  # 물고기가 세로로 배치
                                cv2.putText(im0, fish_name,
                                                (of_x_position_h, of_y_position_h),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)

                    if fish_name == 'Korea rockfish':
                        try:
                            cp_width = min(cp_width, cp_height)
                            if kr_width > kr_height:
                                kr_actual_width = round(float(kr_width / cp_width * 5.5), 2)
                                cv2.putText(im0, fish_name + ' : ' + str(kr_actual_width) + 'cm',
                                                (kr_x_position_w, kr_y_position_w),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)


                                with open(txt_path + '.txt', 'w') as f:
                                    f.write(str(fish_name_number) + ' ' + str(round(kr_actual_width, 2)))
                            else:
                                kr_actual_width = round(float(kr_height / cp_width * 5.5), 2)
                                cv2.putText(im0, fish_name + ' : ' + str(kr_actual_width) + 'cm',
                                                (kr_x_position_h, kr_y_position_h),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)


                                with open(txt_path + '.txt', 'w') as f:
                                    f.write(str(fish_name_number) + ' ' + str(round(kr_actual_width, 2)))

                        except:
                                # 물고기만 있을 때(담뱃갑 x)

                            with open(txt_path + '.txt', 'w') as f:
                                f.write(str(fish_name_number))

                            if kr_width > kr_height:  # 물고기가 가로로 배치
                                cv2.putText(im0, fish_name,
                                                (kr_x_position_w, kr_y_position_w),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)
                            else:  # 물고기가 세로로 배치
                                cv2.putText(im0, fish_name,
                                                (kr_x_position_h, kr_y_position_h),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)

                    if fish_name == 'Rock bream':
                        try:
                            cp_width = min(cp_width, cp_height)
                            if rb_width > rb_height:
                                rb_actual_width = round(float(rb_width / cp_width * 5.5), 2)
                                cv2.putText(im0, fish_name + ' : ' + str(rb_actual_width) + 'cm',
                                                (rb_x_position_w, rb_y_position_w),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)


                                with open(txt_path + '.txt', 'w') as f:
                                    f.write(str(fish_name_number) + ' ' + str(round(rb_actual_width, 2)))
                            else:
                                rb_actual_width = round(float(rb_height / cp_width * 5.5), 2)
                                cv2.putText(im0, fish_name + ' : ' + str(rb_actual_width) + 'cm',
                                                (rb_x_position_h, rb_y_position_h),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)


                                with open(txt_path + '.txt', 'w') as f:
                                    f.write(str(fish_name_number) + ' ' + str(round(rb_actual_width, 2)))

                        except:
                                # 물고기만 있을 때(담뱃갑 x)
                            # fish_size.extend([fish_name_number])
                            with open(txt_path + '.txt', 'w') as f:
                                f.write(str(fish_name_number))

                            if rb_width > rb_height:  # 물고기가 가로로 배치
                                cv2.putText(im0, fish_name,
                                                (rb_x_position_w, rb_y_position_w),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)
                            else:  # 물고기가 세로로 배치
                                cv2.putText(im0, fish_name,
                                                (rb_x_position_h, rb_y_position_h),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 0, 255), 2)



                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            try:
                with open(txt_path + '.txt', 'r') as f:

                    fish_info = f.readlines()[0].split(' ')
                    if len(fish_info) == 1:
                        fish_size.append(int(fish_info[0]))
                    else:
                        fish_info_1 = int(fish_info[0])
                        fish_info_2 = float(fish_info[1])
                        fish_size.extend([fish_info_1, fish_info_2])
                    f.close()

            except:
                pass


            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)



    print(fish_size)
    return fish_size  # 각 어종의 class와 fish_size 출력


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
