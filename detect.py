import math
import os
import sys
import threading
import time
from pathlib import Path
import numpy as np
import torch
import tkinter

from utils_gen.augmentations import letterbox
from models.common import DetectMultiBackend
from utils_gen.general import (non_max_suppression, scale_boxes, xyxy2xywh)
from utils_gen.plots import Annotator
from utils_gen.torch_utils import smart_inference_mode
from ScreenShot import screenshot
from SendInput import *
import pynput
from pynput.mouse import Listener

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT1 = ROOT
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


is_right_pressed = False
result = False


def is_button():
    def is_open():
        global result
        result = not result
        if result:
            label.config(text='开启成功')
        else:
            label.config(text='关闭成功')

    window = tkinter.Tk()
    window.geometry('100x100')
    label = tkinter.Label(window)
    button = tkinter.Button(window, text="狙击自动开枪开关", command=is_open, width=15, height=2)
    label.pack()
    button.pack()
    window.mainloop()


def mouse_click(x, y, button, pressed):
    global is_right_pressed
    # print(x, y, button, pressed)
    if pressed and button == pynput.mouse.Button.right:
        is_right_pressed = True
    elif not pressed and button == pynput.mouse.Button.right:
        is_right_pressed = False


def mouse_listener():
    with Listener(on_click=mouse_click) as listener:
        listener.join()


@smart_inference_mode()
def run():
    # Load model
    global is_right_pressed
    global result
    device = torch.device('cuda:0')
    model = DetectMultiBackend(weights=os.path.join(ROOT1, 'weights/1匪0警.pt'), device=device, dnn=False,
                               data=False, fp16=True)

    while True:
        if is_right_pressed:
            # 读取图片
            im = screenshot()
            im0 = im
            # 处理图片
            im = letterbox(im, (640, 640), stride=32, auto=True)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # 推理
            for i in range(1):
                # start = time.time()
                pred = model(im, augment=False, visualize=False)

                # NMS
                pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45, classes=[0, 1], max_det=1000)
                # end = time.time()
                # print(f'推理所需时间{end - start}s')

            # Process predictions
            for i, det in enumerate(pred):  # per image
                annotator = Annotator(im0, line_width=1)
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    xywh = []
                    distance_list = []
                    target_list = []
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):  # 处理推理出来每个目标的信息
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                        annotator.box_label(xyxy, label=str(int(cls)), color=(34, 139, 34), txt_color=(0, 191, 255))
                        # print(xyxy, line)
                        xywh[1] = xywh[1] - xywh[3] / 6
                        X = xywh[0] - 200
                        Y = xywh[1] - 200
                        distance = math.sqrt(X ** 2 + Y ** 2)
                        distance_list.append(distance)
                        target_list.append(xywh)
                    target_info = target_list[distance_list.index(min(distance_list))]
                    mouse_xy(int(target_info[0] - 200), int(target_info[1] - 200))
                    # 狙击枪自动开枪
                    if abs(target_info[0] - 200) <= xywh[2] / 5 and xywh[3] / 3 >= abs(target_info[1] - 200) and result:
                        ctr = pynput.mouse.Controller()
                        ctr.click(pynput.mouse.Button.left)
                        time.sleep(0.05)
                        ctr1 = pynput.keyboard.Controller()
                        ctr1.press("q")
                        ctr1.release("q")
                        time.sleep(0.1)
                        ctr1.press("q")
                        ctr1.release("q")
                        time.sleep(0.9)

                # im0 = annotator.result()
                # cv2.imshow("window", im0)
                # cv2.waitKey(1)

        else:
            time.sleep(0.00001)


if __name__ == "__main__":
    threading.Thread(target=mouse_listener).start()
    threading.Thread(target=is_button).start()
    run()
