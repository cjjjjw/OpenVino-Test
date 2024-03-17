"""
brief :Editor cjh,ljl
"""

import time
import openvino.runtime as ov
import cv2
import numpy as np
import openvino.preprocess as op
from openvino import *
from collections import deque
import logging
import multiprocessing
from numba import jit


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def rgb_call(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def rgb2hex(rgb):
        '''RGB转HEX
        :param rgb: RGB颜色元组，Tuple[int, int, int]
        :return: int or str
        '''
        r, g, b = rgb
        result = r + (g << 8) + (b << 16)
        return hex(result)[2:]


def cap_capture(stack, source, top) -> None:
    """
    :param source: 视频流参数
    :param stack: Manager.list对象,作为第一通道的视频流
    :param top: 缓冲栈容量
    :return: None
    """
    global cap
    if len(source) <= 2:
        try:
            cap = cv2.VideoCapture(int(source))
        except Exception:
            pass
    else:
        cap = cv2.VideoCapture(source)
    while True:
        ret, Frame = cap.read()
        if ret:
            stack.append(Frame)
            if len(stack) >= top:
                del stack[:]

@jit(nopython=True)
def accle_for(detections:np.ndarray,confidence:float):
    detections_out=[]
    for prediction in detections:
        _confidence_ = prediction[4].item()  # 获取置信度
        # 第一次进行过滤，过滤掉大部分没用信息，但是设定阈值不能太高
        if _confidence_ >= confidence / 2:
            detections_out.append(prediction)
    return detections_out

class Vino(Colors):
    def __init__(self, model_path="/home/nuc2/PycharmProjects/yolov5-master/best_ball.xml"
                 , weights_path="/home/nuc2/PycharmProjects/yolov5-master/best_ball.bin"
                 , conf_thres=0.55
                 , line_thickness=3
                 , iou_thres=0.55
                 , device="GPU"
                 , name_porcess="Vino"):
        super(Vino, self).__init__()
        self.confidence = conf_thres
        self.line_thickness = line_thickness
        self.device = device
        self.iou_thres = iou_thres
        # 在实例化的时候就开辟空间给内核激活
        self.logger = logging.getLogger(name_porcess)
        self.logger.info(f"{model_path, weights_path, conf_thres, line_thickness, iou_thres, device}")
        self.inter = self.Core(model_path=model_path, weights_path=weights_path)
        # 设置多个进程进行并行处理数据

        # 修改图像的亮度，brightness取值0～2 <1表示变暗 >1表示变亮

    def change_brightness(self, img, brightness):
        [averB, averG, averR] = np.array(cv2.mean(img))[:-1] / 3
        k = np.ones(img.shape)
        k[:, :, 0] *= averB
        k[:, :, 1] *= averG
        k[:, :, 2] *= averR
        img = img + (brightness - 1) * k
        img[img > 255] = 255
        img[img < 0] = 0
        return img.astype(np.uint8)

    def letter_box(self, box_, img, shape_re, dw, dh, confidence_box, class_id) -> (np.ndarray, np.uint8):
        color = self.rgb_call(class_id, True)
        rx = img.shape[1] / (shape_re[1] - dw)
        ry = img.shape[0] / (shape_re[0] - dh)
        # box_ 代表着xyxy
        box_[0] = int(rx * box_[0])
        box_[1] = int(box_[1] * ry)
        box_[2] = int(rx * box_[2]) + box_[0]
        box_[3] = int(box_[3] * ry) + box_[1]
        cv2.rectangle(img, (int(box_[0]), int(box_[1])), (int(box_[2]), int(box_[3])), color,
                      self.line_thickness)  # 绘制物体框
        cv2.putText(img, f"confidence:{str(int(confidence_box * 100))}%,id:{class_id}",
                    (int(box_[0]), int(box_[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)
        size_decrease = (int(img.shape[1] / 1), int(img.shape[0] / 1))
        img_decrease = cv2.resize(img, size_decrease, interpolation=cv2.INTER_CUBIC)
        return box_, img_decrease.astype(np.uint8)

    def Core(self, model_path: str, weights_path: str) -> InferRequest:
        try:
            core = ov.Core()
            #  读取用YOLOv5模型转换而来的IR模型
            model = core.read_model(model_path, weights_path)
            ppp = op.PrePostProcessor(model)
            ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")).set_color_format(
                op.ColorFormat.BGR)
            ppp.input().preprocess().convert_element_type(ov.Type.f32).convert_color(op.ColorFormat.RGB).scale(
                [255., 255., 255.])
            ppp.input().model().set_layout(ov.Layout("NCHW"))
            ppp.output(0).tensor().set_element_type(ov.Type.f32)
            _model = ppp.build()
            # 加载模型，可用CPU or GPU
            compilemodel = core.compile_model(model, self.device)
            # 推理结果
            results = compilemodel.create_infer_request()
            self.logger.info("Success to Start")
            return results
        except:
            self.logger.info("Failed to Start")

    def resized(self, image, new_shape):
        old_size = image.shape[:2]
        # 记录新形状和原生图像矩形形状的比率
        ratio = float(new_shape[-1] / max(old_size))
        # 新尺寸
        new_size = tuple([int(x * ratio) for x in old_size])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        dw = new_shape[1] - new_size[1]
        dh = new_shape[0] - new_size[0]
        color = [100, 100, 100]
        new_im = cv2.copyMakeBorder(image, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)
        return new_im, dw, dh

    def run(self, img) -> (np.uint8, deque, deque):
        # 尺寸处理,变成一个贴合矩形
        img_re, dw, dh = self.resized(img, (640, 640))
        shape_re = img_re.shape
        # 获得输入张量
        input_tensor = np.expand_dims(img_re, 0)
        # 输入到推理引擎
        self.inter.infer({0: input_tensor})
        # # 获得推理结果
        output = self.inter.get_output_tensor(0)
        # # 获得检测数据
        detections = output.data[0]
        # # 使用deque，不使用list
        boxes = deque()
        class_ids = deque()
        confidences_deque = deque()
        # 以下是将输出的队列
        det_out = deque()
        detections=accle_for(detections,self.confidence)
        for prediction in detections:
            _confidence_ = prediction[4].item()  # 获取置信度
            # 第一次进行过滤，过滤掉大部分没用信息，但是设定阈值不能太高
            # if _confidence_ >= self.confidence / 2:
            classes_scores = prediction[5:]
            t, w, e, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if classes_scores[class_id] > .25:
                confidences_deque.append(_confidence_)
                class_ids.append(class_id)
                # 得到有用信息
                x, y, w, h = prediction[0].item(), prediction[1].item(), prediction[2].item(), prediction[3].item()
                xmin = x - (w / 2)
                ymin = y - (h / 2)
                box = np.array([xmin, ymin, w, h])
                boxes.append(box)
        # 第二次进行筛选
        indexes = cv2.dnn.NMSBoxes(boxes, confidences_deque, 0.5, self.iou_thres)
        detections = deque()
        for i in indexes:
            j = i.item()
            detections.append({"ci": class_ids[j], "cf": confidences_deque[j], "bx": boxes[j]})
        for detection in detections:
            box = detection["bx"]
            classId = detection["ci"]
            confidence = detection["cf"]
            if confidence < self.confidence:
                break
            box_, img = self.letter_box(box, img, shape_re, dw, dh, confidence, classId)
            # 返回值存在4个,分别是box,置信度,目标物的id,和代表色
            det_out.append((box_, confidence, classId))
        return img.astype(np.uint8), det_out


if __name__ == '__main__':
    model_path, weights_path = "/home/nuc2/PycharmProjects/yolov5-master/weights/best_bucket.xml", "/home/nuc2/PycharmProjects/yolov5-master/weights/best_bucket.bin"
    confidence = 0.7
    vino1 = Vino(model_path, weights_path, confidence)
    # cap = cv2.VideoCapture(0)
    Cap_data = multiprocessing.Manager().list()
    p1 = multiprocessing.Process(target=cap_capture,
                                 args=(Cap_data, "/home/nuc2/PycharmProjects/yolov5-master/source_video/6.mp4", 50))
    p1.start()
    while True:
        if len(Cap_data) != 0:
            start = time.time()
            # ret,frame=cap.read()
            frame = Cap_data.pop()
            frame, det = vino1.run(frame)
            for xyxy, conf, cls in reversed(det):
                # print(vino1.rgb2hex(vino1.rgb_call(cls)))
                mid_pos = [int((xyxy[0] + xyxy[2]) / 2),
                           int((xyxy[1] + xyxy[3]) / 2)]
                cv2.circle(frame, mid_pos, 2, (0, 255, 0), 2)
                # print(xyxy, conf, cls)
            end = time.time()
            fps = 1 / (end - start)
            cv2.putText(frame, f"{round(fps, 1)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break




