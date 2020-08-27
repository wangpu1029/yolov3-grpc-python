from ctypes import *
from concurrent import futures
from PIL import Image
import threading
import pickle
import grpc
import testrpc_pb2 as pb2
import testrpc_pb2_grpc as pb2_grpc
import math
import random
import cv2
import time
import queue as lib_queue

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

draw_detections = lib.draw_detections
draw_detections.argtypes = [IMAGE, POINTER(DETECTION), c_int, c_float, POINTER(c_char_p), POINTER(POINTER(IMAGE)), c_int ]

load_alphabet = lib.load_alphabet
load_alphabet.restype = POINTER(POINTER(IMAGE))

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int ] 

work_lock = threading.Lock()

class Buffer(object):
    """A bounded buffer which prefers newer items. We assume only one producer and one consumer."""

    def __init__(self, maxsize):
        self.buf = lib_queue.Queue(maxsize=maxsize)
        self.maxsize = maxsize

    def get(self, *args, **kwargs):
        return self.buf.get(*args, **kwargs)

    def put(self, item):
        while True:    #untill successfully put
            try:
                self.buf.put(item, timeout=0.02)
                break
            except lib_queue.Full:            
                dropped = self.buf.get()
                #print('Dropped', dropped.msg)
                
    def qsize(self):
        return self.buf.qsize()
        
class Listener(threading.Thread):

    def __init__(self, request_iter, buffer, remote_off):
        super(Listener, self).__init__()
        self.request_iter = request_iter
        self.buffer = buffer
        self.remote_off = remote_off

    def run(self):
        for r in self.request_iter:
            self.buffer.put(r)
            # print("got one")
        self.remote_off.put(True)
        print("stop listening")

class Worker(threading.Thread):
    
    def __init__(self, buffer, results, remote_off, local_off):
        super(Worker, self).__init__()
        self.buffer = buffer
        self.results = results
        self.remote_off = remote_off
        self.local_off = local_off
        
    def run(self):
        while True:
            try:
                start = time.time()
                r = self.buffer.get(timeout=0.02)
                req = pickle.loads(r.msg)
                img = cv2.imdecode(req[0], cv2.IMREAD_COLOR)
                imgpath = '/home/zzz/ramdisk/' + str(req[1]) + '.jpg'
                cv2.imwrite(imgpath,img)
                res = detect(net, meta, imgpath.encode('ascii'))
                self.results.put(res)
                end = time.time()
                # print('latency is %.3f s' % (end-start))
            except lib_queue.Empty:            
                try:
                    self.remote_off.get(block=False)
                    self.local_off.put(True)
                    break
                except lib_queue.Empty:
                    pass
        print("work done!!!!!!!")


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    global work_lock
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    work_lock.acquire()
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    work_lock.release()
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((bytes.decode(meta.names[i]), dets[j].prob[i], (b.x, b.y, b.w, b.h), i))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

class TestRPCServicer(pb2_grpc.TestRPCServicer):

    def __init__(self):
        pass

    def OneToOne(self, request, context):
        return

    def StreamToOne(self, request_iter, context):
        return

    def StreamToStream(self, request_iter, context):
        buffer = Buffer(3)
        results = lib_queue.Queue()
        remote_off = lib_queue.Queue()
        local_off = lib_queue.Queue()
        threads = []
        threads.append(Listener(request_iter, buffer, remote_off))
        threads.append(Worker(buffer, results, remote_off, local_off))
        for t in threads:
            t.start()

        while True:
            try:
                msg = results.get(timeout=0.02)
                yield pb2.Reply(msg=pickle.dumps(msg))
            except lib_queue.Empty:
                if local_off.qsize():
                    break
        for t in threads:
            print("a thread is over!!!")
            t.join()
           
        print("Smoothly stopped.")
    
if __name__ == "__main__":
    
    net = load_net("cfg/yolov3.cfg".encode('ascii'), "./yolov3.weights".encode('ascii'), 0)
    meta = load_meta("cfg/coco.data".encode('ascii'))

    try:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        pb2_grpc.add_TestRPCServicer_to_server(TestRPCServicer(), server)
        server.add_insecure_port('[::]:50051')
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass 
