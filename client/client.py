from ctypes import *
import pickle
import grpc
import cv2
import testrpc_pb2 as pb2
import testrpc_pb2_grpc as pb2_grpc
import numpy as np
import time
import threading
import copy
import sys

np.random.seed(40)

detres = []
frame = None
detres_lock = threading.Lock()
detres_flag = False
frame_lock = threading.Lock()
frame_flag = False

class ShowPicResult(threading.Thread):
    
    def __init__(self, x_scale, y_scale, cap):
        super(ShowPicResult, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.cap = cap
        self.local_detres = []
        self.local_frame = None
        self.COLORS = np.random.randint(0, 255, size=(80, 3), dtype="uint8")
    def run(self):
        global detres_flag
        global detres_lock
        global frame_flag
        global frame_lock
        while not frame_flag:
            pass
        while 1:
            if detres_flag:
                detres_lock.acquire()
                self.local_detres = copy.deepcopy(detres)
                detres_flag = False
                detres_lock.release()

            if frame_flag:
                frame_lock.acquire()
                self.local_frame = copy.deepcopy(frame)
                frame_flag = False
                frame_lock.release()
            
            for i in range(len(self.local_detres)):
                x = self.local_detres[i][2][0]*self.x_scale
                y = self.local_detres[i][2][1]*self.y_scale
                w = self.local_detres[i][2][2]*self.x_scale
                h = self.local_detres[i][2][3]*self.y_scale
                classID = self.local_detres[i][3]

                x1= x - w/2
                y1= y - h/2
                x2= x + w/2
                y2= y + h/2
                color = [int(c) for c in self.COLORS[classID]]

                cv2.rectangle(self.local_frame,(int(x1),int(y1)),(int(x2),int(y2)),color,3)
                cv2.putText(self.local_frame, self.local_detres[i][0], (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                #This is a method that works well. 

            cv2.imshow('yolo_image_detector', self.local_frame)
            time.sleep(0.05)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                frame_lock.acquire()
                self.cap.release()
                cv2.destroyAllWindows()
                frame_lock.release()
                print("user have pressed q in thread")
                break
            if not self.cap.isOpened():
                cv2.destroyAllWindows()
                print("thread know video is over")
                break

def frameget(cap):
    
    global frame
    global frame_flag
    global frame_lock
    sent_frame = 0
    while(cap.isOpened()):
        frame_lock.acquire()
        ret, frame = cap.read()
        frame_flag = True
        frame_lock.release()
        if ret == True:
            output_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)
            is_success, buf = cv2.imencode('.jpg', output_frame)
            if is_success == True:
                frame_pickled = pickle.dumps((buf,1))
                sent_frame += 1
                yield pb2.Request(msg=frame_pickled)
        else:
            cap.release()
            print("Video is over")
            print("We have sent %d frames" % (sent_frame))
            break

def make_OneToStream_call(stub):
    return

def make_StreamToOne_call(stub):
    return
    
def make_StreamToStream_call(stub, arg):
    global detres
    global detres_flag
    global detres_lock
    total_time = 0
    detected_frame = 0
    start = time.time()
    if arg == 0:
        cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, \
        height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, \
        format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    else:
        cap = cv2.VideoCapture("traffic.mp4")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    x_scale = width/640
    y_scale = height/360

    try:
        threads = []
        threads.append(ShowPicResult(x_scale, y_scale, cap))
        for t in threads:
            t.start()
    except:
        print("Error: unbale to start thread")

    for reply in stub.StreamToStream(frameget(cap)):
        detres_lock.acquire()
        detres = pickle.loads(reply.msg)
        detres_flag = True
        detres_lock.release()
        detected_frame += 1
    print("grpc is done!")
    for t in threads:
        print("thread is over!")
        t.join()
    end = time.time()
    total_time = end - start
    print('The number of detected frame is %d' % (detected_frame))
    print('latency is %.3f s' % (total_time/detected_frame))
    print('fps is %.3f /s' % (1/(total_time/detected_frame)))

if __name__ == '__main__':

    arg = int(sys.argv[1])
    with grpc.insecure_channel('192.168.1.214:50051') as channel:
        stub = pb2_grpc.TestRPCStub(channel)
        make_StreamToStream_call(stub, arg)