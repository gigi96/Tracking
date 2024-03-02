import PySimpleGUI as sg
import cv2
import numpy as np
from enum import Enum

window_title = 'OpenCV Tracking Demo'
window_location = (0, 0)
window_size = (1920, 1080)
video_size = (640, 360)
video_path = '../resources/videos/soccer-ball.mp4'

classes_file = "../models/coco.names"
model_configuration = "../models/yolov3.cfg"
model_weights = "../models/yolov3.weights"
# Initialize the parameters
objectnessThreshold = 0.5 # Objectness threshold
confThreshold = 0.5       # Confidence threshold
nmsThreshold = 0.4        # Non-maximum suppression threshold
inpWidth = 416            # Width of network's input image
inpHeight = 416           # Height of network's input image

class_to_detect = "sports ball"

start_point = None
end_point = None
detection_rect = None
tracking_rect = None

class PipelineStatus(Enum):
    DETECTION = 1
    INIT_TRACKING = 2
    TRACKING = 3

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
    
# Draw the predicted bounding box
def drawPred(frame, classes, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
    
def postprocess(frame, outs, classes):
    global start_point, end_point

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            if detection[4] > objectnessThreshold :
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # select only the class of interest
        if classes[classIds[i]] == class_to_detect:
            drawPred(frame, classes, classIds[i], confidences[i], left, top, left + width, top + height)
            start_point = (left, top)
            end_point = (left + width, top + height)  
            
def tracker_create(algo):
    if (algo == "boosting"):
        return cv2.legacy.TrackerBoosting_create();
    elif (algo == "mil"):
        return cv2.legacy.TrackerMIL_create()
    elif (algo == "kcf"):
        return cv2.legacy.TrackerKCF_create()
    elif (algo == "tld"):
        return cv2.legacy.TrackerTLD_create()
    elif (algo == "medianflow"):
        return cv2.legacy.TrackerMedianFlow_create()
    elif (algo == "goturn"):
        return cv2.legacy.TrackerGOTURN_create()
    elif (algo == "csrt"):
        return cv2.legacy.TrackerCSRT_create()
    elif (algo == "mosse"):
        return cv2.legacy.TrackerMOSSE_create()

def main():
    global start_point, end_point, detection_rect, tracking_rect

    # define the window layout
    layout =   [
      [ sg.Text(expand_x=True),
        sg.Frame('Detection', 
         [[sg.Graph(
                canvas_size=video_size,
                graph_bottom_left=(0, video_size[1]),
                graph_top_right=(video_size[0], 0),
                key="-DETECTION_GRAPH-",
                enable_events=True,
                background_color='black',
                motion_events=True,
                drag_submits=True,
                )]],
        key='-DETECTION_FRAME-'),
        sg.Frame('Tracking',
         [[ sg.Graph(
                canvas_size=video_size,
                graph_bottom_left=(0, video_size[1]),
                graph_top_right=(video_size[0], 0),
                key="-TRACKING_GRAPH-",
                enable_events=False,
                background_color='black',
                )]],
        key='-TRACKING_FRAME-'),
       sg.Text(expand_x=True)],
       [sg.Button('Play', size=(10, 1), key='-PLAY_PAUSE_BUTTON-'),
        sg.Button('Exit', size=(10, 1))],
      [sg.Radio('Boosting', 'Radio', size=(10, 1), key='-TRK_BOOSTING-', default=True)],
      [sg.Radio('MIL', 'Radio', size=(10, 1), key='-TRK_MIL-')],
      [sg.Radio('KCF', 'Radio', size=(10, 1), key='-TRK_KCF-')],
      [sg.Radio('TLD', 'Radio', size=(10, 1), key='-TRK_TLD-')],
      [sg.Radio('Median FLow', 'Radio', size=(10, 1), key='-TRK_MEDIANFLOW-'),
      [sg.Radio('GOTURN', 'Radio', size=(10, 1), key='-TRK_GOTURN-')],
      [sg.Radio('CSRT', 'Radio', size=(10, 1), key='-TRK_CSRT-')],
      [sg.Radio('MOSSE', 'Radio', size=(10, 1), key='-TRK_MOSSE-')]]         
    ]

    # create the window and show it without the plot
    window = sg.Window(window_title, layout, location=window_location, size=window_size, finalize=True)
    
    classes = None
    with open(classes_file, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)

    cap = cv2.VideoCapture(video_path)
    if  cap.isOpened:
        ret, frame = cap.read()
        last_frame = frame   
    
    sg.Image(filename='', size=video_size, enable_events=True, key='-IMAGE_DETECTION-')
    detection_area = window['-DETECTION_GRAPH-']
    resized = cv2.resize(last_frame, video_size, fx=1, fy=1, interpolation = cv2.INTER_LINEAR)
    detection_img0 = cv2.imencode('.png', resized)[1].tobytes()    
    detection_img = detection_area.draw_image(data=detection_img0, location=(0, 0))
    
    sg.Image(filename='', size=video_size, enable_events=True, key='-IMAGE_TRACKING-')
    tracking_area = window['-TRACKING_GRAPH-']
    resized = cv2.resize(last_frame, video_size, fx=1, fy=1, interpolation = cv2.INTER_LINEAR)
    tracking_img0 = cv2.imencode('.png', resized)[1].tobytes()    
    tracking_img = tracking_area.draw_image(data=tracking_img0, location=(0, 0))    
    
    tracking_bbox = None
    dragging = False
    is_tracking = False
    
    # default tracking algorithm
    tracking_algo = "boosting"

    status = PipelineStatus.DETECTION
    is_playing = False
    while True:
        event, values = window.read(10)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        
        if event == '-PLAY_PAUSE_BUTTON-':
            is_playing = not is_playing
            last_frame = frame
            
        if  cap.isOpened and is_playing: 
            ret, frame = cap.read()
            is_playing = ret
        else:
            is_playing = False
            
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(video_path)
            if  cap.isOpened:
                ret, frame = cap.read()
                last_frame = frame   

        resized = cv2.resize(frame if is_playing else last_frame, video_size, fx=1, fy=1, interpolation = cv2.INTER_LINEAR)
        resized_trk = resized.copy()
        
        if is_playing:
            match status:
                case PipelineStatus.DETECTION:
                    blob = cv2.dnn.blobFromImage(resized, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
                    net.setInput(blob)
                    outs = net.forward(getOutputsNames(net))
                    postprocess(resized, outs, classes)                
                    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
                    t, _ = net.getPerfProfile()
                    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
                    cv2.putText(resized, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))    
                    status = PipelineStatus.INIT_TRACKING
                    
                case PipelineStatus.INIT_TRACKING:
                    tracker = tracker_create(tracking_algo)
                    tracking_rect = (start_point, end_point)            
                    tracking_bbox = (tracking_rect[0][0], tracking_rect[0][1],
                    tracking_rect[1][0] - tracking_rect[0][0], tracking_rect[1][1] - tracking_rect[0][1])
                    tracker_is_init = tracker.init(frame, tracking_bbox)
                    status = PipelineStatus.TRACKING
                    
                case PipelineStatus.TRACKING:
                    is_tracking, tracking_bbox = tracker.update(resized_trk)
                    if is_tracking:
                        p1 = (int(tracking_bbox[0]), int(tracking_bbox[1]))
                        p2 = (int(tracking_bbox[0] + tracking_bbox[2]), int(tracking_bbox[1] + tracking_bbox[3]))
                        tracking_rect = (p1, p2)
                        tracking_rect = tracking_area.draw_rectangle(tracking_rect[0], tracking_rect[1], line_color='green')
                        cv2.putText(resized_trk, "Tracking: " + str(is_tracking), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2); 
                        #cv2.putText(frame, "Tracking Area: " + str(p1) + " - " + str(p2), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2);                             
                    else:
                        status = PipelineStatus.DETECTION
        else:
            status = PipelineStatus.DETECTION
            
            if values['-TRK_BOOSTING-']:
                tracking_algo = "boosting"
            elif values['-TRK_MIL-']:
                tracking_algo = "mil" 
            elif values['-TRK_KCF-']:
                tracking_algo = "kcf" 
            elif values['-TRK_TLD-']:
                tracking_algo = "tld"
            elif values['-TRK_MEDIANFLOW-']:
                tracking_algo = "medianflow"
            elif values['-TRK_GOTURN-']:
                tracking_algo = "goturn"
            elif values['-TRK_CSRT-']:
                tracking_algo = "csrt"
            elif values['-TRK_MOSSE-']:
                tracking_algo = "mosse"
        
        if (detection_img != None):
            detection_area.delete_figure(detection_img) # delete previous image
        if (tracking_img != None):
            tracking_img = tracking_area.delete_figure(tracking_img) # delete previous image
        
        imgbytes = cv2.imencode('.ppm', resized)[1].tobytes()        
        detection_img = detection_area.draw_image(data=imgbytes, location=(0, 0))        
        imgbytes_trk = cv2.imencode('.ppm', resized_trk)[1].tobytes() 
        tracking_img = tracking_area.draw_image(data=imgbytes_trk, location=(0, 0))
        if detection_rect != None:
            detection_area.bring_figure_to_front(detection_rect)
        if tracking_rect != None:
            tracking_area.bring_figure_to_front(tracking_rect) 

        window['-PLAY_PAUSE_BUTTON-'].update(text='Pause' if is_playing else 'Play')

    cap.release()
    window.close()

main()