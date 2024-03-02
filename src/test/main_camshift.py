import PySimpleGUI as sg
import cv2
import numpy as np

window_title = 'OpenCV Tracking Demo'
window_location = (0, 0)
window_size = (1920, 1080)
video_size = (640, 360)
video_path = '../resources/videos/soccer-ball.mp4'

def main():
    #sg.theme(sg.DEFAULT_BACKGROUND_COLOR)

    # define the window layout
    layout = [
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
      [sg.Radio('None', 'Radio', True, size=(10, 1))],
      [sg.Radio('threshold', 'Radio', size=(10, 1), key='-THRESH-'),
       sg.Slider((0, 255), 128, 1, orientation='h', size=(40, 15), key='-THRESH SLIDER-')],
      [sg.Radio('canny', 'Radio', size=(10, 1), key='-CANNY-'),
       sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='-CANNY SLIDER A-'),
       sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='-CANNY SLIDER B-')],
      [sg.Radio('blur', 'Radio', size=(10, 1), key='-BLUR-'),
       sg.Slider((1, 11), 1, 1, orientation='h', size=(40, 15), key='-BLUR SLIDER-')],
      [sg.Radio('hue', 'Radio', size=(10, 1), key='-HUE-'),
       sg.Slider((0, 225), 0, 1, orientation='h', size=(40, 15), key='-HUE SLIDER-')],
      [sg.Radio('enhance', 'Radio', size=(10, 1), key='-ENHANCE-'),
       sg.Slider((1, 255), 128, 1, orientation='h', size=(40, 15), key='-ENHANCE SLIDER-')]     
    ]

    # create the window and show it without the plot
    window = sg.Window(window_title, layout, location=window_location, size=window_size, finalize=True)

    cap = cv2.VideoCapture(video_path) #cv2.VideoCapture(0)
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
    
    start_point = end_point = detection_rect = tracking_rect = None
    tracking_bbox = None
    dragging = False
    is_tracking = False
    currWindow = None
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    is_playing = False
    while True:
        event, values = window.read(timeout=15)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        
        if event == '-PLAY_PAUSE_BUTTON-':
            is_playing = not is_playing
            last_frame = frame
            
        if  cap.isOpened and is_playing: 
            ret, frame = cap.read()
            is_playing = ret            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backProjectImage = cv2.calcBackProject([hsv], [0], histObject, [0,180], 1)
            rotatedWindow, currWindow = cv2.CamShift(backProjectImage, currWindow, term_crit)
            x,y,w,h = currWindow
            tracking_rect = tracking_area.draw_rectangle((x,y), (x+w,y+h), line_color='green')
        else:
            is_playing = False
            
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(video_path)
            if  cap.isOpened:
                ret, frame = cap.read()
                last_frame = frame             
        
        if event == "-DETECTION_GRAPH-":  # if there's a "Graph" event, then it's a mouse
            x, y = values["-DETECTION_GRAPH-"]
            if not dragging:
                start_point = end_point = (x, y)
                dragging = True
            else:
                end_point = (x, y)
            if detection_rect:
                detection_area.delete_figure(detection_rect)
            if None not in (start_point, end_point):
                detection_rect = detection_area.draw_rectangle(start_point, end_point, line_color='blue')
        elif event.endswith('+UP'):
            tracking_rect = (start_point, end_point)            
            tracking_bbox = (tracking_rect[0][0], tracking_rect[0][1],
                             tracking_rect[1][0] - tracking_rect[0][0], tracking_rect[1][1] - tracking_rect[0][1])
            roiObject = last_frame[tracking_bbox[1]:tracking_bbox[1]+tracking_bbox[3], tracking_bbox[0]:tracking_bbox[0]+tracking_bbox[2]]
            hsvObject =  cv2.cvtColor(roiObject, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsvObject, np.array((0., 50., 50.)), np.array((180.,255.,255.)))
            histObject = cv2.calcHist([hsvObject], [0], mask, [180], [0,180])           
            cv2.normalize(histObject, histObject, 0, 255, cv2.NORM_MINMAX);
            currWindow = tracking_bbox
            
            dragging = False
            start_point = end_point = None            

        # if values['-THRESH-']:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
            # frame = cv2.threshold(frame, values['-THRESH SLIDER-'], 255, cv2.THRESH_BINARY)[1]
        # elif values['-CANNY-']:
            # frame = cv2.Canny(frame, values['-CANNY SLIDER A-'], values['-CANNY SLIDER B-'])
        # elif values['-BLUR-']:
            # frame = cv2.GaussianBlur(frame, (21, 21), values['-BLUR SLIDER-'])
        # elif values['-HUE-']:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # frame[:, :, 0] += int(values['-HUE SLIDER-'])
            # frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        # elif values['-ENHANCE-']:
            # enh_val = values['-ENHANCE SLIDER-'] / 40
            # clahe = cv2.createCLAHE(clipLimit=enh_val, tileGridSize=(8, 8))
            # lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            # lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            # frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        cv2.putText(frame, "Tracking: " + str(is_tracking), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2);        

        if (detection_img != None):
            detection_area.delete_figure(detection_img) # delete previous image
        if (tracking_img != None):
            tracking_img = tracking_area.delete_figure(tracking_img) # delete previous image

        resized = cv2.resize(frame if is_playing else last_frame, video_size, fx=1, fy=1, interpolation = cv2.INTER_LINEAR)
        imgbytes = cv2.imencode('.ppm', resized)[1].tobytes()        
        detection_img = detection_area.draw_image(data=imgbytes, location=(0, 0))
        tracking_img = tracking_area.draw_image(data=imgbytes, location=(0, 0))
        if detection_rect != None:
            detection_area.bring_figure_to_front(detection_rect)
        if tracking_rect != None:
            tracking_area.bring_figure_to_front(tracking_rect) 
        
        #window['-IMAGE_DETECTION-'].update(data=imgbytes)  
        #window['-IMAGE_TRACKING-'].update(data=imgbytes)
        window['-PLAY_PAUSE_BUTTON-'].update(text='Pause' if is_playing else 'Play')

    cap.release()
    window.close()

main()