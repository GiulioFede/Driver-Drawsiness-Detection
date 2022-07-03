import cv2
from playsound import playsound
from scipy.spatial import distance as dist
import dlib
from threading import Thread
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
# deque provides an O(1) time complexity for append and pop operations as compared to list which provides O(n) time complexity
from collections import deque
from time import sleep



"""""""""""""""""""""""""""""""""""""""""""""""""""""
DRAWSINESS SYSTEM PARAMETERS
"""""""""""""""""""""""""""""""""""""""""""""""""""""
WAKE_UP_BUFFER_LENGHT = 100
FPS_CAMERA = 30
TIME_WINDOW = 60 #sec
TOTAL_AMNT_BLINK_FRAMES = int(0.5*FPS_CAMERA*(1/3))
EAR_INIT_THREASHOLD = 0.25 #starting suggested value, dinamically fine tuned
MIN_NUMBER_OF_FRAMES_FOR_EAR_TUNING = FPS_CAMERA*10
""""""""""""""""""""""""""""""""""""""""""""""""""""""

'''
--------------- REPRODUCE AUDIO ALARM ---------------------------------------------------
'''
isAudioPlaying = False
def play_alarm():
    print('playing alarm...')
    global isAudioPlaying
    if isAudioPlaying == False:
        isAudioPlaying = True
        playsound('./multimedia/audio/alarm.mp3')
        isAudioPlaying = False
    


'''
--------------- COMPUTE EAR INDEX ---------------------------------------------------
'''
def EAR_fun(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear



'''
--------------- DETECTOR AND PREDICTOR ---------------------------------------------------
'''
#retrieve default face detector
detector = dlib.get_frontal_face_detector()
#retrieve default predictor
predictor = dlib.shape_predictor("./others/shape_predictor_68_face_landmarks.dat")


'''
--------------- CORE CLASSES ---------------------------------------------------
'''

#loop and process frames from the video stream
#frame acquisition should be in a dedicated thread
#a thread 1 should be dedicated to RT frames acquisition
#main thread should be dedicated to process frames once available


"""this class manage the Reactive Percl, an empowered index to make drawsiness evaluation more effective and responsive"""
class RPERCLOS(object):
    
    def __init__(self):
        #base for RPERLC
        self.perclos = 0.0 #moving avarage perclos
        self.rperclos = 0.0
        #time windows to record last 3 minutes of frames and relative indexes
        self.percl_buffer = deque([])
        #structure to make WakeUp system
        self.wake_up_buffer = deque([])
        self.moving_average_wake_up_buffer = 0
        #for blinking filtering function
        self.consecutive_closed_frames = 0
        self.swap_amount = 0
        #percl params
        self.number_of_closed_frames = 0
        self.number_of_opened_frames = 0
   
    
    """INTERNAL METHODS"""
    
    """swap methods"""
    
    def get_swap_value(self):
        return self.swap_amount
    
    def swap_inc(self):
        self.swap_amount+=1
    
    def swap_clean(self):#once swapped frame into percl_buffer
        self.swap_amount = 0

    """methods to manage blinking"""
    def blink_count_up(self):#every negative frame found
        self.consecutive_closed_frames += 1
        
    def get_blink_counter_values(self):
        return self.consecutive_closed_frames
    
    def blink_count_reset(self):#every positve frame found
        self.consecutive_closed_frames =0
    
    
    """methods for WakeUp """
    def wakeup_add_frame_and_update(self,last_frame_state,last_frame_timestamp):
        if (last_frame_state==0):
            #bulk update sincrono con il percl update
            if len(self.wake_up_buffer) > WAKE_UP_BUFFER_LENGHT:
                            leaving_timestamp = self.wake_up_buffer.popleft()
                            self.moving_average_wake_up_buffer = (self.moving_average_wake_up_buffer * WAKE_UP_BUFFER_LENGHT-leaving_timestamp+last_frame_timestamp)/WAKE_UP_BUFFER_LENGHT
            #se il wake buffer buffer non è saturo
            else:
                            self.moving_average_wake_up_buffer = (self.moving_average_wake_up_buffer*len(self.wake_up_buffer)+last_frame_timestamp) / (len(self.wake_up_buffer)+1)
            #add last frame timestamp 
            self.wake_up_buffer.append(last_frame_timestamp)
        
    """methods to manage percl buffer"""
    def percl_add_frames_and_update(self,last_frame_state,last_frame_timestamp):#aggiunge i frame al buffer per il calcolo del percl
        #I can add swap element only if frame is 1 or blink_count is over thshold and frame is 0
        #add all frames in a swap
        if ( last_frame_state==1 or (last_frame_state==0 and self.get_blink_counter_values() > TOTAL_AMNT_BLINK_FRAMES ) ):   
            
            for x in range(self.get_swap_value()):
                    
                    if (self.number_of_closed_frames + self.number_of_opened_frames < TIME_WINDOW*FPS_CAMERA):
                                #AGGIUNGO FRAMES: lo swap viene aggiunto integralmente,uno swap contiene frame uniformi, il timestamp dello swap si suppone unico per tutti i frame, la dimensione di uno swap va da 1 a TOTAL_AMNT_BLINK_FRAMES
                                if(last_frame_state == 1): #if is opened
                                    self.number_of_opened_frames +=1
                                else:
                                    self.number_of_closed_frames +=1 
                                    
                    else:
                        
                        leaving_frame_state = self.percl_buffer.popleft()
                        if(leaving_frame_state == 1): #if is opened
                            self.number_of_opened_frames -= 1
                        else: #if is closed
                             self.number_of_closed_frames -= 1
                        
                    
                    #percl update
                    self.percl_buffer.append(last_frame_state)
                    #wakeup buffer update
                    self.wakeup_add_frame_and_update(last_frame_state,last_frame_timestamp)
                    #basic perclos update
                    self.perclos = (self.number_of_closed_frames / (self.number_of_closed_frames + self.number_of_opened_frames) )*100
            #clean swap
            self.swap_clean()
        
    def get_perclos(self):
        return self.perclos
    
        
    def blink_filtering(self,last_frame_state,last_frame_timestamp):
        #se il frame è OPENEYE resetta il counter, altrimenti lo incrementa
        if (last_frame_state ==1):
            #resetta counter
            self.blink_count_reset()
            #svuoto swap
            self.swap_clean()

        else:
            self.blink_count_up()
        #add current eleemnt to swap
        self.swap_inc()

    """PUBLIC METHODS"""
            
    def update_rperclos(self, current_frame_state, current_timestamp):
        
        #blinkfiltering
        self.blink_filtering(current_frame_state,current_timestamp)
        #percl update, wakeupbuffer update
        self.percl_add_frames_and_update(current_frame_state,current_timestamp)
        #rperclose update
        base_perclos = self.get_perclos()
        wakeup_coefficent = self.moving_average_wake_up_buffer/(current_timestamp)
        self.rperclos = base_perclos * wakeup_coefficent
        
    
    def get_rperclos(self):
        return self.rperclos
    
        
    
class DrawsinessEvaluator(object):
    def __init__(self, src=0):
        #if false, a fine tuning of ear index was not carried out
        self.isEARFineTuned = False
        self.sum_of_all_ear_during_tuning = EAR_INIT_THREASHOLD
        self.EAR_THREASHOLD = EAR_INIT_THREASHOLD
        #time windows to record last 3 minutes of frame
        '''RPERCL'''
        self.rp = RPERCLOS()
        self.last_frame_state =0
        ''' CAMERA SETTINGS '''
        self.capture = cv2.VideoCapture(src+ cv2.CAP_DSHOW)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        # capture actual time
        self.start_time_frame = time.time()
        self.total_number_of_frames = 0  

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(.01)
    
    def stop_video_streaming(self):
            print("clean all")
            self.capture.release()

    def activate_alarm_thread(self):
            t = Thread(target=play_alarm,
                args=())
            t.deamon = True
            t.start()
            
    def fix_new_frame(self):
        # fix next fram to be processed and displayed
        #cv2.imshow('frame', self.frame)
        self.last_frame = self.frame
        self.last_frame_timestamp = time.time()-self.start_time_frame
            
    def face_detection(self,detector):
        #img PREPROCESSING
        res_frame = imutils.resize(self.last_frame, width=450)
        #convert to white-gray format to mitigate brightness jitter effect
        self.gray_res_frame = cv2.cvtColor(res_frame, cv2.COLOR_BGR2GRAY)
        #perform face detection and returns faces
        face_frames = detector(self.gray_res_frame, 0)
        #update last_frame to be displayed
        self.last_frame = res_frame
        return face_frames
    
    def landmark_predictor(self,predictor,face):
        #perform landmarks prediction
        self.landmarks_idx = predictor(self.gray_res_frame, face)
        self.landmarks_idx = face_utils.shape_to_np(self.landmarks_idx)
    
    
    def EAR_evaluator(self,EAR_fun):
        #indexes for eye landamarks
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        #extract eye-index ### avoid once custum model is fitted 
        self.leftEye = self.landmarks_idx[lStart:lEnd]
        self.rightEye = self.landmarks_idx[rStart:rEnd]
        leftEAR = EAR_fun(self.leftEye)
        rightEAR = EAR_fun(self.rightEye)

        #ear eval
        self.ear = (leftEAR + rightEAR) / 2.0
        if (self.ear>=self.EAR_THREASHOLD):
            self.last_frame_state = 1
        else:
            self.last_frame_state = 0
        
    
    #Calcolo PERCLOS (occorre ridefinire finestra mobile e soglia dinamica per l'EAR)    
    def RPERCLOS_application(self):
    
         #draw timer
        cv2.putText(self.last_frame, "Time: {:.2f}".format(self.elapsed_time()), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #draw fps
        cv2.putText(self.last_frame, "FPS: {:.2f}".format(self.current_fps()), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        #keep track of total number of frames for fps and tuning
        self.increment_number_of_total_frames()

        #if ear is in fine tuning...
        if self.isEARFineTuned == False:
            self.sum_of_all_ear_during_tuning += self.ear
            self.EAR_THREASHOLD = (self.sum_of_all_ear_during_tuning) / (self.total_number_of_frames+1)
            cv2.putText(self.last_frame, "EAR fine tuning... EAR= {:.2f}".format(0.8*self.EAR_THREASHOLD), (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            #if a minute has passed and a sufficient number of frames have been processed
            if self.total_number_of_frames > MIN_NUMBER_OF_FRAMES_FOR_EAR_TUNING: #ho supposto di calcolare EAR in 20 secondi
                self.isEARFineTuned = True
                self.EAR_THREASHOLD = self.EAR_THREASHOLD*0.8
            else:
                return
        

        """rpercl computation"""#lavora su self.last_frame_state e self.last_frame_timestamp
        self.rp.update_rperclos(self.last_frame_state, self.last_frame_timestamp)
    
        cv2.putText(self.last_frame, "Numero frame occhi aperti= {:.2f}".format(self.rp.number_of_opened_frames), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(self.last_frame, "Numero frame occhi chiusi= {:.2f}".format(self.rp.number_of_closed_frames), (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            
        #!!! adapt perclose computation to consider a moving window of X frames corrisponding to last 3 minutes of video 
        rperclos = 0
        if (self.total_number_of_frames >= TIME_WINDOW*FPS_CAMERA):
            rperclos = self.rp.get_rperclos()

        #draw the computed PERCLOS on the frame
        self.last_frame = cv2.putText(self.last_frame, "PERCLOS: {:.2f}".format(rperclos), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        color = (0, 255, 0)
        if(rperclos <= 3.75):
            color = (0, 255, 0)
        elif (rperclos <=7.5):
            color = (0,200,0)
        elif (rperclos <= 11.25):
            color = (222,255,0)
        elif (rperclos <= 15):
            color = (255,196,0)
        else:
            color = (0,0,255)
            self.activate_alarm_thread()
            
        #draw rectangle of specified colour    
        self.last_frame =cv2.rectangle(self.last_frame, (5,5), (445, 335), color, 4)
    
    #wrap the image with facial landmark
    def edit_final_image(self):
        #load image
        #
        leftEyeHull = cv2.convexHull(self.leftEye)
        rightEyeHull = cv2.convexHull(self.rightEye)
        self.last_frame = cv2.drawContours(self.last_frame, [leftEyeHull], -1, (0, 255, 0), 1)
        self.last_frame = cv2.drawContours(self.last_frame, [rightEyeHull], -1, (0, 255, 0), 1)
        # draw the computed eye aspect ratio on the frame
        self.last_frame = cv2.putText(self.last_frame, "EAR: {:.2f}".format(self.ear), (300, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return self.last_frame

    ''' FRAME Utilities functions'''
    def increment_number_of_total_frames(self):
        self.total_number_of_frames += 1
    
    def elapsed_time(self):
        return (time.time() - self.start_time_frame)
    
    def current_fps(self):
        return self.total_number_of_frames / self.elapsed_time()


'''
--------------- MAIN ---------------------------------------------------
'''

drawsiness_videodetector = DrawsinessEvaluator()



while True:
        try:
            
            #retrieve last frame from second thread
            drawsiness_videodetector.fix_new_frame()

            #press 'q' to stop video streaming (close all)  NB: tieni premuto q, non basta cliccare una volta
            if cv2.waitKey(1) & 0xFF == ord('q'):
                drawsiness_videodetector.stop_video_streaming()
                cv2.destroyAllWindows()
                break

            #extract faces from last webcam frame
            face_frames = drawsiness_videodetector.face_detection(detector)
            
            #landmark prediction and texture
            for face in face_frames: #this 'for' is here because multiple frames could be detected, in this case we have to consider just the driver one
                
                #landmark prediction
                drawsiness_videodetector.landmark_predictor(predictor,face)
                
                #EAR eval
                drawsiness_videodetector.EAR_evaluator(EAR_fun)
                
                #PERCL ESTIMATION AND ALERT LEVEL DETECTION
                drawsiness_videodetector.RPERCLOS_application()
                
                #composing final image
                final_frame = drawsiness_videodetector.edit_final_image()
            
               
                """ plot img in a videostream fashion: PROBLEMA: SE SI PROVA AD INTERROMPERE IL SERVIZIO CRASHA LA RUNTIME DI PYTHON"""
                cv2.imshow("Frame", final_frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break
                
                # show the frame
                """SE USIAMO PLT NON CRASHA ALL'ARRESTO DEL PROCESSO LA RUNTIME DI PYTHON
                plt.imshow(final_frame)
                plt.show()
                """

        except AttributeError:
            pass
