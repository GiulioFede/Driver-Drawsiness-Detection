import cv2
from numpy import arange
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
--------------- CORE CLASS ---------------------------------------------------
'''

#loop and process frames from the video stream
#frame acquisition should be in a dedicated thread
#a thread 1 should be dedicated to RT frames acquisition
#main thread should be dedicated to process frames once available

class DrawsinessEvaluator(object):
    def __init__(self, src=0):
        self.EAR_THREASHOLD = 0.25
        self.consecutive_closed_frames = 0
        self.number_of_closed_frames = 0
        self.number_of_opened_frames = 0
        #if false, a fine tuning of ear index was not carried out
        self.isEARFineTuned = False
        self.minimum_number_of_frames_to_process_for_ear_tuning = 450 #almeno 30 secondi di frame (mettere 1 o 3 minuti)
        self.sum_of_all_ear_during_tuning = self.EAR_THREASHOLD
        #time windows to record last 3 minutes of frames
        self.time_window = deque([])
        self.wake_up_buffer = deque([])
        self.WAKE_UP_BUFFER_LENGHT = 100  
        self.moving_average_wake_up_buffer = 0 
        #parametri per il blinking
        self.FPS_CAMERA = 25
        self.total_amnt_blink_frames = 0.5*self.FPS_CAMERA*1/3
        self.blinking_counter = 0 
        self.is_first_swap = True 

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
        """
        key = cv2.waitKey(0)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)
        """
        self.last_frame = self.frame
            
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

        #avg or take the min ???
        self.ear = (leftEAR + rightEAR) / 2.0
        
    
    #Calcolo PERCLOS (occorre ridefinire finestra mobile e soglia dinamica per l'EAR)    
    def PERCLOS_f(self):
    
         #draw timer
        cv2.putText(self.last_frame, "Time: {:.2f}".format(self.elapsed_time()), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #draw fps
        self.increment_number_of_total_frames()
        cv2.putText(self.last_frame, "FPS: {:.2f}".format(self.current_fps()), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        #if ear is in fine tuning...
        if self.isEARFineTuned == False:
            self.sum_of_all_ear_during_tuning += self.ear
            self.EAR_THREASHOLD = (self.sum_of_all_ear_during_tuning) / (self.total_number_of_frames+1)
            cv2.putText(self.last_frame, "EAR fine tuning... EAR= {:.2f}".format(0.8*self.EAR_THREASHOLD), (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            #if a minute has passed and a sufficient number of frames have been processed
            if self.total_number_of_frames > 30*5: #ho supposto di calcolare EAR in 20 secondi
                self.isEARFineTuned = True
                self.EAR_THREASHOLD = self.EAR_THREASHOLD*0.8
            else:
                return
        
        print("CRITICAL_INDEX",self.moving_average_wake_up_buffer/(time.time()-self.start_time_frame))
        print("Moving AVG",self.moving_average_wake_up_buffer, "Tempo attuale:",(time.time()-self.start_time_frame))
        
        #se la lista di frame ha ancora un numero di frame minore di quelli collezionabili in 3 minuti
        if self.number_of_closed_frames+self.number_of_opened_frames < 30*40: #current_fps*60=numero di frame collezionabili in 60 secondi
            #semplicemente aggiungo alla lista
            current_type_of_frame = self.ear >= self.EAR_THREASHOLD #if 1, eye is opened, if 0 is closed

            #se il frame è "occhio aperto"
            if(current_type_of_frame == 1): #if is opened
                self.number_of_opened_frames = self.number_of_opened_frames + 1
                #resetto contatore blinking
                self.blinking_counter = 0
                self.is_first_swap = True
                self.time_window.append(current_type_of_frame)
            else: #if is closed
                 #se il contatore di blinking è < soglia
                 if self.blinking_counter < self.total_amnt_blink_frames:
                     self.blinking_counter += 1
                     self.is_first_swap = True
                 else:
                    if self.is_first_swap == True:
                        self.number_of_closed_frames = self.number_of_closed_frames + self.total_amnt_blink_frames +1
                    else:
                        self.number_of_closed_frames = self.number_of_closed_frames +1

                    tempo = int(time.time()-self.start_time_frame)
                    #se il wake up buffer è saturo, tolgo il primo e aggiungo il nuovo come ultimo (a destra)
                    if len(self.wake_up_buffer) >= self.WAKE_UP_BUFFER_LENGHT:
                        first_value = self.wake_up_buffer.popleft()
                        self.moving_average_wake_up_buffer = (self.moving_average_wake_up_buffer*self.WAKE_UP_BUFFER_LENGHT-first_value+tempo)/self.WAKE_UP_BUFFER_LENGHT
                    #se il wake buffer buffer non è saturo
                    else:
                        self.moving_average_wake_up_buffer = (self.moving_average_wake_up_buffer*len(self.wake_up_buffer)+tempo) / (len(self.wake_up_buffer)+1)
                        self.wake_up_buffer.append(tempo)
                    
                    if self.is_first_swap == True:
                        for x in arange(self.total_amnt_blink_frames+1):
                            self.wake_up_buffer.append(tempo)
                            self.time_window.append(current_type_of_frame) 
                    else:
                        self.wake_up_buffer.append(tempo)
                        self.time_window.append(current_type_of_frame) 

                    self.is_first_swap = False  
            return


        else: #se invece la time window ha già raggiunto il suo limite...
            #elimino il primo elemento
            first_frame_in_time_window = self.time_window.popleft()
            #aggiorno i tipi di frame
            if(first_frame_in_time_window == 1): #if is opened
                self.number_of_opened_frames = self.number_of_opened_frames - 1
            else: #if is closed
                 self.number_of_closed_frames = self.number_of_closed_frames - 1

            #inserisco il nuovo frame
            current_type_of_frame = self.ear >= self.EAR_THREASHOLD #if 1, eye is opened, if 0 is closed
            if(current_type_of_frame == 1): #if is opened
                self.number_of_opened_frames = self.number_of_opened_frames + 1
                #resetto contatore blinking
                self.blinking_counter = 0
                self.is_first_swap = True
                self.time_window.append(current_type_of_frame)
            else: #if is closed
                #se il contatore di blinking è < soglia
                 if self.blinking_counter < self.total_amnt_blink_frames:
                     self.blinking_counter += 1
                     self.is_first_swap = True
                 else:
                    if self.is_first_swap == True:
                        self.number_of_closed_frames = self.number_of_closed_frames + self.total_amnt_blink_frames +1
                    else:
                        self.number_of_closed_frames = self.number_of_closed_frames +1
                    #se il wake up buffer è saturo, tolgo il primo e aggiungo il nuovo come ultimo (a destra)
                    tempo = int(time.time()-self.start_time_frame)
                    if len(self.wake_up_buffer) > self.WAKE_UP_BUFFER_LENGHT:
                        first_value = self.wake_up_buffer.popleft()
                        self.moving_average_wake_up_buffer = (self.moving_average_wake_up_buffer*self.WAKE_UP_BUFFER_LENGHT-first_value+tempo)/self.WAKE_UP_BUFFER_LENGHT
                    #se il wake buffer buffer non è saturo
                    else:
                        self.moving_average_wake_up_buffer = (self.moving_average_wake_up_buffer*len(self.wake_up_buffer)+tempo) / (len(self.wake_up_buffer)+1)
                        self.wake_up_buffer.append(tempo)
            
            self.time_window.append(current_type_of_frame)


        cv2.putText(self.last_frame, "Numero frame occhi aperti= {:.2f}".format(self.number_of_opened_frames), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(self.last_frame, "Numero frame occhi chiusi= {:.2f}".format(self.number_of_closed_frames), (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        #!!! adapt perclose computation to consider a moving window of X frames corrisponding to last 3 minutes of video 
        perclos = (self.number_of_closed_frames / (self.number_of_closed_frames + self.number_of_opened_frames) )*100*(self.moving_average_wake_up_buffer/(time.time()-self.start_time_frame))

        #draw the computed PERCLOS on the frame
        self.last_frame = cv2.putText(self.last_frame, "PERCLOS: {:.2f}".format(perclos), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        color = (0, 255, 0)
        if(perclos <= 3.75):
            color = (0, 255, 0)
        elif (perclos <=7.5):
            color = (0,200,0)
        elif (perclos <= 11.25):
            color = (222,255,0)
        elif (perclos <= 15):
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
                drawsiness_videodetector.PERCLOS_f()
                
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
