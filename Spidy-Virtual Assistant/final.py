import pyttsx3 #pip install pyttsx3
import speech_recognition as sr #pip install speechRecognition
import datetime
import wikipedia #pip install wikipedia
import webbrowser
import os
import smtplib
import cv2
import numpy as np
from PIL import Image 
import pyautogui as p
import sys
from LoginDetails import password, my_gmail, destination

t = password
g = my_gmail
d = destination

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# print(voices[1].id)
engine.setProperty('voice', voices[0].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning!")

    elif hour>=12 and hour<18:
        speak("Good Afternoon!")   

    else:
        speak("Good Evening!")  

    speak("I am SPidy Sir. Please tell me how may I help you")       

def takeCommand():
    #It takes microphone input from the user and returns string output

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...") 
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")    
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")

    except Exception as e:
        # print(e)    
        print("Say that again please...")  
        return "None"
    return query

def sendEmail(to, content):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(g, t)
    server.sendmail(g, to, content)
    server.close()

def TaskExecution():
    p.press('esc')
    speak("Verification Sucessful")
    speak("Welcome back ketan sir")
    wishMe()
    while True:
    # if 1:
        query = takeCommand().lower()

        # Logic for executing tasks based on query
        if 'wikipedia' in query:
            speak('Searching Wikipedia...')
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=1)
            speak("According to Wikipedia")
            print(results)
            speak(results)

        elif 'open youtube' in query:
            webbrowser.open("youtube.com")
        
        elif 'open google' in query:
            webbrowser.open("google.com")

        elif 'open stackoverflow' in query:
            webbrowser.open("stackoverflow.com")   


        elif 'play music' in query:
            music_dir = 'C:\\Uzds'
            songs = os.listdir(music_dir)
            print(songs)    
            os.startfile(os.path.join(music_dir, songs[0]))
        
        elif "open instagram" in query:
            webbrowser.open("https://www.instagram.com/")
     
        elif "open linkedin" in query:
            webbrowser.open("https://www.linkedin.com/feed/")
   
        elif "open drive" in query:
            webbrowser.open("https://drive.google.com/drive/u/1/my-drive")
       
        elif "open meet" in query:
            webbrowser.open("https://meet.google.com/?authuser=1")

        elif "open ca" in query:
            webbrowser.open("https://meet.google.com/ggr-ukqo-bsi?authuser=2/")

        elif "open Documents" in query:
            webbrowser.open("https://docs.google.com/document/u/6/?tgif=d")
        elif "open slides" in query:
            webbrowser.open("https://docs.google.com/presentation/u/0/")
        elif "open sheet" in query:
            webbrowser.open("https://docs.google.com/spreadsheets/?authuser=0")    

        
        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")    
            speak(f"Sir, the time is {strTime}")

        elif 'open notepad' in query:
            codePath = "C:\\WINDOWS\\system32\\notepad.exe"
            os.startfile(codePath)
     
        elif 'open proteus' in query:
            codePath = "C:\Program Files (x86)\Labcenter Electronics\Proteus 8 Professional\BIN\PDS.EXE"
            os.startfile(codePath)
     
        elif 'open vs code' in query:
            codePath = "M:\\Microsoft VS Code\\Code.exe"
            os.startfile(codePath)

        elif 'open code' in query:
            codePath = "C:\\Users\\ketan\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"
            os.startfile(codePath)
       
        elif 'top' in query:
            sys.exit()

        elif 'stop' in query:
            sys.exit()

        elif 'send email' in query:
            try:
                speak("What should I say?")
                content = takeCommand()
                to = d    
                sendEmail(to, content)
                speak("Email has been sent!")
            except Exception as e:
                print(e)
                speak("Sorry ketan sir. I am not able to send this email") 
        
        elif 'whatsapp message' in query:

            name = query.replace("whatsapp ","")
            name1 = name.replace("message ","")
            name2 = name1.replace("send ","")
            name3 = name2.replace("to ","")
            Name= str(name3)
            speak("Whats the message")
            MSG = takeCommand()
            from WhatsappAuto import WhatsappMsg
            WhatsappMsg(Name,MSG)

        elif 'call' in query:
            from WhatsappAuto import WhatsappCall
            name = query.replace("call ","")
            name1 = name.replace("to ","")
            name2 = name1.replace("jarvis ","")
            Name= str(name2)
            WhatsappCall(Name)

        elif 'video call' in query:
            from WhatsappAuto import WhatsappVideoCall
            name = query.replace("call ","")
            name1 = name.replace("to ","")
            name2 = name1.replace("Video ","")
            name3 = name2.replace("do ","")
            Name= str(name3)
            WhatsappVideoCall(Name)

        elif 'show chat' in query:
            speak("With whom ?")
            from WhatsappAuto import WhatsappChat
            name = takeCommand()
            WhatsappChat(name)

          

if __name__ == "__main__":
    recognizer = cv2.face.LBPHFaceRecognizer_create() # Local Binary Patterns Histograms
    recognizer.read('trainer/trainer.yml')   #load trained model
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath) #initializing haar cascade for object detection approach

    font = cv2.FONT_HERSHEY_SIMPLEX #denotes the font type


    id = 2 #number of persons you want to Recognize


    names = ['','ketan']  #names, leave first empty bcz counter starts from 0


    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) #cv2.CAP_DSHOW to remove warning
    cam.set(3, 640) # set video FrameWidht
    cam.set(4, 480) # set video FrameHeight

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    # flag = True

    while True:

        ret, img =cam.read() #read the frames using the above created object

        converted_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #The function converts an input image from one color space to another

        faces = faceCascade.detectMultiScale( 
            converted_image,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) #used to draw a rectangle on any image
    
            id, accuracy = recognizer.predict(converted_image[y:y+h,x:x+w]) #to predict on every single image

            # Check if accuracy is less them 100 ==> "0" is perfect match 
            if (accuracy < 100):
                id = names[id]
                accuracy = "  {0}%".format(round(100 - accuracy))
                TaskExecution()
    
            else:
                id = "unknown"
                accuracy = "  {0}%".format(round(100 - accuracy))
        
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(accuracy), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
        cv2.imshow('camera',img) 

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("Thanks for using this program, have a good day.")
    cam.release()
    cv2.destroyAllWindows()    