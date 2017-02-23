import cv2
import numpy as np

def valid_gb_win(x):
    while (x%2 != 1):
        x = x+1
    return (x,x)

if __name__ == "__main__":
    # Init GUI
    nothing = lambda x: x+1   # place holder function
    
    cv2.namedWindow("CAM")
    cv2.createTrackbar('GB_WIN', "CAM", 5, 25, nothing)
    cv2.createTrackbar('Variance X', "CAM", 0, 10, nothing)
    cv2.createTrackbar('Variance Y', "CAM", 0, 10, nothing)
    #Switches
    switch = '0 : OFF \n1: ON'
    cv2.createTrackbar('EqHist\n'+switch, "CAM", 0, 1, nothing)
    cv2.createTrackbar('GaussB\n'+switch, "CAM", 0, 1, nothing)
    
    cap = cv2.VideoCapture(0)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
    flag, frame = cap.read()
    while True:
        flag, frame = cap.read()
        # Instantiate TrackBars
        gb_winsize = cv2.getTrackbarPos('GB_WIN', "CAM")
        gb_winsize = (gb_winsize, gb_winsize)
        gb_var_x = cv2.getTrackbarPos('Variance X', "CAM")/10
        gb_var_y = cv2.getTrackbarPos('Variance Y', "CAM")/10
        # Switches
        eqHist_flag = cv2.getTrackbarPos('EqHist\n'+switch, "CAM")
        GaussB_flag = cv2.getTrackbarPos('GaussB\n'+switch, "CAM")
        
        # Handle
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if eqHist_flag == 1:
            #frame = cv2.equalizeHist(frame)
            frame = clahe.apply(frame)
        if GaussB_flag == 1:
            gb_winsize = valid_gb_win(gb_winsize[0])
            frame = cv2.GaussianBlur(frame, gb_winsize, float(gb_var_x), float(gb_var_y))
            
        # Finally
        cv2.imshow("CAM", frame)
        if ord('q') == (cv2.waitKey(20) & 0xFF):
            break
    cap.release()