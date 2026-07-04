import numpy as np
import cv2 as cv
import sys

# for loading image
img = cv.imread("photos/rct2.png")

grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Empty callback function for trackbars
def nothing(x):
    pass

# 2. Create the Trackbar Window
cv.namedWindow("Trackbars")
cv.resizeWindow("Trackbars", 400, 300)

# Set the default Canny Max lower (80) to help catch the faint top edge
cv.createTrackbar("Canny Min", "Trackbars", 50, 255, nothing)
cv.createTrackbar("Canny Max", "Trackbars", 80, 255, nothing) 
cv.createTrackbar("Epsilon %", "Trackbars", 2, 10, nothing) 

# FIX: Create a proper NumPy array kernel for dilation
kernel = np.ones((3, 3), np.uint8)

while True:
    # Get current positions of the trackbars
    t1 = cv.getTrackbarPos("Canny Min", "Trackbars")
    t2 = cv.getTrackbarPos("Canny Max", "Trackbars")
    eps_val = cv.getTrackbarPos("Epsilon %", "Trackbars") / 100.0
    if eps_val == 0: 
        eps_val = 0.01

    # 3. Preprocessing Pipeline
    blur = cv.GaussianBlur(grey, (5, 5), 0)
    canny = cv.Canny(blur, t1, t2)
    
    # FIX: Use the numpy array kernel here
    dilated = cv.dilate(canny, kernel, iterations=2)

    # Clone original image for drawing lines
    display_img = img.copy()

    # 4. Find and sort contours
    contours, _ = cv.findContours(dilated, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

    for c in contours:
        
        # Ignore tiny shapes like the signature box
        if cv.contourArea(c) < 5000:
            continue
            
        perimeter = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, eps_val * perimeter, True)
        
        # Draw ALL valid large contours in red to see what it detects
        cv.drawContours(display_img, [approx], -1, (0, 0, 255), 2)
        
        # 5. If it finds a 4-corner shape, draw it in GREEN and warp it
        if len(approx) == 4:
            cv.drawContours(display_img, [approx], -1, (0, 255, 0), 4)
            
            # Warp execution
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
            # Sort the 4 corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
            s, d = pts.sum(axis=1), np.diff(pts, axis=1)
            rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
            rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
            
            # Create the final flat image (400x500 pixels)
            dst = np.array([[0, 0], [400, 0], [400, 500], [0, 500]], dtype="float32")
            matrix = cv.getPerspectiveTransform(rect, dst)
            scanned = cv.warpPerspective(img, matrix, (400, 500))
            
            cv.imshow('Scanned Output', scanned)
            break # Stop looking once we find the main document
            
    else:
        # If no 4-corner shape is found, close the scanned window if it was open
        # Wrapped in a try/except because checking properties of uncreated windows can throw errors
        try:
            if cv.getWindowProperty('Scanned Output', cv.WND_PROP_VISIBLE) >= 1:
                cv.destroyWindow('Scanned Output')
        except cv.error:
            pass

    # 6. Show the debugging windows
    cv.imshow('Edge Map (Dilated)', dilated)
    cv.imshow('Detected Contours (Green=4 corners)', display_img)

    # Break loop if 'q' or 'Esc' is pressed
    key = cv.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cv.destroyAllWindows()