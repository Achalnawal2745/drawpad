import cv2
import numpy as np
import mediapipe as mp
import os
import math
from datetime import datetime

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1280, 720
SIDEBAR_WIDTH = 250   # New Sidebar
SMOOTHING = 0.5
BRUSH_SIZE_DRAW = 5   # Smaller default like web
BRUSH_SIZE_ERASER = 40

# --- COLORED BUTTONS ---
# Simple class to define buttons
class Button:
    def __init__(self, x, y, w, h, text, color, action_type, payload=None, icon=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.text = text
        self.color = color
        self.action_type = action_type
        self.payload = payload
        self.is_hovered = False

buttons = []

# --- LAYOUT --
# 1. Colors (Grid)
colors = [
    ('Pink', (85, 0, 255)),   # BGR
    ('Blue', (255, 102, 0)),
    ('Green', (136, 255, 0)),
    ('Orange', (0, 170, 255)),
    ('Purple', (255, 0, 170)),
    ('White', (255, 255, 255))
]

y_offset = 80
for i, (name, col) in enumerate(colors):
    # Two columns
    col_idx = i % 2
    row_idx = i // 2
    bx = 20 + col_idx * 100
    by = y_offset + row_idx * 60
    buttons.append(Button(bx, by, 90, 50, "", col, 'color', col))

y_offset += 200

# 2. Tools
buttons.append(Button(20, y_offset, 200, 50, "Eraser", (50, 50, 50), 'tool', 'eraser'))
y_offset += 60
buttons.append(Button(20, y_offset, 200, 50, "Undo", (50, 50, 50), 'tool', 'undo'))
y_offset += 60
buttons.append(Button(20, y_offset, 200, 50, "Clear Canvas", (50, 50, 200), 'tool', 'clear'))

# 3. Mode Toggle (Bottom)
cursor_btn_y = HEIGHT - 80
buttons.append(Button(20, cursor_btn_y, 200, 60, "Cursor Mode", (100, 100, 100), 'toggle', 'cursor_mode'))

# --- STATE VARIABLES ---
imgCanvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
undo_history = []
MAX_HISTORY = 20

current_color = (85, 0, 255)  # Pink
brush_size = BRUSH_SIZE_DRAW
is_eraser = False
is_cursor_mode = False
is_drawing = False

# Smoothing vars
smooth_x, smooth_y = 0, 0
prev_x, prev_y = 0, 0

# Dwell Click vars
hover_start_time = 0
hovered_button = None
DWELL_TIME = 2.0  # Seconds

# Cooldown after dwell click (prevents mode flickering)
dwell_click_cooldown = False
dwell_cooldown_start = 0
DWELL_COOLDOWN_TIME = 1.0  # 1 second pause after clicking

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# --- HELPER FUNCTIONS ---
def save_state():
    global undo_history
    if len(undo_history) >= MAX_HISTORY:
        undo_history.pop(0)
    undo_history.append(imgCanvas.copy())

def undo():
    global imgCanvas, undo_history
    if len(undo_history) > 0:
        imgCanvas = undo_history.pop()
    else:
        imgCanvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)


def draw_ui(img):
    # 1. Draw Sidebar (with transparency effect)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (SIDEBAR_WIDTH, HEIGHT), (30, 30, 30), cv2.FILLED)
    
    # Header
    cv2.rectangle(overlay, (0, 0), (SIDEBAR_WIDTH, 60), (20, 20, 20), cv2.FILLED)
    cv2.putText(overlay, "Drawing Pad", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Apply transparency
    alpha = 0.9
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # 2. Draw Buttons
    for btn in buttons:
        color = btn.color
        
        # Hover Effect
        if btn.is_hovered:
            # Add white glow border
            cv2.rectangle(img, (btn.x-2, btn.y-2), (btn.x+btn.w+2, btn.y+btn.h+2), (0, 255, 0), 2)
            
        # Draw Button Body
        cv2.rectangle(img, (btn.x, btn.y), (btn.x + btn.w, btn.y + btn.h), color, cv2.FILLED)
        
        # Text Center
        if btn.text:
            text_size = cv2.getTextSize(btn.text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = btn.x + (btn.w - text_size[0]) // 2
            text_y = btn.y + (btn.h + text_size[1]) // 2
            cv2.putText(img, btn.text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 3. Status Indicators
    # Brush Preview (Bottom Left of sidebar? or near color?)
    cv2.circle(img, (SIDEBAR_WIDTH - 30, 40), brush_size//2, current_color if not is_eraser else (255,255,255), cv2.FILLED)
    
    # Cursor Mode Warning
    if is_cursor_mode:
        cv2.putText(img, "CURSOR MODE ACTIVE", (SIDEBAR_WIDTH + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



def handle_click(btn):
    global current_color, is_eraser, brush_size, imgCanvas, is_cursor_mode
    
    if btn.action_type == 'color':
        current_color = btn.payload
        is_eraser = False
        brush_size = BRUSH_SIZE_DRAW
        
    elif btn.action_type == 'tool':
        if btn.payload == 'eraser':
            is_eraser = not is_eraser
            if is_eraser:
                brush_size = BRUSH_SIZE_ERASER
            else:
                brush_size = BRUSH_SIZE_DRAW
        elif btn.payload == 'undo':
            undo()
        elif btn.payload == 'clear':
            save_state()
            imgCanvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            
    elif btn.action_type == 'toggle':
        if btn.payload == 'cursor_mode':
            is_cursor_mode = not is_cursor_mode

cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

print("═══════════════════════════════════════")
print("    Hand Gesture Drawing Pad")
print("═══════════════════════════════════════")
print("KEYBOARD SHORTCUTS:")
print("  C = Cursor Mode  |  E = Eraser")
print("  U = Undo         |  X = Clear")
print("  1-6 = Colors     |  Q = Quit")
print("═══════════════════════════════════════")

while True:
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1) # Mirror
    
    # Hand Tracking
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # Check gestures
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(img, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get Coordinates
            lmList = []
            for id, lm in enumerate(landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            if len(lmList) != 0:
                # Finger Coordinates
                x1, y1 = lmList[8][1:]   # Index tip
                x2, y2 = lmList[12][1:]  # Middle tip
                x3, y3 = lmList[4][1:]   # Thumb tip
                x4, y4 = lmList[20][1:]  # Pinky tip
                
                ## Checking which fingers are up (IMPROVED DETECTION from aa/app.py)
                tipIds = [4, 8, 12, 16, 20]
                fingers = []
                
                # Thumb - check if tip is to the left of the previous joint
                if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                
                # Other fingers - compare tip to MCP joint (base of finger)
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                # Check cooldown first
                if dwell_click_cooldown:
                    elapsed_cooldown = (datetime.now() - dwell_cooldown_start).total_seconds()
                    if elapsed_cooldown > DWELL_COOLDOWN_TIME:
                        dwell_click_cooldown = False
                    else:
                        # Skip gesture processing during cooldown
                        continue
                
                # --- GESTURE MODES ---
                
                ## CURSOR MODE - Index + Middle fingers up (replaces Selection Mode)
                nonCursor = [0, 3, 4]  # Thumb, Ring, Pinky must be down
                if (fingers[1] and fingers[2]) and all(fingers[i] == 0 for i in nonCursor):
                    if not is_cursor_mode:
                        is_cursor_mode = True
                    
                    # Use middle point between index and middle for cursor
                    cursor_x = (x1 + x2) // 2
                    cursor_y = (y1 + y2) // 2
                    
                    # Smoothing
                    if smooth_x == 0 and smooth_y == 0:
                        smooth_x, smooth_y = cursor_x, cursor_y
                    else:
                        smooth_x = int(smooth_x * SMOOTHING + cursor_x * (1 - SMOOTHING))
                        smooth_y = int(smooth_y * SMOOTHING + cursor_y * (1 - SMOOTHING))
                    
                    # Draw cursor indicator
                    cv2.circle(img, (smooth_x, smooth_y), 10, (0, 255, 0), cv2.FILLED)
                    cv2.rectangle(img, (x1-10, y1-15), (x2+10, y2+23), (0, 255, 0), cv2.FILLED)
                    
                    # Hover Check for buttons
                    hovering_any = False
                    for btn in buttons:
                        if btn.x < smooth_x < btn.x + btn.w and btn.y < smooth_y < btn.y + btn.h:
                            hovering_any = True
                            btn.is_hovered = True
                            
                            # VISUAL SELECTION RECTANGLE (like web version)
                            cv2.rectangle(img, (btn.x - 3, btn.y - 3), 
                                        (btn.x + btn.w + 3, btn.y + btn.h + 3), 
                                        (0, 255, 136), 3)
                            
                            if hovered_button != btn:
                                hovered_button = btn
                                hover_start_time = datetime.now()
                            else:
                                elapsed = (datetime.now() - hover_start_time).total_seconds()
                                
                                # Progress ring
                                radius = 20
                                thickness_ring = 3
                                angle = int((elapsed / DWELL_TIME) * 360)
                                cv2.ellipse(img, (smooth_x, smooth_y), (radius, radius), 0, 0, angle, (0, 255, 0), thickness_ring)
                                
                                if elapsed > DWELL_TIME:
                                    handle_click(btn)
                                    hover_start_time = datetime.now()
                                    cv2.circle(img, (smooth_x, smooth_y), 30, (255, 255, 255), -1)
                                    
                                    # ACTIVATE COOLDOWN (prevents mode flickering)
                                    dwell_click_cooldown = True
                                    dwell_cooldown_start = datetime.now()
                        else:
                            btn.is_hovered = False
                    
                    if not hovering_any:
                        hovered_button = None
                    
                    prev_x, prev_y = 0, 0
                
                ## DRAW MODE - ONLY Index finger up
                elif fingers[1] and all(fingers[i] == 0 for i in [0, 2, 3, 4]):
                    is_cursor_mode = False  # Exit cursor mode
                    
                    # Smoothing
                    if smooth_x == 0 and smooth_y == 0:
                        smooth_x, smooth_y = x1, y1
                    else:
                        smooth_x = int(smooth_x * SMOOTHING + x1 * (1 - SMOOTHING))
                        smooth_y = int(smooth_y * SMOOTHING + y1 * (1 - SMOOTHING))
                    
                    # Draw Mode
                    if is_drawing == False:
                        save_state()
                        is_drawing = True
                        prev_x, prev_y = smooth_x, smooth_y
                    
                    if is_eraser:
                        cv2.line(img, (prev_x, prev_y), (smooth_x, smooth_y), (0,0,0), brush_size)
                        cv2.line(imgCanvas, (prev_x, prev_y), (smooth_x, smooth_y), (0,0,0), brush_size)
                    else:
                        cv2.line(img, (prev_x, prev_y), (smooth_x, smooth_y), current_color, brush_size)
                        cv2.line(imgCanvas, (prev_x, prev_y), (smooth_x, smooth_y), current_color, brush_size)
                    
                    prev_x, prev_y = smooth_x, smooth_y
                    
                    # Indicator circle
                    cv2.circle(img, (smooth_x, smooth_y), brush_size//2, current_color if not is_eraser else (255,255,255), cv2.FILLED)
                
                ## STANDBY MODE - Index + Pinky (pauses drawing)
                elif (fingers[1] and fingers[4]) and all(fingers[i] == 0 for i in [0, 2, 3]):
                    is_cursor_mode = False
                    cv2.line(img, (x1, y1), (x4, y4), current_color, 5)
                    prev_x, prev_y = 0, 0
                    is_drawing = False
                
                ## CLEAR CANVAS - Closed fist
                elif all(fingers[i] == 0 for i in range(0, 5)):
                    is_cursor_mode = False
                    save_state()
                    imgCanvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                    prev_x, prev_y = 0, 0
                    is_drawing = False
                
                ## ADJUST THICKNESS - Thumb + Index
                elif all(fingers[i] == j for i, j in zip(range(0, 5), [1, 1, 0, 0, 0])) or all(fingers[i] == j for i, j in zip(range(0, 5), [1, 1, 0, 0, 1])):
                    is_cursor_mode = False
                    
                    # Calculate radius from distance
                    r = int(math.sqrt((x1-x3)**2 + (y1-y3)**2)/3)
                    x0, y0 = int((x1+x3)/2), int((y1+y3)/2)
                    
                    # Draw thickness indicator
                    cv2.circle(img, (x0, y0), int(r/2), current_color, -1)
                    
                    # Confirm with pinky
                    if fingers[4]:
                        brush_size = r
                        cv2.putText(img, 'Set!', (x4-25, y4-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    
                    prev_x, prev_y = 0, 0
                    is_drawing = False
                
                else:
                    is_drawing = False
                    prev_x, prev_y = 0, 0

    # Draw UI
    draw_ui(img)
    
    # Merge Canvas
    # Create mask for transparency
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    
    # Logic: OR canvas onto img where canvas is colored
    # But for eraser to work (black on canvas), we need different blending
    # Simple addition approach:
    # 1. Image area where canvas is NOT drawing
    img = cv2.bitwise_and(img, imgInv) 
    # 2. Add canvas colors
    img = cv2.bitwise_or(img, imgCanvas)
    
    
    cv2.imshow("Hand Gesture Drawing Pad", img)
    
    # Keyboard Controls
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):  # Toggle Cursor Mode
        is_cursor_mode = not is_cursor_mode
    elif key == ord('e'):  # Toggle Eraser
        is_eraser = not is_eraser
        brush_size = BRUSH_SIZE_ERASER if is_eraser else BRUSH_SIZE_DRAW
    elif key == ord('u'):  # Undo
        undo()
    elif key == ord('x'):  # Clear
        save_state()
        imgCanvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    elif key == ord('1'):  # Pink
        current_color = (85, 0, 255)
        is_eraser = False
    elif key == ord('2'):  # Blue
        current_color = (255, 102, 0)
        is_eraser = False
    elif key == ord('3'):  # Green
        current_color = (136, 255, 0)
        is_eraser = False
    elif key == ord('4'):  # Orange
        current_color = (0, 170, 255)
        is_eraser = False
    elif key == ord('5'):  # Purple
        current_color = (255, 0, 170)
        is_eraser = False
    elif key == ord('6'):  # White
        current_color = (255, 255, 255)
        is_eraser = False

cap.release()
cv2.destroyAllWindows()
