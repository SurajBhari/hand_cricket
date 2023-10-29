from cvzone.HandTrackingModule import HandDetector
import cv2
from random import randint
from time import sleep

cap = cv2.VideoCapture(0)

detector = HandDetector(staticMode=True, maxHands=1, modelComplexity=1, detectionCon=0.3, minTrackCon=0.3)
score = 0
last_choice = None
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=True, flipType=True)
    cpu_choice = randint(1, 5)

    if hands:
        cv2.imshow("Image", img)

        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        center1 = hand1['center']
        handType1 = hand1["type"]

        fingers1 = detector.fingersUp(hand1)
        fingersup = fingers1.count(1)
        if not fingersup:
            continue
        if last_choice == fingersup:
            continue
        last_choice = fingersup
        if fingersup != cpu_choice:
            score += cpu_choice
            print(f"cpu- {cpu_choice}, Yours - {fingersup}")
        else:
            print(f"Out! cpu- {cpu_choice}, Yours - {fingersup}")
            break
        length, info, img = detector.findDistance(
            lmList1[8][0:2], 
            lmList1[12][0:2], 
            img, 
            color=(255, 0, 255), 
            scale=5
        )
        
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(score)
cap.release()
cv2.destroyAllWindows()