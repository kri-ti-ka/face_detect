import cv2
import numpy as np

#face detection



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Frame", frame)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    eye_cache = None



    faces = face_cascade.detectMultiScale(frame,1.3,5)
    for (x, y, w, h) in faces:
        roi_gray= gray_frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        print("Number of eyes detected:", len(eyes))
        print("Eyes:", eyes)
        if len(eyes) == 2:
        # Store the position of the eyes in cache
            eye_cache = eyes
    # If 2 eyes aren't detected, use the eye cache
        elif eye_cache is not None:
            eyes = eye_cache
        img = cv2.imread("images.png", -1)
        img_h = img.shape[0]
        img_w = img.shape[1]
    # Create source matrix
        src_mat = np.array([[0,0], [img_w, 0],  [img_w, img_h], [0, img_h]])
    # define the destination matrix based on eye detected order.
    # Order of points must be top-left, top-right, bottom-left,
    # and bottom-right
        if len(eyes) == 2:
            # Store the positions of the eyes in cache
            eye_cache = eyes
            # Extract the positions of the left and right eyes
            left_eye, right_eye = eyes
        elif len(eyes) == 1:
            # Use the position of the detected eye and the cached position of the other eye
            # to generate the positions of the left and right eyes
            if eye_cache is not None:
                left_eye = eye_cache[0]
                right_eye = [left_eye[0] + left_eye[2], left_eye[1], left_eye[2], left_eye[3]]
            else:
                # If no eyes have been detected before, just use the position of the detected eye twice
                left_eye = eyes[0]
                right_eye = eyes[0]
        else:
            # No eyes detected
            continue

        # Define the destination matrix based on eye detected order.
        # Order of points must be top-left, top-right, bottom-left,
        # and bottom-right
        if left_eye[0] < right_eye[0]:
            dst_mat = np.array([
                [x + left_eye[0], y + left_eye[1]],
                [x + right_eye[0] + right_eye[2], y + right_eye[1]],
                [x + right_eye[0] + right_eye[2], y + right_eye[1] + right_eye[3]],
                [x + left_eye[0], y + left_eye[1] + left_eye[3]]
            ])
        else:
            dst_mat = np.array([
                [x + right_eye[0], y + right_eye[1]],
                [x + left_eye[0] + left_eye[2], y + left_eye[1]],
                [x + left_eye[0] + left_eye[2], y + left_eye[1] + left_eye[3]],
                [x + right_eye[0], y + right_eye[1] + right_eye[3]]
            ])

        #     if len(eyes) >= 2 and len(eyes[1]) >= 4 and eyes[0][0] < eyes[0][1]:
    #         dst_mat = np.array([
    #             [x + eyes[0][0], y + eyes[0][1]],
    #             [x + eyes[1][0] + eyes[1][2], y + eyes[1][2]],
    #             [x + eyes[1][0] + eyes[1][2], y + eyes[1][1] + eyes[1][3]],
    #             [x + eyes[0][0], y + eyes[0][1] + eyes[0][3]]
    #         ])
    #     else:
    #         dst_mat = np.array([
    #             [x + eyes[1][0], y + eyes[1][1]],
    #             [x + eyes[0][0] + eyes[0][2], y + eyes[0][2]],
    #             [x + eyes[0][0] + eyes[0][2], y + eyes[0][1] + eyes[1][3]],
    #             [x + eyes[1][0], y + eyes[1][1] + eyes[1][3]]
    #         ])
        # Get the dimensions of the frame
        face_h = frame.shape[0]
        face_w = frame.shape[1]


        # Find the Homography matrix
        hom = cv2.findHomography(src_mat, dst_mat)[0]
        # Warp the image to fit the homegraphy matrix
        warped = cv2.warpPerspective(img, hom, (face_w, face_h))
    # Grab the alpha channel of the warped image and create a mask
        mask = warped[: ,: ,2]
    # Copy and convert the mask to a float and give it 3 channels
        mask_scale = mask.copy() / 255.0
        mask_scale = np.dstack([mask_scale] * 3)
    # Remove the alpha channel from the warped image
        warped = cv2.cvtColor(warped, cv2.COLOR_BGRA2BGR)
        warped_multiplied = cv2.multiply(mask_scale, warped.astype("float"))
        image_multiplied = cv2.multiply(frame.astype(float), 1.0 - mask_scale)
        output = cv2.add(warped_multiplied, image_multiplied)
        output = output.astype("uint8")
        cv2.imshow("SnapTalk", output)
#
    if cv2.waitKey(60) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()