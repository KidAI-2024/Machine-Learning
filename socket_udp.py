import time
import cv2
import numpy as np
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("localhost", 5065))
print("Listening on port 5065")
data = b""
chunkSize = 60000
frame_count = 0
previous_time = time.time()
while True:
    chunk, addr = sock.recvfrom(chunkSize)
    data += chunk

    if len(chunk) < chunkSize:
        frame_count += 1
        if time.time() - previous_time > 1:
            print(f"FPS: {frame_count}")
            previous_time = time.time()
            frame_count = 0
        # print("Received image with size: ", len(data))
        color_array = np.frombuffer(data, dtype=np.uint8)
        img = color_array.reshape((180, 320, 3))
        # Convert the image from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Flip the image vertically
        img = cv2.flip(img, 0)
        # TODO: Process the image

        # --- Send the processed image back to Unity ---
        # # Encode the grayscale image as JPEG
        # _, jpeg = cv2.imencode('.jpg', gray)
        # sock.sendto(jpeg.tobytes(), addr)
        # --- Show the image ---
        # cv2.imshow("UDP Image", img)
        # while True:
        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break
        # cv2.destroyAllWindows()
        # --- Save the image ---
        # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", img)
        # Send a message back to the client
        sock.sendto(b"Image received", addr)
        # --- Reset the data ---
        data = b""
