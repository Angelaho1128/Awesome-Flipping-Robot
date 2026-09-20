import cv2
import depthai as dai

pipeline = dai.Pipeline()

camera = pipeline.create(dai.node.Camera).build()

output = camera.requestOutput(
    (1280, 720),
    dai.ImgFrame.Type.BGR888p,
    dai.ImgResizeMode.CROP,
    30,
)

queue = output.createOutputQueue()

pipeline.start()

print("OAK-1 started.")
print("Press Q to quit.")

while pipeline.isRunning():
    frame = queue.get()
    image = frame.getCvFrame()

    cv2.imshow("OAK-1", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
