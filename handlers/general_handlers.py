import utils
from decorators import event
from server_utils import Req, Res


@event("predict_frame")
def predict_frame(req: Req, res: Res):
    frame_bytes = req.msg["frame"]
    width_str = req.msg["width"]
    height_str = req.msg["height"]
    try:
        width = int(width_str)
    except ValueError:
        width = 320
        print(f"Invalid width: {width_str}")
    try:
        height = int(height_str)
    except ValueError:
        height = 180
        print(f"Invalid height: {height_str}")
    # Convert the bytes to an image
    image = utils.b64string_to_image(frame_bytes, (height, width, 3))
    # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", image)
    pass
