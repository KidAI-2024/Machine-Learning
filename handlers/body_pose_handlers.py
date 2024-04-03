import utils
from decorators import event
from server_utils import Req, Res


@event("start_body_pose_train")
def start_body_pose_train(req: Req, res: Res):
    path = req.msg["path"]
    # training_data is map {"Class Number(first character in the folder name)" : [images]}
    training_data = utils.read_data(path)
    for key in training_data:
        print(f"Training for class {key} length: {len(training_data[key])}")
    res_msg = {"message": "success"}
    return res.build(req.event, res_msg)
