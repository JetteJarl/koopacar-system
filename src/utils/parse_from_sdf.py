import re
import numpy as np


def _models_from_sdf(xml_string):
    state_tag_reg = re.compile("<state.*?</state>", flags=re.DOTALL)
    model_reg = re.compile("<model.*?</model>", flags=re.DOTALL)

    state = state_tag_reg.findall(xml_string)  # find state tag
    models = model_reg.findall(state[0])  # find all models in state tag

    return models


def cone_position_from_sdf(xml_string):
    """ Reads the world coordinates of cone objects from a given sdf string in xml format. """

    pose_reg = re.compile("<pose>.*?</pose>", flags=re.DOTALL)

    models = _models_from_sdf(xml_string)  # find all models in state tag

    cone_positions = []

    for model in models:
        if "cone" not in model:
            continue

        pose_match = pose_reg.search(model)
        pose_tag = pose_match.group()
        pose_string = pose_tag.replace("<pose>", "").replace("</pose>", "")
        pos = np.fromstring(pose_string, sep=" ")[0:3]

        cone_positions.append(pos)

    return np.array(cone_positions)


def bot_pose_from_sdf(xml_string):
    """ Read the pose (position, orientation) for the Koopacar from the given sdf string in xml format. """

    pose_reg = re.compile("<pose>.*?</pose>", flags=re.DOTALL)

    models = _models_from_sdf(xml_string)  # find all models in state tag

    for model in models:
        if "KoopaCar" in model:
            pose_tag = pose_reg.search(model).group()
            pose_string = pose_tag.replace("<pose>", "").replace("</pose>", "")
            pose = np.fromstring(pose_string, sep=" ")

            return pose

    return None


def set_pose_in_sdf(pose, object_name, file_path):
    """
    Set the pose of a object in the given sdf file.

    pose        --> [x, y, z, roll, pitch, yaw]
    object_name --> [name in model tag]
    file_path   --> [path of sdf file]
    """
    if len(pose) != 6:
        raise Exception("Pose needs to be of form [x, y, z, roll, pitch, yaw]")

    with open(file_path, 'r') as file:
        xml_string = file.read()

    pose_reg = re.compile("<pose>.*?</pose>", flags=re.DOTALL)

    models = _models_from_sdf(xml_string)  # find all models in state tag

    for model in models:
        if f"<model name='{object_name}'>" in model:
            pose_tag_old = pose_reg.search(model).group()
            pose_string_old = pose_tag_old.replace("<pose>", "").replace("</pose>", "")
            pose_string_new = " ".join([str(x) for x in pose])
            pose_tag_new = pose_tag_old.replace(pose_string_old, pose_string_new)

            # replace old pose
            xml_string = xml_string.replace(pose_tag_old, pose_tag_new, 1)

            with open(file_path, 'w') as file:
                file.write(xml_string)

            return

    raise Exception(f"No model tag found with name {object_name}")
