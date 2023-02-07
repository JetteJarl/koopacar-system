import re
import numpy as np


def cone_position_from_sdf(xml_string):
    """ Reads the world coordinates of cone objects from a given sdf string in xml format. """

    state_tag_reg = re.compile("<state.*?</state>", flags=re.DOTALL)
    model_reg = re.compile("<model.*?</model>", flags=re.DOTALL)
    pose_reg = re.compile("<pose>.*?</pose>", flags=re.DOTALL)

    state = state_tag_reg.findall(xml_string)  # find state tag
    models = model_reg.findall(state[0])  # find all models in state tag

    cone_positions = []

    #
    for model in models:
        if "cone" not in model:
            continue
        pose_match = pose_reg.search(model)
        pose_tag = pose_match.group()
        pos_string = pose_tag.replace("<pose>", "").replace("</pose>", "")
        pos = np.fromstring(pos_string, sep=" ")[0:3]

        cone_positions.append(pos)

    return np.array(cone_positions)


def bot_pose_from_sdf(xml_string):
    """ Read the pose (position, orientation) for the Koopacar from the given sdf string in xml format. """
    pass

