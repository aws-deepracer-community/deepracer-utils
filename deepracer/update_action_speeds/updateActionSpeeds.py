import json
import os
import uuid


def load(action, edit_value):
    # Max Speed is 4 so limit there or ignore
    if action['speed'] + edit_value > 4:
        action['speed'] = 4
    # Min Speed is 0 so limit there or ignore
    elif action['speed'] + edit_value < 0:
        action['speed'] = 0
    else:
        # Keep Speeds to 2 decimal values when adding/subtracting
        action['speed'] = round((action['speed'] + edit_value), 2)


def update(filename, edit_value, change_left_steering, change_right_steering, add_indexer):
    with open(filename) as action_space_file:
        action_space_data = json.load(action_space_file)
        index_counter = 0
        for action in action_space_data['action_space']:
            if add_indexer:
                action['index'] = index_counter
                index_counter += 1
            if change_left_steering:
                if action['steering_angle'] < 0:
                    load(action, edit_value)
            if change_right_steering:
                if action['steering_angle'] > 0:
                    load(action, edit_value)

    save(action_space_data, filename)


def save(action_space_data, filename):
    # Create randomly named temporary file to avoid interference with other
    # thread/asynchronous request
    temp_file = os.path.join(os.path.dirname(filename), str(uuid.uuid4()))

    with open(temp_file, 'w') as temp:
        json.dump(action_space_data, temp, indent=4)

    # Rename temporary file replacing old file
    os.rename(temp_file, filename)


class UpdateActionSpeeds:

    def __init__(self, filename='action_space.json', edit_value=0.0, change_left_steering=True,
                 change_right_steering=True, add_indexer=True):

        self.filename = filename
        self.edit_value = edit_value
        self.change_left_steering = change_left_steering
        self.change_right_steering = change_right_steering
        self.add_indexer = add_indexer

        update(filename, edit_value, change_left_steering, change_right_steering, add_indexer)


# Change your parameter values here.
# edit_value can be positive or negative
updated_action_speeds = UpdateActionSpeeds(filename='action_space.json', edit_value=0.5,
                                           change_left_steering=True, change_right_steering=True,
                                           add_indexer=True)


def run():
    return updated_action_speeds
