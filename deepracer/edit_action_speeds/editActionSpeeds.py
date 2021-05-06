import json
import os
import uuid


# README
# Paste in your full action space into the file "action_space.json", it can handle index values if you have them already
# Run this script with desired parameters on the last line"


def edit_action_speeds(filename, edit_value, change_left_steering, change_right_steering, add_indexer):
    with open(filename) as action_space_file:
        action_space_data = json.load(action_space_file)
        index_counter = 0
        for action in action_space_data['action_space']:
            if add_indexer:
                action['index'] = index_counter
                index_counter += 1
            if change_left_steering:
                if action['steering_angle'] < 0:
                    action_update(action, edit_value)
            if change_right_steering:
                if action['steering_angle'] > 0:
                    action_update(action, edit_value)

    create_temp_file(action_space_data, filename)


def create_temp_file(action_space_data, filename):
    # Create randomly named temporary file to avoid interference with other thread/asynchronous request
    tmpfile = os.path.join(os.path.dirname(filename), str(uuid.uuid4()))

    with open(tmpfile, 'w') as temp:
        json.dump(action_space_data, temp, indent=4)

    # Rename temporary file replacing old file
    os.rename(tmpfile, filename)


def action_update(action, edit_value):
    # Max Speed is 4 so limit there or ignore
    if action['speed'] + edit_value > 4:
        action['speed'] = 4
    # Min Speed is 0 so limit there or ignore
    elif action['speed'] + edit_value < 0:
        action['speed'] = 0
    else:
        # Keep Speeds to 2 decimal values when adding/subtracting
        action['speed'] = round((action['speed'] + edit_value), 2)


# 1st is filename of action space (metadata.json)
# 2nd Parameter either a positive or negative value for editing all speeds the same value
# 3rd/4th Parameter boolean for changing left(positive) or right steering angles
# 5th Parameter is for adding index values


# EXAMPLE 1 - Add Speed of 0.5 to all actions, with updating index values
edit_action_speeds('action_space.json', 0.5, True, True, True)

# EXAMPLE 2 - Minus Speed of 0.1 to all actions, with no change to index values
# edit_action_speeds('action_space.json', -0.1, True, True, True, False)
