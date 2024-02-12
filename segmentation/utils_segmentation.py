import csv
import os

############

ratio_dict = dict()
ratio_dict['400x'] = 1
ratio_dict['100x'] = 4
ratio_dict['25x'] = 16

############

def remove_white_background_v3(input_img, PADDING, folder_path):
    # Reset variables
    remove_rows_top = 0
    remove_rows_bottom = 0
    remove_cols_left = 0
    remove_cols_right = 0
    x_list = []
    y_list = []
    white_background_vector = [250, 251, 252, 253, 254, 255]
    csv_override_filename = folder_path + 'override.csv'

    if os.path.isfile(csv_override_filename):
        # Some images need special care. We can override the values of x_inside and y_inside here using the CSV file in the folder
        # csv file in the folder should have the name 'override.csv', and contain values "x,y", e.g. "1000,2500".
        # Read from CSV file
        with open(csv_override_filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for line in reader:
                override_xy = line

        if input_img.height < 25000:
            x_inside = int(override_xy[0])
            y_inside = int(override_xy[1])
        elif input_img.height < 50000:
            x_inside = int(override_xy[0]) // ratio_dict['100x']
            y_inside = int(override_xy[1]) // ratio_dict['100x']
        elif input_img.height > 100000:
            x_inside = int(override_xy[0]) // ratio_dict['25x']
            y_inside = int(override_xy[1]) // ratio_dict['25x']
    else:
        # If there is not a CSV file with coordinates of x_inside and y_inside, this section will find them automatically.
        # Make a grid of all X lines and find minimum values (which indicate a WSI in the large white space) (there may be more than one WSI)
        step_x = int(input_img.width // 100)
        step_y = int(input_img.height // 100)

        for x_pos in range(step_x, input_img.width, step_x):
            tmp = input_img.extract_area(x_pos, 0, 1, input_img.height)
            x_list.append((x_pos, tmp.min()))

        # Go through x_list and find all transitions between "white background" and "WSI image".
        threshold = 250
        dict_of_transitions_x = dict()
        over_under_threshold = 'under'

        for index, value in enumerate(x_list):
            if over_under_threshold == 'under':
                if value[1] < threshold:
                    dict_of_transitions_x[len(dict_of_transitions_x)] = index
                    over_under_threshold = 'over'
            elif over_under_threshold == 'over':
                if value[1] > threshold:
                    dict_of_transitions_x[len(dict_of_transitions_x)] = index
                    over_under_threshold = 'under'

        x_inside = x_list[dict_of_transitions_x[0]][0] + ((x_list[dict_of_transitions_x[1]][0] - x_list[dict_of_transitions_x[0]][0]) // 2)

        # Initial crop (if there are more than one WSI in the image, this crops out the first one)
        if len(dict_of_transitions_x) > 2:
            init_crop_x = x_list[dict_of_transitions_x[1]][0] + ((x_list[dict_of_transitions_x[2]][0] - x_list[dict_of_transitions_x[1]][0]) // 2)
            input_img = input_img.extract_area(0, 0, init_crop_x, input_img.height)

        # Make a grid of all Y lines and find minimum values (which indicate a WSI in the large white space)
        for y_pos in range(step_y, input_img.height, step_y):
            tmp = input_img.extract_area(0, y_pos, input_img.width, 1)
            y_list.append((y_pos, tmp.min()))

        dict_of_transitions_y = dict()
        over_under_threshold = 'under'

        for index, value in enumerate(y_list):
            if over_under_threshold == 'under':
                if value[1] < threshold:
                    dict_of_transitions_y[len(dict_of_transitions_y)] = index
                    over_under_threshold = 'over'
            elif over_under_threshold == 'over':
                if value[1] > threshold:
                    dict_of_transitions_y[len(dict_of_transitions_y)] = index
                    over_under_threshold = 'under'

        y_inside = y_list[dict_of_transitions_y[0]][0] + ((y_list[dict_of_transitions_y[1]][0] - y_list[dict_of_transitions_y[0]][0]) // 2)

    ##### REMOVE HORIZONTAL WHITE LINES (TOP AND DOWN)
    if input_img(x_inside, 0)[1] in white_background_vector:
        first = 0
        last = y_inside
        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            if input_img(x_inside, midpoint)[1] in white_background_vector:
                first = midpoint + 1
            else:
                last = midpoint - 1
        remove_rows_top = midpoint - 1
    ##### REMOVE HORIZONTAL WHITE LINES (BOTTOM AND UP)
    if input_img(x_inside, (input_img.height - 1))[1] in white_background_vector:
        # first = (current_image.height // 2) - 5000
        first = y_inside
        last = input_img.height

        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            # if current_image(((current_image.width // current_divide_constant)-(current_image.width//4)), midpoint)[1] == 255:
            if input_img(x_inside, midpoint)[1] in white_background_vector:
                last = midpoint - 1
            else:
                first = midpoint + 1

        remove_rows_bottom = midpoint
    ##### REMOVE VERTICAL WHITE LINES (VENSTRE MOT HoYRE)
    if input_img(0, y_inside)[1] == 255:
        first = 0
        last = x_inside

        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            if input_img(midpoint, y_inside)[1] == 255:
                first = midpoint + 1
            else:
                last = midpoint - 1
        remove_cols_left = midpoint - 1
    ##### REMOVE VERTICAL WHITE LINES (HOYRE MOT VENSTRE)
    if input_img(input_img.width - 1, y_inside)[1] == 255:
        first = x_inside
        last = input_img.width
        while first <= last:
            midpoint = (first + last) // 2  # Using floor division
            if input_img(midpoint, y_inside)[1] == 255:
                last = midpoint - 1
            else:
                first = midpoint + 1
        remove_cols_right = midpoint + 1

    # Calculate new width/height of image and crop.
    if remove_rows_bottom != 0:
        # Calculate new width/height
        new_width = (input_img.width - remove_cols_left - (input_img.width - remove_cols_right))
        new_height = (input_img.height - remove_rows_top - (input_img.height - remove_rows_bottom))

        # Include a border around image (to extract 25x tiles later)
        remove_cols_left = remove_cols_left - PADDING
        remove_rows_top = remove_rows_top - PADDING
        new_width = new_width + 2 * PADDING
        new_height = new_height + 2 * PADDING

        return remove_cols_left, remove_rows_top, new_width, new_height