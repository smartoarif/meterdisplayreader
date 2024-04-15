def group_lines(detections):
    # Sort the list of dictionaries based on 'y1' coordinate
    sorted_detections = sorted(detections, key=lambda x: x['y1'])

    # Initialize variables
    grouped_lines = []
    current_line = []

    # Iterate through sorted detections
    for detection in sorted_detections:
        if not current_line:
            # If current line is empty, add the first detection
            current_line.append(detection)
        else:
            # Calculate the overlap threshold based on half of the smaller detection's height
            overlap_threshold = 0.5 * min(detection['height'], current_line[-1]['height'])
            # Calculate the actual overlap between the detections
            actual_overlap = min(current_line[-1]['y2'], detection['y2']) - max(current_line[-1]['y1'], detection['y1'])
            if actual_overlap >= overlap_threshold:
                # If overlap is sufficient, add to current line
                current_line.append(detection)
            else:
                # If no sufficient overlap, start a new line
                grouped_lines.append(current_line)
                current_line = [detection]

    # Add the last line to grouped lines
    if current_line:
        grouped_lines.append(current_line)

    return grouped_lines

def get_longer_line(grouped_lines):
    # Initialize variables
    max_size = 0
    bigger_line = []

    # Iterate through grouped lines
    for line in grouped_lines:
        # Calculate the size of the line
        line_size = len(line)
        # Update bigger line if current line is bigger
        if line_size > max_size:
            max_size = line_size
            bigger_line = line

    return bigger_line

def format_output(number_list):

    # Group the detections into lines
    grouped_lines = group_lines(number_list)
    # print(grouped_lines)
    longer_line = get_longer_line(grouped_lines)
    
    # sort by x1
    sorted_list = sorted(longer_line, key=lambda x: x['x1'])
    numbers_string = ''.join([item['number'] for item in sorted_list]) 
    
    return int(numbers_string)/100