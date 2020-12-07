import cv2

def put_text(img, class_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    top_left_corner_text = (10, 30)
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2

    cv2.putText(img, class_name,
                top_left_corner_text,
                font,
                font_scale,
                font_color,
                line_type)
    return img


def put_arrow(img, class_name, face):
    arrow_start_x = int(face[0] + 0.5 * face[2])
    arrow_end_x = int(face[0] + 0.1 * face[2] if class_name == "left" else face[0] + 0.9 * face[2])
    arrow_y = int(face[1] + 0.8 * face[3])
    img = cv2.arrowedLine(img,
                          (arrow_start_x, arrow_y),
                          (arrow_end_x, arrow_y),
                          (0, 255, 0),
                          thickness=3,
                          tipLength=0.4)
    return img


def put_rectangle(popped_frame, face):
    color = (0, 255, 0)  # green
    thickness = 2
    popped_frame = cv2.rectangle(popped_frame,
                                 (face[0], face[1]), (face[0] + face[2], face[1] + face[3]),
                                 color,
                                 thickness)
    return popped_frame