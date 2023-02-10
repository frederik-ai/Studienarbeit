import cv2

#def add_tape(img_tensor, color=(34, 50, 199)):
#    img_tensor_with_tape = img_tensor
#    for img in img_tensor_with_tape:
#        img = img.numpy()
#        img_height = img.shape[0]
#        img_width = img.shape[1]
#        rect_width = int(img_width / 4)
#        rect_height = int(img_height / 8)
#        rect_x = int(img_width / 2 - rect_width / 2)
#        rect_y = int(img_height / 2 - rect_height / 2)
#        color = color
#        thickness = -1      # -1: fill the rectangle
#        img = cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), thickness)

#    return img_tensor_with_tape

def add_tape(img, color=(34, 50, 199)):
    img_with_tape = img.numpy()
    img_height = img_with_tape.shape[0]
    img_width = img_with_tape.shape[1]
    rect_width = int(img_width / 4)
    rect_height = int(img_height / 8)
    rect_x = int(img_width / 2 - rect_width / 2)
    rect_y = int(img_height / 2 - rect_height / 2)
    color = color
    thickness = -1      # -1: fill the rectangle
    img_with_tape = cv2.rectangle(img_with_tape, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), color,
                                  thickness)
    return img_with_tape

def add_cross(img):
    img_with_cross = img.numpy()
    img_height = img_with_cross.shape[0]
    img_width = img_with_cross.shape[1]
    thickness = 40
    color = (235, 146, 52)
    #img_with_cross = cv2.line(img_with_cross, (int(img_width*0.2), int(img_height*0.2)),
    #                          (int(img_width*0.8), int(img_height*0.8)), color, thickness)
    #img_with_cross = cv2.line(img_with_cross, (int(img_width*0.2), int(img_height*0.8)),
    #                          (int(img_width*0.8), int(img_height*0.2)), color, thickness)
    img_with_cross = cv2.drawMarker(img_with_cross, (int(img_width*0.5), int(img_height*0.5)), color,
                                    cv2.MARKER_TILTED_CROSS, thickness=thickness, markerSize=int(img_width/2),
                                    line_type=cv2.LINE_8)
    return img_with_cross
