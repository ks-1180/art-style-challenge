import numpy as np
import cv2
from anime_face_detector import create_detector

import os

class VideoExtractor:
    def __init__(self):
        self.detector = create_detector('yolov3')

    def extract(self, movie_name, source, target, step_size=60, start_frame=0):
        if os.path.exists(source) == False:
            print('video does not exist')
            return
        elif os.path.exists(target) == False:
            print('target folder does not exist')
            return

        cap = cv2.VideoCapture(source)
        count = 0
        img_counter = 0

        if start_frame != 0:
            cap.set(1, start_frame)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                pass

            elif frame is None:
                print('video finished')
                break

            elif count == 0:
                try:
                    face_location = []
                    face = self.detector(frame)[0]
                    face_location.append(tuple(face['bbox']))

                    for left, top, right, bottom, conf in face_location:
                        if conf >= 0.85:
                            top, left, bottom, right = self.__make_square(top, left, bottom, right)
                            cropped = frame[top:bottom, left:right]

                            if right-left >= 150:
                                file_name = f'{target}/{movie_name}_{img_counter}.jpg'
                                cv2.imwrite(file_name, cropped)
                                print(file_name)
                                img_counter += 1

                except KeyboardInterrupt:
                    break
                except:
                    pass

                count = step_size

            else:
                count -= 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        #cv2.destroyAllWindows()


    def __make_square(self, top, left, bottom, right):
        offset = int((bottom-top)*0.1)
        top = int(top - offset*3)
        bottom = int(bottom + offset)
        left = int(left - offset*2)
        right = int(right + offset*2)

        height = bottom-top
        width = right-left

        if width < height:
            diff = height - width
            if  diff % 2:
                bottom += 1
                left -= int(np.ceil(diff/2))
                right += int(np.ceil(diff/2))
            else:
                left -= int(diff/2)
                right += int(diff/2)

        elif (height < width):
            diff = width - height
            if  diff % 2:
                right += 1
                top -= int(np.ceil(diff/2))
                bottom += int(np.ceil(diff/2))
            else:
                top -= int(diff/2)
                bottom += int(diff/2) 

        return top, left, bottom, right

    