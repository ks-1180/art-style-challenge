from anime_face_detector import create_detector
import numpy as np
import os
import cv2

class AnimeExtractor():
    def __init__(self, conf=0.85):
        self.conf = conf
        self.detector = create_detector('yolov3')
        self.img_full = None
        self.img_cropped = None
        
    def extract(self, img_source, output, offset_value):
        filename = os.path.splitext(os.path.basename(img_source))
        img = cv2.imread(img_source)
        if img is not None:
            try:
                face_location = []
                face = self.detector(img)[0]
                face_location.append(tuple(face['bbox']))

                for left, top, right, bottom, conf in face_location:
                    if conf >= self.conf:
                        top, left, bottom, right = self.__make_square(top, left, bottom, right, offset_value)
                        self.img_full = img.copy()
                        cv2.rectangle(self.img_full, (left, top), (right, bottom), (0,255,0), 2)
                        self.img_cropped = img[top:bottom, left:right]
                        file_name = f'{output}/{filename}.jpg'
                        cv2.imwrite(file_name, self.img_cropped)

                        print('success: ', filename)
            except:
                print('face not found in: ', filename)
        else:
            print('img not found: ', img_source)

                        

    def extract_all(self, source, output, offset_value=0.1):
        if os.path.exists(source) == False:
            print(f'source folder not found: {source}')
            return
        if os.path.exists(output) == False:
            print(f'output folder not found: {output}')
            return

        for filename in os.listdir(source):
            path = os.path.join(source, filename)
            self.extract(path, output, offset_value)

    def __make_square(self, top, left, bottom, right, offset_value):
        offset = int((bottom-top)*offset_value)
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