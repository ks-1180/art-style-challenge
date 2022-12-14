import face_recognition
import os
import cv2

class RealisticExtractor():
    def __init__(self):
        self.img_full = None
        self.img_cropped = None

    def extract (self, img_source, output, offset_value=0.1):
        filename = os.path.splitext(os.path.basename(img_source))
        img = cv2.imread(img_source)
        if img is not None:
            try:
                face_locations = face_recognition.face_locations(img)
                self.img_full = img.copy()
                for top, right, bottom, left in face_locations:
                    offset = int((bottom-top)*offset_value)
                    top -= int(offset*3)
                    bottom += offset
                    left -= int(offset*2)
                    right += int(offset*2)

                    cv2.rectangle(self.img_full, (left, top), (right, bottom), (0,255,0), 2)
                    self.img_cropped = img[top:bottom, left:right]

                    cv2.imwrite(f'{output}/{filename}.jpg', self.img_cropped)
                    #os.remove(img_source)
                    print('success: ', filename)
            except:
                print('face not found: ', filename)
                pass
        else:
            print(f'img not found: {img_source}')

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
          