from io import BytesIO
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import numpy as np
from threading import Thread
from queue import Full, Queue
from PIL import Image
import uuid

from art.workers import StyleWorker, FaceWorker, ProjectorWorker

app = Flask(__name__)
CORS(app)

##########
# Worker #
##########

style_workers = Queue()
style_workers.put(StyleWorker())
style_workers.put(StyleWorker())
style_workers.put(StyleWorker())

face_workers = Queue()
face_workers.put(FaceWorker())

projector_workers = Queue()
projector_workers.put(ProjectorWorker())
projector_workers.put(ProjectorWorker())

##########
# Stream #
##########


class Stream:

    def __init__(self, target_img, latent_vector):
        self.queue = Queue(10)
        self.target_img = target_img
        self.latent_vector = latent_vector
        self.mix = [
            [0.63 for _ in range(10)],
            [0.03 for _ in range(10)],
            [0.33 for _ in range(10)]
        ]


style_streams = {
    'test': Stream(Image.open("data/test.jpg"), np.load("data/test.npy")),
}

#########
# FLASK #
#########


@app.route('/api/upload', methods=['POST'])
def upload_image():
    print('uploading image')

    files = request.files
    file = files.get('image')

    if file == None:
        print('no image uploaded!')
        return 'no "image" uploaded', 400

    worker = face_workers.get()

    img = Image.open(file.stream).convert('RGB')
    ret = False
    try:
        ret, img = worker.extract_face(img)
    finally:
        face_workers.put(worker)
    if ret:
        img_id = str(uuid.uuid4())
        response = jsonify(id=img_id)

        style_streams[img_id] = Stream(img, np.load('data/test.npy'))
        img_io = BytesIO()
        img.save(img_io, 'JPEG', quality=100)
        style_streams[img_id].queue.put(img_io.getvalue())
        print(f'sending response: {response}')
        return response
    else:
        return 'unsucessfully extracted image', 400


@app.route('/api/face/<id>', methods=['GET'])
def get_face(id):
    img = style_streams[id].target_img
    img_io = BytesIO()
    img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


def gen_stream(id):
    print(f'starting stream with id {id}')
    while True:
        img_bytes = style_streams[id].queue.get()
        # yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
        yield (b'--frame\r\n'
               b'Content-Type:image/jpeg\r\n'
               b'Content-Length: ' + f"{len(img_bytes)}".encode() + b'\r\n'
               b'\r\n' + img_bytes + b'\r\n')
        yield (b'--frame\r\n'
               b'Content-Type:image/jpeg\r\n'
               b'Content-Length: ' + f"{len(img_bytes)}".encode() + b'\r\n'
               b'\r\n' + img_bytes + b'\r\n')

@app.route('/api/stream/<id>')
def stream(id):
    return Response(gen_stream(id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/project/<id>', methods=['POST'])
def project(id):
    print(id)
    body = request.get_json(force=True)
    stream = style_streams[id]
    projector_worker = projector_workers.get()
    print('start projection')
    queue = Queue(10)
    running = True
    def run(queue):
        try:
            while running:
                style_worker = style_workers.get()
                latent_vector = queue.get(timeout=1)
                img_bytes = style_worker.generate_bytes(latent_vector, stream.mix)
                style_workers.put(style_worker)
                stream.queue.put(img_bytes, timeout=1)
        except Full:
            print('end stream')

    thread = Thread(target=run, args=(queue,))
    thread.start()
    try:
        for latent_vector in projector_worker.run_projection(stream.target_img, body['steps']):
            stream.latent_vector = latent_vector
            queue.put(latent_vector, timeout=1)
    except Full:
        print('end projection')
    running = False
    projector_workers.put(projector_worker)
    return 'successfull', 200

@app.route('/api/download/<id>', methods=['GET'])
def download(id):
    stream = style_streams[id]
    worker = style_workers.get()
    img_bytes = worker.generate_bytes(stream.latent_vector, stream.mix)
    style_workers.put(worker)
    return send_file(BytesIO(img_bytes), mimetype='image/jpeg', as_attachment=True, attachment_filename=f'{id}.jpg')


@app.route('/api/mix/<id>', methods=['POST'])
def mix(id):
    body = request.get_json(force=True)

    # calculate weights
    weights = np.array(body['weights']).astype(np.float64)
    layer_weights = np.array(body['layerWeights']).astype(np.float64)
    layer_weights *= weights.reshape(-1, 1)
    sum = np.sum(layer_weights, axis=0)
    layer_weights[0, sum == 0] = 1
    sum[sum == 0] = 1
    layer_weights /= sum

    stream = style_streams[id]
    stream.mix = layer_weights
    worker = style_workers.get()
    img_bytes = worker.generate_bytes(stream.latent_vector, layer_weights)
    style_streams[id].queue.put(img_bytes)
    style_workers.put(worker)

    return 'successfull', 200


########
# TEST #
########

@app.route('/api/gantest', methods=['GET'])
def gantest():
    worker = style_workers.get()
    try:
        mix = [
            [0.63 for _ in range(10)],
            [0.03 for _ in range(10)],
            [0.33 for _ in range(10)]
        ]
        latent_vector = np.load('data/test.npy')
        pil_img = worker.generate_pil(latent_vector, mix)
    finally:
        style_workers.put(worker)
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


def gen_example():
    latent_vector = np.load('data/test.npy')

    trans_ticks = 50
    trans = [i/trans_ticks for i in range(trans_ticks+1)]
    inv_trans = trans[::-1]
    empty = [0 for i in range(trans_ticks+1)]
    half_trans = [(i/trans_ticks)/2 for i in range(trans_ticks+1)]
    half_inv_trans = half_trans[::-1]
    styles2 = trans+inv_trans+empty+half_trans+half_inv_trans
    styles3 = empty+trans+inv_trans+half_trans+half_inv_trans

    i = 0
    while True:
        style2 = styles2[i]
        style3 = styles3[i]
        style1 = 1-(style2+style3)

        worker = style_workers.get()
        img_bytes = worker.generate_bytes(latent_vector, [
            [style1 for _ in range(10)],
            [style2 for _ in range(10)],
            [style3 for _ in range(10)]
        ])
        style_workers.put(worker)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
        i = (i+1) % len(styles2)


@app.route('/api/example', methods=['GET'])
def examplestream():
    return Response(gen_example(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0')