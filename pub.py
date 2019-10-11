import zmq
import random
import sys
import time
import json
import cv2
from math import floor, hypot
# import dlib
import numpy as np
import time
import random
import json


context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")



eyes_cascades = [
    cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml'),
    cv2.CascadeClassifier('haarcascade_eye.xml')
]

# escolhendo de qual modelo a CNN vai ser carregada
# fonte: https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
# nao achei na documentacao do open cv uma melhor explicacao sobre essas redes
# no link acima explica que seguem de um artigo e usam a resnet-10, mas so isso
# TODO : achar na documentacao do opencv uma melhor explicacao
net = None
DNN = "CAFFE"
if DNN == "CAFFE":
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# threshold para considerar ou nao uma face detectada pela CNN, eps \in [0, 1]
CONF_THRESHOLD = 0.8


# objeto de captura de video sera o com indice 0, aqui eh o meu webcam
video_capture = cv2.VideoCapture(0)

# variaveis para logica de controle de FPS (frames processadas por segundo :P)
current_sec = floor(time.time())
last_sec = floor(time.time())
detected_sec = iters_sec = 1 # variaveis so serao avaliadas no final da iteracao, entao ja marco que houve uma

# variaveis para logica de controle de duracao de frame
frame_last_start = time.time()
frame_start = time.time()

# variaveis para logica de controle de piscada
# valores na primeira dimensao (0) sao referentes ao primeiro metodo
# na segunda (1), ao segundo metodo
blink_duration = 0.0
are_closed = False
were_closed = False

while True:
    # marcando quando a frame iniciou
    frame_start = time.time()

    # marcando quando o segundo iniciou
    current_sec = floor(time.time())

    # booleano de controle para marcar se houve ou nao face detectada
    detected = False

    # booleanos de controle para marcar se ambos os olhos estao piscando ou nao por ambos os metodos
    is_blinking_1 = is_blinking_2 = False

    # recuperando uma frame (imagem) do video
    _, frame = video_capture.read()
    # continue

    # convertendo frame extraida para cinza (sera usada na extracao de face landmarks)
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray_frame = frame

    # nao tive tempo ainda de ler e entender tudo, fica de TODO pra reuniao
    # fonte: https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/

    # pelo o que eu entendi, blob eh um processamento feito na imagem. como isso acontece e porque eu nao sei ainda
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

    # blob sera a entrada da rede
    net.setInput(blob)

    # deteccoes feitas na rede
    detections = net.forward()

    max_area = 0
    x1m = y1m = x2m = y2m = None

    # para cada deteccao
    for i in range(detections.shape[2]): # terceira dimensao de shape retorna quantas deteccoes foram feitas. porque eh assim eu nao sei ainda
        confidence = detections[0, 0, i, 2] # posicao no tensor indicando a confianca da deteccao
        if confidence > CONF_THRESHOLD: # se a confianca esta acima da definida como limite, a deteccao sera considerada
            detected = True # marcando que houve face detectada
            x1 = int(detections[0, 0, i, 3] * frame.shape[1])
            y1 = int(detections[0, 0, i, 4] * frame.shape[0])
            x2 = int(detections[0, 0, i, 5] * frame.shape[1])
            y2 = int(detections[0, 0, i, 6] * frame.shape[0])

            area = abs(x2 - x1) * abs(y2 - y1)

            a,b,c = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (a,b,c), 4)

            if area > max_area:
                x1m = x1
                x2m = x2
                y1m = y1
                y2m = y2
                max_area = area

    if max_area > 0:

        # print(x1m, x2m, y1m, y2m)

        heigth = abs(y2m - y1m)
        new_heigth = int((heigth / 3) * 2)
        y2m = y1m + new_heigth

        # desenhando bounding box na imagem
        # eu nao to mostrando essa frame, mas pra fazer isso basta executar
        # cv2.imshow("NOME_DA_JANELA", frame)
        # tem que executar a linha acima apos as alteracoes serem realizadas, ou seja, apos a linha abaixo
        cv2.rectangle(frame, (x1m, y1m), (x2m, y2m), (0, 255, 0 ), 4)


        eyest = []
        try:
            # variavel com a imagem da face
            face_frame = frame[y1m:y2m,x1m:x2m]
        except:
            face_frame = frame

        faceROIg = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)

        for eyes_cascade in eyes_cascades:
            if len(eyest) == 0:
                eyest += list(eyes_cascade.detectMultiScale(faceROIg))

        for (x2,y2,w2,h2) in eyest:
            face_frame = cv2.rectangle(face_frame, (x2, y2), (x2+w2, y2+h2), (0, 255, 0 ), 4)

        cv2.imshow('Video1', face_frame)
        cv2.imshow('Video2', frame)


        are_closed = (len(eyest) == 0)



    ############################################################################
    ################################ ITERS/SEC  ################################
    ############################################################################

    # se virou o segundo (comparando o segundo atual com da iteracao passada)
    if current_sec != last_sec:
        # log da qntd de iteracoes do ultimo segundo
        # print("Iterações por segundo: {}. Deteccoes por segundo: {}".format(iters_sec, detected_sec))
        last_sec = current_sec # atualizando o segundo atual
        detected_sec = iters_sec = 0 # zerando a contagem de iteracoes do ultimo (agr atual) segundo

    # se houve face detectada, adicionar na contagem de faces detectadas pro segundo
    if detected:
        detected_sec += 1
    iters_sec += 1 # incrementar a qtnd de iteracoes pro segundo



    ############################################################################
    ####################### BLINK SEQUENCE / FRAME DURATION ####################
    ############################################################################

    # tem o que melhorar. sera discutido na reuniao.

    # calcuando duracao da frame
    frame_duration = frame_start - frame_last_start

    # log da duracao da frame
    # print("Duração da frame: {}".format(frame_duration))

    log = False
    if(are_closed and were_closed): # tava fechado e continua fechado
        blink_duration += frame_duration # incremento a duracao da piscada

    if(are_closed and not were_closed): # nao tava fechado e agora esta
        blink_duration = frame_duration # comecou a piscar agr (podia ser um +=, ja q antes tava 0)

    if(not are_closed and were_closed): # estava fechado e agora nao ta mais
        if blink_duration > 0.1: # se a duracao da piscada eh maior que 0.25, considero como uma piscada nao espontanea
            log = True # vai ser impresso algo
            x = {
              "piscada": blink_duration
            }
            y = json.dumps(x)
            msg = ("1" + " " + y).encode('ascii')
            # print(msg)
            socket.send(msg)
            # print("PISCANDO. DURACAO {}".format(blink_duration)) # log
        blink_duration = 0 # reinicio a duracao da piscada

    if(not are_closed and not were_closed): # nao tava fechado e continua sem estar
        blink_duration = 0 # duracao da piscada eh 0

    were_closed = are_closed # atualizo a variavel "estava piscando"

    # se foi impresso algo, escrevo essas linhas (pra ficar organizado)
    if log:
        print("-----------------------------------------------------------------")

    # resetando inicio da frame
    frame_last_start = frame_start

    ############################################################################
    ################################ OPENCV LOGIC ##############################
    ############################################################################


    # logica do opencv, aperta q pra fechar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
