import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
import datetime

#Diccionario colores
powerup_colors = {
    "Laser": (255, 0, 0),              # Azul (en formato BGR → Rojo en OpenCV)
    "Disparo-triple": (0, 255, 255),   # Amarillo
    "Disparo-doble": (0, 165, 255),    # Naranja
    "Bala-perforante": (192, 192, 192),# Gris claro
    "Rafagas": (130, 40, 76),          # Violeta
    "Curacion": (255, 0, 255),         # Rosa
    "Escudo": (255, 255, 0),           # Celeste
    "Inversion-controles": (0, 100, 0) # Verde oscuro
}
#Forma de la nave
shape_pixels = [
    (6, 19), (7, 19), (8, 19), (9, 19), (10, 19), (11, 19), (12, 19), (13, 19),  # Primer nivel
    (2, 18), (3, 18), (4, 18), (5, 18), (6, 18), (7, 18), (8, 18), (9, 18), (10, 18), (11, 18), (12, 18), (13, 18), (14, 18), (15, 18), (16, 18), (17, 18), # Segundo nivel
    (0, 17), (1, 17), (2, 17), (3, 17), (4, 17), (5, 17), (6, 17), (7, 17), (8, 17), (9, 17), (10, 17), (11, 17), (12, 17), (13, 17), (14, 17), (15, 17), (16, 17), (17, 17), (18, 17), (19, 17), # Tercer nivel
    (0, 16), (1, 16), (2, 16), (3, 16), (4, 16), (5, 16), (6, 16), (7, 16), (8, 16), (9, 16), (10, 16), (11, 16), (12, 16), (13, 16), (14, 16), (15, 16), (16, 16), (17, 16), (18, 16), (19, 16), # Cuarto nivel
    (0, 15), (1, 15), (2, 15), (3, 15), (4, 15), (5, 15), (6, 15), (7, 15), (8, 15), (9, 15), (10, 15), (11, 15), (12, 15), (13, 15), (14, 15), (15, 15), (16, 15), (17, 15), (18, 15), (19, 15), # Quinto nivel
    (0, 14), (1, 14), (2, 14), (3, 14), (4, 14), (5, 14), (6, 14), (7, 14), (8, 14), (9, 14), (10, 14), (11, 14), (12, 14), (13, 14), (14, 14), (15, 14), (16, 14), (17, 14), (18, 14), (19, 14), # Sexto nivel
    (1, 13), (2, 13), (3, 13), (4, 13), (5, 13), (6, 13), (7, 13), (8, 13), (9, 13), (10, 13), (11, 13), (12, 13), (13, 13), (14, 13), (15, 13), (16, 13), (17, 13), (18, 13), # Séptimo nivel
    (2, 12), (3, 12), (4, 12), (5, 12), (7, 12), (8, 12), (9, 12), (10, 12), (11, 12), (12, 12), (14, 12), (15, 12), (16, 12), (17, 12), # Octavo nivel
    (2, 11), (3, 11), (4, 11), (5, 11), (7, 11), (8, 11), (9, 11), (10, 11), (11, 11), (12, 11), (14, 11), (15, 11), (16, 11), (17, 11), # Noveno nivel
    (3, 10), (4, 10), (7, 10), (8, 10), (9, 10), (10, 10), (11, 10), (12, 10), (15, 10), (16, 10), # Decimo nivel
    (3, 9), (4, 9), (8, 9), (9, 9), (10, 9), (11, 9), (15, 9), (16, 9), # Undecimo nivel
    (3, 8), (4, 8), (8, 8), (9, 8), (10, 8), (11, 8), (15, 8), (16, 8), # Duodecimo nivel
    (3, 7), (4, 7), (9, 7), (10, 7), (15, 7), (16, 7), # Decimotercer nivel
    (3, 6), (4, 6), (9, 6), (10, 6), (15, 6), (16, 6), # Decimocuarto nivel
    (3, 5), (4, 5), (9, 5), (10, 5), (15, 5), (16, 5), # Decimoquinto nivel
    (9, 4), (10, 4), # Decimosexto nivel
    (9, 3), (10, 3), # Decimoseptimo nivel
    (9, 2), (10, 2), # Decimooctavo nivel
    (9, 1), (10, 1), # Decimonoveno nivel
    (9, 0), (10, 0), # Vigesimo nivel
]


# Dibujar la nave en la pantalla
def draw_ship(frame, nave_x, nave_y, shape_pixels):
    for dx, dy in shape_pixels:
        # Ajustar las coordenadas relativas para que se dibujen en la pantalla
        x = int(nave_x + dx*2.1)
        y = int(nave_y + dy*2.1)  # Restar porque y aumenta hacia abajo
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Usamos verde para dibujar la nave


# Configuración de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(landmarks, eye_indices, img_width, img_height):
    pts = []
    for idx in eye_indices:
        pts.append((int(landmarks[idx].x * img_width), int(landmarks[idx].y * img_height)))
    A = euclidean_distance(pts[1], pts[5])
    B = euclidean_distance(pts[2], pts[4])
    C = euclidean_distance(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks, img_width, img_height):
    top_lip = (int(landmarks[13].x * img_width), int(landmarks[13].y * img_height))
    bottom_lip = (int(landmarks[14].x * img_width), int(landmarks[14].y * img_height))
    left_mouth = (int(landmarks[78].x * img_width), int(landmarks[78].y * img_height))
    right_mouth = (int(landmarks[308].x * img_width), int(landmarks[308].y * img_height))
    vertical = euclidean_distance(top_lip, bottom_lip)
    horizontal = euclidean_distance(left_mouth, right_mouth)
    return vertical / horizontal if horizontal else 0 

RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [263, 387, 385, 362, 380, 373]
EAR_THRESHOLD = 0.15  
MOUTH_THRESHOLD = 0.4

# Crear el fondo con líneas en movimiento

win_width, win_height = 400, 600
game_screen = np.zeros((win_height, win_width, 3), dtype=np.uint8)

# Lista para las líneas
lines = []
line_speed_range = [(3, 6), (1, 2)]  # Rango de velocidades para las líneas (más rápido y más lento)
line_count = 100  # Cantidad de líneas

for i in range(line_count):  # Crear líneas
    x = random.randint(0, win_width)  # Posición aleatoria en el eje x
    y = random.randint(-win_height, 0)  # Posición inicial fuera de la pantalla (arriba)
    speed = random.randint(line_speed_range[0][0], line_speed_range[0][1]) if random.random() < 0.5 else random.randint(line_speed_range[1][0], line_speed_range[1][1])
    lines.append([x, y, speed])

nave_width, nave_height = 50, 30
nave_x, nave_y = win_width // 2, win_height - nave_height - 10  

balas = []
bloques = []
powerups = []  # Lista para los powerups
vel_bala, vel_bloque = 10, 2
score = 0
vidas = 3
boca_abierta_anterior = False
last_bloque_spawn = time.time()
destruccion_bloques = 0  # Contador de bloques destruidos

powerup_types = ["Laser", "Disparo-triple", "Disparo-doble", "Bala-perforante",
                 "Rafagas", "Curacion", "Escudo", "Inversion-controles"]

tiempo_power_up = 5  # Duración de los powerups en milisegundos

power_up_activo = None
tiempo_inicio_mensaje = None  # Para mostrar "+1❤" al recoger curación
tiempo_inicio_power_up = 0
ultimo_disparo = time.time() * 1000  # Tiempo en milisegundos

cap = cv2.VideoCapture(0)

while True:
    ret, frame_cam = cap.read()
    if not ret:
        break
    frame_cam = cv2.flip(frame_cam, 1)
    rgb_frame = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    img_h, img_w = frame_cam.shape[:2]

    mover_derecha, mover_izquierda, disparar = False, False, False

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, img_w, img_h)
        ear_left = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, img_w, img_h)
        mar = mouth_aspect_ratio(landmarks, img_w, img_h)

        if ear_right < EAR_THRESHOLD:
            mover_derecha = True
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if ear_left < EAR_THRESHOLD:
            mover_izquierda = True
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if mar > MOUTH_THRESHOLD:
            if power_up_activo == "Laser": # Disparo láser
                tiempo_actual = time.time() * 1000  # Convertimos a milisegundos
                if tiempo_actual - ultimo_disparo >= 1:  # Disparo cada 1 ms
                    balas.append([nave_x + 20, nave_y - nave_height // 2, 0])
                    ultimo_disparo = tiempo_actual
                
            elif not boca_abierta_anterior:  # Disparo normal, doble, triple y perforante
                if power_up_activo == "Rafagas":
                
                    for offset in [-30, 0, 30]:
                        balas.append([nave_x + 20, nave_y - nave_height + offset // 2, 0])

                elif power_up_activo == "Disparo-triple":
                    angulo = 20  # Ángulo de desviación en grados
                    offset_x = 10  # Separación horizontal

                    # Disparo central (recto)
                    balas.append([nave_x + 20, nave_y - nave_height // 2, 0])  # El tercer valor es la inclinación (0 = recto)

                    # Disparo inclinado a la izquierda
                    balas.append([nave_x - offset_x + 10, nave_y - nave_height // 2, -angulo])

                    # Disparo inclinado a la derecha
                    balas.append([nave_x + offset_x + 30, nave_y - nave_height // 2, angulo])

                elif power_up_activo == "Disparo-doble":
                    balas.append([nave_x + 5, nave_y - nave_height // 2, 0])
                    balas.append([nave_x + 35, nave_y - nave_height // 2, 0])
                else:
                    balas.append([nave_x + 20, nave_y - nave_height // 2, 0])

        boca_abierta_anterior = mar > MOUTH_THRESHOLD


        mp_drawing.draw_landmarks(frame_cam, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)

    # Aplicar efecto de inversión de controles si está activo
    if power_up_activo == "Inversion-controles":
        mover_derecha, mover_izquierda = mover_izquierda, mover_derecha  # Intercambiar controles


    if mover_derecha:
        nave_x -= 7  
    if mover_izquierda:
        nave_x += 7

    nave_x = max(nave_width // 2 - 20, min(win_width - nave_width + 6, nave_x))

    if disparar:
        balas.append([nave_x, nave_y - nave_height // 2])

    balas_temp = []
    for x, y, angulo in balas:
        x += int(math.tan(math.radians(angulo)) * vel_bala)  # Ajuste de dirección
        y -= vel_bala  # Movimiento hacia arriba

        if y > 0 and 0 < x < win_width:  # Mantener las balas dentro de la pantalla
            balas_temp.append([x, y, angulo])

    balas = balas_temp
  

    if time.time() - last_bloque_spawn > 2.5:
        bloques.append([random.randint(20, win_width - 20), 0, random.randint(20, 50)])
        last_bloque_spawn = time.time()

    nuevos_bloques = []  # Lista temporal para guardar los bloques que siguen en la pantalla

    for bloque in bloques:
        x, y, tam = bloque  # Extraer las coordenadas y tamaño del bloque
        y += vel_bloque  # Mover el bloque hacia abajo
        
        if y + tam < win_height - nave_height - 10:  # Solo añadimos los bloques que no han salido de la pantalla
            nuevos_bloques.append([x, y, tam])
        else:
            if power_up_activo == "Escudo":
                escudo_activo = None
            else:
                vidas -= 1

            if vidas == 0:
                cv2.putText(game_screen, "GAME OVER", (win_width // 2 - 180, win_height // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                cv2.imshow("Juego Space Shooter", game_screen)
                cv2.imshow("Camara (controles)", frame_cam)
                cv2.waitKey(0)
                score = 0
                vidas = 3
                bloques.clear()
                balas.clear()
                break

    bloques = nuevos_bloques  # Actualizamos la lista original

    nuevos_bloques = []  # Lista para los bloques no destruidos
    nuevas_balas = []  # Lista para las balas que no han destruido bloques
    nuevos_powerups = []  # Lista temporal para powerups

    for bx, by, tam in bloques:
        colision = False
        for bala in balas:
            bala_x, bala_y, angulo = bala
            if bx - tam // 2 < bala_x < bx + tam // 2 and by - tam // 2 < bala_y < by + tam // 2:
                colision = True
                score += 1
                destruccion_bloques += 1

                if destruccion_bloques >= 3:  
                    #tipo = random.choice(powerup_types)  # Elegir un tipo aleatorio
                    tipo = "Escudo"
                    powerups.append([bx, by, 20, tipo])  # Guardar con su tipo
                    destruccion_bloques = 0  # Reiniciar el contador
                break

        if not colision:
            nuevos_bloques.append([bx, by, tam])

        else:
            for b in balas:
                if bx - tam // 2 < b[0] < bx + tam // 2 and by - tam // 2 < b[1] < by + tam // 2:
                    if power_up_activo == "Bala-perforante":
                        continue
                    else:
                        balas.remove(b)

    bloques = nuevos_bloques

    # Mover powerups hacia abajo y eliminar los que salen de la pantalla
    nuevos_powerups = []  # Lista temporal para almacenar los powerups que siguen en pantalla

    for px, py, tam, tipo in powerups:
        py += 2  # Mover el powerup hacia abajo

        # Verificar colisión mejorada usando distancia
        distancia = math.hypot(nave_x - px, nave_y - py)
        radio_colision = (nave_width // 2) + (tam // 2) + 10  # Se suma un margen extra

        if distancia < radio_colision:  # Si la distancia es menor que el radio, recogemos el power-up
            print(f"Power-up {tipo} recogido!")  
            power_up_activo = tipo  
            tiempo_inicio_power_up = time.time()    

            if tipo == "Curacion" and vidas < 3:
                vidas += 1  # Aumenta la vida si es menor a 3
                tiempo_inicio_mensaje = time.time()  # Guardamos el tiempo del mensaje para "+1❤"

            continue  # No agregamos el power-up a la nueva lista, ya que ha sido recogido



        if py < win_height - 10:  # Si el powerup aún está en la pantalla, lo mantenemos
            nuevos_powerups.append([px, py, tam, tipo])

    powerups = nuevos_powerups  # Actualizar la lista original

    activar_texto = False

    # Control de tiempo para desactivar el power-up
    if power_up_activo and (time.time() - tiempo_inicio_power_up) > tiempo_power_up:
        power_up_activo = None  # Desactivar cualquier power-up activo
            

    # Pantalla de Juego
    game_screen.fill(0)

    # Actualizar y dibujar las líneas en movimiento
    for i in range(len(lines)):
        x, y, speed = lines[i]
        y += speed  # Mover la línea hacia abajo

        if y >= win_height:  # Si la línea salió de la pantalla, ponerla de nuevo en la parte superior
            y = -10
            x = random.randint(0, win_width)  # Cambiar posición horizontal aleatoria

        lines[i] = [x, y, speed]  # Actualizar la posición de la línea

        # Dibujar la línea en el fondo (con un color gris)
        cv2.line(game_screen, (x, y), (x, y + 10), (25, 25, 25), 2)


    # Dibujar la nave
    draw_ship(game_screen, nave_x, nave_y, shape_pixels)

    # Dibujar las balas, bloques y powerups
    for x, y, angulo in balas:
        if power_up_activo == "Laser":
            cv2.circle(game_screen, (x, y), 5, (255, 0, 0), -1)
        else:
            cv2.circle(game_screen, (x, y), 5, (255, 255, 255), -1)

    for bx, by, tam in bloques:
        cv2.rectangle(game_screen, (bx - tam // 2, by - tam // 2), 
                      (bx + tam // 2, by + tam // 2), (0, 0, 255), -1)

    for px, py, tam, tipo in powerups:
        color = powerup_colors.get(tipo, (255, 255, 255))  # Blanco por defecto si hay error
        cv2.rectangle(game_screen, (px - tam // 2, py - tam // 2), 
                    (px + tam // 2, py + tam // 2), color, -1)


    cv2.putText(game_screen, f"Score: {str(score).zfill(2)}", (win_width - 160, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.putText(game_screen, f"Lifes: {vidas}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if power_up_activo and (time.time() - tiempo_inicio_power_up) < tiempo_power_up and not power_up_activo == "Curacion":
        cv2.putText(game_screen, f"Power-up: {(5 - (time.time() - tiempo_inicio_power_up)):.2f}", 
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Mostrar "+1 vida" en el centro de la pantalla por 1 segundo tras recoger curación
    if tiempo_inicio_mensaje and (time.time() - tiempo_inicio_mensaje) < 1:
        cv2.putText(game_screen, "+1 vida", (win_width - nave_width // 2 - 30, win_height - nave_height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    # Dibujar el escudo si está activo con efecto de parpadeo
    if power_up_activo == "Escudo":
        radio = nave_width  # Radio del escudo, del mismo tamaño que la nave
        grosor = 2  # Grosor del círculo
        cv2.circle(game_screen, (nave_x + 20, nave_y + 25), radio - 10, (255, 255, 0), grosor)  # Amarillo (BGR)




    cv2.rectangle(game_screen, (0, win_height - nave_height - 30), (win_width, win_height), (255, 255, 255), 2)
    cv2.imshow("Juego Space Shooter", game_screen)
    cv2.imshow("Camara (controles)", frame_cam)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
