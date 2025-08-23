import cv2
import numpy as np
from ultralytics import YOLO

def update_tracking(current_ids, active_ids, collected_ids, disappeared, max_disappear):
    """
    Atualiza o rastreamento de IDs, movendo os que sumiram por muito tempo para 'coletados'.
    """
    current_set = set(current_ids)

    # Verifica IDs que desapareceram
    for an_id in list(active_ids):
        if an_id not in current_set:
            disappeared[an_id] += 1
            if disappeared[an_id] > max_disappear:
                collected_ids.add(an_id)
                active_ids.remove(an_id)
                del disappeared[an_id]
        else:
            disappeared[an_id] = 0  # Reseta o contador se o ID reaparecer

    # Adiciona novos IDs detectados
    for an_id in current_set:
        if an_id not in active_ids and an_id not in collected_ids:
            active_ids.add(an_id)
            disappeared[an_id] = 0

def process_frame(frame, model, roi, active_ids, collected_ids, disappeared, max_disappear):
    """
    Processa um único quadro: detecta, rastreia e desenha as informações na tela.
    """
    # Cria a máscara da ROI e aplica no frame
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [roi], (255, 255, 255))
    roi_frame = cv2.bitwise_and(frame, mask)

    # Rastreia objetos na ROI
    results = model.track(roi_frame, persist=True, conf=0.3, verbose=False)
    
    current_ids = []
    if results[0].boxes.id is not None:
        current_ids = results[0].boxes.id.int().cpu().tolist()

    update_tracking(current_ids, active_ids, collected_ids, disappeared, max_disappear)

    # Prepara o frame de saída
    annotated = results[0].plot()
    output = np.where(mask == 255, annotated, frame)
    
    # Desenha a sobreposição e informações
    overlay = output.copy()
    cv2.fillPoly(overlay, [roi], (0, 255, 0))
    output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
    cv2.polylines(output, [roi], True, (0, 255, 0), 2)
    cv2.putText(output, f"Na ROI: {len(active_ids)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(output, f"Coletados: {len(collected_ids)}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return output

def generate_report(active_ids, collected_ids):
    """Imprime o relatório final da contagem."""
    print("\n=== RELATÓRIO FINAL ===")
    print(f"Total de tênis únicos detectados: {len(active_ids) + len(collected_ids)}")
    print(f"Tênis coletados: {len(collected_ids)}")
    print(f"Tênis restantes na ROI: {len(active_ids)}")

def main():
    """Função principal para inicializar e rodar o programa."""
    model = YOLO("./my_model2/train/weights/best.pt")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise IOError("Não foi possível abrir a webcam")

    w, h = int(cap.get(3)), int(cap.get(4))
    roi = np.array(
        [[0, 80], 
        [w, 80], 
        [w, h], 
        [0, h]], 
    np.int32)

    # Variáveis de rastreamento
    active_ids, collected_ids = set(), set()
    disappeared = dict()
    max_disappear_frames = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = process_frame(frame, model, roi, active_ids, collected_ids, disappeared, max_disappear_frames)
        
        cv2.imshow("YOLOv8 - Detecção de Tênis", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    generate_report(active_ids, collected_ids)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
