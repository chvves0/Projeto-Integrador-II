import cv2
import numpy as np
import json
import os
import time

# ─── CONFIGURAÇÕES ───────────────────────────────────────────
ARQUIVO_VAGAS  = "vagas.json"
ARQUIVO_REF    = "referencia.json"
FOTO_TEMP      = "frame_temp.jpg"
INTERVALO_SEG  = 5       # segundos entre cada análise
CAMERA_INDEX   = 0       # 0 = webcam padrão
LIMITE_SAT     = 15
LIMITE_BRILHO  = 18
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
#  DEFINIR VAGAS  (roda só se vagas.json não existir)
# ══════════════════════════════════════════════════════════════
vagas_temp = []
ponto_inicio = None

def mouse_callback(event, x, y, flags, param):
    global ponto_inicio, vagas_temp
    if event == cv2.EVENT_LBUTTONDOWN:
        ponto_inicio = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and ponto_inicio:
        x1, y1 = ponto_inicio
        w = abs(x - x1)
        h = abs(y - y1)
        if w > 10 and h > 10:
            vagas_temp.append({"id": len(vagas_temp) + 1,
                               "x": min(x, x1), "y": min(y, y1),
                               "w": w, "h": h})
            print(f"  ✅ Vaga {len(vagas_temp)} marcada")
        ponto_inicio = None

def definir_vagas(img):
    """Recebe um frame da webcam para marcar as vagas."""
    janela = "Marque as vagas | Clique e arraste | 's' salva | 'u' desfaz | 'q' sai"
    cv2.namedWindow(janela)
    cv2.setMouseCallback(janela, mouse_callback)
    print("\n🖱️  Clique e arraste para marcar cada vaga.")
    print("     's' = salvar  |  'u' = desfazer última  |  'q' = sair\n")

    while True:
        exibir = img.copy()
        for v in vagas_temp:
            x, y, w, h = v["x"], v["y"], v["w"], v["h"]
            cv2.rectangle(exibir, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(exibir, f"V{v['id']}", (x+5, y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.rectangle(exibir, (0, 0), (420, 40), (20, 20, 20), -1)
        cv2.putText(exibir, f"{len(vagas_temp)} vagas | 's' salva | 'u' desfaz",
                    (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.imshow(janela, exibir)
        tecla = cv2.waitKey(20) & 0xFF

        if tecla == ord('u') and vagas_temp:
            removida = vagas_temp.pop()
            print(f"  ↩️  Vaga {removida['id']} removida")
        elif tecla == ord('s'):
            if not vagas_temp:
                print("  ⚠️  Nenhuma vaga marcada ainda.")
            else:
                with open(ARQUIVO_VAGAS, "w") as f:
                    json.dump(vagas_temp, f, indent=2)
                print(f"\n💾 {len(vagas_temp)} vagas salvas em '{ARQUIVO_VAGAS}'")
                break
        elif tecla == ord('q'):
            vagas_temp.clear()
            break

    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════
#  CALIBRAR REFERÊNCIA  (roda só se referencia.json não existir)
# ══════════════════════════════════════════════════════════════
def calibrar_referencia(img):
    if os.path.exists(ARQUIVO_REF):
        with open(ARQUIVO_REF) as f:
            ref = json.load(f)
        print(f"✅ Referência carregada: brilho={ref['brilho']:.1f}, sat={ref['sat']:.1f}")
        return ref["brilho"], ref["sat"]

    clique = {}
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clique["x"], clique["y"] = x, y

    janela = "Clique em ASFALTO VAZIO para calibrar | 's' confirma"
    cv2.namedWindow(janela)
    cv2.setMouseCallback(janela, on_click)
    print("\n🎯 Clique em uma área de asfalto VAZIO e pressione 's'.")

    while True:
        exibir = img.copy()
        if "x" in clique:
            x, y = clique["x"], clique["y"]
            cv2.drawMarker(exibir, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            patch = img[max(0,y-15):y+15, max(0,x-15):x+15]
            hsv_p = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            b = np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
            s = np.mean(hsv_p[:,:,1])
            cv2.putText(exibir, f"brilho={b:.0f}  sat={s:.0f}  's' confirma",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        cv2.imshow(janela, exibir)
        tecla = cv2.waitKey(20) & 0xFF

        if tecla == ord('s') and "x" in clique:
            x, y = clique["x"], clique["y"]
            patch = img[max(0,y-15):y+15, max(0,x-15):x+15]
            hsv_p = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            ref_brilho = float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)))
            ref_sat    = float(np.mean(hsv_p[:,:,1]))
            with open(ARQUIVO_REF, "w") as f:
                json.dump({"brilho": ref_brilho, "sat": ref_sat}, f)
            print(f"💾 Referência salva: brilho={ref_brilho:.1f}, sat={ref_sat:.1f}")
            cv2.destroyAllWindows()
            return ref_brilho, ref_sat
        elif tecla == ord('q'):
            break

    cv2.destroyAllWindows()
    return 40.0, 5.0  # fallback


# ══════════════════════════════════════════════════════════════
#  ANALISAR UM FRAME
# ══════════════════════════════════════════════════════════════
def analisar_frame(img, vagas, ref_brilho, ref_sat):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    resultado = img.copy()
    livres, ocupadas = 0, 0

    for v in vagas:
        x, y, w, h = v["x"], v["y"], v["w"], v["h"]
        recorte     = img[y:y+h, x:x+w]
        recorte_hsv = hsv[y:y+h, x:x+w]

        brilho = np.mean(cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY))
        sat    = np.mean(recorte_hsv[:, :, 1])

        ocupada = (sat - ref_sat) > LIMITE_SAT or abs(brilho - ref_brilho) > LIMITE_BRILHO

        cor   = (0, 0, 255) if ocupada else (0, 255, 0)
        label = "OCUPADA" if ocupada else "LIVRE"

        overlay = resultado.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), cor, -1)
        cv2.addWeighted(overlay, 0.2, resultado, 0.8, 0, resultado)
        cv2.rectangle(resultado, (x, y), (x+w, y+h), cor, 2)
        cv2.putText(resultado, f"V{v['id']}", (x+5, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.putText(resultado, label, (x+5, y+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 2)

        if ocupada: ocupadas += 1
        else:       livres += 1

    # Painel resumo
    cv2.rectangle(resultado, (0, 0), (480, 45), (20, 20, 20), -1)
    cv2.putText(resultado,
                f"Total: {len(vagas)}  |  Livres: {livres}  |  Ocupadas: {ocupadas}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

    return resultado, livres, ocupadas


# ══════════════════════════════════════════════════════════════
#  LOOP PRINCIPAL — WEBCAM
# ══════════════════════════════════════════════════════════════
def loop_webcam(vagas, ref_brilho, ref_sat):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Não foi possível abrir a webcam.")
        return

    print(f"\n🎥 Monitoramento iniciado (análise a cada {INTERVALO_SEG}s)")
    print("     Pressione 'q' na janela para encerrar.\n")

    ultimo = time.time() - INTERVALO_SEG  # força análise imediata na 1ª vez

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro ao capturar frame.")
            break

        agora = time.time()

        if agora - ultimo >= INTERVALO_SEG:
            ultimo = agora

            resultado, livres, ocupadas = analisar_frame(frame, vagas, ref_brilho, ref_sat)

            # Printa no terminal
            horario = time.strftime("%H:%M:%S")
            print(f"[{horario}] Total: {len(vagas)} | 🟢 Livres: {livres} | 🔴 Ocupadas: {ocupadas}")

            cv2.imshow("Estacionamento — Monitoramento", resultado)
        else:
            # Mostra o frame ao vivo enquanto aguarda
            cv2.imshow("Estacionamento — Monitoramento", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 Monitoramento encerrado.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════
#  EXECUÇÃO
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # 1. Abre a webcam para capturar um frame de referência
    print("📷 Abrindo webcam para configuração...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Webcam não encontrada.")
        exit(1)

    ret, frame_ref = cap.read()
    cap.release()
    if not ret:
        print("❌ Não foi possível capturar frame inicial.")
        exit(1)

    # 2. Definir vagas (só na primeira vez)
    if not os.path.exists(ARQUIVO_VAGAS):
        print("⚙️  Primeira execução: defina as vagas.")
        definir_vagas(frame_ref)

    if not os.path.exists(ARQUIVO_VAGAS):
        print("❌ Nenhuma vaga definida. Encerrando.")
        exit(1)

    with open(ARQUIVO_VAGAS) as f:
        vagas = json.load(f)
    print(f"✅ {len(vagas)} vagas carregadas.")

    # 3. Calibrar referência (só na primeira vez)
    ref_brilho, ref_sat = calibrar_referencia(frame_ref)

    # 4. Inicia o loop de monitoramento
    loop_webcam(vagas, ref_brilho, ref_sat)

    # Para redefinir vagas:      delete vagas.json
    # Para redefinir referência: delete referencia.json
