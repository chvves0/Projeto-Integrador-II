import cv2
import numpy as np
import json
import os

# ─── CONFIGURAÇÕES ───────────────────────────────────────────
IMAGEM       = "estacionamento.jpg"   # sua imagem ou frame da câmera
ARQUIVO_VAGAS = "vagas.json"          # onde as vagas ficam salvas
LIMITE_SAT   = 15    # diferença de saturação para detectar carro colorido
LIMITE_BRILHO = 18   # diferença de brilho para detectar carro branco/escuro
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
#  MODO 1 — DEFINIR VAGAS (roda só na primeira vez)
#  Clique e arraste para marcar cada vaga
#  's' = salvar | 'u' = desfazer | 'q' = sair
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
            vagas_temp.append({
                "id": len(vagas_temp) + 1,
                "x": min(x, x1),
                "y": min(y, y1),
                "w": w,
                "h": h
            })
            print(f"  ✅ Vaga {len(vagas_temp)} marcada")
        ponto_inicio = None

def definir_vagas(imagem_path):
    img = cv2.imread(imagem_path)
    if img is None:
        print(f"❌ Imagem '{imagem_path}' não encontrada.")
        return []

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

        cv2.rectangle(exibir, (0, 0), (400, 40), (20, 20, 20), -1)
        cv2.putText(exibir, f"{len(vagas_temp)} vagas marcadas | 's' para salvar",
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
            print("  Saindo sem salvar.")
            vagas_temp.clear()
            break

    cv2.destroyAllWindows()
    return vagas_temp


# ══════════════════════════════════════════════════════════════
#  MODO 2 — ANALISAR (roda automaticamente toda vez)
#  Carrega as vagas salvas e verifica cada uma
# ══════════════════════════════════════════════════════════════
def calibrar_referencia(imagem_path):
    """Usuário clica em uma área de asfalto VAZIA para calibrar."""
    ARQUIVO_REF = "referencia.json"

    if os.path.exists(ARQUIVO_REF):
        with open(ARQUIVO_REF) as f:
            ref = json.load(f)
        print(f"✅ Referência carregada: brilho={ref['brilho']:.1f}, sat={ref['sat']:.1f}")
        return ref["brilho"], ref["sat"]

    img = cv2.imread(imagem_path)
    clique = {}

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clique["x"], clique["y"] = x, y

    janela = "Clique em uma area de ASFALTO VAZIO para calibrar | 's' confirma"
    cv2.namedWindow(janela)
    cv2.setMouseCallback(janela, on_click)
    print("\n🎯 Clique em uma área de asfalto VAZIO (sem carro) e pressione 's'.")

    while True:
        exibir = img.copy()
        if "x" in clique:
            cv2.drawMarker(exibir, (clique["x"], clique["y"]),
                           (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            # Amostra 30x30 pixels ao redor do clique
            x, y = clique["x"], clique["y"]
            patch = img[max(0,y-15):y+15, max(0,x-15):x+15]
            hsv_p = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            b = np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
            s = np.mean(hsv_p[:,:,1])
            cv2.putText(exibir, f"brilho={b:.0f}  sat={s:.0f}  's' para confirmar",
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
    return 40.0, 5.0  # fallback padrão


def analisar(imagem_path, vagas):
    img = cv2.imread(imagem_path)
    if img is None:
        print(f"❌ Imagem '{imagem_path}' não encontrada.")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    resultado = img.copy()

    ref_brilho, ref_sat = calibrar_referencia(imagem_path)

    livres, ocupadas = 0, 0

    print("\n📊 Resultado da análise:")
    print("─" * 35)

    for v in vagas:
        x, y, w, h = v["x"], v["y"], v["w"], v["h"]

        recorte     = img[y:y+h, x:x+w]
        recorte_hsv = hsv[y:y+h, x:x+w]

        brilho = np.mean(cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY))
        sat    = np.mean(recorte_hsv[:, :, 1])

        delta_brilho = abs(brilho - ref_brilho)
        delta_sat    = sat - ref_sat

        ocupada = delta_sat > LIMITE_SAT or delta_brilho > LIMITE_BRILHO

        # Desenha resultado na imagem
        cor   = (0, 0, 255) if ocupada else (0, 255, 0)
        label = "OCUPADA" if ocupada else "LIVRE"

        overlay = resultado.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), cor, -1)
        cv2.addWeighted(overlay, 0.2, resultado, 0.8, 0, resultado)
        cv2.rectangle(resultado, (x, y), (x+w, y+h), cor, 2)
        cv2.putText(resultado, f"V{v['id']}", (x+5, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(resultado, label, (x+5, y+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 2)

        print(f"  Vaga {v['id']:2d}: {label}")
        if ocupada: ocupadas += 1
        else:       livres += 1

    print("─" * 35)
    print(f"  Total: {len(vagas)} | 🟢 Livres: {livres} | 🔴 Ocupadas: {ocupadas}\n")

    # Painel resumo no topo da imagem
    cv2.rectangle(resultado, (0, 0), (450, 45), (20, 20, 20), -1)
    cv2.putText(resultado,
                f"Total: {len(vagas)}  |  Livres: {livres}  |  Ocupadas: {ocupadas}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("Estacionamento — Resultado", resultado)
    cv2.imwrite("resultado.jpg", resultado)
    print("💾 Resultado salvo em 'resultado.jpg'")
    print("     Pressione qualquer tecla para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════
#  EXECUÇÃO
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # Se ainda não tem vagas salvas → abre para definir
    if not os.path.exists(ARQUIVO_VAGAS):
        print("⚙️  Primeira execução: defina as vagas.")
        definir_vagas(IMAGEM)

    # Carrega vagas e analisa
    if os.path.exists(ARQUIVO_VAGAS):
        with open(ARQUIVO_VAGAS) as f:
            vagas = json.load(f)

        print(f"✅ {len(vagas)} vagas carregadas de '{ARQUIVO_VAGAS}'")
        analisar(IMAGEM, vagas)

    # Para redefinir as vagas:      delete vagas.json e rode novamente
    # Para redefinir a referência:  delete referencia.json e rode novamente
