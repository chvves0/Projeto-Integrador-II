"""
=============================================================================
  SMART PARKING - Sistema de Detecção de Vagas por Visão Computacional
=============================================================================
  Autor     : Projeto Smart Parking
  Versão    : 2.0
  Tecnologias: Python, OpenCV, NumPy

  DESCRIÇÃO:
    Sistema modular para identificar vagas livres e ocupadas em
    estacionamentos usando processamento de imagem clássico.
    Exporta status das vagas em JSON para integração com site.

  COMO USAR:
    1. Coloque "referencia_vazia.jpg" na pasta do projeto
    2. Execute: python smart_parking.py
       → Na primeira execução abre o calibrador interativo de vagas
    3. Coloque fotos na pasta "fotos_pendentes/"
       → O script processa 1 foto por ciclo (a mais antiga),
         salva o resultado visual em "resultados/",
         gera/atualiza "status_vagas.json" para o site
         e deleta a foto processada.

  COMANDOS:
    python smart_parking.py             → loop de produção
    python smart_parking.py --calibrar  → re-calibrar vagas
    python smart_parking.py --teste foto.jpg → teste rápido

=============================================================================
"""

import cv2
import numpy as np
import os
import glob
import time
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

# --- Ciclo de análise ---
INTERVALO_ANALISE_MINUTOS = 4          # Tempo entre verificações (minutos)
INTERVALO_ANALISE_SEGUNDOS = INTERVALO_ANALISE_MINUTOS * 60

# --- Arquivos e pastas ---
ARQUIVO_REFERENCIA   = "referencia_vazia.jpg"   # Estacionamento 100% vazio
PASTA_ENTRADA        = "fotos_pendentes"         # Câmera despeja fotos aqui
PASTA_RESULTADOS     = "resultados"              # Imagens com retângulos de saída
ARQUIVO_STATUS_SITE  = "status_vagas.json"       # JSON lido pelo site

# --- Modo de operação ---
# True  → usa APENAS bordas + textura (ideal quando a referência tem
#          ângulo/perspectiva diferente da foto atual, ou para testes
#          com imagens cartoon/ilustradas)
# False → usa os três scores: diferença de referência + bordas + textura
MODO_SEM_REFERENCIA = False

# --- Pesos do Score Combinado ---
# Modo COM referência (devem somar 1.0):
PESO_DIFERENCA_REFERENCIA = 0.50   # Comparação pixel a pixel com imagem vazia
PESO_BORDAS_CANNY         = 0.30   # Densidade de bordas (contornos de carros)
PESO_TEXTURA              = 0.20   # Variância de textura da ROI

# Modo SEM referência (devem somar 1.0):
PESO_BORDAS_SEM_REF       = 0.55   # Bordas têm mais peso sem a âncora da referência
PESO_TEXTURA_SEM_REF      = 0.45   # Variância de textura complementa as bordas

# --- Limiar de decisão ---
# Score > LIMIAR_OCUPADO  → OCUPADA
# Score < LIMIAR_LIVRE    → LIVRE
# Entre os dois           → INCERTA
# Score ≤ LIMIAR_LIVRE   → LIVRE    (ex: 0.15 = scores até 15%)
# Score ≥ LIMIAR_OCUPADO → OCUPADA  (ex: 0.30 = scores a partir de 30%)
# Entre os dois          → INCERTA
LIMIAR_LIVRE   = 0.23
LIMIAR_OCUPADO = 0.27

# False = usa os limiares fixos acima (recomendado quando já calibrado)
# True  = calcula limiares automaticamente pelo gap dos scores da imagem
USAR_LIMIAR_AUTOMATICO = False

# --- Parâmetros de processamento de imagem ---
CLAHE_CLIP_LIMIT    = 2.0    # Controle do CLAHE (1.0 a 4.0)
CLAHE_TILE_GRID     = (8, 8) # Grade de tiles do CLAHE
BLUR_KERNEL_SIZE    = (5, 5) # Kernel do desfoque Gaussiano
CANNY_LIMIAR_1      = 30     # Limiar inferior do Canny (30 = mais sensível p/ cartoon)
CANNY_LIMIAR_2      = 100    # Limiar superior do Canny

# --- Visualização ---
COR_LIVRE    = (0, 220, 80)    # Verde
COR_OCUPADA  = (0, 50, 220)    # Vermelho (BGR)
COR_INCERTA  = (0, 180, 255)   # Amarelo/Laranja
COR_TEXTO    = (255, 255, 255) # Branco
ESPESSURA_RETANGULO = 2
FONTE = cv2.FONT_HERSHEY_SIMPLEX


# =============================================================================
# COORDENADAS DAS VAGAS
# =============================================================================
# Este dicionário é preenchido automaticamente pelo CalibradorVagas.
# Você pode pré-definir vagas aqui OU deixar vazio e usar o modo
# de calibração interativa (--calibrar) para desenhá-las com o mouse.
#
# Formato manual (opcional):  "ID": [x, y, largura, altura]
COORDENADAS_VAGAS = {}

# Arquivo onde as coordenadas calibradas são salvas/carregadas
ARQUIVO_CALIBRACAO = "vagas_calibradas.json"


# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("smart_parking.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SmartParking")


# =============================================================================
# ESTRUTURA DE DADOS DE UMA VAGA
# =============================================================================
@dataclass
class ResultadoVaga:
    """Armazena o resultado da análise de uma vaga individual."""
    id_vaga: str
    coordenadas: list              # [x, y, w, h]
    score_diferenca: float = 0.0   # 0.0 (igual à ref) → 1.0 (muito diferente)
    score_bordas: float    = 0.0   # 0.0 (sem bordas)  → 1.0 (muitas bordas)
    score_textura: float   = 0.0   # 0.0 (asfalto liso) → 1.0 (muito complexo)
    score_final: float     = 0.0   # Score combinado ponderado
    status: str            = "INCERTA"  # "LIVRE", "OCUPADA" ou "INCERTA"


# =============================================================================
# MÓDULO 1: CARREGAMENTO DE IMAGENS
# =============================================================================
class GerenciadorImagens:
    """Responsável pelo carregamento e validação das imagens."""

    @staticmethod
    def carregar_imagem(caminho: str) -> Optional[np.ndarray]:
        """
        Carrega uma imagem do disco.
        Retorna None se o arquivo não existir ou não puder ser lido.
        """
        if not os.path.exists(caminho):
            return None
        imagem = cv2.imread(caminho)
        if imagem is None:
            logger.error(f"Falha ao decodificar imagem: {caminho}")
        return imagem

    @staticmethod
    def redimensionar_para_referencia(imagem: np.ndarray,
                                      referencia: np.ndarray) -> np.ndarray:
        """
        Garante que a imagem atual tenha o mesmo tamanho da referência.
        Essencial para a comparação pixel a pixel funcionar corretamente.
        """
        h_ref, w_ref = referencia.shape[:2]
        h_img, w_img = imagem.shape[:2]
        if (h_img, w_img) != (h_ref, w_ref):
            logger.warning(
                f"Tamanhos diferentes! Ref={w_ref}x{h_ref}, "
                f"Atual={w_img}x{h_img}. Redimensionando..."
            )
            imagem = cv2.resize(imagem, (w_ref, h_ref),
                                interpolation=cv2.INTER_AREA)
        return imagem

    @staticmethod
    def deletar_imagem(caminho: str) -> bool:
        """
        Remove a imagem do disco após o processamento.
        Retorna True em caso de sucesso.
        """
        try:
            os.remove(caminho)
            logger.info(f"Arquivo deletado: {caminho}")
            return True
        except OSError as e:
            logger.error(f"Erro ao deletar {caminho}: {e}")
            return False


# =============================================================================
# MÓDULO 2: PRÉ-PROCESSAMENTO (NORMALIZAÇÃO DE ILUMINAÇÃO)
# =============================================================================
class Preprocessador:
    """
    Aplica normalização de iluminação para tornar a análise robusta
    a sombras, variações climáticas e diferentes horários do dia.
    """

    def __init__(self):
        # Cria o objeto CLAHE uma única vez (reutilização eficiente)
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID
        )

    def normalizar_roi(self, roi_colorida: np.ndarray) -> np.ndarray:
        """
        Pipeline de normalização aplicado em cada ROI (recorte de vaga):
          1. Conversão para escala de cinza
          2. Desfoque Gaussiano (suaviza ruídos/granulação)
          3. CLAHE (equalização adaptativa de histograma)
             → compensa sombras locais e variações de luminosidade

        Retorna imagem em escala de cinza normalizada.
        """
        # Passo 1: Tons de cinza
        cinza = cv2.cvtColor(roi_colorida, cv2.COLOR_BGR2GRAY)

        # Passo 2: Suavização Gaussiana (reduz ruído antes do CLAHE)
        suavizada = cv2.GaussianBlur(cinza, BLUR_KERNEL_SIZE, 0)

        # Passo 3: CLAHE — equalizador adaptativo por regiões
        normalizada = self.clahe.apply(suavizada)

        return normalizada


# =============================================================================
# MÓDULO 3: EXTRATOR DE ROI (RECORTE DAS VAGAS)
# =============================================================================
class ExtratoreROI:
    """Recorta a região de interesse correspondente a cada vaga."""

    @staticmethod
    def recortar(imagem: np.ndarray, coordenadas: list) -> Optional[np.ndarray]:
        """
        Recorta uma vaga da imagem com base nas coordenadas [x, y, w, h].
        Valida os limites para evitar erros com coordenadas fora da imagem.
        """
        x, y, w, h = coordenadas
        alt_img, larg_img = imagem.shape[:2]

        # Validação dos limites
        if x < 0 or y < 0 or (x + w) > larg_img or (y + h) > alt_img:
            logger.warning(
                f"Coordenadas {coordenadas} fora dos limites da imagem "
                f"({larg_img}x{alt_img}). Ajuste COORDENADAS_VAGAS."
            )
            return None

        roi = imagem[y:y+h, x:x+w]

        # Verificação de ROI vazia
        if roi.size == 0:
            logger.warning(f"ROI vazia para coordenadas {coordenadas}")
            return None

        return roi


# =============================================================================
# MÓDULO 4: ANÁLISE — COMPARAÇÃO COM REFERÊNCIA
# =============================================================================
class AnalisadorDiferenca:
    """
    Compara a vaga atual com a vaga na imagem de referência (estacionamento vazio).
    Quanto mais diferente da referência, maior a chance de estar ocupada.
    """

    @staticmethod
    def calcular_diferenca(roi_atual: np.ndarray,
                           roi_referencia: np.ndarray) -> float:
        """
        Calcula a diferença média absoluta entre a vaga atual e a referência.

        Processo:
          1. Garante que ambas as ROIs têm o mesmo tamanho
          2. Calcula diferença absoluta pixel a pixel
          3. Normaliza para o intervalo [0.0, 1.0]

        Retorna score: 0.0 = idêntica à referência, 1.0 = completamente diferente
        """
        # Garante tamanhos iguais (redundância de segurança)
        if roi_atual.shape != roi_referencia.shape:
            roi_referencia = cv2.resize(
                roi_referencia, (roi_atual.shape[1], roi_atual.shape[0])
            )

        # Diferença absoluta pixel a pixel
        diferenca = cv2.absdiff(roi_atual, roi_referencia)

        # Média da diferença normalizada (0-255 → 0.0-1.0)
        score = float(np.mean(diferenca)) / 255.0

        return score


# =============================================================================
# MÓDULO 5: ANÁLISE — DETECÇÃO DE BORDAS (CANNY)
# =============================================================================
class AnalisadorBordas:
    """
    Usa o detector de bordas Canny para identificar estruturas visuais.
    Carros possuem muitas bordas (linhas, contornos); chão vazio tem poucas.
    """

    @staticmethod
    def calcular_densidade_bordas(roi_normalizada: np.ndarray) -> float:
        """
        Aplica o filtro Canny e calcula a proporção de pixels de borda.

        Processo:
          1. Aplica detector de bordas Canny
          2. Conta pixels "acesos" (bordas detectadas)
          3. Divide pelo total de pixels da ROI → densidade [0.0, 1.0]

        Retorna score: 0.0 = nenhuma borda, 1.0 = totalmente preenchido de bordas
        """
        bordas = cv2.Canny(roi_normalizada, CANNY_LIMIAR_1, CANNY_LIMIAR_2)

        # Densidade = pixels de borda / total de pixels
        total_pixels = bordas.size
        pixels_borda = float(np.count_nonzero(bordas))

        densidade = pixels_borda / total_pixels if total_pixels > 0 else 0.0

        return densidade


# =============================================================================
# MÓDULO 5b: ANÁLISE — TEXTURA (VARIÂNCIA LOCAL)
# =============================================================================
class AnalisadorTextura:
    """
    Mede a complexidade visual de uma ROI pela variância dos pixels.

    Por que funciona:
      - Asfalto vazio → tons de cinza uniformes → variância BAIXA
      - Carro          → cores, reflexos, janelas, detalhes → variância ALTA

    Isso é especialmente robusto em imagens cartoon/ilustradas, onde as
    bordas do Canny podem ser enganadas pelas linhas de demarcação do chão,
    mas a variância distingue bem o asfalto liso do carro colorido.
    """

    # Percentil máximo observado de variância em ROIs de asfalto puro.
    # Usado para normalizar o score para [0.0, 1.0].
    # Valor empírico: ~2500 para fotos reais; ~1500 para imagens cartoon.
    _VARIANCIA_MAX_REFERENCIA = 2000.0

    @classmethod
    def calcular_variancia(cls, roi_normalizada: np.ndarray) -> float:
        """
        Calcula a variância dos pixels da ROI normalizada e retorna
        um score entre 0.0 (uniforme) e 1.0 (muito variado).
        """
        variancia = float(np.var(roi_normalizada.astype(np.float32)))
        score = min(variancia / cls._VARIANCIA_MAX_REFERENCIA, 1.0)
        return round(score, 4)


# =============================================================================
# MÓDULO 6: TOMADA DE DECISÃO (SCORE COMBINADO)
# =============================================================================
class TomadorDecisao:
    """
    Combina os scores individuais em uma decisão final por vaga.
    Suporta dois modos: com referência (3 scores) e sem referência (2 scores).
    """

    @staticmethod
    def calcular_score_final(score_diferenca: float,
                              score_bordas: float,
                              score_textura: float) -> float:
        """
        Combina os três scores com pesos configuráveis.

        Modo COM referência:
          Score = (PESO_DIF × dif) + (PESO_BORD × bordas) + (PESO_TEX × textura)

        Modo SEM referência (MODO_SEM_REFERENCIA = True):
          Score = (PESO_BORD_SR × bordas) + (PESO_TEX_SR × textura)
          → ignora score_diferenca pois a referência tem perspectiva diferente
        """
        if MODO_SEM_REFERENCIA:
            score = (PESO_BORDAS_SEM_REF   * score_bordas +
                     PESO_TEXTURA_SEM_REF  * score_textura)
        else:
            score = (PESO_DIFERENCA_REFERENCIA * score_diferenca +
                     PESO_BORDAS_CANNY         * score_bordas     +
                     PESO_TEXTURA              * score_textura)

        return round(min(score, 1.0), 4)

    @staticmethod
    def classificar(score_final: float,
                    limiar_livre: float,
                    limiar_ocupado: float) -> str:
        """
        Converte o score numérico em status textual.
        Recebe os limiares como parâmetro para suportar calibração automática.
        """
        if score_final >= limiar_ocupado:
            return "OCUPADA"
        elif score_final <= limiar_livre:
            return "LIVRE"
        else:
            return "INCERTA"

    @staticmethod
    def calibrar_limiares_automaticos(scores: list[float]) -> tuple[float, float]:
        """
        Encontra o ponto de corte natural entre vagas livres e ocupadas
        usando detecção de gap — o maior intervalo entre scores consecutivos.

        Por que funciona:
          Scores de vagas vazias e ocupadas formam dois grupos bem separados.
          Ex: [0.03, 0.04, 0.05, 0.08, 0.09 | GAP | 0.35, 0.36, 0.41, 0.42]
          O maior salto entre valores consecutivos é exatamente o divisor
          natural entre "sem carro" e "com carro", independente da escala.

        Limiares gerados:
          LIVRE   = ponto_corte × 0.80  (20% abaixo do gap)
          OCUPADO = ponto_corte × 1.20  (20% acima do gap)
        """
        if not scores:
            return LIMIAR_LIVRE, LIMIAR_OCUPADO

        arr = sorted(scores)

        if len(arr) < 3:
            return LIMIAR_LIVRE, LIMIAR_OCUPADO

        # Encontra o maior gap entre scores consecutivos
        gaps       = [(arr[i+1] - arr[i], i) for i in range(len(arr) - 1)]
        maior_gap, idx_gap = max(gaps, key=lambda x: x[0])

        # Ponto de corte = meio do maior gap
        ponto_corte = (arr[idx_gap] + arr[idx_gap + 1]) / 2

        limiar_livre   = round(ponto_corte * 0.80, 3)
        limiar_ocupado = round(ponto_corte * 1.20, 3)

        # Garante separação mínima
        if limiar_ocupado - limiar_livre < 0.04:
            limiar_livre   = round(ponto_corte - 0.02, 3)
            limiar_ocupado = round(ponto_corte + 0.02, 3)

        logger.info(
            f"Limiares automáticos (gap={maior_gap:.3f}) → "
            f"LIVRE ≤ {limiar_livre} | OCUPADA ≥ {limiar_ocupado}  "
            f"(corte={ponto_corte:.3f} | scores={[round(s,2) for s in arr]})"
        )
        return limiar_livre, limiar_ocupado


# =============================================================================
# MÓDULO 7: VISUALIZAÇÃO DOS RESULTADOS
# =============================================================================
class Visualizador:
    """Renderiza os resultados sobre a imagem do estacionamento."""

    @staticmethod
    def desenhar_vagas(imagem: np.ndarray,
                       resultados: list[ResultadoVaga]) -> np.ndarray:
        """
        Desenha retângulos coloridos e informações sobre cada vaga.
        Também exibe o resumo total no canto superior da imagem.
        """
        output = imagem.copy()
        vagas_livres   = sum(1 for r in resultados if r.status == "LIVRE")
        vagas_ocupadas = sum(1 for r in resultados if r.status == "OCUPADA")
        vagas_incertas = sum(1 for r in resultados if r.status == "INCERTA")
        total          = len(resultados)

        for resultado in resultados:
            x, y, w, h = resultado.coordenadas

            # Seleciona cor baseada no status
            if resultado.status == "LIVRE":
                cor = COR_LIVRE
            elif resultado.status == "OCUPADA":
                cor = COR_OCUPADA
            else:
                cor = COR_INCERTA

            # Retângulo da vaga
            cv2.rectangle(output, (x, y), (x + w, y + h), cor, ESPESSURA_RETANGULO)

            # Fundo semitransparente para o texto
            overlay = output.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + 28), cor, -1)
            cv2.addWeighted(overlay, 0.55, output, 0.45, 0, output)

            # ID e status da vaga
            cv2.putText(
                output,
                f"{resultado.id_vaga}: {resultado.status}",
                (x + 4, y + 18),
                FONTE, 0.42, COR_TEXTO, 1, cv2.LINE_AA
            )

            # Score numérico abaixo do retângulo
            cv2.putText(
                output,
                f"S:{resultado.score_final:.2f}",
                (x + 4, y + h - 6),
                FONTE, 0.38, cor, 1, cv2.LINE_AA
            )

        # --- Painel de resumo no topo ---
        cv2.rectangle(output, (0, 0), (imagem.shape[1], 40), (30, 30, 30), -1)
        resumo = (
            f"SMART PARKING  |  "
            f"Total: {total}  "
            f"Livres: {vagas_livres}  "
            f"Ocupadas: {vagas_ocupadas}  "
            f"Incertas: {vagas_incertas}  |  "
            f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        )
        cv2.putText(
            output, resumo, (10, 27),
            FONTE, 0.52, (220, 220, 220), 1, cv2.LINE_AA
        )

        return output

    @staticmethod
    def salvar_resultado(imagem: np.ndarray, pasta: str) -> str:
        """Salva a imagem de resultado com timestamp único."""
        os.makedirs(pasta, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = os.path.join(pasta, f"resultado_{timestamp}.jpg")
        cv2.imwrite(nome_arquivo, imagem, [cv2.IMWRITE_JPEG_QUALITY, 92])
        logger.info(f"Resultado salvo em: {nome_arquivo}")
        return nome_arquivo


# =============================================================================
# MÓDULO 7b: EXPORTADOR DE STATUS PARA O SITE (JSON)
# =============================================================================
class ExportadorJSON:
    """
    Gera (ou sobrescreve) o arquivo status_vagas.json após cada análise.
    Esse arquivo é consumido pelo site para exibir o status em tempo real.

    Estrutura gerada:
    {
      "ultima_atualizacao": "2026-06-02 22:30:15",
      "total_vagas": 10,
      "vagas_livres": 7,
      "vagas_ocupadas": 3,
      "vagas_incertas": 0,
      "detalhes": {
        "V01": "LIVRE",
        "V02": "OCUPADA"
      }
    }
    """

    @staticmethod
    def exportar(resultados: list) -> bool:
        """
        Recebe a lista de ResultadoVaga e grava o JSON no disco.
        Retorna True em caso de sucesso.
        """
        vagas_livres   = sum(1 for r in resultados if r.status == "LIVRE")
        vagas_ocupadas = sum(1 for r in resultados if r.status == "OCUPADA")
        vagas_incertas = sum(1 for r in resultados if r.status == "INCERTA")

        payload = {
            "ultima_atualizacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_vagas"       : len(resultados),
            "vagas_livres"      : vagas_livres,
            "vagas_ocupadas"    : vagas_ocupadas,
            "vagas_incertas"    : vagas_incertas,
            "detalhes"          : {r.id_vaga: r.status for r in resultados},
        }

        try:
            with open(ARQUIVO_STATUS_SITE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info(
                f"JSON exportado → '{ARQUIVO_STATUS_SITE}' "
                f"(livres={vagas_livres} ocupadas={vagas_ocupadas} "
                f"incertas={vagas_incertas})"
            )
            return True
        except OSError as e:
            logger.error(f"Erro ao exportar JSON: {e}")
            return False


# =============================================================================
# MÓDULO 8: MOTOR PRINCIPAL DE ANÁLISE
# =============================================================================
class MotorAnalise:
    """
    Orquestra todos os módulos e executa o pipeline completo de análise
    para uma imagem do estacionamento.
    """

    def __init__(self, imagem_referencia: np.ndarray):
        self.referencia         = imagem_referencia
        self.gerenciador        = GerenciadorImagens()
        self.preprocessador     = Preprocessador()
        self.extrator_roi       = ExtratoreROI()
        self.analisador_dif     = AnalisadorDiferenca()
        self.analisador_bordas  = AnalisadorBordas()
        self.analisador_textura = AnalisadorTextura()
        self.decisao            = TomadorDecisao()
        self.visualizador       = Visualizador()
        modo = "SEM referência" if MODO_SEM_REFERENCIA else "COM referência"
        logger.info(f"Motor de análise inicializado — modo: {modo}.")

    def analisar_vaga(self, id_vaga: str, coordenadas: list,
                      imagem_atual: np.ndarray) -> ResultadoVaga:
        """
        Pipeline completo para uma única vaga:
          1. Recorta ROI da imagem atual (e da referência, se disponível)
          2. Normaliza iluminação com CLAHE
          3. Calcula score de diferença com referência (se modo COM ref)
          4. Calcula densidade de bordas Canny
          5. Calcula variância de textura
          6. Combina os scores — retorna ResultadoVaga sem status final
             (o status é definido em analisar_estacionamento após calibração)
        """
        resultado = ResultadoVaga(id_vaga=id_vaga, coordenadas=coordenadas)

        # Recorte da ROI atual
        roi_atual = self.extrator_roi.recortar(imagem_atual, coordenadas)
        if roi_atual is None:
            logger.warning(f"Vaga {id_vaga}: ROI inválida, marcando como INCERTA.")
            resultado.status = "INCERTA"
            return resultado

        # Normalização de iluminação
        roi_atual_norm = self.preprocessador.normalizar_roi(roi_atual)

        # Score de diferença com referência (zerado se modo SEM referência)
        if not MODO_SEM_REFERENCIA and self.referencia is not None:
            roi_ref = self.extrator_roi.recortar(self.referencia, coordenadas)
            if roi_ref is not None:
                roi_ref_norm = self.preprocessador.normalizar_roi(roi_ref)
                resultado.score_diferenca = self.analisador_dif.calcular_diferenca(
                    roi_atual_norm, roi_ref_norm
                )

        # Score de bordas Canny
        resultado.score_bordas = self.analisador_bordas.calcular_densidade_bordas(
            roi_atual_norm
        )

        # Score de textura (variância)
        resultado.score_textura = self.analisador_textura.calcular_variancia(
            roi_atual_norm
        )

        # Score combinado ponderado (sem status ainda — definido depois)
        resultado.score_final = self.decisao.calcular_score_final(
            resultado.score_diferenca,
            resultado.score_bordas,
            resultado.score_textura
        )

        logger.debug(
            f"  Vaga {id_vaga}: dif={resultado.score_diferenca:.3f} | "
            f"bord={resultado.score_bordas:.3f} | "
            f"tex={resultado.score_textura:.3f} | "
            f"final={resultado.score_final:.3f}"
        )

        return resultado

    def analisar_estacionamento(self, imagem_atual: np.ndarray) -> list[ResultadoVaga]:
        """
        Analisa todas as vagas e aplica calibração automática de limiares.

        Fluxo:
          1. Calcula os scores de todas as vagas (sem decidir ainda)
          2. Se USAR_LIMIAR_AUTOMATICO: deriva limiares pela distribuição
             dos scores para separar naturalmente vagas cheias de vazias
          3. Aplica os limiares e define o status final de cada vaga
        """
        logger.info(f"Iniciando análise de {len(COORDENADAS_VAGAS)} vagas...")
        resultados = []

        # Passo 1: calcula todos os scores
        for id_vaga, coordenadas in COORDENADAS_VAGAS.items():
            resultado = self.analisar_vaga(id_vaga, coordenadas, imagem_atual)
            resultados.append(resultado)

        # Passo 2: define limiares (automático ou fixo)
        # Usa todos os scores calculados (status ainda é "INCERTA" por padrão —
        # não filtra por status aqui para não criar deadlock)
        scores_todos = [r.score_final for r in resultados]

        if USAR_LIMIAR_AUTOMATICO and len(scores_todos) >= 3:
            lim_livre, lim_ocupado = self.decisao.calibrar_limiares_automaticos(
                scores_todos
            )
        else:
            lim_livre, lim_ocupado = LIMIAR_LIVRE, LIMIAR_OCUPADO
            logger.info(f"Limiares fixos → LIVRE ≤ {lim_livre} | OCUPADA ≥ {lim_ocupado}")

        # Passo 3: classifica TODAS as vagas com os limiares definidos
        # (ROIs inválidas já têm score_final=0.0 e serão marcadas como LIVRE,
        #  mas também logamos um aviso se a ROI foi inválida)
        for resultado in resultados:
            resultado.status = self.decisao.classificar(
                resultado.score_final, lim_livre, lim_ocupado
            )

        livres   = sum(1 for r in resultados if r.status == "LIVRE")
        ocupadas = sum(1 for r in resultados if r.status == "OCUPADA")
        incertas = sum(1 for r in resultados if r.status == "INCERTA")
        logger.info(
            f"Análise concluída → Livres: {livres} | "
            f"Ocupadas: {ocupadas} | Incertas: {incertas}"
        )

        return resultados


# =============================================================================
# MÓDULO 9: LOOP PRINCIPAL (CICLO DE ANÁLISE)
# =============================================================================
def loop_principal():
    """
    Loop contínuo que:
      1. Garante que PASTA_ENTRADA existe
      2. A cada ciclo de INTERVALO_ANALISE_MINUTOS:
         a. Lista arquivos .jpg/.jpeg em PASTA_ENTRADA (ordem alfabética)
         b. Se vazia → loga e aguarda
         c. Se houver fotos → processa APENAS A PRIMEIRA (mais antiga),
            salva resultado visual, exporta status_vagas.json, deleta a foto
      3. Aguarda o intervalo completo antes do próximo ciclo
    """
    logger.info("=" * 60)
    logger.info("   SMART PARKING v2.0 - Sistema de Detecção de Vagas")
    logger.info("=" * 60)

    # Garante que a pasta de entrada existe
    os.makedirs(PASTA_ENTRADA, exist_ok=True)
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    logger.info(f"Pasta de entrada : {os.path.abspath(PASTA_ENTRADA)}")
    logger.info(f"Pasta de saída   : {os.path.abspath(PASTA_RESULTADOS)}")
    logger.info(f"JSON do site     : {os.path.abspath(ARQUIVO_STATUS_SITE)}")

    # Carrega imagem de referência (uma única vez)
    logger.info(f"Carregando referência: {ARQUIVO_REFERENCIA}")
    referencia = GerenciadorImagens.carregar_imagem(ARQUIVO_REFERENCIA)
    if referencia is None:
        logger.error(
            f"ERRO CRÍTICO: '{ARQUIVO_REFERENCIA}' não encontrado! "
            "Coloque a imagem do estacionamento vazio na pasta do projeto."
        )
        return

    logger.info(
        f"Referência carregada: {referencia.shape[1]}x{referencia.shape[0]} pixels"
    )
    logger.info(f"Vagas configuradas   : {list(COORDENADAS_VAGAS.keys())}")
    logger.info(f"Intervalo de ciclo   : {INTERVALO_ANALISE_MINUTOS} minuto(s)")
    logger.info("-" * 60)

    motor        = MotorAnalise(referencia)
    gerenciador  = GerenciadorImagens()
    visualizador = Visualizador()
    exportador   = ExportadorJSON()
    ciclo        = 0

    while True:
        ciclo += 1

        # --- Lista fotos pendentes (jpg e jpeg, ordem alfabética) ----------
        padroes = (
            glob.glob(os.path.join(PASTA_ENTRADA, "*.jpg"))  +
            glob.glob(os.path.join(PASTA_ENTRADA, "*.jpeg")) +
            glob.glob(os.path.join(PASTA_ENTRADA, "*.JPG"))  +
            glob.glob(os.path.join(PASTA_ENTRADA, "*.JPEG"))
        )
        fotos_pendentes = sorted(set(padroes))  # set remove duplicatas de case

        logger.info(
            f"[Ciclo #{ciclo}] {len(fotos_pendentes)} foto(s) em '{PASTA_ENTRADA}'."
        )

        if not fotos_pendentes:
            logger.info(
                f"  Pasta vazia. "
                f"Próximo ciclo em {INTERVALO_ANALISE_MINUTOS} minuto(s)."
            )
        else:
            # Pega APENAS a primeira foto (a mais antiga pela ordem alfabética)
            caminho_foto = fotos_pendentes[0]
            nome_foto    = os.path.basename(caminho_foto)
            logger.info(f"  Processando: {nome_foto}")

            foto = gerenciador.carregar_imagem(caminho_foto)
            if foto is None:
                logger.error(f"  Não foi possível ler '{nome_foto}'. Deletando.")
                gerenciador.deletar_imagem(caminho_foto)
            else:
                logger.info(
                    f"  Dimensões: {foto.shape[1]}x{foto.shape[0]} pixels"
                )

                # Normaliza tamanho para o da referência
                foto = gerenciador.redimensionar_para_referencia(foto, referencia)

                # ── Análise completa ──────────────────────────────────────
                resultados = motor.analisar_estacionamento(foto)

                # ── Exporta JSON para o site ──────────────────────────────
                exportador.exportar(resultados)

                # ── Salva imagem de resultado visual ──────────────────────
                nome_base    = os.path.splitext(nome_foto)[0]
                timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
                nome_saida   = f"resultado_{nome_base}_{timestamp}.jpg"
                caminho_saida = os.path.join(PASTA_RESULTADOS, nome_saida)
                cv2.imwrite(
                    caminho_saida,
                    visualizador.desenhar_vagas(foto, resultados),
                    [cv2.IMWRITE_JPEG_QUALITY, 92]
                )
                logger.info(f"  Resultado salvo: {nome_saida}")

                # ── Deleta a foto processada da pasta de entrada ──────────
                gerenciador.deletar_imagem(caminho_foto)

                restantes = len(fotos_pendentes) - 1
                if restantes > 0:
                    logger.info(
                        f"  {restantes} foto(s) ainda pendente(s) — "
                        "serão processadas nos próximos ciclos."
                    )

        # Aguarda o intervalo completo antes do próximo ciclo
        logger.info(
            f"[Ciclo #{ciclo}] Aguardando {INTERVALO_ANALISE_MINUTOS} minuto(s)..."
        )
        time.sleep(INTERVALO_ANALISE_SEGUNDOS)


# =============================================================================
# MODO DE TESTE RÁPIDO (sem loop, para validação manual)
# =============================================================================
def modo_teste_rapido(caminho_foto: str):
    """
    Executa UMA análise diretamente em uma imagem específica.
    Ideal para testar o algoritmo sem esperar o loop.
    Também exporta o status_vagas.json.

    Uso:
        python smart_parking.py --teste minha_foto.jpg
    """
    logger.info(f"=== MODO TESTE RÁPIDO: {caminho_foto} ===")

    referencia = GerenciadorImagens.carregar_imagem(ARQUIVO_REFERENCIA)
    if referencia is None:
        logger.error(f"Referência '{ARQUIVO_REFERENCIA}' não encontrada!")
        return

    foto = GerenciadorImagens.carregar_imagem(caminho_foto)
    if foto is None:
        logger.error(f"Foto de teste '{caminho_foto}' não encontrada!")
        return

    foto = GerenciadorImagens.redimensionar_para_referencia(foto, referencia)

    motor        = MotorAnalise(referencia)
    visualizador = Visualizador()
    exportador   = ExportadorJSON()

    resultados       = motor.analisar_estacionamento(foto)
    exportador.exportar(resultados)
    imagem_resultado = visualizador.desenhar_vagas(foto, resultados)

    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_saida   = os.path.join(
        PASTA_RESULTADOS,
        f"resultado_{os.path.splitext(os.path.basename(caminho_foto))[0]}_{timestamp}.jpg"
    )
    cv2.imwrite(nome_saida, imagem_resultado, [cv2.IMWRITE_JPEG_QUALITY, 92])
    logger.info(f"Resultado salvo em: {nome_saida}")
    logger.info("Pressione qualquer tecla para fechar a janela...")
    cv2.imshow("Smart Parking - Teste", imagem_resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# MÓDULO 10: CALIBRADOR INTERATIVO DE VAGAS
# =============================================================================
class CalibradorVagas:
    """
    Abre a imagem de referência e permite ao usuário desenhar as vagas
    com o mouse (clique e arraste). As coordenadas são salvas em JSON
    e carregadas automaticamente nas próximas execuções.

    CONTROLES DA JANELA DE CALIBRAÇÃO:
      Clique e arraste   → desenha um retângulo de vaga
      Z                  → desfaz a última vaga desenhada
      C                  → limpa todas as vagas e recomeça
      ENTER / S          → salva e encerra a calibração
      ESC                → cancela sem salvar
    """

    # Cores usadas apenas na tela de calibração
    _COR_RASCUNHO  = (255, 200, 0)    # Ciano — retângulo sendo desenhado
    _COR_SALVO     = (0, 220, 80)     # Verde — vagas já confirmadas
    _COR_LABEL     = (255, 255, 255)  # Branco — texto do ID
    _COR_OVERLAY   = (20, 20, 20)     # Painel de instruções

    def __init__(self):
        self._vagas: dict         = {}   # {ID: [x, y, w, h]}
        self._contador: int       = 0    # Próximo número de vaga
        self._pt_inicio           = None # Ponto de início do clique
        self._pt_fim              = None # Ponto atual do mouse
        self._desenhando: bool    = False
        self._imagem_base         = None # Cópia limpa da referência
        self._canvas              = None # Imagem exibida (com desenhos)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def calibrar(self) -> dict:
        """
        Abre a janela de calibração.
        Retorna o dicionário de vagas desenhadas (pode estar vazio se cancelado).
        """
        referencia = GerenciadorImagens.carregar_imagem(ARQUIVO_REFERENCIA)
        if referencia is None:
            logger.error(
                f"Calibração cancelada: '{ARQUIVO_REFERENCIA}' não encontrado.\n"
                "Coloque a imagem do estacionamento vazio na pasta do projeto."
            )
            return {}

        # Redimensiona para caber na tela (máximo 1280×720) mantendo proporção
        referencia = self._ajustar_para_tela(referencia, max_w=1280, max_h=720)

        self._imagem_base = referencia.copy()
        self._canvas      = referencia.copy()
        self._vagas       = {}
        self._contador    = 0

        nome_janela = "Smart Parking — Calibracao de Vagas"
        cv2.namedWindow(nome_janela, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(nome_janela, self._callback_mouse)

        logger.info("=" * 60)
        logger.info("  MODO CALIBRAÇÃO INTERATIVA")
        logger.info("  Clique e arraste para marcar cada vaga.")
        logger.info("  Z = desfaz  |  C = limpa  |  ENTER/S = salvar  |  ESC = cancelar")
        logger.info("=" * 60)

        while True:
            self._renderizar(nome_janela)
            tecla = cv2.waitKey(20) & 0xFF

            if tecla in (13, ord('s'), ord('S')):   # ENTER ou S → salvar
                cv2.destroyWindow(nome_janela)
                if self._vagas:
                    self._salvar_json(self._vagas)
                    logger.info(f"Calibração salva: {len(self._vagas)} vaga(s).")
                else:
                    logger.warning("Nenhuma vaga foi desenhada. Calibração vazia.")
                return self._vagas

            elif tecla == 27:                        # ESC → cancelar
                cv2.destroyWindow(nome_janela)
                logger.info("Calibração cancelada pelo usuário.")
                return {}

            elif tecla in (ord('z'), ord('Z')):      # Z → desfaz última
                if self._vagas:
                    ultimo_id = list(self._vagas.keys())[-1]
                    del self._vagas[ultimo_id]
                    self._contador = max(0, self._contador - 1)
                    logger.info(f"Vaga {ultimo_id} removida.")
                    self._redesenhar_canvas()

            elif tecla in (ord('c'), ord('C')):      # C → limpa tudo
                self._vagas    = {}
                self._contador = 0
                self._redesenhar_canvas()
                logger.info("Todas as vagas foram limpas.")

    @staticmethod
    def carregar_json() -> dict:
        """
        Carrega o arquivo JSON de calibração salvo anteriormente.
        Retorna dicionário vazio se o arquivo não existir.
        """
        if not os.path.exists(ARQUIVO_CALIBRACAO):
            return {}
        try:
            with open(ARQUIVO_CALIBRACAO, "r", encoding="utf-8") as f:
                dados = json.load(f)
            logger.info(
                f"Calibração carregada de '{ARQUIVO_CALIBRACAO}': "
                f"{len(dados)} vaga(s)."
            )
            return dados
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Erro ao carregar calibração: {e}")
            return {}

    # ------------------------------------------------------------------
    # Internos — mouse
    # ------------------------------------------------------------------

    def _callback_mouse(self, evento, x, y, flags, param):
        """Gerencia os eventos do mouse para desenhar retângulos."""

        if evento == cv2.EVENT_LBUTTONDOWN:
            # Inicia o desenho
            self._desenhando = True
            self._pt_inicio  = (x, y)
            self._pt_fim     = (x, y)

        elif evento == cv2.EVENT_MOUSEMOVE and self._desenhando:
            # Atualiza o ponto final enquanto arrasta
            self._pt_fim = (x, y)

        elif evento == cv2.EVENT_LBUTTONUP:
            # Finaliza o retângulo
            self._desenhando = False
            self._pt_fim     = (x, y)

            # Normaliza para garantir x1 < x2, y1 < y2
            x1 = min(self._pt_inicio[0], self._pt_fim[0])
            y1 = min(self._pt_inicio[1], self._pt_fim[1])
            x2 = max(self._pt_inicio[0], self._pt_fim[0])
            y2 = max(self._pt_inicio[1], self._pt_fim[1])
            w  = x2 - x1
            h  = y2 - y1

            # Ignora retângulos muito pequenos (clique acidental)
            if w > 10 and h > 10:
                self._contador += 1
                id_vaga = f"V{self._contador:02d}"
                self._vagas[id_vaga] = [x1, y1, w, h]
                logger.info(f"  Vaga {id_vaga} marcada: x={x1} y={y1} w={w} h={h}")
                self._redesenhar_canvas()

            self._pt_inicio = None
            self._pt_fim    = None

    # ------------------------------------------------------------------
    # Internos — renderização
    # ------------------------------------------------------------------

    def _renderizar(self, nome_janela: str):
        """Compõe o frame final e exibe na janela."""
        frame = self._canvas.copy()

        # Retângulo de rascunho (sendo desenhado agora)
        if self._desenhando and self._pt_inicio and self._pt_fim:
            cv2.rectangle(frame, self._pt_inicio, self._pt_fim,
                          self._COR_RASCUNHO, 2)

        # Painel de instruções (canto inferior)
        self._desenhar_painel_instrucoes(frame)

        cv2.imshow(nome_janela, frame)

    def _redesenhar_canvas(self):
        """Reconstrói o canvas a partir da imagem base + vagas salvas."""
        self._canvas = self._imagem_base.copy()
        for id_vaga, (x, y, w, h) in self._vagas.items():
            # Retângulo da vaga confirmada
            cv2.rectangle(self._canvas,
                          (x, y), (x + w, y + h),
                          self._COR_SALVO, 2)
            # Fundo semitransparente do label
            overlay = self._canvas.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + 22), self._COR_SALVO, -1)
            cv2.addWeighted(overlay, 0.50, self._canvas, 0.50, 0, self._canvas)
            # ID da vaga
            cv2.putText(
                self._canvas, id_vaga,
                (x + 4, y + 15),
                FONTE, 0.50, self._COR_LABEL, 1, cv2.LINE_AA
            )

    def _desenhar_painel_instrucoes(self, frame: np.ndarray):
        """Desenha o painel de atalhos na parte inferior da imagem."""
        h, w = frame.shape[:2]
        painel_h = 30
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, h - painel_h), (w, h),
                      self._COR_OVERLAY, -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        instrucoes = (
            f"Vagas: {len(self._vagas)}   |   "
            "Clique+arraste=marcar   Z=desfaz   C=limpa   "
            "ENTER/S=salvar   ESC=cancelar"
        )
        cv2.putText(
            frame, instrucoes,
            (10, h - 9),
            FONTE, 0.44, (200, 200, 200), 1, cv2.LINE_AA
        )

    # ------------------------------------------------------------------
    # Internos — utilitários
    # ------------------------------------------------------------------

    @staticmethod
    def _ajustar_para_tela(imagem: np.ndarray,
                           max_w: int, max_h: int) -> np.ndarray:
        """Redimensiona a imagem para caber na tela mantendo a proporção."""
        h, w = imagem.shape[:2]
        escala = min(max_w / w, max_h / h, 1.0)  # nunca aumenta
        if escala < 1.0:
            novo_w = int(w * escala)
            novo_h = int(h * escala)
            imagem = cv2.resize(imagem, (novo_w, novo_h),
                                interpolation=cv2.INTER_AREA)
            logger.info(
                f"Imagem redimensionada para exibição: {novo_w}x{novo_h} "
                f"(escala {escala:.2f})"
            )
        return imagem

    @staticmethod
    def _salvar_json(vagas: dict):
        """Persiste as coordenadas das vagas em arquivo JSON."""
        try:
            with open(ARQUIVO_CALIBRACAO, "w", encoding="utf-8") as f:
                json.dump(vagas, f, indent=2, ensure_ascii=False)
            logger.info(f"Coordenadas salvas em '{ARQUIVO_CALIBRACAO}'.")
        except OSError as e:
            logger.error(f"Erro ao salvar calibração: {e}")


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    import sys

    # ------------------------------------------------------------------
    # Resolve as coordenadas das vagas antes de qualquer modo de execução.
    # Prioridade:
    #   1. COORDENADAS_VAGAS preenchido manualmente no código
    #   2. Arquivo vagas_calibradas.json (calibração anterior)
    #   3. Abre a calibração interativa automaticamente
    # ------------------------------------------------------------------
    calibrador = CalibradorVagas()

    if not COORDENADAS_VAGAS:
        vagas_json = CalibradorVagas.carregar_json()

        if vagas_json:
            COORDENADAS_VAGAS.update(vagas_json)
        else:
            logger.info(
                "Nenhuma vaga configurada. Abrindo calibração interativa..."
            )
            vagas_novas = calibrador.calibrar()
            if not vagas_novas:
                logger.error(
                    "Calibração vazia ou cancelada. "
                    "O sistema não pode rodar sem vagas definidas."
                )
                sys.exit(1)
            COORDENADAS_VAGAS.update(vagas_novas)

    # ------------------------------------------------------------------
    # Modos de execução
    # ------------------------------------------------------------------
    if len(sys.argv) == 2 and sys.argv[1] == "--calibrar":
        # Força re-calibração mesmo que já exista JSON salvo
        # Uso: python smart_parking.py --calibrar
        vagas_novas = calibrador.calibrar()
        if vagas_novas:
            COORDENADAS_VAGAS.clear()
            COORDENADAS_VAGAS.update(vagas_novas)
            logger.info("Calibração concluída. Execute sem argumentos para iniciar.")
        sys.exit(0)

    elif len(sys.argv) == 3 and sys.argv[1] == "--teste":
        # Teste rápido em uma foto específica (sem loop, sem deletar)
        # Uso: python smart_parking.py --teste foto_com_carros.jpg
        modo_teste_rapido(sys.argv[2])

    else:
        # Loop contínuo de produção
        # Uso: python smart_parking.py
        try:
            loop_principal()
        except KeyboardInterrupt:
            logger.info("\nSistema encerrado pelo usuário (Ctrl+C).")
            cv2.destroyAllWindows()