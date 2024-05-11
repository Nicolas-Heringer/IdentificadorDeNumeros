import pygame
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model('meu_modelo.h5')

# Defina as constantes de tela
LARGURA, ALTURA = 640, 590
N = 28  # Tamanho do grid NxN
TAMANHO_DA_CELULA = ALTURA // N
LARGURA_CANVA_SECUNDARIO = 50
LARGURA_CANVA_PRIMARIO = LARGURA - LARGURA_CANVA_SECUNDARIO
LARGURA_CANVA_TRES = 640
ALTURA_CANVA_TRES = 200

# Circulos
RAIO_DO_CIRCULO = 20

# Inicialize o Pygame
pygame.init()
screen = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Desenho com Grid")

# Crie uma matriz para armazenar os valores do grid
grid = np.zeros((N, N), dtype=int)

# Defina a fonte
font = pygame.font.Font(None, 24)

# Crie um objeto Clock para controlar o framerate
clock = pygame.time.Clock()
framerate_desejado = 60  # Altere este valor para definir o framerate desejado

# Função para desenhar o grid no canvas principal
def draw_grid_primary():
    for i in range(N):
        for j in range(N):
            color = (0, 0, 0) if grid[i][j] else (200, 200, 50)
            pygame.draw.rect(
                screen, # Tela em que sera posicionado
                color, # Cor
                (j * TAMANHO_DA_CELULA, i * TAMANHO_DA_CELULA, TAMANHO_DA_CELULA, TAMANHO_DA_CELULA) # Posição e tamanho
            )

# Função para desenhar os círculos no canvas secundário
def draw_circles_secondary(previsao):
    lista_previsao=previsao.T
    minimo=min(lista_previsao)
    lista_previsao=lista_previsao-(minimo[0])
    maximo=max(lista_previsao)
    lista_previsao=lista_previsao/maximo[0]
    for i in range(10):
        circle_y = (i+0.5) * ALTURA/10
        pygame.draw.circle(
            screen, # Tela em que sera posicionado
            (255-(lista_previsao[i]*200),lista_previsao[i]*255, lista_previsao[i]*200), # Cor
            (LARGURA_CANVA_PRIMARIO + LARGURA_CANVA_SECUNDARIO // 2, circle_y), # Posição
            RAIO_DO_CIRCULO
        )
        text = font.render(str(i), True, (255-lista_previsao[i]*255, 255-lista_previsao[i]*255, 255-lista_previsao[i]*255))
        text_rect = text.get_rect(center=(LARGURA_CANVA_PRIMARIO + LARGURA_CANVA_SECUNDARIO // 2, circle_y))
        screen.blit(text, text_rect)

# Função para atualizar a matriz do grid com base na posição do mouse
def update_grid(mouse_pos, is_clicked):
    i, j = mouse_pos[1] // TAMANHO_DA_CELULA, mouse_pos[0] // TAMANHO_DA_CELULA
    if 0 <= i < N and 0 <= j < N:
        grid[i][j] = is_clicked

# Loop principal
running = True
is_clicked = False
clear_canvas = False  # Variável para verificar se o canvas precisa ser limpo
while running:
    screen.fill((0, 0, 0))
    
    # Limpar o canvas se necessário
    if clear_canvas:
        grid = np.zeros((N, N), dtype=int)
        clear_canvas = False  # Redefinir a variável para evitar limpezas repetidas

    draw_grid_primary()
    pygame.draw.rect(screen, (0, 0, 0), (LARGURA_CANVA_PRIMARIO, 0, LARGURA_CANVA_SECUNDARIO, ALTURA))
    grid_vetorizado = grid.reshape(-1,28*28).astype("float32")
    previsao = model.predict(grid_vetorizado)

    draw_circles_secondary(previsao)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            is_clicked = True
        elif event.type == pygame.MOUSEBUTTONUP:
            is_clicked = False
        elif event.type == pygame.MOUSEMOTION and is_clicked:
            update_grid(pygame.mouse.get_pos(), is_clicked)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Se a tecla 'c' for pressionada
                clear_canvas = True

    pygame.display.flip()

    clock.tick(framerate_desejado)

pygame.quit()
