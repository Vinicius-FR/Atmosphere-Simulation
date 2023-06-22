import numpy as np
import matplotlib.pyplot as plt

# Parâmetros da simulação
num_particulas = 1000  # Número de partículas na simulação
dimensao = 2  # Dimensão do espaço (2D ou 3D)
num_passos = 1000  # Número de passos da simulação
dt = 0.1  # Intervalo de tempo entre os passos da simulação

# Parâmetros físicos
gravidade = np.array([0, -9.8])  # Vetor de gravidade
coef_restituicao = 1  # Coeficiente de restituição para colisões

# Limites da região de simulação
limite_x = [0, 10]
limite_y = [0, 10]

# Inicialização das posições e velocidades
posicoes = np.random.uniform(low=[limite_x[0], limite_y[0]], high=[limite_x[1], limite_y[1]], size=(num_particulas, dimensao))
velocidades = np.zeros((num_particulas, dimensao))

# Simulação
for t in range(num_passos):
    # Cálculo da força de gravidade em cada partícula
    forca_gravidade = gravidade
    
    # Cálculo da força de agitação aleatória em cada partícula
    forca_agitacao = np.random.randn(num_particulas, dimensao)
    
    # Cálculo da aceleração resultante em cada partícula
    aceleracoes = forca_gravidade + forca_agitacao
    
    # Atualização das velocidades e posições usando o método de Euler
    velocidades += aceleracoes * dt
    posicoes += velocidades * dt
    
    # Verificação e tratamento de colisões com as paredes
    for i in range(num_particulas):
        for j in range(dimensao):
            if posicoes[i, j] < limite_x[0]:
                posicoes[i, j] = limite_x[0]
                velocidades[i, j] *= -coef_restituicao
            elif posicoes[i, j] > limite_x[1]:
                posicoes[i, j] = limite_x[1]
                velocidades[i, j] *= -coef_restituicao
            if posicoes[i, j] < limite_y[0]:
                posicoes[i, j] = limite_y[0]
                velocidades[i, j] *= -coef_restituicao
            elif posicoes[i, j] > limite_y[1]:
                posicoes[i, j] = limite_y[1]
                velocidades[i, j] *= -coef_restituicao
    
    # Verificação e tratamento de colisões entre as partículas
    for i in range(num_particulas):
        for j in range(i+1, num_particulas):
            distancia = np.linalg.norm(posicoes[i] - posicoes[j])
            if distancia < 1.0:  # Defina o limite de distância para detectar colisões
                # Cálculo do vetor de direção entre as partículas
                if distancia == 0:
                    direcao = np.zeros(dimensao)
                else:
                    direcao = (posicoes[j] - posicoes[i]) / distancia

                
                # Cálculo das velocidades relativas entre as partículas
                velocidade_relativa = np.dot(velocidades[j] - velocidades[i], direcao)
                
                # Cálculo das forças de colisão
                forca_colisao = coef_restituicao * velocidade_relativa * direcao
                
                # Atualização das velocidades
                velocidades[i] += forca_colisao
                velocidades[j] -= forca_colisao
    
    # Visualização da simulação
    plt.figure()
    plt.scatter(posicoes[:, 0], posicoes[:, 1])
    plt.xlim(limite_x[0], limite_x[1])
    plt.ylim(limite_y[0], limite_y[1])
    plt.xlabel('Posição X')
    plt.ylabel('Posição Y')
    plt.title('Simulação da Atmosfera - Passo {}'.format(t+1))
    plt.show()
