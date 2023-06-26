import numpy as np
import matplotlib.pyplot as plt

# Parâmetros da simulação
num_particulas = 1000  # Número de partículas
dimensao = 2  # Dimensão do espaço (2D ou 3D)
num_passos = 1000  # Número de passos da simulação
dt = 0.1  # Intervalo de tempo entre os passos

# Parâmetros físicos
gravidade = np.array([0, -9.8])  # Vetor gravidade
coef_restituicao = 1  # Coeficiente de restituição para colisões

# Limites da região
limite_x = [0, 10]
limite_y = [0, 10]

# Inicialização das posições e velocidades
posicoes = np.random.uniform(low=[limite_x[0], limite_y[0]], high=[limite_x[1], limite_y[1]], size=(num_particulas, dimensao))
velocidade_magnitude = 10.0  # Valor da magnitude da velocidade
phi = np.random.uniform(0, 2*np.pi, size=num_particulas)  # Ângulos aleatórios
theta = np.random.uniform(0, np.pi, size=num_particulas)
velocidades = velocidade_magnitude * np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi))).T

# Simulação
for t in range(num_passos):
    # Cálculo da aceleração devido a uma agitação aleatória em cada partícula
    agitacao = 10.0 * np.random.randn(num_particulas, dimensao)
    
    # Cálculo da aceleração resultante em cada partícula
    aceleracao = gravidade + agitacao
    
    # Atualização das velocidades e posições usando o método de Euler
    velocidades += aceleracao * dt
    posicoes += velocidades * dt
    
    # Verificação e tratamento de colisões com as paredes
    for i in range(num_particulas):
        for j in range(dimensao):
            if posicoes[i, 0] < limite_x[0]:
                posicoes[i, 0] = limite_x[0]
                velocidades[i, 0] *= -coef_restituicao
            elif posicoes[i, 0] > limite_x[1]:
                posicoes[i, 0] = limite_x[1]
                velocidades[i, 0] *= -coef_restituicao
            if posicoes[i, 1] < limite_y[0]:
                posicoes[i, 1] = limite_y[0]
                velocidades[i, 1] *= -coef_restituicao
            elif posicoes[i, 1] > limite_y[1]:
                posicoes[i, 1] = limite_y[1]
                velocidades[i, 1] *= -coef_restituicao
    
    # Verificação e tratamento de colisões entre as partículas
    for i in range(num_particulas):
        for j in range(i+1, num_particulas):
            distancia = np.linalg.norm(posicoes[i] - posicoes[j])
            if distancia < 0.1:  # Limite de distância para detectar colisões (depende da dimensão e quantidade de partículas)
                # Cálculo do versor de direção entre as partículas
                if distancia == 0:
                    direcao = np.zeros(dimensao)
                else:
                    direcao = (posicoes[j] - posicoes[i]) / distancia

                
                # Cálculo das velocidades relativas entre as partículas
                velocidade_relativa = np.dot(velocidades[j] - velocidades[i], direcao)
                
                # Cálculo da variação de momento/velocidade devido à colisão
                forca_colisao = coef_restituicao * velocidade_relativa * direcao
                
                # Atualização das velocidades
                velocidades[i] += forca_colisao
                velocidades[j] -= forca_colisao
    
    # Visualização da simulação
    if t % 50 == 0:
        plt.figure()
        plt.scatter(posicoes[:, 0], posicoes[:, 1])
        plt.xlim(limite_x[0], limite_x[1])
        plt.ylim(limite_y[0], limite_y[1])
        plt.xlabel('Posição X')
        plt.ylabel('Posição Y')
        plt.title('Simulação da Atmosfera - Passo {}'.format(t+1))
        plt.show()