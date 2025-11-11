# savi-2025-2026-trabalho1-grupoX

## Tarefa 1
A tarefa 1 consiste em criar duas nuvens de pontos (uma fonte e um alvo), cada uma a partir de uma imagem "típica" (que contém os canais correspondentes às cordenadas x e y e à cor) e de uma imagem que contém os valores de profundidade. Para alinhar estas duas nuvens de pontos usa-se o método PointToPlane, uma vez que é mais eficiente e produz melhores resultados que o método PointToPoint para nuvens de pontos mais complexas. Isto porque o método PointToPoint tenta reduzir a distância entre os pontos da fonte e do alvo, o que funciona bem para nuvens de pontos simples e com pouco ruído. Já o PointToPlane procura diminuir a distância entre os pontos da fonte e os planos tangentes dos pontos do alvo correspondentes.

Para tal é necessário:
1. Carregar e filtrar as duas imagens (imagem rgb e imagem profundidade) para a criação de cada uma das nuvens de pontos;
   - Carregar a imagem rgb lembrando de alterar de bgr para rgb usando cv2.COLOR_BGR2RGB;
   - Carregar a imagem profunidade mantendo o número de bits (16 bits ao invés dos 8 bits típicos) usando o argumento cv2.IMREAD_UNCHANGED;
   - Aplicar um filtro de mediana à imagem profundidade para reduzir o ruído (cv2.medianBlur);
   - Converter a imagem profundidade para o tipo float32 e para metros, assumindo um fator de ampliação de 5000;
   - Remover valores de profundidade inválidos (inferiores a 0.1 m e superiores a 3 m).
2. Criação das nuvens de pontos fonte e alvo usando o Open3D;
   - Criar as imagens rgbd (imagens com cor e profundidade) através do open3d.geometry.RGBDImage.create_from_color_and_depth. Esta função tem como parâmetros (...)
- Pré-processamento realizando um down_sample e uma estimativa de normais definindo uma superfície local de 30 pontos no máximo num raio de 5 cm;
- Através da visualização das nuvens de pontos sobrepostas antes de qualquer transformação, estimou-se uma translação de 80 cm no eixo x e -5 cm nos eixos y e z, assim como uma rotação de -10º em torno de y e -5º em torno de z;
- Aplicação do ICP com o parâmetro Point to Plane usando como transformação inicial a matriz resultante das transformações estimadas anteriormente;
- Realização de 20 iterações, com um threshold = 0.02 e um número máximo de 500 iterações para convergir;
- Visualização do melhor resultado.

Os parâmetros como número de iterações, threshold, valores a atribuir no down_sample e nos filtros usados foram estimados por tentativa e erro. A matriz de transformação inicial foi calculada por IA tendo por base a estimativa feita, como solução alternativa ao uso do cloud compare.

Seguem abaixo algumas imagens do resultado obtido:

<p float="left">

 <img src="https://github.com/user-attachments/assets/94c1618c-a598-4b2a-b352-b8fa25dfc2d6" width="400" />
 <img src="https://github.com/user-attachments/assets/70e0b708-aba7-4ee9-9339-9a8175d4e054" width="400" /> 
 <img src="https://github.com/user-attachments/assets/0e3490ee-9695-4137-9bda-f5c5550a079d" width="400" />
 <img src="https://github.com/user-attachments/assets/0d0eddb9-a63c-49f1-b680-205ae1228837" width="400" />

</p>

