# Verificação Biométrica com CNN + Justificação com LLM

Este repositório contém o desenvolvimento de um sistema de verificação biométrica baseada em imagens oculares, implementado com redes neuronais convolucionais (CNN), complementado com justificações interpretáveis geradas por modelos de linguagem multimodal (LLM/VLM).

---

## Objetivos

### Parte 1

Desenvolver, com Python + Keras, um modelo de verificação biométrica que:
- Recebe pares de imagens oculares;
- Retorna 1 se forem da mesma pessoa, ou 0 se forem de pessoas diferentes.

### Parte 2 

Desenvolver um plano de interação entre o classificador anterior e um modelo de linguagem multimodal (LLM/VLM) à escolha, de modo a obter um sistema explicável:
- Que justifique a decisão da CNN;
- Com linguagem simples e acessível ao utilizador final.

---

## Tecnologias Utilizadas

- Python + Keras — Treino e avaliação da CNN
- OpenCV / PIL — Pré-processamento de imagens (tons de cinzento + filtros)
- CLIP (ViT-L/14) — Extração de embeddings visuais
- InternVL-Chat (via OpenRouter) — Justificação textual com base visual
- T-SNE / UMAP / K-Means — Redução de dimensionalidade e clustering
- APIs: OpenRouter, Hugging Face

---

## Arquitetura e Pipeline

1. Pré-processamento das imagens
   - Convertidas para tons de cinzento
   - Aplicado filtro de deteção de arestas

2. CNN
   - Recebe dois inputs (imagem 1 e imagem 2)
   - Decide: 0 (pessoas diferentes) ou 1 (mesma pessoa)

3. Justificação com LLM
   - Geração de embeddings com CLIP
   - Criação de prompt textual com base na previsão
   - Envio ao InternVL-Chat para obter justificação
   - Exemplo de prompt usado:
     > "As imagens são da mesma pessoa. Inicia a tua justificação apenas em tópicos, em português de Portugal, baseando-te nas características e forma dos olhos, na forma das sobrancelhas, na presença ou ausência de óculos..."

4. Análise de justificações
   - Aplicação de T-SNE/UMAP aos embeddings
   - Clustering com K-Means para agrupar justificações
   - Seleção da justificativa mais próxima ao centróide como representativa

---

## Resultados Obtidos

### Mesma Pessoa — Justificação Gerada

- Olhos: Formato, cor e posição semelhantes;
- Sobrancelhas: Espessura e alinhamento equivalentes;
- Óculos: Ausência em ambas;
- Cabelo: Posição e moldura facial idêntica.

O uso de prompts mais descritivos melhorou significativamente a qualidade das respostas.

### Pessoas Diferentes — Justificação Gerada

- Forma dos Olhos: Diferente entre imagens;
- Sobrancelhas: Uma mais curva e estilizada;
- Textura: Variações subtis na região ocular.

---

## Visualizações

Foram utilizados gráficos com:
- Projeções bidimensionais dos embeddings (via T-SNE / UMAP);
- Clusters resultantes de K-Means para distinguir tipos de justificações;
- Evidência de que as melhores respostas estavam mais próximas dos centróides.

---

## Conclusão

- O sistema desenvolvido realiza verificação biométrica robusta usando CNNs.
- Com o apoio de LLMs, é possível gerar explicações acessíveis, úteis para interpretabilidade.
- Técnicas de agrupamento (K-Means) e seleção automática de justificações melhoraram a clareza das respostas.
- Limitações encontradas:
  - Restrições de tokens da API
  - Necessidade de prompts bem estruturados para obter respostas úteis

---

## Referências

- CLIP - OpenAI  
- InternVL via OpenRouter  
- Sentence Transformers  
- Keras  
- T-SNE & UMAP

---

## Autor

Tomás Azevedo Jorge Silva Gomes  
Universidade da Beira Interior  
Curso: Inteligência Artificial e Ciência de Dados  
Disciplina: Interação com Modelos de Larga Escala  
Docente: Prof. Hugo Proença  
```
