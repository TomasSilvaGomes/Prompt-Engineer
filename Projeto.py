import os
import random
import re

import pandas as pd

# Diretório com as imagens
diretoria = "both_eyes"

# Carrega os ficheiros
ficheiros_tv = [f for f in os.listdir(diretoria) if f.endswith('.jpg') and 'S2' not in f]  # S1
ficheiros_teste = [f for f in os.listdir(diretoria) if f.endswith('.jpg') and 'S2' in f]  # S2

# Número total de comparações
comparacoes_total = 10000
positivos_necessarios = comparacoes_total // 2
negativos_necessarios = comparacoes_total // 2

# Distribuição de splits
split_distrib = {
    0: int(comparacoes_total * 0.7),  # treino
    1: int(comparacoes_total * 0.15),  # validação
    2: int(comparacoes_total * 0.15),  # teste
}

# Contadores
split_counters = {0: 0, 1: 0, 2: 0}
positivos = 0
negativos = 0

# Função para extrair ID da pessoa (Cxx)
def get_id_pessoa(ficheiro):
    match = re.match(r'(C\d+)', ficheiro)
    return match.group(1) if match else None

# Função para atribuir split com base no tipo da imagem (S1 ou S2)
def get_split(img1, img2):
    if 'S2' in img1 or 'S2' in img2:
        if 'S2' in img1 and 'S2' in img2:
            return 2  # Teste
        else:
            return None  # Não pode comparar S1 com S2
    else:
        return random.choice([0, 1])  # Treino ou Validação, aleatoriamente

# Geração das comparações
comparacoes_final = []

while len(comparacoes_final) < comparacoes_total:
    if positivos < positivos_necessarios and negativos < negativos_necessarios:
        tipo = random.choice([0, 1])  # Escolher aleatoriamente entre positivo e negativo
    elif positivos < positivos_necessarios:
        tipo = 1  # Gerar mais positivos
    else:
        tipo = 0  # Gerar mais negativos

    if tipo == 1:
        img1, img2 = random.sample(ficheiros_tv, 2) if random.random() < 0.5 else random.sample(ficheiros_teste, 2)
        while get_id_pessoa(img1) != get_id_pessoa(img2):
            img1, img2 = random.sample(ficheiros_tv, 2) if random.random() < 0.5 else random.sample(ficheiros_teste, 2)
    else:
        img1 = random.choice(ficheiros_tv) if random.random() < 0.5 else random.choice(ficheiros_teste)
        img2 = random.choice(ficheiros_tv) if random.random() < 0.5 else random.choice(ficheiros_teste)
        while get_id_pessoa(img1) == get_id_pessoa(img2):
            img2 = random.choice(ficheiros_tv) if random.random() < 0.5 else random.choice(ficheiros_teste)

    label = tipo
    split = get_split(img1, img2)
    if split is None or split_counters[split] >= split_distrib[split]:
        continue

    comparacoes_final.append((img1, img2, label, split))
    split_counters[split] += 1
    if label == 1:
        positivos += 1
    else:
        negativos += 1

# Criar DataFrame e misturar aleatoriamente as instâncias
df = pd.DataFrame(comparacoes_final, columns=['img1', 'img2', 'identicas', 'fase'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Garantir que o número final de comparações seja exatamente 10.000
while len(df) < comparacoes_total:
    falta = comparacoes_total - len(df)
    amostras = df.sample(n=falta, replace=True)
    df = pd.concat([df, amostras], ignore_index=True)

# Gravação do CSV misturado
csv_path = "comparacoes_10000_shuffled.csv"
df.to_csv(csv_path, index=False)

# Verificação final
print("✅ CSV criado e misturado com sucesso!")
print(f"Total comparações: {len(df)}")
print(f"Positivos: {df['identicas'].sum()}, Negativos: {len(df) - df['identicas'].sum()}")
