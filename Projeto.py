import os
import re
import csv
import random

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
    0: int(comparacoes_total * 0.3),  # treino
    1: int(comparacoes_total * 0.2),  # validação
    2: int(comparacoes_total * 0.5),  # teste
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
    # Verificar se as imagens são do tipo S1 ou S2
    if 'S2' in img1 or 'S2' in img2:
        # Ambas devem ser S2 para serem comparadas
        if 'S2' in img1 and 'S2' in img2:
            return 2  # Teste
        else:
            return None  # Não pode comparar S1 com S2
    else:
        # Ambas devem ser S1 para serem comparadas
        return random.choice([0, 1])  # Treino ou Validação, aleatoriamente

# Geração das comparações
comparacoes_final = []

while len(comparacoes_final) < comparacoes_total:
    # Amostra aleatória entre as imagens de S1 ou S2
    if random.random() < 0.5:
        img1, img2 = random.sample(ficheiros_tv, 2)  # S1
    else:
        img1 = random.choice(ficheiros_teste)  # S2
        img2 = random.choice(ficheiros_teste)  # S2

    pessoa1 = get_id_pessoa(img1)
    pessoa2 = get_id_pessoa(img2)
    if not pessoa1 or not pessoa2:
        continue

    label = 1 if pessoa1 == pessoa2 else 0
    if label == 1 and positivos >= positivos_necessarios:
        continue
    if label == 0 and negativos >= negativos_necessarios:
        continue

    split = get_split(img1, img2)
    if split is None or split_counters[split] >= split_distrib[split]:
        continue

    comparacoes_final.append((img1, img2, label, split))
    split_counters[split] += 1
    if label == 1:
        positivos += 1
    else:
        negativos += 1

# Gravação do CSV
with open("comparacoes_10000.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['img1', 'img2', 'identicas', 'fase'])
    writer.writerows(comparacoes_final)

# Verificação final
print("✅ CSV criado com sucesso!")
print(f"Total comparações: {len(comparacoes_final)}")
print(f"Positivos: {positivos}, Negativos: {negativos}")

