import os
import re
import re
from itertools import combinations

diretoria = "both_eyes"

ficheiros_tv =[f for f in os.listdir(diretoria) if f.endswith('.jpg') and 'S2' not in f]
# os ficheiros_teste sao os ficheiros que possuem uma na sua string S2:
ficheiros_teste = [f for f in os.listdir(diretoria) if 'S2' in f and f.endswith('.jpg')]

print(ficheiros_tv)
# numero de linhas do novo csv
comparacoes = 10000


train = 0.7
validation = 0.2
test = 0.1

def comparar_array(arr):
    padrao_C = r"^(C\d+)"  # Captura a parte que começa com C até o primeiro "_"
    padrao_I = r"I(\d+)"   # Captura a parte que começa com "I" seguido de números

    resultados = []  # Lista para armazenar os pares comparados e seu resultado

    for ficheiro1, ficheiro2 in combinations(arr, 2):  # Compara todas as combinações possíveis
        match_C1 = re.match(padrao_C, ficheiro1)
        match_C2 = re.match(padrao_C, ficheiro2)

        match_I1 = re.search(padrao_I, ficheiro1)
        match_I2 = re.search(padrao_I, ficheiro2)

        if match_C1 and match_C2 and match_I1 and match_I2:
            C1, C2 = match_C1.group(1), match_C2.group(1)
            I1, I2 = match_I1.group(1), match_I2.group(1)

            if C1 == C2 and I1 != I2:
                resultados.append((ficheiro1, ficheiro2, 1))  # Adiciona o par à lista com 1

    return resultados



resultado = comparar_array(ficheiros_tv)
# Exibir os resultados
for r in resultado:
    print(r)
