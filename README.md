# Banco OPzS de 1.200 Ah com MLP

Estimar o **estado de carga (SoC)** de uma célula OPzS de 1.200 Ah a partir da tensão, da corrente e da temperatura, utilizando uma MLP do scikit-learn, e caracterizar o **estado de saúde (SoH)** com base na capacidade medida em descargas completas.

## Dados

`data/Cell N/Cell N/*.csv` - 6 células, 14 ensaios por célula (110 arquivos).

Os arquivos CSV utilizam `;` como separador, `,` como separador decimal e a codificação `cp1252`. Os cabeçalhos variam entre os arquivos e são normalizados em `src/data_io.py`.

## Como o alvo de SoC é construído

1. Segmentar cada arquivo em trechos de descarga (corrente negativa).
2. Manter apenas as descargas completas (≥ 1.000 Ah). Resultado: **30 descargas completas** (5 por célula, da 2ª à 6ª), totalizando aproximadamente 28.800 amostras.
3. Em cada descarga completa, assumir que a bateria começa totalmente carregada e termina descarregada, integrando a corrente para obter o SoC em cada instante. O SoC é normalizado para o intervalo [0, 1].

4. Treinar com as células **1 a 4** e testar com as células **5 e 6**, realizando a divisão por célula para evitar vazamento de dados entre treino e teste.

Características de entrada (*features*): `voltage`, `current` e `temperature`. Alvo: `soc`.

## SoH

`SOH = Q_descarga / 1.200 Ah`.

Como as seis células estão ligadas em série, a corrente é a mesma em todas elas, e a capacidade medida também é comum ao banco.

Não há uma faixa significativa de degradação, pois o banco está em condições de fábrica e foram realizados apenas 14 ciclos. Será apresentada a tabela com os valores de capacidade e SoH medidos.