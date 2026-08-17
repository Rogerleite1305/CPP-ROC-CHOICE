Readme 

# CPP-ROC CHOICE — Decision Support System (DSS)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

O **CPP-ROC CHOICE** é um Sistema de Apoio à Decisão Multicritério (DSS) que combina modelagem estocástica e ponderação ordinal para apoiar escolhas estratégicas sob condições de incerteza e preferências de múltiplos decisores.

A aplicação utiliza o método **Probabilistic Composition of Preferences (CPP)** com simulação de Monte Carlo para tratar a imprecisão das avaliações e o método **Rank Order Centroid (ROC)** para converter rankings ordinais de prioridade em pesos matemáticos objetivos.

---

## 📋 Sumário###

- [Visão Geral e Metodologia](#-visão-geral-e-metodologia)
- [Funcionalidades Principais](#-funcionalidades-principais)
- [Pré-requisitos](#-pré-requisitos)
- [Passo a Passo de Instalação](#-passo-a-passo-de-instalação)
- [Manual de Operação (Passo a Passo)](#-manual-de-operação-passo-a-passo)
- [Interpretação dos Resultados](#-interpretação-dos-resultados)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Créditos e Licença](#-créditos-e-licença)

---

## 🔬 Visão Geral e Metodologia###

O sistema resolve o desafio de agregar a opinião de múltiplos decisores (DMs) sem a necessidade de atribuição arbitrária de pesos numéricos para os critérios.

1. **Rank Order Centroid (ROC):** Converte a ordenação de prioridade dos critérios ($1º, 2º, \dots, n$) em pesos ponderados via fórmula:
   $$w_k = \frac{1}{n} \sum_{j=k}^{n} \frac{1}{j}$$
2. **Probabilistic Composition of Preferences (CPP):** Trata as pontuações atribuídas como modas de distribuições de probabilidade Triangulares ($\pm 10\%$ de variação). Executa 5.000 simulações de Monte Carlo para calcular:
   - **$M_i$ (Probabilidade de Excelência):** Chance da alternativa ser a melhor no cenário global (perfil maximotimizador).
   - **$m_i$ (Probabilidade de Pior Desempenho):** Chance da alternativa ser a pior no cenário global (perfil conservador/gestão de risco).

---

## ✨ Funcionalidades Principais###

-  **Suporte Multi-Decisor:** Customização de quantidade de decisores, alternativas e critérios.
-  **Escalas Flexíveis:** Suporte a matrizes normalizadas `[0, 1]`, notas de `[0, 10]` e pontuações `[0, 100]`.
-  **Memória de Cálculo (Drill-Down):** Transparência total dos pesos ROC individuais e contribuição de cada decisor no resultado final.
- 📄 **Relatório Executivo PDF (A4):** Geração automática de relatórios formatados com gráfico, quadro explicativo e dados institucionais personalizáveis (CNPJ, CEP com busca automática via ViaCEP e Logomarca).
-  **Personalização Institucional:** Upload de logo, banner e dados cadastrais para adequação ao padrão visual corporativo.

---

##  Pré-requisitos###

Antes de iniciar, certifique-se de ter instalado em sua máquina:
- **Python 3.9** ou superior ([Download Python](https://www.python.org/downloads/))
- **Git** ([Download Git](https://git-scm.com/))

---

##  Passo a Passo de Instalação###

### 1. Clonar o Repositório
Abra o seu terminal (PowerShell, Command Prompt ou Terminal do Linux/macOS) e execute:

```bash
git clone [https://github.com/seu-usuario/cpp-roc-choice.git](https://github.com/seu-usuario/cpp-roc-choice.git)
cd cpp-roc-choice