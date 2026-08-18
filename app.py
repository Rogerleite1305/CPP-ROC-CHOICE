import base64
import io
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
import requests
from scipy.stats import triang
import streamlit as st

# ==========================================
# 1. FUNÇÕES AUXILIARES
# ==========================================


def redimensionar_imagem(image_file, largura, altura):
    img = Image.open(image_file)
    resample_filter = getattr(Image, "Resampling", Image).LANCZOS
    img_res = img.resize((largura, altura), resample_filter)
    return img_res


def imagem_para_base64(image_file, largura=44, altura=44):
    try:
        if hasattr(image_file, "getvalue"):
            image_file = io.BytesIO(image_file.getvalue())
        img_res = redimensionar_imagem(image_file, largura, altura)
        buf = io.BytesIO()
        img_res.save(buf, format="PNG")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def formatar_cnpj(cnpj_raw):
    if not cnpj_raw:
        return ""
    limpo = re.sub(r"[^a-zA-Z0-9]", "", str(cnpj_raw)).upper()
    if len(limpo) == 14:
        return (
            f"{limpo[:2]}.{limpo[2:5]}.{limpo[5:8]}/{limpo[8:12]}-{limpo[12:]}"
        )
    return limpo


def callback_atualizar_cnpj():
    cnpj_digitado = st.session_state.get("emp_cnpj_input", "")
    st.session_state["emp_cnpj"] = formatar_cnpj(cnpj_digitado)


@st.cache_data(ttl=3600, show_spinner=False)
def buscar_endereco_hibrido(cep_raw):
    if not cep_raw:
        return "", ""
    cep_limpo = re.sub(r"[^a-zA-Z0-9]", "", str(cep_raw)).upper()
    if len(cep_limpo) < 3:
        return cep_limpo, ""
    if len(cep_limpo) == 8 and cep_limpo.isdigit():
        try:
            res = requests.get(
                f"https://viacep.com.br/ws/{cep_limpo}/json/", timeout=4
            )
            if res.status_code == 200:
                dados = res.json()
                if "erro" not in dados:
                    partes = [
                        p
                        for p in [
                            dados.get("logradouro", ""),
                            dados.get("bairro", ""),
                            f"{dados.get('localidade', '')} - {dados.get('uf', '')}",
                            "Brasil",
                        ]
                        if p
                    ]
                    return cep_limpo, ", ".join(partes)
        except Exception:
            pass
    return cep_limpo, ""


def callback_atualizar_cep():
    cep_digitado = st.session_state.get("input_cep", "")
    cep_limpo, end_encontrado = buscar_endereco_hibrido(cep_digitado)
    st.session_state["cep_formatado"] = cep_limpo
    if end_encontrado:
        st.session_state["emp_end"] = end_encontrado


# ==========================================
# 2. MOTOR MATEMÁTICO COM RASTREABILIDADE (CPP-ROC)
# ==========================================


def calcular_pesos_roc(n_criterios):
    pesos = np.zeros(n_criterios)
    for k in range(1, n_criterios + 1):
        pesos[k - 1] = (1 / n_criterios) * np.sum(
            [1 / j for j in range(k, n_criterios + 1)]
        )
    return pesos


@st.cache_data(show_spinner=False)
def estimar_probabilidades_cpp(matriz_avaliacoes, val_max=1.0, n_simulacoes=5000):
    n_alt, n_crit = matriz_avaliacoes.shape
    M_ij = np.zeros((n_alt, n_crit))
    m_ij = np.zeros((n_alt, n_crit))

    for c in range(n_crit):
        valores = matriz_avaliacoes[:, c]
        amostras = np.zeros((n_alt, n_simulacoes))

        for a in range(n_alt):
            c_val = valores[a]
            a_min = max(0.0, c_val * 0.9) if c_val > 0 else 0.0
            b_max = min(val_max, c_val * 1.1) if c_val < val_max else val_max
            if a_min == b_max:
                b_max += 1e-5
            scale = b_max - a_min
            c_param = (c_val - a_min) / scale if scale > 0 else 0.5
            amostras[a, :] = triang.rvs(
                c_param, loc=a_min, scale=scale, size=n_simulacoes
            )

        melhor_idx = np.argmax(amostras, axis=0)
        pior_idx = np.argmin(amostras, axis=0)

        for a in range(n_alt):
            M_ij[a, c] = np.sum(melhor_idx == a) / n_simulacoes
            m_ij[a, c] = np.sum(pior_idx == a) / n_simulacoes

    return M_ij, m_ij


def executar_cpp_roc_choice(matrizes_dms, ordens_criterios_dms, val_max=1.0):
    n_dms = len(matrizes_dms)
    n_alt, n_crit = matrizes_dms[0].shape

    M_agregado = np.zeros(n_alt)
    m_agregado = np.zeros(n_alt)

    detalhes_dms = []

    for d in range(n_dms):
        matriz = np.array(matrizes_dms[d])
        ordem = ordens_criterios_dms[d]

        pesos_roc_base = calcular_pesos_roc(n_crit)
        pesos_ordenados = np.zeros(n_crit)
        for rank, crit_idx in enumerate(ordem):
            pesos_ordenados[crit_idx] = pesos_roc_base[rank]

        M_ij, m_ij = estimar_probabilidades_cpp(matriz, val_max=val_max)

        M_dm_contribuicao = np.zeros(n_alt)
        m_dm_contribuicao = np.zeros(n_alt)
        for a in range(n_alt):
            M_dm_contribuicao[a] = np.sum(pesos_ordenados * M_ij[a, :])
            m_dm_contribuicao[a] = np.sum(pesos_ordenados * m_ij[a, :])

        M_agregado += M_dm_contribuicao
        m_agregado += m_dm_contribuicao

        detalhes_dms.append(
            {
                "decisor": d + 1,
                "pesos_criterios": pesos_ordenados,
                "M_ij": M_ij,
                "m_ij": m_ij,
                "M_contribuicao": M_dm_contribuicao,
                "m_contribuicao": m_dm_contribuicao,
            }
        )

    M_final = M_agregado / n_dms
    m_final = m_agregado / n_dms

    melhor_opcao_max = int(np.argmax(M_final))
    melhor_opcao_min = int(np.argmin(m_final))

    return {
        "M_i": M_final,
        "m_i": m_final,
        "otima_max_Mi": melhor_opcao_max,
        "otima_min_mi": melhor_opcao_min,
        "detalhes_dms": detalhes_dms,
    }


def gerar_grafico_membro(df_resultados):
    fig, ax = plt.subplots(figsize=(6.5, 2.6), dpi=200)
    x = np.arange(len(df_resultados))
    largura = 0.35

    ax.bar(
        x - largura / 2,
        df_resultados["M_i (Probabilidade de Excelência)"],
        largura,
        label="M_i (Excelência)",
        color="#2563eb",
    )
    ax.bar(
        x + largura / 2,
        df_resultados["m_i (Probabilidade de Pior Desempenho)"],
        largura,
        label="m_i (Pior Desempenho)",
        color="#ef4444",
    )

    ax.set_ylabel("Probabilidade Agregada", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df_resultados["Alternativa"], fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ==========================================
# 3. GERADOR DE RELATÓRIO PDF A4
# ==========================================


def gerar_pdf_relatorio(
    df_resultados,
    resultado_completo,
    opt_max_nome,
    opt_min_nome,
    n_dms,
    n_alt,
    n_crit,
    nomes_criterios,
    nomes_dms,
    empresa_nome,
    empresa_cnpj,
    empresa_contato,
    empresa_cep,
    empresa_end,
    empresa_link,
    logo_file,
):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    elements = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=4,
    )
    analise_style = ParagraphStyle(
        "AnaliseText",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#1e293b"),
        leading=11,
    )
    footer_style = ParagraphStyle(
        "FooterText",
        parent=styles["Normal"],
        fontSize=7.5,
        textColor=colors.HexColor("#475569"),
        leading=10,
    )

    elements.append(
        Paragraph("RELATÓRIO EXECUTIVO DE TOMADA DE DECISÃO", title_style)
    )
    elements.append(
        Paragraph(
            "SISTEMA CPP-ROC CHOICE | RASTREABILIDADE E MEMÓRIA DE CÁLCULO",
            subtitle_style,
        )
    )
    elements.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#2563eb"),
            spaceAfter=8,
        )
    )

    sorted_mi = np.sort(df_resultados["M_i (Probabilidade de Excelência)"])[
        ::-1
    ]
    margem_dominancia = (
        (sorted_mi[0] - sorted_mi[1]) if len(sorted_mi) > 1 else sorted_mi[0]
    )

    rec_data = [
        ["Perfil Decisório", "Alternativa Ótima", "Métrica Agregada"],
        [
            "Maximotimizador (argmax Mi)",
            str(opt_max_nome),
            f"Mi = {sorted_mi[0]:.4f}",
        ],
        [
            "Conservador / Menor Risco (argmin mi)",
            str(opt_min_nome),
            f"mi = {df_resultados['m_i (Probabilidade de Pior Desempenho)'].min():.4f}",
        ],
        [
            "Margem de Dominância (Robustez)",
            f"{margem_dominancia*100:.2f}% de vantagem sobre 2º lugar",
            "Alta Estabilidade",
        ],
    ]
    t_rec = Table(rec_data, colWidths=[180, 180, 120])
    t_rec.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(t_rec)
    elements.append(Spacer(1, 6))

    elements.append(
        Paragraph("Análise Gráfica e Diretriz Decisória", section_style)
    )
    buf_grafico = gerar_grafico_membro(df_resultados)
    img_grafico = RLImage(buf_grafico, width=440, height=170)

    texto_analitico = f"""
    <b>Interpretação Executiva:</b><br/>
    • A alternativa <b>{opt_max_nome}</b> apresenta a maior probabilidade de alcançar o desempenho superior (<i>M<sub>i</sub></i> = {sorted_mi[0]:.4f}).
    A diferença de <b>{margem_dominancia*100:.2f}%</b> em relação à segunda colocada indica uma preferência estocástica consolidada.<br/>
    • Para gestores com aversão ao risco, a alternativa <b>{opt_min_nome}</b> oferece a menor probabilidade de falha ou pior desempenho relativo (<i>m<sub>i</sub></i>).
    """
    p_analitico = Paragraph(texto_analitico, analise_style)

    quadro_conteudo = [[img_grafico], [Spacer(1, 3)], [p_analitico]]
    t_quadro = Table(quadro_conteudo, colWidths=[480])
    t_quadro.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(t_quadro)
    elements.append(Spacer(1, 6))

    elements.append(
        Paragraph("Rastreabilidade: Pesos ROC por Decisor", section_style)
    )
    roc_headers = ["Critério"] + [
        f"{dm} (Peso)" for dm in nomes_dms
    ] + ["Peso Médio ROC"]
    roc_rows = [roc_headers]

    detalhes = resultado_completo["detalhes_dms"]
    for c_idx, c_nome in enumerate(nomes_criterios):
        row = [str(c_nome)]
        pesos_crit = []
        for d in range(n_dms):
            p_val = detalhes[d]["pesos_criterios"][c_idx]
            pesos_crit.append(p_val)
            row.append(f"{p_val:.4f}")
        row.append(f"{np.mean(pesos_crit):.4f}")
        roc_rows.append(row)

    t_roc = Table(
        roc_rows, colWidths=[120] + [360 // (n_dms + 1)] * (n_dms + 1)
    )
    t_roc.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f8f9fa")],
                ),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    elements.append(t_roc)

    elements.append(Spacer(1, 10))
    elements.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.HexColor("#cbd5e1"),
            spaceAfter=6,
        )
    )

    logo_cell = ""
    if logo_file:
        try:
            logo_bytes = io.BytesIO(logo_file.getvalue())
            mini_logo = redimensionar_imagem(logo_bytes, 40, 40)
            mini_buffer = io.BytesIO()
            mini_logo.save(mini_buffer, format="PNG")
            mini_buffer.seek(0)
            logo_cell = RLImage(mini_buffer, width=40, height=40)
        except Exception:
            logo_cell = ""

    info_text = (
        f"<b>{empresa_nome if empresa_nome else 'Empresa Registrada'}</b>"
    )
    if empresa_link:
        url_val = (
            empresa_link
            if empresa_link.startswith("http")
            else f"https://{empresa_link}"
        )
        info_text += f" | <a href='{url_val}' color='#2563eb'><u>Página Oficial / Rede Social</u></a>"

    info_text += f"<br/><b>CNPJ:</b> {empresa_cnpj if empresa_cnpj else 'Não informado'} | <b>Contato:</b> {empresa_contato if empresa_contato else 'Não informado'}<br/>"
    info_text += f"<b>CEP:</b> {empresa_cep if empresa_cep else 'Não informado'} | <b>Endereço:</b> {empresa_end if empresa_end else 'Não informado'}"

    text_p = Paragraph(info_text, footer_style)

    if logo_cell != "":
        footer_table_data = [[logo_cell, text_p]]
        col_widths = [45, 435]
    else:
        footer_table_data = [[text_p]]
        col_widths = [480]

    t_footer = Table(footer_table_data, colWidths=col_widths)
    t_footer.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    elements.append(t_footer)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ==========================================
# 4. INTERFACE STREAMLIT
# ==========================================

st.set_page_config(
    page_title="DSS | CPP-ROC CHOICE", page_icon="📈", layout="wide"
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .main-header { background-color: #0f172a; padding: 20px; border-radius: 8px; color: #ffffff; margin-bottom: 20px; }
    .main-header h1 { color: #ffffff !important; font-size: 24px; font-weight: 600; margin: 0; }
    .main-header p { color: #94a3b8; font-size: 13px; margin-top: 4px; margin-bottom: 0; }
    .stButton>button { background-color: #2563eb !important; color: white !important; font-weight: 600 !important; border-radius: 6px !important; border: none !important; width: 100%; }
    
    .company-card {
        background-color: #ffffff;
        border-left: 4px solid #2563eb;
        padding: 16px;
        border-radius: 8px;
        margin-top: 30px;
        margin-bottom: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .company-logo-img { width: 44px; height: 44px; object-fit: cover; border-radius: 6px; border: 1px solid #e2e8f0; }
    .company-info-container { flex-grow: 1; }
    .company-header-title { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    .company-link-btn {
        background-color: #eff6ff; color: #2563eb !important; padding: 3px 10px; border-radius: 16px;
        font-size: 11px; font-weight: 600; text-decoration: none !important; border: 1px solid #bfdbfe;
    }
    .company-details-text { margin: 4px 0 0 0; color: #475569; font-size: 12px; line-height: 1.4; }
    
    .copyright-footer { 
        background-color: #0f172a; 
        color: #94a3b8; 
        text-align: center; 
        padding: 12px; 
        border-radius: 8px; 
        font-size: 12px; 
        margin-top: 10px; 
    }
    .copyright-footer a { color: #60a5fa; text-decoration: none; font-weight: bold; }
    </style>
""",
    unsafe_allow_html=True,
)

if "cep_formatado" not in st.session_state:
    st.session_state["cep_formatado"] = ""
if "emp_end" not in st.session_state:
    st.session_state["emp_end"] = ""
if "emp_cnpj" not in st.session_state:
    st.session_state["emp_cnpj"] = ""
if "calculo_executado" not in st.session_state:
    st.session_state["calculo_executado"] = False

# BARRA LATERAL
st.sidebar.markdown("### **Configuração do Problema**")
n_dms = st.sidebar.number_input("Quantidade de Decisores (DMs)", 1, 10, 2)
n_alt = st.sidebar.number_input("Quantidade de Alternativas", 2, 20, 3)
n_crit = st.sidebar.number_input("Quantidade de Critérios", 2, 10, 3)

st.sidebar.markdown("#### **Escala de Avaliação**")
tipo_escala = st.sidebar.selectbox(
    "Escolha o intervalo da escala:",
    [
        "[0, 1] - Normalizada",
        "[0, 10] - Notas de 0 a 10",
        "[0, 100] - Porcentagem/Pontos",
    ],
    index=0,
)

if "[0, 10]" in tipo_escala:
    val_max = 10.0
    rotulo_matriz = "Matriz de Avaliação [0, 10]"
elif "[0, 100]" in tipo_escala:
    val_max = 100.0
    rotulo_matriz = "Matriz de Avaliação [0, 100]"
else:
    val_max = 1.0
    rotulo_matriz = "Matriz de Avaliação Normalizada [0, 1]"

st.sidebar.markdown("#### **Nomes Globais de Referência**")
nomes_dms_globais = [
    st.sidebar.text_input(
        f"Nome Decisor {d+1}", value=f"Decisor {d+1}", key=f"glob_dm_{d}"
    )
    for d in range(n_dms)
]
nomes_alt_globais = [
    st.sidebar.text_input(
        f"Nome Alt. {a+1}", value=f"Alternativa {a+1}", key=f"glob_alt_{a}"
    )
    for a in range(n_alt)
]
nomes_crit_globais = [
    st.sidebar.text_input(
        f"Nome Crit. {c+1}", value=f"Critério {c+1}", key=f"glob_crit_{c}"
    )
    for c in range(n_crit)
]

st.sidebar.divider()

with st.sidebar.expander(
    "⚙️ Configurações Visuais e Cadastrais", expanded=False
):
    logo_file = st.file_uploader(
        "Logo da Empresa (Rodapé e PDF)", type=["png", "jpg", "jpeg"], key="logo"
    )

    st.text_input(
        "Nome da Empresa",
        placeholder="Ex: Prefeitura de Maceió",
        key="emp_nome",
    )
    st.text_input(
        "CNPJ (Numérico ou Alfanumérico)",
        placeholder="Digite 14 caracteres (ex: 12ABC345000199)",
        key="emp_cnpj_input",
        on_change=callback_atualizar_cnpj,
    )
    st.text_input("Contato", placeholder="E-mail / Telefone", key="emp_contato")
    st.text_input(
        "Link Oficial",
        placeholder="instagram.com/rise.ufal",
        key="emp_link",
    )
    st.text_input(
        "CEP",
        placeholder="Digite o CEP",
        key="input_cep",
        on_change=callback_atualizar_cep,
    )
    st.text_area("Endereço Completo", key="emp_end")

# EXIBIÇÃO DA LOGO DO CPP-ROC CHOICE CENTRALIZADA COM LARGURA FIXA
col_l1, col_l2, col_l3 = st.columns([1, 1, 1])
with col_l2:
    try:
        url_logo_github = "https://raw.githubusercontent.com/Rogerleite1305/CPP-ROC-CHOISE/main/LOGO%20CPP-ROC-CHOICE.png"
        st.image(url_logo_github, width=150)  # Defina o tamanho em pixels desejado aqui
    except Exception:
        st.warning("Não foi possível carregar a logo oficial do topo.")

st.markdown(
    """
    <div class="main-header">
        <h1>DECISION SUPPORT SYSTEM — CPP-ROC CHOICE</h1>
        <p>Apoio à Decisão Multicritério com Rastreabilidade de Cálculos e Análise de Sensibilidade</p>
    </div>
""",
    unsafe_allow_html=True,
)

# ABA PRINCIPAL / MANUAL DO MÉTODO
tab_app, tab_manual = st.tabs(["📊 Painel de Avaliação", "📖 Manual do Método"])

with tab_manual:
    st.markdown(
        """
    ### **MANUAL DO USUÁRIO — MÉTODO CPP-ROC CHOICE**

    O **CPP-ROC CHOICE** é um Sistema de Apoio à Decisão Multicritério (DSS) estocástico e ordino-cardinal. Ele foi desenvolvido para auxiliar equipes na escolha de alternativas sob condições de incerteza, integrando a avaliação probabilística de preferências (**CPP**) com a ponderação ordinal de critérios (**ROC**).

    ---

    #### **1. Fundamentação Teórica dos Métodos**

    **A. Método ROC (Rank Order Centroid)**  
    O **ROC** elimina a subjetividade na atribuição de pesos numéricos diretos aos critérios. Os decisores precisam apenas ordenar os critérios por prioridade ($1º, 2º, \dots, n$). O sistema converte automaticamente essa ordem em pesos matemáticos via fórmula:

    $$w_k = \\frac{1}{n} \\sum_{j=k}^{n} \\frac{1}{j}$$

    * **$n$**: Número total de critérios.
    * **$k$**: Posição do critério no ranking de prioridade (onde $k=1$ é o mais importante).

    ---

    **B. Método CPP (Probabilistic Composition of Preferences)**  
    Em vez de tratar as avaliações dos decisores como números fixos, o **CPP** assume que cada nota é o centro de uma distribuição de probabilidade. Para modelar a imprecisão humana, o sistema executa **Simulações de Monte Carlo** (5.000 iterações por célula) utilizando a **Distribuição Triangular**:

    * **Moda ($c$):** A nota atribuída pelo decisor.
    * **Limite Inferior ($a$):** $0.9 \\times c$ (variação de $-10\%$).
    * **Limite Superior ($b$):** $1.1 \\times c$ (variação de $+10\%$).

    A partir das simulações, o sistema calcula duas probabilidades estocásticas agregadas para cada alternativa $i$:

    * **$M_i$ (Probabilidade de Excelência / Maximotimizador):** Chance da alternativa apresentar o melhor desempenho global no cenário analisado.
    * **$m_i$ (Probabilidade de Pior Desempenho / Conservador):** Chance da alternativa apresentar o pior desempenho global no cenário analisado.

    ---

    #### **2. Passo a Passo de Operação no Software**

    | Etapa | Ação do Usuário | O que o Sistema Faz |
    | :--- | :--- | :--- |
    | **1. Configuração** | Define na barra lateral o nº de DMs, Alternativas, Critérios, Escala e Dados Cadastrais. | Instancia as matrizes de entrada e campos personalizados. |
    | **2. Avaliação** | Cada Decisor (DM) preenche sua matriz de notas e estabelece o ranking de prioridade dos critérios. | Armazena as preferências ordinais e cardinais separadamente. |
    | **3. Processamento** | Clica no botão **"EXECUTAR ANÁLISE DE DECISÃO"**. | Calcula pesos ROC, roda Monte Carlo (CPP) e agrega os resultados globais. |
    | **4. Análise** | Inspeciona os indicadores $M_i$/$m_i$, a tabela *drill-down* e faz o download do PDF. | Renderiza relatórios executivos no padrão A4 com memória de cálculo. |

    ---

    #### **3. Guia de Interpretação dos Resultados**

    * **Perfil Maximotimizador ($\text{argmax } M_i$):** Indicado para cenários de inovação ou concorrência, onde o objetivo é escolher a alternativa com maior potencial de liderança.
    * **Perfil Conservador ($\text{argmin } m_i$):** Indicado para gerenciamento de risco e governança pública, escolhendo a opção com menor probabilidade de falha crítica.
    * **Margem de Dominância:** Mede a diferença entre o 1º e o 2º lugar no indicador $M_i$. Valores acima de **$5.00\%$** indicam alta estabilidade e clareza na escolha.
    """
    )

with tab_app:
    matrizes_dms = []
    ordens_dms = []

    st.markdown("### **Painel de Avaliação dos Decisores**")
    abas_dms = st.tabs([f"👤 {nome_dm}" for nome_dm in nomes_dms_globais])

    for d, aba in enumerate(abas_dms):
        with aba:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.caption(rotulo_matriz)
                valores_iniciais = np.round(
                    np.random.rand(n_alt, n_crit) * val_max, 2
                )
                df_init = pd.DataFrame(
                    valores_iniciais,
                    columns=nomes_crit_globais,
                    index=nomes_alt_globais,
                )
                df_editada = st.data_editor(df_init, key=f"matriz_indiv_{d}")
                matrizes_dms.append(df_editada.values)

            with col2:
                st.caption("Ranking de Importância dos Critérios (ROC)")
                ordem = []
                criterios_disp = list(nomes_crit_globais)
                for r in range(n_crit):
                    default_idx = min(r, len(criterios_disp) - 1)
                    escolha = st.selectbox(
                        f"Prioridade {r+1}",
                        criterios_disp,
                        index=default_idx,
                        key=f"dm_{d}_rank_{r}",
                    )
                    ordem.append(nomes_crit_globais.index(escolha))
                ordens_dms.append(ordem)

    st.divider()

    if st.button("EXECUTAR ANÁLISE DE DECISÃO"):
        with st.spinner("Processando simulação e rastreando cálculos..."):
            res = executar_cpp_roc_choice(
                matrizes_dms, ordens_dms, val_max=val_max
            )

            alt_opt_max_nome = nomes_alt_globais[res["otima_max_Mi"]]
            alt_opt_min_nome = nomes_alt_globais[res["otima_min_mi"]]

            df_res = pd.DataFrame(
                {
                    "Alternativa": nomes_alt_globais,
                    "M_i (Probabilidade de Excelência)": res["M_i"],
                    "m_i (Probabilidade de Pior Desempenho)": res["m_i"],
                }
            )

            pdf_bytes = gerar_pdf_relatorio(
                df_res,
                res,
                alt_opt_max_nome,
                alt_opt_min_nome,
                n_dms,
                n_alt,
                n_crit,
                nomes_crit_globais,
                nomes_dms_globais,
                st.session_state.get("emp_nome", ""),
                st.session_state.get("emp_cnpj", ""),
                st.session_state.get("emp_contato", ""),
                st.session_state.get("cep_formatado", ""),
                st.session_state.get("emp_end", ""),
                st.session_state.get("emp_link", ""),
                logo_file,
            )

            st.session_state["res"] = res
            st.session_state["df_res"] = df_res
            st.session_state["pdf_bytes"] = pdf_bytes
            st.session_state["alt_opt_max_nome"] = alt_opt_max_nome
            st.session_state["alt_opt_min_nome"] = alt_opt_min_nome
            st.session_state["calculo_executado"] = True

    if st.session_state.get("calculo_executado", False):
        res = st.session_state["res"]
        df_res = st.session_state["df_res"]
        pdf_bytes = st.session_state["pdf_bytes"]
        alt_opt_max_nome = st.session_state["alt_opt_max_nome"]
        alt_opt_min_nome = st.session_state["alt_opt_min_nome"]

        st.markdown("### **Resultados & Interpretação Decisória**")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                label="Recomendação Maximotimizadora",
                value=str(alt_opt_max_nome),
            )
        with c2:
            st.metric(
                label="Recomendação de Menor Risco",
                value=str(alt_opt_min_nome),
            )
        with c3:
            sorted_mi = np.sort(res["M_i"])[::-1]
            vantagem = (
                (sorted_mi[0] - sorted_mi[1])
                if len(sorted_mi) > 1
                else sorted_mi[0]
            )
            st.metric(
                label="Margem de Dominância", value=f"{vantagem*100:.2f}%"
            )

        st.dataframe(df_res, use_container_width=True, hide_index=True)

        with st.expander(
            "🔍 Explorar Origem dos Cálculos (Drill-Down)", expanded=True
        ):
            st.markdown(
                "#### **Pesos ROC Gerados por Decisor a partir da Ordenação:**"
            )
            data_roc = {"Critério": nomes_crit_globais}
            for d in range(n_dms):
                data_roc[f"Peso ({nomes_dms_globais[d]})"] = res[
                    "detalhes_dms"
                ][d]["pesos_criterios"]
            st.dataframe(
                pd.DataFrame(data_roc),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                "#### **Desagregação de Contribuição por Decisor no Resultado Final:**"
            )
            dm_contrib_data = {"Alternativa": nomes_alt_globais}
            for d in range(n_dms):
                dm_contrib_data[f"Suporte {nomes_dms_globais[d]} (M_i)"] = res[
                    "detalhes_dms"
                ][d]["M_contribuicao"]
            st.dataframe(
                pd.DataFrame(dm_contrib_data),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()
        st.download_button(
            label="📄 BAIXAR RELATÓRIO EXECUTIVO E MEMÓRIA DE CÁLCULO (PDF A4)",
            data=pdf_bytes,
            file_name="Relatorio_CPP_ROC_CHOICE.pdf",
            mime="application/pdf",
        )

# ==========================================
# RODAPÉ ÚNICO: CARTÃO EMPRESA + DIREITOS AUTORAIS RISE-UFAL
# ==========================================
logo_b64 = imagem_para_base64(logo_file) if logo_file else None
emp_nome = st.session_state.get("emp_nome", "")
emp_cnpj = st.session_state.get("emp_cnpj", "")
emp_contato = st.session_state.get("emp_contato", "")
emp_cep = st.session_state.get("cep_formatado", "")
emp_end = st.session_state.get("emp_end", "")
emp_link = st.session_state.get("emp_link", "")

link_html = ""
if emp_link:
    url_formatada = (
        emp_link if emp_link.startswith("http") else f"https://{emp_link}"
    )
    link_html = f'<a href="{url_formatada}" target="_blank" class="company-link-btn">Página Oficial / Rede Social</a>'

logo_html = f'<img src="{logo_b64}" class="company-logo-img"/>' if logo_b64 else ""

card_empresa_footer_html = (
    f'<div class="company-card">'
    f"{logo_html}"
    f'<div class="company-info-container">'
    f'<div class="company-header-title">'
    f'<strong style="color: #0f172a; font-size: 15px;">{emp_nome if emp_nome else "Empresa Registrada"}</strong>'
    f"{link_html}"
    f"</div>"
    f'<p class="company-details-text">'
    f'<b>CNPJ:</b> {emp_cnpj if emp_cnpj else "Não informado"} &nbsp;|&nbsp; '
    f'<b>Contato:</b> {emp_contato if emp_contato else "Não informado"}<br/>'
    f'<b>CEP:</b> {emp_cep if emp_cep else "Não informado"} &nbsp;|&nbsp; '
    f'<b>Endereço:</b> {emp_end if emp_end else "Não informado"}'
    f"</p>"
    f"</div>"
    f"</div>"
)

copyright_footer_html = """
    <div class="copyright-footer">
        Desenvolvido pelo Laboratório de Pesquisa <b>RISE / UFAL</b> — Todos os direitos reservados.
    </div>
"""

st.markdown(card_empresa_footer_html, unsafe_allow_html=True)
st.markdown(copyright_footer_html, unsafe_allow_html=True)
