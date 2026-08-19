import base64
import datetime
import io
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
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
# 0. CLASSE NUMBERED CANVAS (PAGINAÇÃO DINÂMICA)
# ==========================================


class NumberedCanvas(canvas.Canvas):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#475569"))

        data_hora = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        texto_rodape = (
            f"Relatório Gerado em: {data_hora} | Página {self._pageNumber} de {page_count}"
        )

        self.drawRightString(A4[0] - 36, 20, texto_rodape)
        self.drawString(
            36,
            20,
            "DSS CPP-ROC CHOICE — Desenvolvido por RISE/UFAL © Todos os direitos reservados",
        )

        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 32, A4[0] - 36, 32)

        self.restoreState()


# ==========================================
# 1. FUNÇÕES AUXILIARES E FORMATADORES
# ==========================================


def redimensionar_imagem(image_file, largura, altura):
    img = Image.open(image_file)
    resample_filter = getattr(Image, "Resampling", Image).LANCZOS
    img_res = img.resize((largura, altura), resample_filter)
    return img_res


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
# 2. MOTOR MATEMÁTICO (TABELA 2 DE PERFIS)
# ==========================================


def calcular_pesos_roc(n_criterios):
    pesos = np.zeros(n_criterios)
    for k in range(1, n_criterios + 1):
        pesos[k - 1] = (1 / n_criterios) * np.sum(
            [1 / j for j in range(k, n_criterios + 1)]
        )
    return pesos


def obter_fatores_perfil(
    perfil_col1, perfil_col2, pct_incerteza_base=0.10
):
    """
    Ajusta o suporte triangular de acordo com a Tabela 2 do artigo:
    Coluna 1: Optimistic vs. Progressive
    Coluna 2: Pessimistic vs. Conservative
    """
    fator_inf = 1.0 - pct_incerteza_base
    fator_sup = 1.0 + pct_incerteza_base

    # Ajuste Coluna 1
    if perfil_col1 == "Optimistic (Otimista)":
        fator_sup += 0.08
    elif perfil_col1 == "Progressive (Progressista)":
        fator_sup += 0.12

    # Ajuste Coluna 2
    if perfil_col2 == "Pessimistic (Pessimista)":
        fator_inf -= 0.08
    elif perfil_col2 == "Conservative (Conservador)":
        fator_inf = max(0.01, fator_inf - 0.05)

    return fator_inf, fator_sup


def normalizar_matriz_avaliacao(matriz, tipos_criterios, val_max=1.0):
    matriz_norm = np.array(matriz, dtype=float) / float(val_max)
    n_alt, n_crit = matriz_norm.shape

    for c in range(n_crit):
        tipo = tipos_criterios[c]
        if tipo == "Custo":
            matriz_norm[:, c] = 1.0 - matriz_norm[:, c]

    matriz_norm = np.clip(matriz_norm, 0.0, 1.0)
    return matriz_norm


@st.cache_data(show_spinner=False)
def estimar_probabilidades_cpp_custom(
    matriz_norm, fator_inf, fator_sup, n_simulacoes=5000
):
    n_alt, n_crit = matriz_norm.shape
    M_ij = np.zeros((n_alt, n_crit))
    m_ij = np.zeros((n_alt, n_crit))

    for c in range(n_crit):
        valores = matriz_norm[:, c]
        amostras = np.zeros((n_alt, n_simulacoes))

        for a in range(n_alt):
            c_val = valores[a]
            a_min = max(0.0, c_val * fator_inf) if c_val > 0 else 0.0
            b_max = min(1.0, c_val * fator_sup) if c_val < 1.0 else 1.0

            if a_min >= b_max:
                b_max = min(1.0, a_min + 1e-4)

            scale = b_max - a_min
            c_param = (c_val - a_min) / scale if scale > 0 else 0.5
            c_param = np.clip(c_param, 1e-4, 0.9999)

            amostras[a, :] = triang.rvs(
                c_param, loc=a_min, scale=scale, size=n_simulacoes
            )

        melhor_idx = np.argmax(amostras, axis=0)
        pior_idx = np.argmin(amostras, axis=0)

        for a in range(n_alt):
            M_ij[a, c] = np.sum(melhor_idx == a) / n_simulacoes
            m_ij[a, c] = np.sum(pior_idx == a) / n_simulacoes

    return M_ij, m_ij


def executar_cpp_roc_choice(
    matrizes_dms,
    ordens_criterios_dms,
    perfis_dms,
    tipos_criterios,
    val_max=1.0,
    pct_incerteza=0.10,
    n_simulacoes=5000,
):
    n_dms = len(matrizes_dms)
    n_alt, n_crit = matrizes_dms[0].shape

    M_agregado = np.zeros(n_alt)
    m_agregado = np.zeros(n_alt)

    detalhes_dms = []

    for d in range(n_dms):
        matriz_bruta = np.array(matrizes_dms[d])
        matriz_norm = normalizar_matriz_avaliacao(
            matriz_bruta, tipos_criterios, val_max=val_max
        )

        ordem = ordens_criterios_dms[d]
        col1_perfil, col2_perfil = perfis_dms[d]

        fator_inf, fator_sup = obter_fatores_perfil(
            col1_perfil, col2_perfil, pct_incerteza_base=pct_incerteza
        )

        pesos_roc_base = calcular_pesos_roc(n_crit)
        pesos_ordenados = np.zeros(n_crit)
        for rank, crit_idx in enumerate(ordem):
            pesos_ordenados[crit_idx] = pesos_roc_base[rank]

        M_ij, m_ij = estimar_probabilidades_cpp_custom(
            matriz_norm,
            fator_inf,
            fator_sup,
            n_simulacoes=n_simulacoes,
        )

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
                "perfil": f"{col1_perfil} - {col2_perfil}",
                "pesos_criterios": pesos_ordenados,
                "matriz_norm": matriz_norm,
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


def gerar_grafico_membro(df_resultados, para_impressao=False):
    figsize = (6.5, 3.2) if not para_impressao else (6.5, 2.5)
    dpi = 300 if para_impressao else 150

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x = np.arange(len(df_resultados))
    largura = 0.35

    ax.bar(
        x - largura / 2,
        df_resultados["M_i (Probabilidade de Excelência)"],
        largura,
        label="M_i (Excelência / Dominância)",
        color="#2563eb",
    )
    ax.bar(
        x + largura / 2,
        df_resultados["m_i (Probabilidade de Pior Desempenho)"],
        largura,
        label="m_i (Pior Desempenho / Risco)",
        color="#ef4444",
    )

    ax.set_ylabel("Probabilidade Agregada", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        df_resultados["Alternativa"], fontsize=8, rotation=0, ha="center"
    )
    ax.legend(fontsize=8, loc="upper right")
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
    tipos_criterios,
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
        bottomMargin=45,
    )
    elements = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=13,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=10,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=6,
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
            "DECISION SUPPORT SYSTEM - CPP-ROC-CHOICE<br/>Sistema de Apoio à Decisão Multicritério com Múltiplos Decisores Sob Incerteza. Para a problemática de escolha.",
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
            "Maximotimizador (argmax M_i)",
            str(opt_max_nome),
            f"M_i = {sorted_mi[0]:.4f}",
        ],
        [
            "Conservador / Menor Risco (argmin m_i)",
            str(opt_min_nome),
            f"m_i = {df_resultados['m_i (Probabilidade de Pior Desempenho)'].min():.4f}",
        ],
        [
            "Margem de Dominância",
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

    buf_grafico = gerar_grafico_membro(df_resultados, para_impressao=True)
    img_grafico = RLImage(buf_grafico, width=460, height=160)

    texto_analitico = f"""
    <b>Interpretação Executiva para a Problemática de Escolha:</b><br/>
    • A alternativa recomendada <b>{opt_max_nome}</b> obteve a maior probabilidade agregada de excelência (<i>M<sub>i</sub></i> = {sorted_mi[0]:.4f}) considerando as preferências combinadas dos {n_dms} decisores.<br/>
    • Sob a ótica de minimização de risco, a alternativa <b>{opt_min_nome}</b> apresentou o menor índice de vulnerabilidade (<i>m<sub>i</sub></i>).
    """
    p_analitico = Paragraph(texto_analitico, analise_style)

    quadro_conteudo = [
        [Paragraph("<b>Análise Gráfica e Diretriz Decisória</b>", section_style)],
        [img_grafico],
        [Spacer(1, 2)],
        [p_analitico],
    ]
    t_quadro = Table(quadro_conteudo, colWidths=[480])
    t_quadro.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    elements.append(KeepTogether([t_quadro]))
    elements.append(Spacer(1, 8))

    elements.append(
        Paragraph(
            "Rastreabilidade: Pesos ROC, Perfis e Tipos de Critérios",
            section_style,
        )
    )
    roc_headers = (
        ["Critério", "Tipo"]
        + [f"{dm}" for dm in nomes_dms]
        + ["Peso Médio ROC"]
    )
    roc_rows = [roc_headers]

    detalhes = resultado_completo["detalhes_dms"]
    for c_idx, c_nome in enumerate(nomes_criterios):
        c_tipo = tipos_criterios[c_idx]
        row = [str(c_nome), str(c_tipo)]
        pesos_crit = []
        for d in range(n_dms):
            p_val = detalhes[d]["pesos_criterios"][c_idx]
            pesos_crit.append(p_val)
            row.append(f"{p_val:.4f}")
        row.append(f"{np.mean(pesos_crit):.4f}")
        roc_rows.append(row)

    col_w_base = 320 // (n_dms + 1)
    t_roc = Table(
        roc_rows, colWidths=[110, 50] + [col_w_base] * (n_dms + 1)
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
    elements.append(KeepTogether([t_roc]))

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

    url_rise = "https://www.instagram.com/rise.ufal/"

    info_text = f"<b>Organização Solicitante:</b> {empresa_nome if empresa_nome else 'Não informada'}"
    if empresa_link:
        url_val = empresa_link if empresa_link.startswith("http") else f"https://{empresa_link}"
        info_text += f" | <a href='{url_val}' color='#2563eb'><u>Site Oficial</u></a>"

    info_text += f"<br/><b>CNPJ:</b> {empresa_cnpj if empresa_cnpj else 'N/A'} | <b>Contato:</b> {empresa_contato if empresa_contato else 'N/A'}"
    info_text += f" | <b>CEP:</b> {empresa_cep if empresa_cep else 'N/A'}<br/>"
    info_text += f"<b>Desenvolvimento & Direitos Autorais:</b> RISE — Laboratório de Pesquisa (UFAL) | <a href='{url_rise}' color='#2563eb'><u>@rise.ufal</u></a>"

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
    elements.append(KeepTogether([t_footer]))

    doc.build(elements, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()


# ==========================================
# 4. INTERFACE STREAMLIT
# ==========================================

st.set_page_config(
    page_title="DSS | CPP-ROC-CHOICE", page_icon="📈", layout="wide"
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .main-header { background-color: #0f172a; padding: 22px; border-radius: 8px; color: #ffffff; margin-bottom: 20px; }
    .main-header h1 { color: #ffffff !important; font-size: 22px; font-weight: 700; margin: 0; }
    .main-header p { color: #94a3b8; font-size: 13px; margin-top: 6px; margin-bottom: 0; }
    .stButton>button { background-color: #2563eb !important; color: white !important; font-weight: 600 !important; border-radius: 6px !important; border: none !important; width: 100%; height: 45px; }
    
    .step-badge {
        background-color: #2563eb;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 8px;
    }
    
    .recommendation-card {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .footer-rights {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding: 20px 0 10px 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 40px;
    }
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
st.sidebar.markdown(
    """
    <div style="padding-bottom: 10px;">
        <h3 style="margin: 0; color: #0f172a; font-size: 18px;">⚙️ Configurações Técnicas</h3>
        <p style="margin: 0; color: #64748b; font-size: 11px;">Ajustes de Monte Carlo & Empresa</p>
    </div>
""",
    unsafe_allow_html=True,
)

with st.sidebar.expander("Escala e Incerteza do Modelo", expanded=True):
    tipo_escala = st.selectbox(
        "Escala dos Dados Brutos:",
        [
            "[0, 1] - Normalizada",
            "[0, 10] - Notas de 0 a 10",
            "[0, 100] - Porcentagem/Pontos",
        ],
        index=0,
        key="cfg_tipo_escala",
    )

    if "[0, 10]" in tipo_escala:
        val_max = 10.0
    elif "[0, 100]" in tipo_escala:
        val_max = 100.0
    else:
        val_max = 1.0

    pct_incerteza = (
        st.slider(
            "Nível de Incerteza / Variação Perturbativa (%)",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
        )
        / 100.0
    )

    n_simulacoes = st.select_slider(
        "Simulações de Monte Carlo",
        options=[1000, 2500, 5000, 10000, 20000],
        value=5000,
    )

with st.sidebar.expander("Identidade Corporativa (Relatório PDF)", expanded=False):
    logo_file = st.file_uploader(
        "Logomarca do Cliente", type=["png", "jpg", "jpeg"], key="logo"
    )
    st.text_input("Empresa / Organização", placeholder="Ex: RISE / UFAL", key="emp_nome")
    st.text_input(
        "CNPJ",
        placeholder="Apenas números ou formatado",
        key="emp_cnpj_input",
        on_change=callback_atualizar_cnpj,
    )
    st.text_input("Contato / E-mail", placeholder="contato@empresa.com", key="emp_contato")
    st.text_input("Website", value="https://www.instagram.com/rise.ufal/", key="emp_link")
    st.text_input(
        "CEP",
        placeholder="00000-000",
        key="input_cep",
        on_change=callback_atualizar_cep,
    )
    st.text_area("Endereço", key="emp_end", height=70)

# CABEÇALHO PRINCIPAL COM TEXTO EXACTO DO PROFESSOR
col_l1, col_l2, col_l3 = st.columns([1, 1, 1])
with col_l2:
    try:
        url_logo_github = "https://raw.githubusercontent.com/Rogerleite1305/CPP-ROC-CHOISE/main/LOGO%20CPP-ROC-CHOICE.png"
        st.image(url_logo_github, width=220)
    except Exception:
        pass

st.markdown(
    """
    <div class="main-header">
        <h1>Decision Support System - CPP-ROC-CHOICE</h1>
        <p>Sistema de Apoio à Decisão Multicritério com Múltiplos Decisores Sob Incerteza. Para a problemática de escolha.</p>
    </div>
""",
    unsafe_allow_html=True,
)

tab_app, tab_sens, tab_manual = st.tabs(
    ["Painel de Avaliação Decisória", "Análise de Sensibilidade", "Manual do Método"]
)

with tab_manual:
    st.markdown(
        """
    ### **MANUAL DO USUÁRIO — CPP-ROC CHOICE**
    
    O **CPP-ROC CHOICE** combina a **Probabilidade de Composição Probabilística (CPP)** com a ponderação por **Rank Order Centroid (ROC)**.

    ---

    #### **Matriz de Perfis Estratégicos dos Decisores (Tabela 2)**
    Para cada decisor, deve ser associado um perfil combinando duas perspectivas:
    * **Coluna 1 (Tendência):** `Optimistic` (Otimista) ou `Progressive` (Progressista).
    * **Coluna 2 (Comportamento):** `Pessimistic` (Pessimista) ou `Conservative` (Conservador).
    """
    )

with tab_app:
    # -------------------------------------------------------------
    # ETAPA 1: ESTRUTURA DO PROBLEMA (DECISORES, CRITÉRIOS, ALTERNATIVAS)
    # -------------------------------------------------------------
    st.markdown("### <span class='step-badge'>ETAPA 1</span> **Definição da Estrutura do Problema**", unsafe_allow_html=True)
    st.caption("Cadastre o número de decisores, a quantidade de critérios e alternativas envolvidas.")
    
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        n_dms = st.number_input("Nº de Decisores", min_value=1, max_value=20, value=2, step=1, key="cfg_n_dms")
    with col_e2:
        n_crit = st.number_input("Nº de Critérios", min_value=2, max_value=10, value=3, step=1, key="cfg_n_crit")
    with col_e3:
        n_alt = st.number_input("Nº de Alternativas", min_value=2, max_value=20, value=3, step=1, key="cfg_n_alt")

    st.markdown("---")

    # Cadastramento de Alternativas e Critérios
    col_cad1, col_cad2 = st.columns(2)
    with col_cad1:
        st.markdown("#### **📌 Cadastro das Alternativas**")
        nomes_alt_globais = [
            st.text_input(
                f"Alternativa {a+1}:",
                value=st.session_state.get(f"glob_alt_{a}", f"Alternativa {a+1}"),
                key=f"glob_alt_{a}",
            )
            for a in range(n_alt)
        ]

    with col_cad2:
        st.markdown("#### **📊 Cadastro dos Critérios e Direção**")
        nomes_crit_globais = []
        tipos_crit_globais = []
        for c in range(n_crit):
            col_c1, col_c2 = st.columns([2, 1])
            with col_c1:
                nome_c = st.text_input(
                    f"Critério {c+1}:",
                    value=st.session_state.get(f"glob_crit_{c}", f"Critério {c+1}"),
                    key=f"glob_crit_{c}",
                )
                nomes_crit_globais.append(nome_c)
            with col_c2:
                tipo_c = st.selectbox(
                    "Tipo",
                    ["Benefício", "Custo"],
                    index=0,
                    key=f"tipo_crit_{c}",
                )
                tipos_crit_globais.append(tipo_c)

    st.markdown("---")

    # -------------------------------------------------------------
    # ETAPA 2: ENTRADA DOS DECISORES (PERFIS, NOTAS E PESOS ROC)
    # -------------------------------------------------------------
    st.markdown("### <span class='step-badge'>ETAPA 2</span> **Avaliação, Perfis dos Decisores e Ponderação ROC**", unsafe_allow_html=True)
    st.caption("Cada decisor informa sua matriz de desempenho, define sua ordenação de preferência dos critérios (ROC) e seleciona seu Perfil Decisório.")

    matrizes_dms = []
    ordens_dms = []
    perfis_dms = []
    nomes_dms_finais = []

    abas_dms = st.tabs([f"Decisor {d+1}" for d in range(n_dms)])

    # OPÇÕES ESTRITAMENTE EMBASADAS NA TABELA 2
    opcoes_coluna_1 = ["Optimistic (Otimista)", "Progressive (Progressista)"]
    opcoes_coluna_2 = ["Pessimistic (Pessimista)", "Conservative (Conservador)"]

    for d, aba in enumerate(abas_dms):
        with aba:
            col_id1, col_id2 = st.columns([1, 2])
            with col_id1:
                nome_dm = st.text_input(
                    f"Nome/Identificação do Decisor {d+1}:",
                    value=f"Decisor {d+1}",
                    key=f"nome_dm_input_{d}",
                )
                nomes_dms_finais.append(nome_dm)

            # PERFIL DO DECISOR (TABELA 2 DO ARTIGO)
            st.markdown(f"##### 👤 **Perfil Decisório do {nome_dm} (Tabela 2)**")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                perfil_col1 = st.selectbox(
                    f"Perfil Coluna 1 (Tendência) — {nome_dm}:",
                    opcoes_coluna_1,
                    index=0 if d % 2 == 0 else 1,
                    key=f"dm_perfil_col1_{d}",
                )
            with col_p2:
                perfil_col2 = st.selectbox(
                    f"Perfil Coluna 2 (Comportamento) — {nome_dm}:",
                    opcoes_coluna_2,
                    index=0 if d % 2 == 0 else 1,
                    key=f"dm_perfil_col2_{d}",
                )

            perfis_dms.append((perfil_col1, perfil_col2))
            st.divider()

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"##### 📝 **Matriz de Avaliação de Alternativas**")
                valores_iniciais = np.round(
                    np.random.rand(n_alt, n_crit) * val_max, 2
                )
                df_init = pd.DataFrame(
                    valores_iniciais,
                    columns=nomes_crit_globais,
                    index=nomes_alt_globais,
                )

                df_editada = st.data_editor(
                    df_init, key=f"matriz_indiv_{d}_v2"
                )
                matrizes_dms.append(df_editada.values)

            with col2:
                st.markdown("##### ⚖️ **Ordenação ROC dos Critérios**")
                st.caption("Ordene do mais importante (1º) ao menos importante:")
                ordem_indices = []
                criterios_disponiveis = list(range(n_crit))
                
                for r in range(n_crit):
                    crit_selecionado_idx = st.selectbox(
                        f"{r+1}º Lugar de Importância:",
                        options=criterios_disponiveis,
                        format_func=lambda x: nomes_crit_globais[x],
                        key=f"dm_{d}_rank_{r}",
                    )
                    ordem_indices.append(crit_selecionado_idx)
                
                ordens_dms.append(ordem_indices)

    st.markdown("---")

    # -------------------------------------------------------------
    # ETAPA 3: EXECUÇÃO E RECOMENDAÇÃO FINAL
    # -------------------------------------------------------------
    st.markdown("### <span class='step-badge'>ETAPA 3</span> **Processamento & Recomendações**", unsafe_allow_html=True)

    if st.button("🚀 EXECUTAR ANÁLISE DE DECISÃO", key="btn_executar"):
        st.session_state["calculo_executado"] = True

    if st.session_state.get("calculo_executado", False):
        res = executar_cpp_roc_choice(
            matrizes_dms=matrizes_dms,
            ordens_criterios_dms=ordens_dms,
            perfis_dms=perfis_dms,
            tipos_criterios=tipos_crit_globais,
            val_max=val_max,
            pct_incerteza=pct_incerteza,
            n_simulacoes=n_simulacoes,
        )

        st.session_state["ultimo_resultado"] = res

        df_res = pd.DataFrame(
            {
                "Alternativa": nomes_alt_globais,
                "M_i (Probabilidade de Excelência)": res["M_i"],
                "m_i (Probabilidade de Pior Desempenho)": res["m_i"],
            }
        )

        opt_max_nome = nomes_alt_globais[res["otima_max_Mi"]]
        opt_min_nome = nomes_alt_globais[res["otima_min_mi"]]

        st.markdown("#### **Resultados da Análise Multicritério**")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(
                f"""
            <div class="recommendation-card">
                <h4 style="margin:0; color:#16a34a;">🏆 Recomendação Principal (Maximotimizador)</h4>
                <p style="margin:5px 0 0 0; font-size:16px;"><b>{opt_max_nome}</b></p>
                <span style="font-size:12px; color:#15803d;">Maior probabilidade de excelência (M<sub>i</sub> = {res['M_i'][res['otima_max_Mi']]:.4f})</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col_r2:
            st.markdown(
                f"""
            <div class="recommendation-card" style="background-color:#eff6ff; border-color:#bfdbfe; border-left-color:#2563eb;">
                <h4 style="margin:0; color:#1d4ed8;">🛡️ Opção de Menor Risco (Conservadora)</h4>
                <p style="margin:5px 0 0 0; font-size:16px;"><b>{opt_min_nome}</b></p>
                <span style="font-size:12px; color:#1e40af;">Menor probabilidade de pior desempenho (m<sub>i</sub> = {res['m_i'][res['otima_min_mi']]:.4f})</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.dataframe(df_res.style.highlight_max(subset=["M_i (Probabilidade de Excelência)"], color="#dcfce7"), use_container_width=True)

        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            fig_buf = gerar_grafico_membro(df_res)
            st.image(fig_buf, caption="Distribuição de Probabilidades M_i vs m_i por Alternativa", use_container_width=True)

        with col_g2:
            st.markdown("#### **Ações de Exportação**")
            
            pdf_bytes = gerar_pdf_relatorio(
                df_resultados=df_res,
                resultado_completo=res,
                opt_max_nome=opt_max_nome,
                opt_min_nome=opt_min_nome,
                n_dms=n_dms,
                n_alt=n_alt,
                n_crit=n_crit,
                nomes_criterios=nomes_crit_globais,
                tipos_criterios=tipos_crit_globais,
                nomes_dms=nomes_dms_finais,
                empresa_nome=st.session_state.get("emp_nome", ""),
                empresa_cnpj=st.session_state.get("emp_cnpj", ""),
                empresa_contato=st.session_state.get("emp_contato", ""),
                empresa_cep=st.session_state.get("cep_formatado", ""),
                empresa_end=st.session_state.get("emp_end", ""),
                empresa_link=st.session_state.get("emp_link", ""),
                logo_file=logo_file,
            )

            st.download_button(
                label="📄 Baixar Relatório Completo em PDF",
                data=pdf_bytes,
                file_name=f"relatorio_decisao_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                key="btn_pdf",
            )

with tab_sens:
    st.markdown("### **Análise de Sensibilidade Estocástica**")
    if st.session_state.get("calculo_executado", False):
        st.info("Altere o nível de variação perturbativa (%) e execute a simulação para observar a estabilidade dos rankings de M_i e m_i.")
    else:
        st.warning("Execute o cálculo na aba 'Painel de Avaliação Decisória' primeiro.")

st.markdown(
    """
    <div class="footer-rights">
        DSS CPP-ROC CHOICE — Desenvolvido por <a href="https://www.instagram.com/rise.ufal/" target="_blank">RISE/UFAL</a> © Todos os direitos reservados.
    </div>
""",
    unsafe_allow_html=True,
)
