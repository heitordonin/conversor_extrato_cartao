import io
import re
import yaml
import pdfplumber
import pandas as pd
from decimal import Decimal, InvalidOperation
from dateutil import parser
from datetime import datetime
from unidecode import unidecode
import streamlit as st

st.set_page_config(page_title="Extrator de Extratos – MVP", page_icon="💳", layout="wide")
st.title("💳 Extrator de Extratos → CSV (MVP)")

# ----------------------
# Helpers gerais
# ----------------------

def load_template(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_decimal_pt(text: str) -> Decimal:
    """Normaliza números em formatos PT/EN de forma inteligente.
    Aceita:
    - PT-BR: 1.234,56 ou -55,90
    - EN:    1,234.56 ou 733.33
    Regras:
    - Se existir vírgula e ponto, assume que o *último separador* é o decimal.
      Remove o outro como milhar.
    - Se existir só vírgula → vírgula é decimal (remove pontos como milhar).
    - Se existir só ponto → ponto é decimal (remove vírgulas como milhar).
    - Remove espaços, símbolo de moeda e normaliza sinal unicode.
    """
    t = text.strip()
    t = t.replace("−", "-")  # sinal unicode
    t = t.replace("R$", "")
    t = t.replace(" ", "")

    has_comma = "," in t
    has_dot = "." in t

    if has_comma and has_dot:
        last_comma = t.rfind(',')
        last_dot = t.rfind('.')
        if last_comma > last_dot:
            # vírgula é decimal
            t = t.replace('.', '')
            t = t.replace(',', '.')
        else:
            # ponto é decimal
            t = t.replace(',', '')
            # ponto fica como decimal
    elif has_comma:
        # vírgula é decimal
        t = t.replace('.', '')
        t = t.replace(',', '.')
    elif has_dot:
        # ponto é decimal
        t = t.replace(',', '')
        # mantém ponto
    # senão: dígitos puros → inteiro

    try:
        return Decimal(t)
    except InvalidOperation:
        raise ValueError(f"Valor inválido: {text}")



def extract_text_lines(pdf_bytes: bytes) -> list[str]:
    """Extrai todas as linhas de texto (sequenciais) do PDF."""
    lines = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for ln in text.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)
    return lines


def keep_only_between_markers(lines: list[str], start_markers: list[str], end_markers: list[str] | None = None) -> list[str]:
    """Mantém apenas as linhas entre um dos *start_markers* e, opcionalmente, antes de um *end_marker*."""
    start_idx = None
    end_idx = None

    # normaliza para busca mais tolerante
    norm = [unidecode(l).upper() for l in lines]
    sm = [unidecode(s).upper() for s in start_markers]
    em = [unidecode(e).upper() for e in (end_markers or [])]

    for i, ln in enumerate(norm):
        if any(s in ln for s in sm):
            start_idx = i + 1  # começa na próxima linha
            break

    if start_idx is None:
        return lines  # não encontrou marcador; devolve tudo (modo tolerante)

    if em:
        for j in range(start_idx, len(norm)):
            if any(e in norm[j] for e in em):
                end_idx = j
                break

    return lines[start_idx:end_idx] if end_idx else lines[start_idx:]


# ----------------------
# Funções de data auxiliares (PT-BR)
# ----------------------

def parse_ptbr_day_month(raw_date, year_hint=None):
    """Converte datas como '07 FEV' ou '07/02' em datetime(YYYY,MM,DD) usando mapa PT-BR.
    Não depende do parser para meses abreviados PT-BR.
    """
    s = unidecode(str(raw_date)).strip().upper()

    # Normaliza separadores
    s_norm = s.replace('-', '/').replace('  ', ' ')

    months = {
        'JAN': 1, 'JANEIRO': 1,
        'FEV': 2, 'FEVEREIRO': 2,
        'MAR': 3, 'MARCO': 3, 'MARÇO': 3,
        'ABR': 4, 'ABRIL': 4,
        'MAI': 5, 'MAIO': 5,
        'JUN': 6, 'JUNHO': 6,
        'JUL': 7, 'JULHO': 7,
        'AGO': 8, 'AGOSTO': 8,
        'SET': 9, 'SETEMBRO': 9,
        'OUT': 10, 'OUTUBRO': 10,
        'NOV': 11, 'NOVEMBRO': 11,
        'DEZ': 12, 'DEZEMBRO': 12,
    }

    # Caso numérico: dd/mm ou dd/mm/aaaa
    if '/' in s_norm and s_norm.split('/')[0].isdigit():
        parts = [p.strip() for p in s_norm.split('/') if p.strip()]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            day = int(parts[0])
            month = int(parts[1])
            year = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else int(year_hint or datetime.today().year)
            return datetime(year, month, day)

    # Caso com mês por extenso/abreviado: "07 FEV" ou "7 FEVEREIRO"
    tokens = s.split()
    if len(tokens) >= 2 and tokens[0].isdigit():
        day = int(tokens[0])
        mon_token = tokens[1]
        mon_key = mon_token if mon_token in months else mon_token[:3]
        if mon_key in months:
            month = months[mon_key]
            year = int(year_hint or datetime.today().year)
            return datetime(year, month, day)

    # Fallback: tenta parser padrão
    return parser.parse(str(raw_date), dayfirst=True, fuzzy=True)


# ----------------------
# Parsers por banco
# ----------------------

class ParseResult(pd.DataFrame):
    @property
    def _constructor(self):
        return ParseResult


def parse_nubank(pdf_bytes: bytes, cfg: dict) -> ParseResult:
    lines = extract_text_lines(pdf_bytes)

    # 1) detectar período vigente (1ª página normalmente)
    period = None
    for ln in lines[:40]:
        m = re.search(cfg["header"]["period_regex"], ln, flags=re.IGNORECASE)
        if m:
            period = m.group("period")
            break

    # 2) ano de referência (procura um token 20xx nas primeiras linhas)
    year_hint = None
    for ln in lines[:120]:
        for tok in ln.split():
            if tok.isdigit() and len(tok) == 4 and tok.startswith('20'):
                year_hint = int(tok)
                break
        if year_hint:
            break

    # 3) manter somente a seção TRANSAÇÕES
    section_lines = keep_only_between_markers(
        lines,
        start_markers=cfg["section"]["start_markers"],
        end_markers=cfg["section"].get("end_markers", []),
    )

    row_re = re.compile(cfg["row_pattern"], flags=re.IGNORECASE)
    parsed = []

    for ln in section_lines:
        m = row_re.search(ln)
        if not m:
            continue
        raw_date = m.group("date").strip()
        raw_desc = m.group("desc").strip()
        raw_amount = m.group("amount").strip()

        # Data robusta para meses PT-BR (JAN/FEV/MAR...)
        try:
            dt = parse_ptbr_day_month(raw_date, year_hint)
        except Exception:
            continue

        # Valor com detecção de sinal antes do R$ (ex.: "-R$ 0,25").
        idx = ln.find(raw_amount)
        neg = False
        if idx != -1:
            before = ln[max(0, idx - 6):idx]
            if "-" in before or "−" in before:
                neg = True
        val = normalize_decimal_pt(raw_amount)
        if neg:
            val = -val

        # Regra do cartão: inverter saldo
        val = -val

        # Parcelas (usa regex do template)
        parc_atual, parc_total = "", ""
        inst_rx = cfg.get("installment_regex")
        if inst_rx:
            pm = re.search(inst_rx, raw_desc)
            if pm:
                parc_atual, parc_total = pm.group(1), pm.group(2)

        parsed.append({
            "data": dt.strftime("%d/%m/%Y"),
            "descricao": raw_desc,
            "valor": f"{val:.2f}".replace(".", ","),
            "parcela_atual": parc_atual,
            "parcelas_totais": parc_total,
            "categoria": "",
            "observacao": f"Nubank | Periodo: {period or ''}".strip()
        })

    return ParseResult(parsed)


def parse_bb(pdf_bytes: bytes, cfg: dict) -> ParseResult:
    lines = extract_text_lines(pdf_bytes)

    # 1) "Fatura fechada em: dd/mm/aaaa"
    closing_date = None
    for ln in lines[:200]:
        m = re.search(cfg["header"]["closed_regex"], ln, flags=re.IGNORECASE)
        if m:
            try:
                closing_date = parser.parse(m.group("closed"), dayfirst=True)
                break
            except Exception:
                pass
    if not closing_date:
        closing_date = datetime.today()

    close_m = closing_date.month
    close_y = closing_date.year
    prev_m = 12 if close_m == 1 else close_m - 1

    # 2) Somente "Detalhes da fatura"
    section_lines = keep_only_between_markers(
        lines,
        start_markers=cfg["section"]["start_markers"],
        end_markers=cfg["section"].get("end_markers", []),
    )

    row_re = re.compile(cfg["row_pattern"], flags=re.IGNORECASE)
    parsed = []

    for ln in section_lines:
        m = row_re.search(ln)
        if not m:
            continue
        raw_date = m.group("date").strip()
        raw_desc = m.group("desc").strip()
        raw_amount = m.group("amount").strip()

        # 3) Data com regra especial
        #    a) Se vier SEM ano (ex.: 18/11):
        #       - definimos o ano para NÃO ficar "no futuro" do fechamento:
        #         year = close_y se month <= close_m, senão year = close_y - 1
        #       - depois calculamos months_diff; se >=2, ajustamos para mês do fechamento.
        #    b) Se vier COM ano:
        #       - se months_diff >= 2 (muito antiga) ou months_diff < 0 (futura), ajusta para mês/ano do fechamento.
        mdate = re.match(r'(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?', raw_date)
        no_year = False
        if mdate:
            day = int(mdate.group(1))
            month = int(mdate.group(2))
            if mdate.group(3):
                year = int(mdate.group(3))
                if year < 100:
                    year += 2000
            else:
                no_year = True
                year = close_y if month <= close_m else close_y - 1
            try:
                tx_date = datetime(year, month, day)
            except ValueError:
                continue
        else:
            try:
                tx_date = parser.parse(raw_date, dayfirst=True)
            except Exception:
                continue

        if no_year:
            # Já garantimos que não fica "no futuro". Agora aplicamos a janela de até 1 mês.
            months_diff = (close_y - tx_date.year) * 12 + (close_m - tx_date.month)
            if months_diff >= 2:
                tx_date = tx_date.replace(month=close_m, year=close_y)
        else:
            months_diff = (close_y - tx_date.year) * 12 + (close_m - tx_date.month)
            if months_diff >= 2 or months_diff < 0:
                tx_date = tx_date.replace(month=close_m, year=close_y)

        # 4) Valor (regra do cartão: inverter sinal)
        val = normalize_decimal_pt(raw_amount)
        val = -val

        # 5) Monta linha (a UI depois já reduz para data/descricao/valor)
        parsed.append({
            "data": tx_date.strftime("%d/%m/%Y"),
            "descricao": raw_desc,
            "valor": f"{val:.2f}".replace(".", ","),
        })

    return ParseResult(parsed)



def parse_sumup(pdf_bytes: bytes, cfg: dict) -> ParseResult:
    lines = extract_text_lines(pdf_bytes)

    # manter apenas a seção Depósitos recebidos
    section_lines = keep_only_between_markers(
        lines,
        start_markers=cfg["section"]["start_markers"],
        end_markers=cfg["section"].get("end_markers", []),
    )

    row_re = re.compile(cfg["row_pattern"], flags=re.IGNORECASE)
    parsed = []

    def yyyymmdd_to_ddmmyyyy(s: str) -> str:
        # entrada: "20250331" → "31/03/2025"
        y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
        return f"{d:02d}/{m:02d}/{y}"

    for ln in section_lines:
        m = row_re.search(ln)
        if not m:
            continue
        code = m.group("code").strip()
        raw_total = m.group("total").strip()
        raw_fee = m.group("fee").strip()

        # data extraída dos 8 primeiros dígitos do código (aaaa mm dd)
        date_str = yyyymmdd_to_ddmmyyyy(code[:8])

        total_val = normalize_decimal_pt(raw_total)
        fee_val = normalize_decimal_pt(raw_fee)  # já vem negativo no PDF (ex.: -R$31,53)

        # regra do Heitor: não inverter no SumUp; duplicar lançamentos (Total e Taxas)
        parsed.append({
            "data": date_str,
            "descricao": f"SUMUP – Depósito {code}",
            "valor": f"{total_val:.2f}".replace(".", ","),
            "parcela_atual": "",
            "parcelas_totais": "",
            "categoria": "Receitas",
            "observacao": "Valor Total"
        })
        parsed.append({
            "data": date_str,
            "descricao": f"SUMUP – Taxa {code}",
            "valor": f"{fee_val:.2f}".replace(".", ","),
            "parcela_atual": "",
            "parcelas_totais": "",
            "categoria": "Taxas",
            "observacao": "Taxa"
        })

    return ParseResult(parsed)


# ----------------------
# UI – seleção e processamento
# ----------------------

BANKS = {
    "Nubank (v1)": {
        "template": "templates/nubank_v1.yaml",
        "parser": parse_nubank,
    },
    "Banco do Brasil (v1)": {
        "template": "templates/bb_v1.yaml",
        "parser": parse_bb,
    },
    "SumUp (v1)": {
        "template": "templates/sumup_v1.yaml",
        "parser": parse_sumup,
    },
}

col1, col2 = st.columns([2, 1])
with col1:
    bank_label = st.selectbox("Banco / Layout", list(BANKS.keys()))
with col2:
    client = st.text_input("Cliente (referência)", placeholder="Ex.: Clínica Rosa")

uploaded = st.file_uploader("PDF do extrato / fatura", type=["pdf"])
process_btn = st.button("Processar", type="primary", disabled=not uploaded)

if process_btn and uploaded is not None:
    cfg = load_template(BANKS[bank_label]["template"])
    parser_fn = BANKS[bank_label]["parser"]

    pdf_bytes = uploaded.read()

    try:
        df = parser_fn(pdf_bytes, cfg)

        # Manter apenas colunas essenciais (data, descricao, valor)
        try:
            df = df[["data", "descricao", "valor"]]
        except Exception:
            pass
    except Exception as e:
        st.error(f"Erro ao processar: {e}")
        st.stop()

    if df.empty:
        st.warning("Nenhuma linha reconhecida. Ajuste o template YAML ou verifique se a seção correta foi identificada.")
    else:
        st.subheader("Prévia dos lançamentos")
        st.dataframe(df, use_container_width=True, height=420)

        # CSV com separador ; (mais amigável ao Excel PT-BR) e BOM UTF-8
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False, sep=";")
        csv_bytes = csv_buf.getvalue().encode("utf-8-sig")

        fname = f"{client or 'cliente'}_{bank_label.split('(')[0].strip().replace(' ', '_')}.csv"
        st.download_button(
            label="⬇️ Baixar CSV",
            data=csv_bytes,
            file_name=fname,
            mime="text/csv",
        )

        # Resuminho
        st.info(f"Linhas geradas: {len(df)} | Banco: {bank_label}")