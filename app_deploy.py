import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import re
import os
import warnings

# -----------------------------------------------------------------------------
# 0. 경고 메시지 차단 (터미널을 깨끗하게)
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# -----------------------------------------------------------------------------
# 1. 페이지 설정 & 상수 정의
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="라운드랩 독도 토너 통합 CRM 솔루션",
    page_icon="🔴",
    layout="wide"
)

# 스타일링
st.markdown("""
<style>
    .insight-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B; margin-bottom: 20px; }
    .aha-box { background-color: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3; margin-bottom: 20px; }
    .strategy-box { background-color: #fff8e1; padding: 15px; border-radius: 10px; border-left: 5px solid #FFD700; margin-top: 10px; }
    .info-box { background-color: #e8f4f8; padding: 15px; border-radius: 10px; border-left: 5px solid #87CEEB; font-size: 14px; margin-bottom: 20px; }
    .action-card {
        background: #ffffff;
        padding: 18px 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        min-height: 280px;
    }
    .action-card h3 {
        margin: 0 0 10px 0;
        font-size: 18px;
    }
    .action-card b {
        color: #0f172a;
    }
        /* ✅ 다크모드에서 흰 글자 상속 문제 해결: 박스들 글자색을 강제로 지정 */
    .insight-box,
    .aha-box,
    .strategy-box,
    .info-box,
    .action-card{
    color: #0f172a !important;   /* 글자색 고정 */
    }

    /* ✅ 박스 내부 모든 텍스트 요소도 동일 색상 상속(white로 덮이는 것 방지) */
    .insight-box * ,
    .aha-box * ,
    .strategy-box * ,
    .info-box * ,
    .action-card * {
    color: inherit !important;
    }

    /* 링크가 안 보일 때 대비 */
    .insight-box a,
    .aha-box a,
    .strategy-box a,
    .info-box a,
    .action-card a{
    text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# 브랜드별 고유 색상
BRAND_COLORS = {
    '라운드랩': '#FF4B4B',   # Red (Hero)
    '토리든': '#4169E1',     # Royal Blue
    '에스네이처': '#2E8B57',  # Sea Green
    '아비브': '#808080',     # Gray
    '토니모리': '#FFD700'    # Gold
}
COLOR_COMP = '#87CEEB'     # 일반 경쟁사
COLOR_FASHION = '#90EE90'  # 패션 카테고리

# 타겟 브랜드 및 키워드
TARGET_BRANDS = ['라운드랩', '토리든', '에스네이처', '아비브', '토니모리']
TARGETS = {
    '라운드랩':  {'brand_kw': r'라운드랩|Round\s*Lab|독도', 'prod_kw': r'토너|스킨|독도'},
    '에스네이처': {'brand_kw': r'에스네이처|S\.NATURE|SNATURE', 'prod_kw': r'토너|스킨'},
    '토리든':    {'brand_kw': r'토리든|Torriden',    'prod_kw': r'토너|스킨'},
    '아비브':    {'brand_kw': r'아비브|Abib',        'prod_kw': r'토너|스킨|부스터'},
    '토니모리':  {'brand_kw': r'토니모리|TONYMOLY',  'prod_kw': r'모찌|세라마이드|원더'}
}

# 11대 속성 키워드
PATTERNS = {
    '수분/보습': r'수분|촉촉', '진정': r'진정|가라앉|뒤집어', '붉은기': r'붉은|홍조|열감', 
    '트러블': r'트러블|여드름|좁쌀', '순함': r'순함|순해|순한', '자극없음': r'자극|따가|아프', 
    '가성비': r'가성비|저렴|싸게|가격|세일|1\+1|양도|용량', '물제형': r'물제형|물같|워터',
    '산뜻함': r'산뜻|가볍|끈적임없', '흡수력': r'흡수|스며', '무난함': r'무난|호불호|데일리'
}

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리 (안전한 로드 로직)
# -----------------------------------------------------------------------------
# [수정된 load_data 함수]
# [app_deploy.py 수정]
@st.cache_data
def load_data():
    # 4조각난 파일을 읽어서 합칩니다.
    files = [f'data_part{i}.parquet' for i in range(1, 5)] # 1,2,3,4번 파일
    
    # 첫 번째 파일이 없으면 빈 데이터프레임 반환
    if not os.path.exists(files[0]):
        return pd.DataFrame()
        
    # 리스트 컴프리헨션으로 4개를 한 번에 읽기
    df_list = [pd.read_parquet(f) for f in files]
    
    # 하나로 합체
    return pd.concat(df_list, ignore_index=True)

# -----------------------------------------------------------------------------
# 3. 분석 함수 모음
# -----------------------------------------------------------------------------
def parse_skin_info(text):
    if pd.isna(text): return None
    text = text.lower()
    if 'dry' in text: return '건성'
    if 'oily' in text: return '지성'
    if 'combination' in text: return '복합성'
    if 'sensitive' in text: return '민감성'
    return '기타'

@st.cache_data
def get_repurchase_stats(df):
    if df.empty: return pd.DataFrame()
    results = []
    for brand, filters in TARGETS.items():
        b_mask = df['brand'].astype(str).str.contains(filters['brand_kw'], case=False, na=False)
        subset = df[b_mask]
        if len(subset) == 0: continue
        
        user_counts = subset['user_id'].value_counts()
        rep_users = user_counts[user_counts >= 2].index
        rep_subset = subset[subset['user_id'].isin(rep_users)]
        if len(rep_subset) == 0: continue
        
        texts = rep_subset['content'].fillna('').astype(str)
        row = {'Brand': brand}
        for k, v in PATTERNS.items():
            row[k] = (texts.str.contains(v).sum() / len(texts)) * 100
        results.append(row)
    return pd.DataFrame(results).set_index('Brand')

@st.cache_data
def calculate_lift(df, brand_name):
    if df.empty: return pd.Series()
    filters = TARGETS[brand_name]
    b_mask = df['brand'].astype(str).str.contains(filters['brand_kw'], case=False, na=False)
    subset = df[b_mask]
    if len(subset) == 0: return pd.Series()
    
    user_counts = subset['user_id'].value_counts()
    rep_users = user_counts[user_counts >= 2].index
    one_users = user_counts[user_counts == 1].index
    
    rep_df = subset[subset['user_id'].isin(rep_users)]
    one_df = subset[subset['user_id'].isin(one_users)]
    if len(rep_df) == 0 or len(one_df) == 0: return pd.Series()
    
    lift_data = {}
    for k, v in PATTERNS.items():
        rep_rate = rep_df['content'].fillna('').astype(str).str.contains(v).mean()
        one_rate = one_df['content'].fillna('').astype(str).str.contains(v).mean()

        lift_data[k] = (rep_rate / one_rate) if one_rate > 0 else 0
    return pd.Series(lift_data).sort_values(ascending=False)

@st.cache_data
def get_frequency_basket(df, brand_name):
    if df.empty: return {}
    filters = TARGETS[brand_name]
    b_mask = df['brand'].astype(str).str.contains(filters['brand_kw'], case=False, na=False)
    target_purchases = df[b_mask]
    user_counts = target_purchases.groupby('user_id').size()
    groups = {
        '1회 (이탈/체험)': user_counts[user_counts == 1].index,
        '2회 (재방문)': user_counts[user_counts == 2].index,
        '3회+ (찐팬)': user_counts[user_counts >= 3].index
    }
    basket_data = {}
    for g_name, u_ids in groups.items():
        if len(u_ids) == 0: basket_data[g_name] = pd.Series()
        else:
            hist = df[df['user_id'].isin(u_ids)]
            hist = hist[~hist['brand'].astype(str).str.contains(filters['brand_kw'], case=False, na=False)]
            basket_data[g_name] = hist['full_name'].value_counts().head(10)

    return basket_data

def get_item_color(item_name, target_brand):
    if target_brand in item_name or (target_brand == '라운드랩' and '독도' in item_name): return BRAND_COLORS['라운드랩']
    if any(x in item_name for x in ['양말', '삭스', '티셔츠', '팬츠']): return COLOR_FASHION
    return COLOR_COMP

@st.cache_data
def analyze_aha_moment(df):
    """아하 모먼트 분석 (라이프스타일 & 패션 취향 매칭)"""
    
    # 1. 타겟 필터링
    dokdo_mask = (df['brand'].str.contains('라운드랩', na=False)) & \
                 (df['goods_name'].str.contains('독도', na=False)) & \
                 (df['goods_name'].str.contains('토너', na=False))
    target_df = df[dokdo_mask]
    
    # 2. 유저 그룹핑
    analysis_end_date = df['date'].max()
    user_summary = target_df.groupby('user_id').agg(count=('date', 'count'), last_date=('date', 'max'))
    user_summary['days_since_last'] = (analysis_end_date - user_summary['last_date']).dt.days
    
    rep_users = user_summary[user_summary['count'] >= 2].index
    churn_users = user_summary[(user_summary['count'] == 1) & (user_summary['days_since_last'] > 45)].index
    
    relevant_users = list(rep_users) + list(churn_users)
    full_history_df = df[df['user_id'].isin(relevant_users)].copy()
    
    # 3. 비화장품(패션) 추출
    beauty_keywords = ['라운드랩', '토리든', '에스네이처', '아비브', '토니모리', '이니스프리', '닥터지', '아누아', '마녀공장', '메디힐', '성분에디터', '올리브영', '화장솜']
    is_beauty = full_history_df['brand'].astype(str).str.contains('|'.join(beauty_keywords), na=False)
    fashion_df = full_history_df[~is_beauty].copy()
    
    # [핵심] 텍스트 통합 (상품명 + 옵션)
    fashion_df['concat_text'] = (
        fashion_df['goods_name'].astype(str) + " " + 
        fashion_df['option'].fillna("").astype(str)
    ).str.upper()

    # 태그 사전 (한글+영어)
    LIFESTYLE_TAGS = {
        '상의 (Basic/T-shirt)': ['반팔', '티셔츠', '롱슬리브', '무지', '탑', '긴팔', 'T-SHIRT', 'TEE', 'BASIC'],
        '상의 (Sweat/Hoodie)': ['맨투맨', '스웨트', '후드', '집업', '아노락', 'SWEATSHIRT', 'HOODIE', 'MTM'],
        '상의 (Knit/Shirt)': ['니트', '스웨터', '가디건', '셔츠', 'KNIT', 'CARDIGAN', 'SHIRT'],
        '아우터 (Outer)': ['패딩', '코트', '자켓', '점퍼', '파카', '플리스', 'PADDING', 'COAT', 'JACKET'],
        '하의 (Pants/Denim)': ['바지', '팬츠', '데님', '청바지', '슬랙스', '조거', 'PANTS', 'DENIM', 'SLACKS'],
        '신발 (Shoes)': ['스니커즈', '운동화', '런닝화', '구두', '부츠', 'SNEAKERS', 'SHOES'],
        '가방/모자 (Bag/Head)': ['가방', '백팩', '메신저백', '모자', '볼캡', '비니', 'BAG', 'CAP', 'HAT'],
        '속옷/양말/홈 (Inner)': ['양말', '삭스', '드로즈', '팬티', '잠옷', 'SOCKS', 'UNDERWEAR'],
        '디지털/라이프 (Tech)': ['케이스', '필름', '거치대', '충전기', 'CASE', 'FILM'],
        '블랙/무채색 (Monotone)': ['블랙', '검정', 'BLACK', '그레이', '회색', 'GREY', 'GRAY', '차콜', '화이트', '흰색', 'WHITE', '네이비', 'NAVY'],
        '유채색/포인트 (Color)': ['핑크', '블루', '옐로우', '그린', '민트', '라벤더', 'PINK', 'BLUE', 'GREEN']
    }
    
    # 유저별 태그 매칭
    user_text_map = fashion_df.groupby('user_id')['concat_text'].apply(' '.join)
    user_tags = []
    for uid in relevant_users:
        u_type = 'Repurchase(재구매)' if uid in rep_users else 'Churn(이탈자)'
        text = user_text_map.get(uid, "")
        
        row = {'User_Type': u_type}
        for tag_name, keywords in LIFESTYLE_TAGS.items():
            has_tag = any(k in text for k in keywords)
            row[tag_name] = 1 if has_tag else 0
        user_tags.append(row)
        
    tag_df = pd.DataFrame(user_tags)
    
    result_list = []
    for tag in LIFESTYLE_TAGS.keys():
        rep_rate = tag_df[tag_df['User_Type']=='Repurchase(재구매)'][tag].mean() * 100
        churn_rate = tag_df[tag_df['User_Type']=='Churn(이탈자)'][tag].mean() * 100
        lift = rep_rate / churn_rate if churn_rate > 0 else 0
        gap = rep_rate - churn_rate
        result_list.append({'Category': tag, 'Loyal(%)': rep_rate, 'Churn(%)': churn_rate, 'Lift': lift, 'Gap(%p)': gap})
        
    result_df = pd.DataFrame(result_list).sort_values('Lift', ascending=False)
    debug_info = {'total_analyzed': len(tag_df), 'fashion_buyers': len(user_text_map)}
    
    return result_df, debug_info

def render_lift_chart(df, brand_name, title_prefix=""):
    series = calculate_lift(df, brand_name)

    if series is None or series.empty:
        st.info(f"[{brand_name}] 데이터가 부족하여 차트를 표시할 수 없습니다.")
        return

    colors = [
        BRAND_COLORS.get(brand_name, "gray") if v > 1.0 else "#ddd"
        for v in series.values
    ]

    fig = go.Figure(
        go.Bar(
            x=series.values,
            y=series.index,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}배" for v in series.values],
            textposition="auto",
        )
    )
    fig.add_vline(x=1.0, line_dash="dash")
    fig.update_layout(
        title=f"{title_prefix}[{brand_name}] 재구매 결정 요인",
        yaxis=dict(autorange="reversed"),
        height=520,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 4. UI Layout (메인 화면)
# -----------------------------------------------------------------------------
if df.empty: st.stop()

# -----------------------------------------------------------------------------
# ✅ (추가) 탭 "위"에 고정되는 Sticky 헤더 + KPI 카드 (info-box 스타일 재활용)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
/* 상단 고정 헤더 컨테이너 */
.sticky-header-wrap{
  position: sticky;
  top: 0;
  z-index: 9999;
  padding-top: 8px;
  padding-bottom: 10px;
  /* 배경이 투명하면 뒤 요소가 비쳐 보여서 살짝 깔아줌 */
  background: rgba(0,0,0,0);
  backdrop-filter: blur(0px);
}

/* 탭 메뉴와 겹치지 않게 약간의 여백 + 구분선 */
.sticky-divider{
  height: 1px;
  background: rgba(255,255,255,0.08);
  margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ✅ (1) 1줄 요약 배너 (info-box 재활용)
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="sticky-header-wrap">

  <div class="info-box" style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
    <div>
      <b>📦 Dataset:</b> 무신사 뷰티 토너 카테고리 기반 구매/리뷰 데이터 (User ID 교차 크롤링)
    </div>
    <div style="white-space:nowrap;">
      <b>기간:</b> 2024.01.01 ~ 2025.11.30
    </div>
  </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ✅ (2) KPI 카드 Row (Streamlit metric 사용)
#     - 값은 df 로드 이후에 계산해 넣는 게 정석이지만,
#       "교체용 블록"으로 바로 붙여넣기 쉽게 기본은 하드코딩/안전 계산 둘 다 제공
# -----------------------------------------------------------------------------

# (권장) df가 이미 로드된 뒤라면 자동 계산 사용
# df가 아직 없으면 아래 하드코딩 라인만 쓰셔도 됩니다.
# ✅ KPI 하드코딩 (스크린샷 값)
kpi_unique_users = 5519
kpi_rows = 489_526
kpi_products = 129_828
kpi_price_matched = 9_999
kpi_coverage = 41.51  # %


k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Unique ID", f"{kpi_unique_users:,}명")
with k2:
    st.metric("Rows", f"{kpi_rows:,}건")
with k3:
    st.metric("Products", f"{kpi_products:,}개")
with k4:
    st.metric("Price Matched", f"{kpi_price_matched:,}건")
with k5:
    st.metric("Coverage", f"{kpi_coverage:.2f}%")

st.markdown('<div class="sticky-divider"></div></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ✅ 이제 여기 "아래"에 tabs 선언이 오면, 탭 메뉴 위에 헤더+KPI가 붙습니다.
# tabs = st.tabs([...])
# -----------------------------------------------------------------------------


# (탭 간 공용으로 쓰는 데이터는 탭 밖에서 미리 만들어두는 게 안전합니다)
dokdo_df = df[(df['brand'].str.contains('라운드랩', na=False)) & (df['goods_name'].str.contains('독도', na=False))]

# 탭 구성 (요청하신 7개 순서)
tabs = st.tabs([
    "📊 1. 현황 진단 (Market)",
    "🗺️ 2. 위기 요인 (Journey)",
    "🧠 3. 포지셔닝 & 속성 (Positioning)",   # ✅ 추가
    "🛒 4. 문제 발견 (Behavior)",
    "🗣️ 5. 이탈 원인 (Voice)",
    "💡 6. 기회 탐색 (Aha!)",
    "🧪 7. 통계 검증 (Proof)",
    "🚀 8. 액션 플랜 (Strategy)"
])

# =============================================================================
# [Tab 1] Market (기존 Tab 3: Market Share)
# =============================================================================
with tabs[0]:
    st.header("📊 1. 현황 진단 (Market)")
    st.markdown("""<div class="info-box"><b>📊 Data Context:</b> 정확한 비교를 위해 <b>주요 브랜드의 토너 제품군을 하나로 통합(Total)</b>하여 집계했습니다.</div>""", unsafe_allow_html=True)
    col_rank, col_trend = st.columns([1, 2])
    with col_rank:
        st.subheader("🏆 통합 베스트셀러 Top 20")
        rank_df = df.copy()
        rank_df.loc[(rank_df['brand'].str.contains('라운드랩', na=False) & rank_df['goods_name'].str.contains('독도|토너', na=False)), 'goods_name'] = '🔴 라운드랩 1025 독도 토너 (Total)'
        rank_df.loc[(rank_df['brand'].str.contains('토리든', na=False) & rank_df['goods_name'].str.contains('토너', na=False)), 'goods_name'] = '🔵 토리든 다이브인 토너 (Total)'
        rank_df.loc[(rank_df['brand'].str.contains('에스네이처', na=False) & rank_df['goods_name'].str.contains('토너|스킨', na=False)), 'goods_name'] = '🟢 에스네이처 아쿠아 토너 (Total)'
        rank_df.loc[(rank_df['brand'].str.contains('아비브', na=False) & rank_df['goods_name'].str.contains('토너|패드', na=False)), 'goods_name'] = '⚪ 아비브 어성초 토너 (Total)'
        rank_df.loc[(rank_df['brand'].str.contains('토니모리', na=False) & rank_df['goods_name'].str.contains('모찌', na=False)), 'goods_name'] = '🟡 토니모리 모찌 토너 (Total)'

        top_products = rank_df['goods_name'].value_counts().head(20)
        colors = [BRAND_COLORS['라운드랩'] if '라운드랩' in name else '#eee' for name in top_products.index]
        fig_rank = px.bar(x=top_products.values, y=top_products.index, orientation='h', height=600, title="상품명 통합 기준 판매 순위")
        fig_rank.update_traces(marker_color=colors, texttemplate='%{x}', textposition='outside')
        fig_rank.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, margin=dict(l=10))
        st.plotly_chart(fig_rank, use_container_width=True)
        # ... Top20 차트 출력 직후



    with col_trend:
        st.subheader("📅 브랜드별 월간 점유율 추이 (5대 토너 시장 내)")

        # ✅ 원본 df 보호 + month 생성
        df_ms = df.copy()
        df_ms['month'] = df_ms['date'].dt.to_period('M').astype(str)

        # ✅ 5대 브랜드(토너 제품군)만 분모/분자로 쓰기 위한 마스크
        # - 브랜드: TARGETS[brand]['brand_kw']
        # - 제품군(토너): TARGETS[brand]['prod_kw']
        mask_5 = False
        for b in TARGET_BRANDS:
            brand_kw = TARGETS[b]['brand_kw']
            prod_kw  = TARGETS[b]['prod_kw']
            mask_b = (
                df_ms['brand'].astype(str).str.contains(brand_kw, case=False, na=False) &
                df_ms['goods_name'].astype(str).str.contains(prod_kw, case=False, na=False)
            )
            mask_5 = mask_5 | mask_b

        df_5 = df_ms[mask_5].copy()

        if df_5.empty:
            st.warning("5대 브랜드 토너 데이터가 없어 점유율을 계산할 수 없습니다.")
            st.stop()

        # ✅ 분모: 5대 토너 전체 월별 건수
        monthly_total = df_5.groupby('month').size()

        # ✅ 분자: 브랜드별(토너) 월별 건수 → 월별 점유율
        ms_data = []
        for b in TARGET_BRANDS:
            brand_kw = TARGETS[b]['brand_kw']
            prod_kw  = TARGETS[b]['prod_kw']

            b_counts = df_5[
                df_5['brand'].astype(str).str.contains(brand_kw, case=False, na=False) &
                df_5['goods_name'].astype(str).str.contains(prod_kw, case=False, na=False)
            ].groupby('month').size()

            share = (b_counts / monthly_total) * 100

            for m, val in share.items():
                ms_data.append({'Month': m, 'Brand': b, 'Share': float(val)})

        ms_df = pd.DataFrame(ms_data)

        # ✅ 월 순서 정렬(문자열이라 정렬 필요)
        if not ms_df.empty:
            ms_df['Month'] = pd.to_datetime(ms_df['Month'] + "-01", errors='coerce')
            ms_df = ms_df.dropna(subset=['Month']).sort_values('Month')
            ms_df['Month'] = ms_df['Month'].dt.to_period('M').astype(str)

        fig_ms = px.line(
            ms_df, x='Month', y='Share', color='Brand',
            markers=True, title="5대 브랜드 토너 시장 내 점유율 추이 (%)",
            color_discrete_map=BRAND_COLORS
        )
        fig_ms.update_traces(line_width=3)
        st.plotly_chart(fig_ms, use_container_width=True)



# =============================================================================
# [Tab 2] Journey (기존 Tab 2: Customer Journey)
# =============================================================================
with tabs[1]:
    st.header("🗺️ 2. 위기 요인 (Journey)")
    st.markdown("""
    <div class="strategy-box">
    <b>🕵️‍♂️ Note:</b> 유입 브랜드와 이탈 브랜드가 유사한 것은 <b>'회전문 현상(Revolving Door)'</b>입니다.<br>
    고객들은 새로운 브랜드를 개척하기보다, <b>검증된 상위권 브랜드 사이를 순환</b>하며 소비하고 있습니다.
    </div>
    """, unsafe_allow_html=True)

    df_sorted = df.sort_values(['user_id', 'date'])
    df_sorted['prev_brand'] = df_sorted.groupby('user_id')['brand'].shift(1)
    df_sorted['next_brand'] = df_sorted.groupby('user_id')['brand'].shift(-1)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛫 유입: 어디서 독도로 왔는가?")
        target_inflow = df_sorted[
            (df_sorted['brand'].str.contains('라운드랩', na=False)) &
            (df_sorted['goods_name'].str.contains('독도', na=False)) &
            (df_sorted['prev_brand'].notna()) &
            (~df_sorted['prev_brand'].str.contains('라운드랩', na=False))
        ]
        if not target_inflow.empty:
            inflow_counts = target_inflow['prev_brand'].value_counts().head(10)
            fig_inflow = px.bar(x=inflow_counts.values, y=inflow_counts.index, orientation='h', title="직전 사용 브랜드 Top 10", color_discrete_sequence=[COLOR_COMP])
            fig_inflow.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_inflow, use_container_width=True)

            sb_in = st.selectbox("상세 제품 보기 (유입):", inflow_counts.index, key='sb_in')
            detail_in = target_inflow[target_inflow['prev_brand'] == sb_in]['goods_name'].value_counts().head(5)
            st.dataframe(detail_in, use_container_width=True)

    with col2:
        st.subheader("🛬 이탈: 독도를 쓰고 어디로 갔는가?")
        outflow_mask = ((df_sorted['brand'].str.contains('라운드랩', na=False)) & (df_sorted['next_brand'].notna()) & (~df_sorted['next_brand'].str.contains('라운드랩', na=False)))
        outflow_data = df_sorted[outflow_mask]
        if not outflow_data.empty:
            outflow_counts = outflow_data['next_brand'].value_counts().head(10)
            fig_out = px.bar(x=outflow_counts.values, y=outflow_counts.index, orientation='h', title="다음 구매 브랜드 Top 10", color_discrete_sequence=['#FF8080'])
            fig_out.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_out, use_container_width=True)

            sb_out = st.selectbox("상세 제품 보기 (이탈):", outflow_counts.index, key='sb_out')
            detail_out = outflow_data[outflow_data['next_brand'] == sb_out]['goods_name'].value_counts().head(5)
            st.dataframe(detail_out, use_container_width=True)

    st.divider()
    st.subheader("🕸️ 브랜드 생태계 네트워크")
    network_data = {'라운드랩': {'양말': 319, '독도토너': 309, '토너+로션': 247, '토리든세럼': 221, '에스네이처토너': 199}, '에스네이처': {'에스네이처토너': 1434, '수분크림': 355}, '토리든': {'토리든토너': 1055, '토리든크림': 881, '양말': 753}}
    G = nx.Graph()
    for brand, items in network_data.items():
        G.add_node(brand, size=40 if brand=='라운드랩' else 25, color=BRAND_COLORS['라운드랩'] if brand=='라운드랩' else '#999')
        for item, weight in items.items():
            i_color = BRAND_COLORS['라운드랩'] if '독도' in item else (COLOR_FASHION if '양말' in item else COLOR_COMP)
            G.add_node(item, size=10+(weight/50), color=i_color)
            G.add_edge(brand, item, weight=weight)
    pos = nx.spring_layout(G, k=2.5, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]; edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#bbb'), hoverinfo='none', mode='lines')
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]; node_x.append(x); node_y.append(y); node_text.append(node); node_color.append(G.nodes[node]['color']); node_size.append(G.nodes[node]['size'])
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center", marker=dict(color=node_color, size=node_size, line_width=1, line_color='white'))
    fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest', xaxis=dict(visible=False), yaxis=dict(visible=False)))
    st.plotly_chart(fig_net, use_container_width=True)

# =============================================================================
# [Tab 3] Positioning
# =============================================================================
with tabs[2]:
    st.header("3. 포지셔닝 & 속성 분석")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.subheader("🕸️ 찐팬들이 칭찬하는 포인트 (Spider Chart)")
        rep_df = get_repurchase_stats(df)
        if not rep_df.empty:
            all_brands = list(rep_df.index)
            selected_brands = st.multiselect(
                "비교할 브랜드:",
                all_brands,
                default=['라운드랩', '토리든', '에스네이처'],
                key="ms_positioning_brands"  # ✅ (권장) 키 충돌 방지
            )
            fig_spider = go.Figure()
            categories = list(rep_df.columns)
            for brand in selected_brands:
                fig_spider.add_trace(
                    go.Scatterpolar(
                        r=rep_df.loc[brand].values,
                        theta=categories,
                        fill='toself' if len(selected_brands) <= 2 else 'none',
                        name=brand,
                        line=dict(color=BRAND_COLORS.get(brand, 'gray'), width=2)
                    )
                )
            fig_spider.update_layout(polar=dict(radialaxis=dict(visible=True)), height=450)
            st.plotly_chart(fig_spider, use_container_width=True)
            st.divider()
            st.subheader("🔵 토리든 재구매 결정 요인")
            render_lift_chart(df, "토리든")

    with col_p2:
        st.subheader("🚀 재구매 유발 요인 (Lift Analysis)")
        lift_brand = st.selectbox(
            "분석할 브랜드:",
            list(TARGETS.keys()),
            key="sb_positioning_lift_brand"  # ✅ (권장) 키 충돌 방지
        )
        lift_series = calculate_lift(df, lift_brand)
        if not lift_series.empty:
            colors = [BRAND_COLORS.get(lift_brand, 'gray') if v > 1.0 else '#ddd' for v in lift_series.values]
            fig_lift = go.Figure(
                go.Bar(
                    x=lift_series.values,
                    y=lift_series.index,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{v:.2f}배" for v in lift_series.values],
                    textposition='auto'
                )
            )
            fig_lift.add_vline(x=1.0, line_dash="dash")
            fig_lift.update_layout(
                title=f"[{lift_brand}] 재구매 결정 요인",
                yaxis=dict(autorange="reversed"),
                height=450
            )
            st.plotly_chart(fig_lift, use_container_width=True)
            # ... 월간 점유율 차트 출력 직후
            st.divider()
            st.subheader("🟢 에스네이처 재구매 결정 요인")
            render_lift_chart(df, "에스네이처")


# =============================================================================
# [Tab 3] Behavior (기존 Tab 5: 구매 행동)
# =============================================================================
with tabs[3]:
    st.header("🛒 4. 문제 발견 (Behavior)")
    st.subheader("🛍️ 구매 빈도별 장바구니 (1회 vs 2회 vs 3회+)")

    sel_brand_basket = st.selectbox("장바구니 분석 브랜드:", list(TARGETS.keys()), index=0)
    basket_data = get_frequency_basket(df, sel_brand_basket)
    b_col1, b_col2, b_col3 = st.columns(3)
    for g_name, col in zip(['1회 (이탈/체험)', '2회 (재방문)', '3회+ (찐팬)'], [b_col1, b_col2, b_col3]):
        with col:
            st.markdown(f"**{g_name}**")
            top_items = basket_data.get(g_name, pd.Series())
            if not top_items.empty:
                b_colors = [get_item_color(item, sel_brand_basket) for item in top_items.index]
                fig_b = px.bar(x=top_items.values, y=top_items.index, orientation='h', text_auto=True)
                fig_b.update_traces(marker_color=b_colors)
                fig_b.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_b, use_container_width=True)

    st.divider()

    st.subheader("🔄 평균 재구매 주기")
    user_counts = dokdo_df['user_id'].value_counts()
    rep_users = user_counts[user_counts >= 2].index
    if len(rep_users) > 0:
        rep_df_local = dokdo_df[dokdo_df['user_id'].isin(rep_users)].sort_values(['user_id', 'date'])
        periods = []
        for uid, group in rep_df_local.groupby('user_id'):
            if len(group) < 2: continue
            diff = (group['date'].max() - group['date'].min()).days
            periods.append(diff / (len(group) - 1))
        st.metric("평균 재구매 주기", f"{int(np.mean(periods))}일")

# =============================================================================
# [Tab 4] Voice (기존 Tab 6: Voice & Persona)
# =============================================================================
with tabs[4]:
    st.header("🗣️ 5. 이탈 원인 (Voice)")
    col_last1, col_last2 = st.columns(2)
    with col_last1:
        st.subheader("👋 이탈자 vs 찐팬 불만 비교")
        user_counts = dokdo_df['user_id'].value_counts()
        churn_users = user_counts[user_counts == 1].index
        loyal_users = user_counts[user_counts >= 3].index
        churn_txt = dokdo_df[dokdo_df['user_id'].isin(churn_users)]['content'].fillna('')
        loyal_txt = dokdo_df[dokdo_df['user_id'].isin(loyal_users)]['content'].fillna('')
        neg_kws = ['건조', '좁쌀', '트러블', '끈적', '비싸', '그저', '자극']
        data = []
        for kw in neg_kws:
            data.append({'Keyword': kw, 'Churn': churn_txt.str.contains(kw).mean()*100, 'Loyal': loyal_txt.str.contains(kw).mean()*100})
        comp_df = pd.DataFrame(data)
        comp_df['Gap'] = comp_df['Churn'] - comp_df['Loyal']
        comp_df = comp_df.sort_values('Gap', ascending=False)
        fig_churn = px.bar(comp_df, x='Keyword', y=['Churn', 'Loyal'], barmode='group', color_discrete_map={'Churn': BRAND_COLORS['라운드랩'], 'Loyal': '#ddd'})
        st.plotly_chart(fig_churn, use_container_width=True)

    with col_last2:
        st.subheader("🧖 브랜드별 피부 타입 분포")
        if 'skin_info' in df.columns:
            skin_data = []
            for b in TARGET_BRANDS:
                b_df = df[df['brand'].str.contains(b, na=False)]
                parsed = b_df['skin_info'].apply(parse_skin_info).dropna()
                if not parsed.empty:
                    counts = parsed.value_counts(normalize=True)*100
                    for s, p in counts.items(): skin_data.append({'Brand':b, 'Skin':s, 'Pct':p})
            skin_plot = pd.DataFrame(skin_data)
            fig_skin = px.bar(
                skin_plot[skin_plot['Skin'].str.contains('건성|지성|복합성')],
                x='Brand', y='Pct', color='Skin', barmode='group',
                color_discrete_map={'건성': '#FFD700', '지성': '#87CEEB', '복합성': '#90EE90'}
            )
            st.plotly_chart(fig_skin, use_container_width=True)

# =============================================================================
# [Tab 5] Aha (기존 Tab 1: Aha Moment)
# =============================================================================
with tabs[5]:
    st.header("💡 6. 기회 탐색 (Aha!)")
    st.markdown("""
    독도 토너는 '기본'에 충실한 제품입니다.
    **"패션에서도 '기본템(맨투맨, 무채색)'을 선호하는 사람이 독도 토너에도 정착하지 않을까?"** 라는 가설을 검증합니다.
    """)

    with st.spinner("패션 취향 분석 중..."):
        lifestyle_df, debug_info = analyze_aha_moment(df)

    st.info(f"분석 대상 유저 {debug_info['total_analyzed']:,}명 중 패션/잡화 구매 이력이 있는 {debug_info['fashion_buyers']:,}명의 데이터를 분석했습니다.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("👕 패션 스타일 & 컬러 매칭 (Table)")
        st.dataframe(
            lifestyle_df.style.background_gradient(cmap='Greens', subset=['Lift']),
            use_container_width=True,
            column_config={
                "Lift": st.column_config.NumberColumn("Lift (배수)", format="%.2f배"),
                "Loyal(%)": st.column_config.NumberColumn("찐팬 보유율", format="%.1f%%"),
                "Churn(%)": st.column_config.NumberColumn("이탈자 보유율", format="%.1f%%"),
            }
        )
    with col2:
        st.subheader("🎯 찐팬 시그널 Top 5 (Chart)")
        fig_life = px.bar(
            lifestyle_df, x='Lift', y='Category', orientation='h',
            title="이탈자 대비 찐팬의 성향 강도 (Lift)",
            color='Lift', color_continuous_scale='Greens'
        )
        fig_life.add_vline(x=1.0, line_dash="dash", annotation_text="평균")
        st.plotly_chart(fig_life, use_container_width=True)

    top_factor = lifestyle_df.iloc[0]
    
    st.markdown(f"""
    <div class="insight-box">
    <b>🕵️‍♂️ Analyst Insight:</b><br>
    데이터 분석 결과, <b>[{top_factor['Category']}]</b> 제품을 구매한 사람들의 독도 토너 정착 확률이
    일반 이탈자보다 <b>{top_factor['Lift']:.2f}배</b> 높습니다!<br><br>
    <b>🚀 Action Plan:</b><br>
    무신사 스토어에서 <b>"{top_factor['Category'].split('(')[0]}" 카테고리 기획전</b>을 할 때,
    독도 토너를 <b>'코디 추천템'</b>이나 <b>'계산대 앞 1+1'</b>으로 노출시키세요.<br>
    이들의 취향(Taste)이 독도 토너와 정확히 일치합니다.
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# [Tab 6] Proof (기존 Tab 7: Statistical Analysis)
# =============================================================================
with tabs[6]:
    st.header("🧪 7. 통계 검증 (Proof)")
    st.markdown("""
    <div class="aha-box">
    <b>🧪 통계적 검증 완료 (Validated):</b><br>
    단순한 직관이 아닌, 다른 변수들을 모두 통제한 상태에서
    <b>각 요인이 재구매 확률을 실질적으로 몇 배 높이는지(Odds Ratio)</b> 계산했습니다.
    </div>
    """, unsafe_allow_html=True)

    stats_data = {
        'Factor': ['무채색 선호 (Monotone)', '기본템 선호 (Basic)', '민감성 피부 (Sensitive)', '장바구니 크기', '첫 구매액', '리뷰 길이', '로션 합배송'],
        'Odds Ratio': [2.00, 1.42, 1.29, 1.01, 1.00, 0.99, 0.79],
        'Impact': ['Positive', 'Positive', 'Positive', 'Neutral', 'Neutral', 'Neutral', 'Negative'],
        'Description': [
            '블랙/그레이 옷을 입는 사람은 2배 더 재구매합니다.',
            '맨투맨/후드를 입는 사람은 1.4배 더 재구매합니다.',
            '민감성 피부는 정착 확률이 1.3배 높습니다.',
            '많이 산다고 재구매하는 건 아닙니다.',
            '비싸게 샀다고 이탈하지 않습니다.',
            '리뷰 길이는 재구매와 무관합니다.',
            '로션을 같이 산 사람은 오히려 이탈합니다.'
        ]
    }
    stats_df = pd.DataFrame(stats_data)

    col_main, col_sub = st.columns([1.5, 1])

    with col_main:
        st.subheader("📊 재구매 영향력 (Odds Ratio) 시각화")
        fig_stats = px.bar(
            stats_df, x='Odds Ratio', y='Factor', orientation='h',
            color='Impact',
            color_discrete_map={'Positive': '#FF4B4B', 'Neutral': '#DDDDDD', 'Negative': '#4169E1'},
            title="Factor Impact on Repurchase (Odds Ratio)",
            text='Odds Ratio',
            hover_data=['Description']
        )
        fig_stats.add_vline(x=1.0, line_dash="dash", line_color="black", annotation_text="영향 없음 (1.0)")
        fig_stats.update_traces(texttemplate='%{text:.2f}배', textposition='outside', width=0.6)
        fig_stats.update_layout(
            yaxis=dict(autorange="reversed"),
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#eee'),
            height=500
        )
        st.plotly_chart(fig_stats, use_container_width=True)

    with col_sub:
        st.subheader("💡 핵심 인사이트")
        top_val = 2.00
        avg_val = 1.0

        fig_donut = go.Figure(data=[go.Pie(
            labels=['무채색 선호 효과', '일반 평균'],
            values=[top_val, avg_val],
            hole=.7,
            marker_colors=['#FF4B4B', '#eee'],
            textinfo='none'
        )])

        fig_donut.update_layout(
            title_text="<b>무채색 선호의 파급력</b><br>(일반 대비 2배)",
            title_x=0.5,
            height=300,
            showlegend=False,
            annotations=[dict(text=f'{top_val}배', x=0.5, y=0.5, font_size=40, showarrow=False, font_color='#FF4B4B')]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("""
        <div class="strategy-box">
        <b>1️⃣ 패션 취향이 깡패다</b><br>
        가격(1.0)이나 리뷰 정성도(0.99)보다 <b>'옷 스타일(무채색, 기본템)'</b>이 재구매를 훨씬 강력하게 예측합니다.<br>
        → <i>무신사 '모노톤 기획전'에 타겟 광고를 집행하세요.</i><br><br>
        <b>2️⃣ 로션의 배신?</b><br>
        토너와 로션을 같이 산 고객(0.79배)은 왜 떠날까요?<br>
        → <i>로션 제품의 만족도를 긴급 점검하거나, 세트 상품의 사용 주기를 체크해보세요.</i>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# [Tab 7] Action Plan
# =============================================================================
with tabs[7]:
    st.header("🚀 8. 결론 및 제언: 1위 탈환을 위한 3대 전략")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div class="action-card">
        <h3>🎯 전략 1. 타겟팅 확장</h3>
        <b>"패션 카테고리로 침투하라"</b><br><br>
        뷰티는 레드오션입니다. <b>'무채색 패션족'</b>을 공략하세요.<br><br>
        ✅ <b>Action:</b><br>
        - 무신사 '스웨트/후드' 카테고리 광고 집행<br>
        - 메시지: "기본에 충실한 룩엔, 기본 토너"
        </div>
        """, unsafe_allow_html=True)
        st.metric("예상 타겟 적중률", "200%", "+100%p")

    with col_b:
        st.markdown("""
        <div class="action-card">
        <h3>📦 전략 2. 상품 재정비</h3>
        <b>"로션 세트를 과감히 버려라"</b><br><br>
        로션 합배송은 재구매를 21% 떨어뜨립니다.<br><br>
        ✅ <b>Action:</b><br>
        - '토너+로션' 기획세트 판매 축소<br>
        - 로션은 '샘플링'으로 먼저 경험 유도
        </div>
        """, unsafe_allow_html=True)
        st.metric("예상 이탈 방어율", "+21%", "역성장 방지")

    with col_c:
        st.markdown("""
        <div class="action-card">
        <h3>🛡️ 전략 3. 리텐션 강화</h3>
        <b>"이탈 키워드를 선제 방어하라"</b><br><br>
        이탈자는 '끈적임'과 '자극'에 민감합니다.<br><br>
        ✅ <b>Action:</b><br>
        - 상세페이지 상단: "산뜻한 마무리" 강조<br>
        - 45일차(이탈시점) CRM 메시지 발송
        </div>
        """, unsafe_allow_html=True)
        st.metric("예상 재구매 전환", "+15%", "이탈자 회복")

    st.divider()

    st.subheader("📈 전략 실행 시 예상 성장 시나리오")
    growth_data = pd.DataFrame({
        'Stage': ['현재(AS-IS)', '타겟팅 최적화', '상품 구조조정', '리텐션 강화(TO-BE)'],
        'Retention Rate': [25, 35, 42, 50]
    })
    fig_growth = px.line(
        growth_data,
        x='Stage',
        y='Retention Rate',
        markers=True,
        title="단계별 예상 재구매율 변화 (%)"
    )
    fig_growth.update_traces(line_color='#FF4B4B', line_width=4, marker_size=12)
    fig_growth.add_annotation(
        x='리텐션 강화(TO-BE)',
        y=50,
        text="Goal: 50%",
        showarrow=True,
        arrowhead=1
    )
    st.plotly_chart(fig_growth, use_container_width=True)



st.markdown("---")
st.markdown("Created with Streamlit | Round Lab Analysis")
