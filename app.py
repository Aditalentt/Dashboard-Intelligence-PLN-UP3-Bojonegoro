import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utils import load_data, aggregate_unit, segment, add_segmentation, apply_segmentation, segment_action, get_segment_comparison, segment_distribution, apply_action, train_model, detect_anomaly, train_nlp_model, clean_text, tokenize

st.set_page_config(layout="wide")

st.title("Dashboard Intelligence PLN UP3 Bojonegoro")


# LOAD DATA
@st.cache_data
def load_main_data():
    df = load_data()
    df = add_segmentation(df)
    df = apply_segmentation(df)
    df = apply_action(df)
    df = detect_anomaly(df)
    return df

@st.cache_data
def load_nlp_data():
    return pd.read_excel('keluhan baru.xlsx')

df_keluhan = load_nlp_data()

def run_model(df):
    return train_model(df)

df = load_main_data()

# FILTER
df['UNITUP'] = df['UNITUP'].astype(str)

unit = st.selectbox("Pilih UNITUP", df['UNITUP'].unique())
filtered = df[df['UNITUP'] == unit]

if filtered.empty:
    st.error("Data kosong")
    st.stop()

# Tab 1
def show_dashboard(filtered):

    # KPI
    st.subheader("KPI")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Data", len(filtered))
    total_revenue = filtered['TOTAL_RP'].sum()
    col2.metric("Total Tagihan", f'Rp{total_revenue:,.0f}'.replace(',','.'))
    total_kwh = filtered['TOTAL_KWH'].sum()
    col3.metric("Total kWh", f'{total_kwh:,.0f}'.replace(',','.'))

    # Tabel
    tarif_summary = (filtered.groupby(['TARIP','DAYA']).agg(Total_KWH=('TOTAL_KWH','sum')).reset_index())
    total_kwh = tarif_summary['Total_KWH'].sum()
    tarif_summary['Kontribusi (%)'] = (tarif_summary['Total_KWH'] / total_kwh * 100).round(2)
    st.dataframe(tarif_summary, width='stretch')

    # Bar Chart per Tarif
    tarif_chart = (filtered.groupby('TARIP')['TOTAL_KWH'].sum().reset_index())
    tarif_chart = tarif_chart.sort_values('TOTAL_KWH', ascending = False)
    top3 = tarif_chart.head(3)['TARIP']
    tarif_chart['Kategori'] = 'Lainnya'
    tarif_chart.loc[tarif_chart.index[0], 'Kategori'] = 'Top 1'
    tarif_chart.loc[tarif_chart.index[1], 'Kategori'] = 'Top 2'
    tarif_chart.loc[tarif_chart.index[2], 'Kategori'] = 'Top 3'
    fig = px.bar(tarif_chart, x='TARIP', y='TOTAL_KWH', color = 'Kategori', color_discrete_map = {'Top 1': '#FF6B00', 'Top 2': '#008FD5', 'Top 3': '#5DADE2', 'Lainnya': '#D9D9D9'}, title='Total Konsumsi Berdasarkan Tarif', text_auto='.2s')
    fig.update_layout(xaxis_title='Tarif', yaxis_title='Total KWH')
    st.plotly_chart(fig, use_container_width=True)

    # Pareto
    tagihan_tarif = (filtered.groupby('TARIP')['TOTAL_RP'].sum().reset_index().sort_values('TOTAL_RP', ascending=False))
    total_tagihan = tagihan_tarif['TOTAL_RP'].sum()
    tagihan_tarif['Kontribusi (%)'] = (tagihan_tarif['TOTAL_RP'] / total_tagihan * 100).round(2)
    top3 = tagihan_tarif.head(3)['TARIP']
    tagihan_tarif['Kategori'] = 'Lainnya'
    tagihan_tarif.loc[tagihan_tarif.index[0], 'Kategori'] = 'Top 1'
    tagihan_tarif.loc[tagihan_tarif.index[1], 'Kategori'] = 'Top 2'
    tagihan_tarif.loc[tagihan_tarif.index[2], 'Kategori'] = 'Top 3'
    st.dataframe(tagihan_tarif, width='stretch')

    fig = px.bar(tagihan_tarif, x='TARIP', y='Kontribusi (%)', color = 'Kategori', color_discrete_map = {'Top 1': '#FF6B00', 'Top 2': '#008FD5', 'Top 3': '#5DADE2', 'Lainnya': '#D9D9D9'}, title='Kontribusi Tagihan per Tarif',text='Kontribusi (%)')
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title = 'Tarif', yaxis_title = 'Kontribusi Tagihan (%)', showlegend = False)
    st.plotly_chart(fig, use_container_width=True)

    # Segmentasi Pelanggan
    st.subheader('Distribusi Segmentasi Pelanggan')
    segment_counts = filtered['SEGMENT'].value_counts().reset_index()
    segment_counts.columns = ['SEGMENT', 'COUNT']
    fig = px.pie(segment_counts, names = 'SEGMENT', values = 'COUNT')
    fig.update_traces(textinfo = 'percent+label')
    st.plotly_chart(fig)

    st.subheader('Pendapatan tagihan per Segmen')
    segment_revenue = filtered.groupby('SEGMENT')['TOTAL_RP'].sum()
    st.bar_chart(segment_revenue)

    # Segmentation Validation
    dist = segment_distribution(df)
    for seg, val in dist.items():
        with st.expander(f"Segment: {seg}"):
            st.dataframe(val, width = 'stretch')

    # Segment Summary
    st.subheader('Segment Summary')
    segment_summary = df.groupby('SEGMENT').agg({
        'TOTAL_RP': ['sum', 'mean'],
        'TOTAL_KWH': ['mean'],
        'RP_PER_KWH': ['mean']
    }).round(2)

    st.dataframe(segment_summary)

    # Comparison
    st.subheader('Comparison')
    compare = get_segment_comparison(df, filtered)

    st.subheader('Gap Analysis (Local vs Global)')

    compare['gap'] = compare['Local'] - compare['Global']
    st.dataframe(compare)

    # Top Contributor
    top_real = filtered.sort_values('TOTAL_RP', ascending = False).head(15)
    st.subheader('Top Real Revenue Driver')
    st.dataframe(top_real[['NAMA', 'TOTAL_RP', 'TOTAL_KWH']])

    # Rekomendasi Aksi
    st.subheader('Rekomendasi Aksi')
    st.dataframe(df[['NAMA', 'UNITUP', 'SEGMENT', 'ACTION']])

# Tab 2
def show_modeling(df, unit):
    st.subheader('Modelling')
    
    if st.button('Run Model'):
        df_model, model, mae, mape = run_model(df)

        # Evaluasi Model
        col1, col2 = st.columns(2)
        col1.metric('MAE', f'{mae:,.0f}'.replace(',','.'))
        col2.metric('MAPE (%)', f'{mape:.2f}%')

        # Scatter plot
        fig = px.scatter(df_model, x = 'TOTAL_RP', y = 'PRED_RP')
        st.plotly_chart(fig, use_container_width=True)

        # Residual Analysis
        df_model['RESIDUAL'] = df_model['TOTAL_RP'] - df_model['PRED_RP']

        st.subheader('Analisis Residual')

        fig = px.scatter(df_model, x = 'TOTAL_KWH', y = 'RESIDUAL')
        st.plotly_chart(fig, use_container_width=True)
        fig = px.box(df_model, x = 'SEGMENT', y = 'RESIDUAL')
        st.plotly_chart(fig, use_container_width=True)

        # Feature Importance
        importance = pd.DataFrame({'feature': ['TOTAL_KWH', 'DAYA', 'JAMNYALA', 'TARIP'], 'importance': model.feature_importances_}).sort_values(by = 'importance', ascending=False)
        importance['Kategori'] = 'Lainnya'
        importance.loc[importance['importance'].idxmax(), 'Kategori'] = 'Terpenting'
        st.subheader('Feature Importance')
        st.dataframe(importance)

        fig = px.bar(importance, x='feature', y='importance', color = 'Kategori', color_discrete_map={'Terpenting': '#FF6B00', 'Lainnya': '#D9D9D9'}, text_auto = '.3f')
        fig.update_layout(xaxis_title = 'Feature', yaxis_title = 'Importance', showlegend = False)
        fig.update_traces(textposition = 'outside')
        st.plotly_chart(fig, use_container_width=True)

        # Anomaly Detection
        anomali_tagihan = (filtered[filtered['ANOMALI_TAGIHAN']][['NAMA', 'TARIP', 'DAYA', 'TOTAL_KWH', 'TOTAL_RP']].sort_values('TOTAL_RP', ascending=False))
        st.subheader('Anomali Tagihan')
        st.dataframe(anomali_tagihan, width = 'stretch')

        anomali_jam = (filtered[filtered['ANOMALI_JAM']][['NAMA', 'TARIP', 'DAYA', 'JAMNYALA', 'TOTAL_KWH']].sort_values('JAMNYALA', ascending=True))
        st.subheader('Anomali Jam Nyala')
        st.dataframe(anomali_jam, width = 'stretch')

# Tab 3
def show_nlp(df_keluhan, unit):
    st.subheader('Analisis Keluhan Pelanggan')

    df_keluhan['UNITUP'] = df_keluhan['UNITUP'].astype(str).str.strip()

    df_k = df_keluhan[df_keluhan['UNITUP'] == unit].copy()

    # Distribusi Kategori
    st.subheader('Distribusi Jenis Keluhan')
    issue_counts = df_k['LABEL'].value_counts()
    st.bar_chart(issue_counts)

    # Keluhan vs Segmentasi
    st.subheader('Keluhan vs Segmentasi')
    cross = pd.crosstab(df_k['SEGMENT'], df_keluhan['LABEL'])
    st.dataframe(cross)

    # Sampel Teks
    st.subheader('Contoh Keluhan')
    st.dataframe(df_k[['KELUHAN', 'LABEL', 'UNITUP']].head(10))

    st.subheader("Tokenizing")

    df_k['TOKENS'] = df_k['KELUHAN'].apply(
    lambda x: tokenize(clean_text(x))
    )

    st.dataframe(df_k[['KELUHAN', 'TOKENS']].head())

    # Word Cloud
    st.subheader('Word Cloud Keluhan')

    text = " ".join(df_k['KELUHAN'].dropna().astype(str))

    if text.strip() == "":
        st.warning("Tidak ada teks untuk dibuat word cloud")
    else:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate(text)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis("off")

        st.pyplot(fig)

    # Evaluasi
    st.subheader('Evaluasi Model NLP')
    accuracy, report, y_test, y_pred = train_nlp_model(df_k)
    st.metric('Accuracy', f'{accuracy:.2%}')
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)


# Tab Control
tab1, tab2, tab3 = st.tabs(['Dashboard', 'Modelling', 'NLP'])

with tab1:
    show_dashboard(filtered)
with tab2:
    show_modeling(df, unit)
with tab3:
    show_nlp(df_keluhan, unit)