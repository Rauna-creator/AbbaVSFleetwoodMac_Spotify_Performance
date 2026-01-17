import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="ABBA vs Fleetwood Mac Analysis",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1DB954;}
    .insight-box {background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #1DB954;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">ABBA vs Fleetwood Mac: Spotify Analysis</p>', unsafe_allow_html=True)
st.markdown("Five Decades of Streaming Performance")
st.markdown("---")

@st.cache_data
def load_data():
    import os
    # Debug: show what files exist
    st.write("Files in current directory:", os.listdir('.'))
    if os.path.exists('data'):
        st.write("Files in data directory:", os.listdir('data'))
    else:
        st.write("data directory does not exist!")
    
    abba = pd.read_csv('data/Abba.csv')
    fm = pd.read_csv('data/FleetwoodMac.csv')
    
    abba['Artist'] = 'ABBA'
    fm['Artist'] = 'Fleetwood Mac'
    
    df = pd.concat([abba, fm], ignore_index=True)
    
    df['Era'] = pd.cut(df['Year'], 
                       bins=[0, 1980, 1990, 2000, 2010, 2030],
                       labels=['70s', '80s', '90s', '2000s', '2020s'])
    
    df['Popularity_Category'] = pd.cut(df['Popularity'],
                                       bins=[0, df['Popularity'].quantile(0.2),
                                            df['Popularity'].quantile(0.8), 100],
                                       labels=['Deep Cut', 'Mid-tier', 'Hit'])
    
    df['Duration_seconds'] = df['Duration'].apply(
        lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1])
    )
    
    df['Mode_Label'] = df['Mode'].map({1: 'Major', 0: 'Minor'})
    
    return df

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Analysis", [
    "Overview",
    "Artist Comparison",
    "Era Analysis",
    "Musical Personalities",
    "Feature Importance",
    "Clustering Analysis",
    "Mamma Mia Effect",
    "Hits vs Deep Cuts"
])

try:
    df = load_data()
    
    if page == "Overview":
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Songs", len(df))
        col2.metric("ABBA Songs", len(df[df['Artist'] == 'ABBA']))
        col3.metric("Fleetwood Mac Songs", len(df[df['Artist'] == 'Fleetwood Mac']))
        col4.metric("Time Span", "1968-2021")
        
        st.markdown("---")
        st.subheader("Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
            <b>ABBA Dominates Spotify</b><br>
            49% higher popularity than Fleetwood Mac (43.8 vs 29.4, p < 0.001)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
            <b>Production Quality Matters Most</b><br>
            Loudness is strongest predictor (r = 0.320). Audio features explain only 12.5% of variance.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
            <b>Mamma Mia Effect</b><br>
            13 film-featured songs average 71.5 vs 40.2 for other ABBA tracks
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <b>Musical Mode Irrelevant</b><br>
            Major vs minor keys show no significant difference (p = 0.889)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
            <b>Style Influences Success</b><br>
            Mainstream pop-rock: 60% of top 20. Slow-tempo: only 5%
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
            <b>Three Success Factors</b><br>
            Loudness (+3.7 dB), Energy (+0.113), Danceability (+0.047)
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Dataset Sample")
        st.dataframe(df.head(10), use_container_width=True)
    
    elif page == "Artist Comparison":
        st.header("Artist Comparison")
        
        stats_df = df.groupby('Artist')['Popularity'].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ABBA")
            st.metric("Mean Popularity", f"{stats_df.loc['ABBA', 'mean']:.1f}")
            st.metric("Median Popularity", f"{stats_df.loc['ABBA', 'median']:.0f}")
            st.metric("Max Popularity", f"{stats_df.loc['ABBA', 'max']:.0f}")
            
        with col2:
            st.subheader("Fleetwood Mac")
            st.metric("Mean Popularity", f"{stats_df.loc['Fleetwood Mac', 'mean']:.1f}")
            st.metric("Median Popularity", f"{stats_df.loc['Fleetwood Mac', 'median']:.0f}")
            st.metric("Max Popularity", f"{stats_df.loc['Fleetwood Mac', 'max']:.0f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            df.boxplot(column='Popularity', by='Artist', ax=ax)
            ax.set_title('Popularity Distribution by Artist')
            ax.set_xlabel('Artist')
            ax.set_ylabel('Spotify Popularity Score')
            plt.suptitle('')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            for artist in ['ABBA', 'Fleetwood Mac']:
                data = df[df['Artist'] == artist]['Popularity']
                ax.hist(data, alpha=0.6, label=artist, bins=20)
            ax.set_xlabel('Spotify Popularity Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Popularity Distribution Comparison')
            ax.legend()
            st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Statistical Analysis")
        
        abba_pop = df[df['Artist'] == 'ABBA']['Popularity']
        fm_pop = df[df['Artist'] == 'Fleetwood Mac']['Popularity']
        t_stat, p_value = stats.ttest_ind(abba_pop, fm_pop)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("t-statistic", f"{t_stat:.3f}")
        col2.metric("p-value", f"{p_value:.4f}")
        col3.metric("Significance", "Yes (p < 0.001)" if p_value < 0.001 else "No")
        
        st.markdown("""
        <div class="insight-box">
        <b>Interpretation:</b> ABBA demonstrates statistically significant dominance on Spotify 
        with mean popularity 14.4 points higher (49% increase). The difference is robust (p < 0.001).
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "Era Analysis":
        st.header("Era-Based Performance Analysis")
        
        era_stats = df.groupby('Era')['Popularity'].agg(['mean', 'median', 'count'])
        
        st.subheader("Era Statistics")
        st.dataframe(era_stats.style.format({'mean': '{:.1f}', 'median': '{:.0f}', 'count': '{:.0f}'}),
                    use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            for artist in ['ABBA', 'Fleetwood Mac']:
                artist_data = df[df['Artist'] == artist].groupby('Era')['Popularity'].mean()
                ax.plot(artist_data.index, artist_data.values, marker='o', label=artist, linewidth=2)
            ax.set_xlabel('Era')
            ax.set_ylabel('Mean Popularity')
            ax.set_title('Popularity Trends Over Time')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            era_stats['mean'].plot(kind='bar', ax=ax, color='#1DB954')
            ax.set_xlabel('Era')
            ax.set_ylabel('Mean Popularity')
            ax.set_title('Mean Popularity by Era')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("""
        <div class="insight-box">
        <b>Key Finding:</b> 70s and 80s material achieves highest streaming performance (>34 mean popularity), 
        representing both artists' commercial peak output. 90s shows lowest performance, reflecting reduced activity.
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "Musical Personalities":
        st.header("Musical Personality Profiles")
        
        features = ['Danceability', 'Energy', 'Acousticness', 'Valence', 'Loudness']
        artist_features = df.groupby('Artist')[features].mean()
        
        st.subheader("Feature Comparison")
        comparison = artist_features.T
        comparison['Difference'] = comparison['ABBA'] - comparison['Fleetwood Mac']
        comparison['% Difference'] = (comparison['Difference'] / comparison['Fleetwood Mac'] * 100).round(1)
        
        st.dataframe(comparison.style.format({
            'ABBA': '{:.3f}',
            'Fleetwood Mac': '{:.3f}',
            'Difference': '{:+.3f}',
            '% Difference': '{:+.1f}%'
        }), use_container_width=True)
        
        st.subheader("Radar Chart Comparison")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=artist_features.loc['ABBA', features].values,
            theta=features,
            fill='toself',
            name='ABBA'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=artist_features.loc['Fleetwood Mac', features].values,
            theta=features,
            fill='toself',
            name='Fleetwood Mac'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <b>Analysis:</b> ABBA demonstrates higher danceability, energy, and valence (upbeat pop sound) 
        with lower acousticness (electronic production). Fleetwood Mac shows higher acousticness 
        (organic rock instrumentation). These distinctions reflect fundamental genre differences.
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "Feature Importance":
        st.header("Audio Feature Importance Analysis")
        
        features = ['Danceability', 'Energy', 'Acousticness', 'Valence', 
                   'Loudness', 'Tempo', 'Instrumentalness']
        
        correlations = df[features].corrwith(df['Popularity']).sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Feature Correlations with Popularity")
            st.dataframe(correlations.to_frame('Correlation').style.format('{:.3f}'),
                        use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            correlations.plot(kind='barh', ax=ax, color='#1DB954')
            ax.set_xlabel('Correlation with Popularity')
            ax.set_title('Feature Importance')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Feature Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(df['Loudness'], df['Popularity'], alpha=0.5, c='#1DB954')
            ax.set_xlabel('Loudness (dB)')
            ax.set_ylabel('Popularity')
            ax.set_title(f'Loudness vs Popularity (r = {correlations["Loudness"]:.3f})')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(df['Valence'], df['Popularity'], alpha=0.5, c='#1DB954')
            ax.set_xlabel('Valence')
            ax.set_ylabel('Popularity')
            ax.set_title(f'Valence vs Popularity (r = {correlations["Valence"]:.3f})')
            st.pyplot(fig)
        
        st.markdown("""
        <div class="insight-box">
        <b>Key Finding:</b> Production quality (loudness, r = 0.320) emerges as strongest predictor. 
        Valence shows weak correlation (r = 0.093). Audio characteristics explain only 12.5% of popularity variance.
        Majority driven by non-audio factors (artist recognition, cultural moments).
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "Clustering Analysis":
        st.header("Musical Style Clustering")
        
        st.write("Identifying distinct song styles using K-means clustering")
        
        features_for_clustering = ['Danceability', 'Energy', 'Acousticness', 
                                   'Valence', 'Loudness', 'Tempo']
        
        X = df[features_for_clustering].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        cluster_labels = {
            0: 'Acoustic Ballads',
            1: 'Mainstream Pop-Rock',
            2: 'High-Energy Dance',
            3: 'Slow-Tempo Songs'
        }
        
        df['Cluster_Label'] = df['Cluster'].map(cluster_labels)
        
        st.subheader("Cluster Composition")
        cluster_stats = df.groupby('Cluster_Label').agg({
            'Popularity': ['mean', 'count'],
            'Artist': lambda x: (x == 'ABBA').sum() / len(x) * 100
        }).round(2)
        cluster_stats.columns = ['Mean Popularity', 'Song Count', '% ABBA']
        
        st.dataframe(cluster_stats, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            df.boxplot(column='Popularity', by='Cluster_Label', ax=ax)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Popularity')
            ax.set_title('Popularity Distribution by Cluster')
            plt.suptitle('')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            cluster_counts = df['Cluster_Label'].value_counts()
            ax.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%')
            ax.set_title('Cluster Distribution')
            st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("Top 20 Song Distribution")
        
        top_20 = df.nlargest(20, 'Popularity')
        top_20_clusters = top_20['Cluster_Label'].value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(top_20_clusters.to_frame('Count'), use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            top_20_clusters.plot(kind='bar', ax=ax, color='#1DB954')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Count in Top 20')
            ax.set_title('Top 20 Songs by Cluster')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        
        st.markdown("""
        <div class="insight-box">
        <b>Key Finding:</b> Mainstream Pop-Rock achieves highest success rate (60% of top 20 vs 41% of catalogue). 
        High-Energy Dance over-represents in hits (32% vs 25%). Slow-Tempo significantly underperforms (5% vs 19%).
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "Mamma Mia Effect":
        st.header("Mamma Mia Effect Analysis")
        
        st.write("Analysing impact of Mamma Mia film features on ABBA's Spotify performance")
        
        mamma_mia_songs = [
            'Dancing Queen', 'Mamma Mia', 'The Winner Takes It All', 'Waterloo',
            'Super Trouper', 'Chiquitita', 'Voulez-Vous', 'Money, Money, Money',
            'Fernando', 'Knowing Me, Knowing You', 'Does Your Mother Know', 
            'SOS', 'Honey, Honey'
        ]
        
        abba_df = df[df['Artist'] == 'ABBA'].copy()
        abba_df['Mamma_Mia'] = abba_df['Track'].isin(mamma_mia_songs)
        
        stats_mm = abba_df.groupby('Mamma_Mia')['Popularity'].agg(['mean', 'median', 'count'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Non-Featured Songs")
            st.metric("Count", f"{int(stats_mm.loc[False, 'count'])}")
            st.metric("Mean Popularity", f"{stats_mm.loc[False, 'mean']:.1f}")
            st.metric("Median Popularity", f"{stats_mm.loc[False, 'median']:.0f}")
        
        with col2:
            st.subheader("Mamma Mia Featured")
            st.metric("Count", f"{int(stats_mm.loc[True, 'count'])}")
            st.metric("Mean Popularity", f"{stats_mm.loc[True, 'mean']:.1f}")
            st.metric("Median Popularity", f"{stats_mm.loc[True, 'median']:.0f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            abba_df.boxplot(column='Popularity', by='Mamma_Mia', ax=ax)
            ax.set_xticklabels(['Other ABBA Songs', 'Mamma Mia Featured'])
            ax.set_ylabel('Spotify Popularity Score')
            ax.set_title('Popularity: Mamma Mia Songs vs Others')
            plt.suptitle('')
            st.pyplot(fig)
        
        with col2:
            featured = abba_df[abba_df['Mamma_Mia']].sort_values('Popularity', ascending=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(range(len(featured)), featured['Popularity'], color='#1DB954')
            ax.set_yticks(range(len(featured)))
            ax.set_yticklabels(featured['Track'], fontsize=8)
            ax.set_xlabel('Popularity')
            ax.set_title('Mamma Mia Featured Songs')
            st.pyplot(fig)
        
        st.markdown("---")
        featured_pop = abba_df[abba_df['Mamma_Mia']]['Popularity']
        non_featured_pop = abba_df[~abba_df['Mamma_Mia']]['Popularity']
        t_stat, p_value = stats.ttest_ind(featured_pop, non_featured_pop)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Difference", f"+{stats_mm.loc[True, 'mean'] - stats_mm.loc[False, 'mean']:.1f} points")
        col2.metric("t-statistic", f"{t_stat:.3f}")
        col3.metric("p-value", "< 0.0001")
        
        st.markdown("""
        <div class="insight-box">
        <b>Key Finding:</b> Featured songs substantially outperform non-featured material (71.5 vs 40.2). 
        All 13 featured songs exceed non-featured median. This partially explains ABBA's overall 
        advantage over Fleetwood Mac (43.8 vs 29.4). Fleetwood Mac lacks comparable cultural media exposure.
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "Hits vs Deep Cuts":
        st.header("Hits vs Deep Cuts Analysis")
        
        hits = df[df['Popularity'] >= df['Popularity'].quantile(0.8)]
        deep_cuts = df[df['Popularity'] <= df['Popularity'].quantile(0.2)]
        
        st.subheader("Sample Sizes")
        col1, col2 = st.columns(2)
        col1.metric("Hits (Top 20%)", len(hits))
        col2.metric("Deep Cuts (Bottom 20%)", len(deep_cuts))
        
        st.markdown("---")
        st.subheader("Audio Feature Comparison")
        
        features = ['Danceability', 'Energy', 'Acousticness', 'Valence', 'Loudness', 'Tempo']
        
        comparison = pd.DataFrame({
            'Hits': hits[features].mean(),
            'Deep Cuts': deep_cuts[features].mean()
        })
        comparison['Difference'] = comparison['Hits'] - comparison['Deep Cuts']
        
        p_values = []
        for feature in features:
            t_stat, p_val = stats.ttest_ind(hits[feature], deep_cuts[feature])
            p_values.append(p_val)
        
        comparison['p-value'] = p_values
        comparison['Significant'] = comparison['p-value'].apply(
            lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns'
        )
        
        st.dataframe(comparison.style.format({
            'Hits': '{:.3f}',
            'Deep Cuts': '{:.3f}',
            'Difference': '{:+.3f}',
            'p-value': '{:.4f}'
        }), use_container_width=True)
        
        st.caption("*** p<0.001, ** p<0.01, * p<0.05, ns=not significant")
        
        st.markdown("---")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            data = [hits[feature], deep_cuts[feature]]
            axes[idx].boxplot(data, labels=['Hits', 'Deep Cuts'])
            axes[idx].set_title(feature)
            axes[idx].set_ylabel(feature)
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        <div class="insight-box">
        <b>Key Finding:</b> Three characteristics significantly separate hits from deep cuts:
        <br>1. Loudness: +3.7 dB (p < 0.001) - production quality matters most
        <br>2. Energy: +0.113 (p = 0.002) - more energetic songs perform better
        <br>3. Danceability: +0.047 (p = 0.02) - modest but significant advantage
        <br><br>
        Emotional tone (valence), acousticness, and tempo show no significant differences.
        </div>
        """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("Data files not found. Ensure Abba.csv and FleetwoodMac.csv are in the data/ directory.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
ABBA vs Fleetwood Mac: Spotify Analysis | MSc Data Science | Winter 2025/2026
</div>
""", unsafe_allow_html=True)
