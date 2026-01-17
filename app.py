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
    .stRadio > label {font-size: 1.1rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">ABBA vs Fleetwood Mac: Spotify Analysis</p>', unsafe_allow_html=True)
st.markdown("Five Decades of Streaming Performance")
st.markdown("---")

@st.cache_data
def load_data():
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

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

try:
    df = load_data()
    
    # Global filters
    selected_artists = st.sidebar.multiselect(
        "Select Artists",
        options=['ABBA', 'Fleetwood Mac'],
        default=['ABBA', 'Fleetwood Mac']
    )
    
    selected_eras = st.sidebar.multiselect(
        "Select Eras",
        options=['70s', '80s', '90s', '2000s', '2020s'],
        default=['70s', '80s', '90s', '2000s', '2020s']
    )
    
    min_pop, max_pop = st.sidebar.slider(
        "Popularity Range",
        min_value=int(df['Popularity'].min()),
        max_value=int(df['Popularity'].max()),
        value=(int(df['Popularity'].min()), int(df['Popularity'].max()))
    )
    
    # Apply filters
    df_filtered = df[
        (df['Artist'].isin(selected_artists)) &
        (df['Era'].isin(selected_eras)) &
        (df['Popularity'] >= min_pop) &
        (df['Popularity'] <= max_pop)
    ]
    
    if len(df_filtered) == 0:
        st.warning("No data matches the selected filters. Please adjust your selection.")
        st.stop()
    
    if page == "Overview":
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Songs", len(df_filtered))
        col2.metric("ABBA Songs", len(df_filtered[df_filtered['Artist'] == 'ABBA']))
        col3.metric("Fleetwood Mac Songs", len(df_filtered[df_filtered['Artist'] == 'Fleetwood Mac']))
        col4.metric("Avg Popularity", f"{df_filtered['Popularity'].mean():.1f}")
        
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
        st.dataframe(df_filtered.head(10), use_container_width=True)
    
    elif page == "Artist Comparison":
        st.header("Artist Comparison")
        
        if len(df_filtered['Artist'].unique()) < 2:
            st.warning("Select both artists to compare.")
        else:
            stats_df = df_filtered.groupby('Artist')['Popularity'].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ABBA")
                if 'ABBA' in stats_df.index:
                    st.metric("Mean Popularity", f"{stats_df.loc['ABBA', 'mean']:.1f}")
                    st.metric("Median Popularity", f"{stats_df.loc['ABBA', 'median']:.0f}")
                    st.metric("Max Popularity", f"{stats_df.loc['ABBA', 'max']:.0f}")
                
            with col2:
                st.subheader("Fleetwood Mac")
                if 'Fleetwood Mac' in stats_df.index:
                    st.metric("Mean Popularity", f"{stats_df.loc['Fleetwood Mac', 'mean']:.1f}")
                    st.metric("Median Popularity", f"{stats_df.loc['Fleetwood Mac', 'median']:.0f}")
                    st.metric("Max Popularity", f"{stats_df.loc['Fleetwood Mac', 'max']:.0f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(df_filtered, x='Artist', y='Popularity', 
                            title='Popularity Distribution by Artist',
                            color='Artist',
                            color_discrete_map={'ABBA': '#1DB954', 'Fleetwood Mac': '#FF6B6B'})
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(df_filtered, x='Popularity', color='Artist',
                                  title='Popularity Distribution Comparison',
                                  nbins=20, barmode='overlay',
                                  color_discrete_map={'ABBA': '#1DB954', 'Fleetwood Mac': '#FF6B6B'})
                fig.update_traces(opacity=0.6)
                st.plotly_chart(fig, use_container_width=True)
            
            if 'ABBA' in stats_df.index and 'Fleetwood Mac' in stats_df.index:
                st.markdown("---")
                st.subheader("Statistical Analysis")
                
                abba_pop = df_filtered[df_filtered['Artist'] == 'ABBA']['Popularity']
                fm_pop = df_filtered[df_filtered['Artist'] == 'Fleetwood Mac']['Popularity']
                t_stat, p_value = stats.ttest_ind(abba_pop, fm_pop)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("t-statistic", f"{t_stat:.3f}")
                col2.metric("p-value", f"{p_value:.4f}")
                col3.metric("Significance", "Yes (p < 0.05)" if p_value < 0.05 else "No")
    
    elif page == "Era Analysis":
        st.header("Era-Based Performance Analysis")
        
        era_stats = df_filtered.groupby('Era')['Popularity'].agg(['mean', 'median', 'count'])
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Era Statistics")
            st.dataframe(era_stats.style.format({'mean': '{:.1f}', 'median': '{:.0f}', 'count': '{:.0f}'}),
                        use_container_width=True)
        
        with col2:
            fig = px.line(df_filtered.groupby(['Era', 'Artist'])['Popularity'].mean().reset_index(),
                         x='Era', y='Popularity', color='Artist',
                         title='Popularity Trends Over Time',
                         markers=True,
                         color_discrete_map={'ABBA': '#1DB954', 'Fleetwood Mac': '#FF6B6B'})
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Musical Personalities":
        st.header("Musical Personality Profiles")
        
        features = ['Danceability', 'Energy', 'Acousticness', 'Valence', 'Loudness']
        artist_features = df_filtered.groupby('Artist')[features].mean()
        
        st.subheader("Feature Comparison")
        
        if len(artist_features) == 2:
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
        
        for artist in artist_features.index:
            fig.add_trace(go.Scatterpolar(
                r=artist_features.loc[artist, features].values,
                theta=features,
                fill='toself',
                name=artist
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Feature Importance":
        st.header("Audio Feature Importance Analysis")
        
        features = ['Danceability', 'Energy', 'Acousticness', 'Valence', 
                   'Loudness', 'Tempo', 'Instrumentalness']
        
        correlations = df_filtered[features].corrwith(df_filtered['Popularity']).sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlations")
            st.dataframe(correlations.to_frame('Correlation').style.format('{:.3f}'),
                        use_container_width=True)
        
        with col2:
            fig = px.bar(x=correlations.values, y=correlations.index, orientation='h',
                        title='Feature Importance',
                        labels={'x': 'Correlation with Popularity', 'y': 'Feature'},
                        color=correlations.values,
                        color_continuous_scale='RdYlGn')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Feature Relationships")
        
        selected_feature = st.selectbox("Select feature to explore", features)
        
        fig = px.scatter(df_filtered, x=selected_feature, y='Popularity', color='Artist',
                        title=f'{selected_feature} vs Popularity',
                        trendline='ols',
                        color_discrete_map={'ABBA': '#1DB954', 'Fleetwood Mac': '#FF6B6B'})
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Clustering Analysis":
        st.header("Musical Style Clustering")
        
        features_for_clustering = ['Danceability', 'Energy', 'Acousticness', 
                                   'Valence', 'Loudness', 'Tempo']
        
        X = df_filtered[features_for_clustering].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)
        
        cluster_labels = {
            0: 'Acoustic Ballads',
            1: 'Mainstream Pop-Rock',
            2: 'High-Energy Dance',
            3: 'Slow-Tempo Songs'
        }
        
        df_filtered['Cluster_Label'] = df_filtered['Cluster'].map(cluster_labels)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Cluster Composition")
            cluster_stats = df_filtered.groupby('Cluster_Label').agg({
                'Popularity': ['mean', 'count']
            }).round(2)
            cluster_stats.columns = ['Mean Popularity', 'Song Count']
            st.dataframe(cluster_stats, use_container_width=True)
        
        with col2:
            fig = px.box(df_filtered, x='Cluster_Label', y='Popularity', color='Cluster_Label',
                        title='Popularity Distribution by Cluster')
            fig.update_layout(showlegend=False, xaxis_title='', xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Mamma Mia Effect":
        st.header("Mamma Mia Effect Analysis")
        
        mamma_mia_songs = [
            'Dancing Queen', 'Mamma Mia', 'The Winner Takes It All', 'Waterloo',
            'Super Trouper', 'Chiquitita', 'Voulez-Vous', 'Money, Money, Money',
            'Fernando', 'Knowing Me, Knowing You', 'Does Your Mother Know', 
            'SOS', 'Honey, Honey'
        ]
        
        abba_df = df_filtered[df_filtered['Artist'] == 'ABBA'].copy()
        abba_df['Mamma_Mia'] = abba_df['Track'].isin(mamma_mia_songs)
        
        if len(abba_df) > 0:
            stats_mm = abba_df.groupby('Mamma_Mia')['Popularity'].agg(['mean', 'median', 'count'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                if False in stats_mm.index:
                    st.subheader("Non-Featured Songs")
                    st.metric("Count", f"{int(stats_mm.loc[False, 'count'])}")
                    st.metric("Mean Popularity", f"{stats_mm.loc[False, 'mean']:.1f}")
                    st.metric("Median Popularity", f"{stats_mm.loc[False, 'median']:.0f}")
            
            with col2:
                if True in stats_mm.index:
                    st.subheader("Mamma Mia Featured")
                    st.metric("Count", f"{int(stats_mm.loc[True, 'count'])}")
                    st.metric("Mean Popularity", f"{stats_mm.loc[True, 'mean']:.1f}")
                    st.metric("Median Popularity", f"{stats_mm.loc[True, 'median']:.0f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(abba_df, x='Mamma_Mia', y='Popularity',
                            title='Popularity: Mamma Mia Songs vs Others',
                            labels={'Mamma_Mia': 'Song Type'})
                fig.update_xaxes(ticktext=['Other ABBA Songs', 'Mamma Mia Featured'],
                                tickvals=[False, True])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                featured = abba_df[abba_df['Mamma_Mia']].sort_values('Popularity', ascending=True)
                if len(featured) > 0:
                    fig = px.bar(featured, x='Popularity', y='Track', orientation='h',
                                title='Mamma Mia Featured Songs',
                                color='Popularity',
                                color_continuous_scale='Greens')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select ABBA to view Mamma Mia analysis")
    
    elif page == "Hits vs Deep Cuts":
        st.header("Hits vs Deep Cuts Analysis")
        
        hits = df_filtered[df_filtered['Popularity'] >= df_filtered['Popularity'].quantile(0.8)]
        deep_cuts = df_filtered[df_filtered['Popularity'] <= df_filtered['Popularity'].quantile(0.2)]
        
        col1, col2 = st.columns(2)
        col1.metric("Hits (Top 20%)", len(hits))
        col2.metric("Deep Cuts (Bottom 20%)", len(deep_cuts))
        
        st.markdown("---")
        
        features = ['Danceability', 'Energy', 'Acousticness', 'Valence', 'Loudness', 'Tempo']
        
        comparison = pd.DataFrame({
            'Hits': hits[features].mean(),
            'Deep Cuts': deep_cuts[features].mean()
        })
        comparison['Difference'] = comparison['Hits'] - comparison['Deep Cuts']
        
        p_values = []
        for feature in features:
            if len(hits) > 0 and len(deep_cuts) > 0:
                t_stat, p_val = stats.ttest_ind(hits[feature], deep_cuts[feature])
                p_values.append(p_val)
            else:
                p_values.append(1.0)
        
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

except FileNotFoundError:
    st.error("Data files not found. Ensure Abba.csv and FleetwoodMac.csv are in the data/ directory.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
ABBA vs Fleetwood Mac: Spotify Analysis | MSc Data Science | Winter 2025/2026
</div>
""", unsafe_allow_html=True)
