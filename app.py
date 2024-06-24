import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

# 1. Data Preparation
df = pd.read_csv('permanently_cleaned_geolocation_data.csv')
df = df.rename(columns={'id_left': 'id', 'geolocation_state': 'state', 'geolocation_lat': 'latitude', 'geolocation_lng': 'longitude'})
df['object'] = df['object'].astype(str).str.capitalize()

# 2. Streamlit App
st.title("Olist Warehouse Location Optimization Tool")

# Filters (in the sidebar)
with st.sidebar:
    st.subheader("Filters")
    selected_objects = st.multiselect("Select Object:", df['object'].unique())
    
    # State Filter with "Select All"
    all_states = df['state'].unique().tolist()
    all_states.insert(0, "All")
    selected_states = st.multiselect("Select State:", all_states)
    if "All" in selected_states:
        selected_states = all_states[1:] 

    num_clusters = st.number_input("Number of Warehouses:", min_value=1, value=3)

# 3. Data Filtering and Display
if not selected_objects or not selected_states:
    st.warning("Please select at least one object and one state.")
else:
    filtered_df = df[df['object'].isin(selected_objects) & df['state'].isin(selected_states)]

    # 4. Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(filtered_df[['longitude', 'latitude']])
    filtered_df['cluster'] = kmeans.labels_

    # 5. Cluster Centers
    cluster_centers = filtered_df.groupby('cluster')[['longitude', 'latitude']].mean().reset_index()

    # 6. Mapbox Plot
    fig = px.scatter_mapbox(filtered_df, lat="latitude", lon="longitude", color="cluster",
                            hover_name="object",
                            mapbox_style="carto-positron",
                            color_discrete_sequence=["green"])
    fig.add_scattermapbox(
        lat=cluster_centers["latitude"],
        lon=cluster_centers["longitude"],
        mode="markers",
        marker=dict(size=20, color="red"),
        name="Warehouses",
        hoverinfo="text",
        text=[
            f"Warehouse {i + 1}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}"
            for i, lat, lon in zip(
                cluster_centers.index, cluster_centers["latitude"], cluster_centers["longitude"]
            )
        ],
        showlegend=False,
    )

    # Display Plot
    st.plotly_chart(fig, use_container_width=True)  # Expand plot to full width

