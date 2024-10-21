import pandas as pd
import networkx as nx
import folium

# Load data
traffic_stations = pd.read_csv('data/traffic_stations.csv')
station_included = pd.read_csv('data/stations_included.csv')
station_included = station_included[['0']].rename(columns={'0':'id'})
traffic_stations = traffic_stations.merge(station_included, on=['id'])
graph_df = pd.read_pickle('data/graph.pkl')

all_stations = list(traffic_stations['id'])
graph_df = graph_df.loc[all_stations, all_stations]

# Extract node (station) positions
df = traffic_stations.copy()
edge_data = graph_df

# Create a NetworkX graph
G = nx.Graph()

# Add nodes (stations)
for idx, row in df.iterrows():
    G.add_node(row['id'], pos=(row['longitude'], row['latitude']))

# Add edges based on adjacency matrix
for i in edge_data.index:
    for j in edge_data.columns:
        if edge_data.loc[i, j] == 1:
            G.add_edge(i, j)

# Initialize a Folium map centered around the average latitude and longitude of the stations
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
map_graph = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Add nodes (stations) as circle markers (with no icon)
for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,  # Size of the marker
        color='blue',
        fill=True,
        fill_color='blue',
        popup=f"Station ID: {row['id']}"
    ).add_to(map_graph)

# Add edges (as polylines) connecting the stations, with clearer lines
for i, j in G.edges():
    lon1, lat1 = G.nodes[i]['pos']
    lon2, lat2 = G.nodes[j]['pos']
    folium.PolyLine(
        locations=[(lat1, lon1), (lat2, lon2)],
        color='red',
        weight=3,  # Thickness of the line
        opacity=0.8  # Transparency of the line
    ).add_to(map_graph)

# Save the map to an HTML file
map_graph.save('img/map_stations_network.html')

# # To display the map in a Jupyter notebook (if applicable)
# map_graph
