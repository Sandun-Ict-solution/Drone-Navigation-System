"""
Enhanced Autonomous Drone Monitoring System - Streamlined Edition
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import threading, time, json, random
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt

st.set_page_config(layout="wide", page_title="Elite Drone Command", page_icon="🚁")

st.markdown("""
<style>
.main-header {font-size: 3rem; font-weight: bold; background: linear-gradient(90deg, #1f77b4, #ff7f0e);
-webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin: 2rem 0;
animation: glow 2s ease-in-out infinite alternate;}
@keyframes glow {from {filter: drop-shadow(0 0 5px #1f77b4);} to {filter: drop-shadow(0 0 20px #ff7f0e);}}
.success-box {background: linear-gradient(135deg, #d4edda, #c3e6cb); padding: 1.5rem; border-radius: 10px;
border-left: 5px solid #28a745; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
.danger-box {background: linear-gradient(135deg, #f8d7da, #f5c6cb); padding: 1.5rem; border-radius: 10px;
border-left: 5px solid #dc3545; animation: pulse 1s infinite;}
@keyframes pulse {0%, 100% {opacity: 1;} 50% {opacity: 0.7;}}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚁 Autonomus Drone Navigation System</h1>', unsafe_allow_html=True)

# Initialize State
if 'initialized' not in st.session_state:
    st.session_state.update({
        'initialized': True, 'mission_active': False, 'emergency_stop': False, 'mission_start_time': None,
        'alerts': [], 'drone_configs': [], 'connection_status': {}, 'drones_data': {}, 'drone_ids': [],
        'missions_completed': 0,
        'drone_names': {}, 'flight_altitude': 20, 'mission_speed': 7, 'min_battery_threshold': 20,
        'auto_rth_low_battery': True, 'mission_type': 'Survey', 'weather_condition': 'Clear',
        'selected_area': [], 'generated_waypoints': None
    })

# Helper Functions
def geodesic_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2)**2
    return 2 * asin(sqrt(a)) * 6371000

def initialize_drone_data(drone_id):
    st.session_state.drones_data[drone_id] = {
        "telemetry": {"battery": 100, "altitude": st.session_state.flight_altitude,
            "latitude": 6.9271 + np.random.uniform(-0.001, 0.001),
            "longitude": 79.8612 + np.random.uniform(-0.001, 0.001),
            "speed": st.session_state.mission_speed, "heading": 0, "gps_fix": "3D",
            "satellites": 12, "voltage": 12.6, "status": "READY"},
        "flight_history": [], "detected_objects": [], "alerts": [], "home_position": (6.9271, 79.8612),
        "rth_active": False, "statistics": {"total_distance": 0, "objects_detected": 0, "waypoints_completed": 0},
        "waypoints": pd.DataFrame({"Waypoint": ["WP1", "WP2", "WP3"], "Status": ["Pending"]*3,
            "Latitude": [6.9272, 6.9275, 6.9278], "Longitude": [79.8615, 79.8618, 79.8621]}),
        "camera_url": ""  # IP camera URL
    }

def add_alert(drone_id, level, message):
    alert = {"drone": drone_id, "level": level, "message": message, "timestamp": datetime.now()}
    if drone_id in st.session_state.drones_data:
        st.session_state.drones_data[drone_id]["alerts"].append(alert)
    st.session_state.alerts.append(alert)

def simulate_drone_telemetry(drone_id):
    while st.session_state.mission_active and drone_id in st.session_state.drones_data:
        try:
            data = st.session_state.drones_data[drone_id]
            data["telemetry"]["battery"] -= 0.05
            data["telemetry"]["altitude"] += np.random.uniform(-0.5, 0.5)
            data["telemetry"]["latitude"] += np.random.uniform(-0.00001, 0.00001)
            data["telemetry"]["longitude"] += np.random.uniform(-0.00001, 0.00001)
            data["telemetry"]["heading"] = (data["telemetry"]["heading"] + np.random.uniform(-5, 5)) % 360
            data["flight_history"].append((data["telemetry"]["longitude"], data["telemetry"]["latitude"],
                data["telemetry"]["altitude"], time.time()))
            if len(data["flight_history"]) > 500:
                data["flight_history"] = data["flight_history"][-500:]
            if np.random.random() < 0.05:
                data["detected_objects"].append({"lon": data["telemetry"]["longitude"],
                    "lat": data["telemetry"]["latitude"], "class": np.random.randint(0, 5)})
                data["statistics"]["objects_detected"] += 1
            if st.session_state.auto_rth_low_battery and data["telemetry"]["battery"] < st.session_state.min_battery_threshold:
                data["rth_active"] = True
                add_alert(drone_id, "CRITICAL", "Low battery - RTH")
            if len(data["flight_history"]) > 1:
                last, curr = data["flight_history"][-2], data["flight_history"][-1]
                data["statistics"]["total_distance"] += geodesic_distance(last[1], last[0], curr[1], curr[0])
            time.sleep(1)
        except: time.sleep(1)

def connect_drone(config):
    drone_id = config['id']
    config['connected'] = True
    if drone_id not in st.session_state.drone_ids:
        st.session_state.drone_ids.append(drone_id)
    if drone_id not in st.session_state.drones_data:
        initialize_drone_data(drone_id)
    st.success(f"✅ {drone_id} connected")

def disconnect_drone(config):
    config['connected'] = False
    if config['id'] in st.session_state.drone_ids:
        st.session_state.drone_ids.remove(config['id'])

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎛️ Control", "📊 Analytics", "🗺️ Map & Tracking", "📹 Live Cameras", "⚙️ Settings", "📄 Reports"])

# CONTROL TAB
with tab1:
    st.header("🎮 Mission Control")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.session_state.mission_active and st.session_state.mission_start_time:
            elapsed = datetime.now() - st.session_state.mission_start_time
            st.metric("⏱️ Time", str(elapsed).split('.')[0])
        else:
            st.metric("⏱️ Time", "00:00:00")
    with col2:
        st.metric("🚁 Drones", len(st.session_state.drone_ids))
    with col3:
        if st.session_state.drones_data:
            avg_bat = np.mean([d["telemetry"]["battery"] for d in st.session_state.drones_data.values()])
            st.metric("🔋 Avg Battery", f"{avg_bat:.1f}%")
        else:
            st.metric("🔋 Avg Battery", "N/A")
    with col4:
        total_obj = sum(d["statistics"]["objects_detected"] for d in st.session_state.drones_data.values())
        st.metric("👁️ Detections", total_obj)
    
    st.divider()
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🚀 START", use_container_width=True, disabled=st.session_state.mission_active, type="primary"):
            st.session_state.mission_active = True
            st.session_state.mission_start_time = datetime.now()
            st.session_state.emergency_stop = False
            for drone_id in st.session_state.drone_ids:
                threading.Thread(target=simulate_drone_telemetry, args=(drone_id,), daemon=True).start()
            st.rerun()
    with col2:
        if st.button("⏸️ PAUSE", use_container_width=True, disabled=not st.session_state.mission_active):
            st.session_state.mission_active = False
            st.rerun()
    with col3:
        if st.button("🏠 RTH", use_container_width=True):
            for d_id in st.session_state.drone_ids:
                if d_id in st.session_state.drones_data:
                    st.session_state.drones_data[d_id]["rth_active"] = True
    with col4:
        if st.button("🛑 STOP", use_container_width=True):
            st.session_state.emergency_stop = True
            st.session_state.mission_active = False
            st.rerun()
    
    # Status
    if st.session_state.mission_active:
        st.markdown('<div class="success-box">✅ MISSION ACTIVE</div>', unsafe_allow_html=True)
    elif st.session_state.emergency_stop:
        st.markdown('<div class="danger-box">🛑 EMERGENCY STOP</div>', unsafe_allow_html=True)
    else:
        st.info("⏸️ Ready to launch")
    
    st.divider()
    
    # Fleet status
    st.subheader("🚁 Fleet Monitor")
    if st.session_state.drones_data:
        cols = st.columns(min(len(st.session_state.drones_data), 3))
        for idx, (d_id, data) in enumerate(st.session_state.drones_data.items()):
            with cols[idx % 3]:
                name = st.session_state.drone_names.get(d_id, d_id)
                st.markdown(f"### 🚁 {name}")
                battery = data["telemetry"]["battery"]
                st.metric("Battery", f"{battery:.1f}%")
                st.progress(battery / 100)
                c1, c2 = st.columns(2)
                c1.metric("Alt", f"{data['telemetry']['altitude']:.1f}m")
                c2.metric("Speed", f"{data['telemetry']['speed']:.1f}m/s")
                if battery < 20:
                    st.error("⚠️ LOW BATTERY")
                elif data["rth_active"]:
                    st.warning("🏠 RTH")
                else:
                    st.success("✅ OK")
    else:
        st.info("No drones connected")
    
    st.divider()
    
    # Alerts
    st.subheader("🔔 Live Alerts")
    if st.session_state.alerts:
        for alert in reversed(st.session_state.alerts[-5:]):
            ts = alert["timestamp"].strftime("%H:%M:%S")
            if alert["level"] == "CRITICAL":
                st.error(f"🚨 [{ts}] {alert['drone']}: {alert['message']}")
            elif alert["level"] == "WARNING":
                st.warning(f"⚠️ [{ts}] {alert['drone']}: {alert['message']}")
            else:
                st.info(f"ℹ️ [{ts}] {alert['drone']}: {alert['message']}")
    else:
        st.success("✅ All clear")
    
    st.divider()
    
    # 3D View
    st.subheader("🗺️ 3D View")
    view_mode = st.radio("View:", ["Isometric", "Top-Down", "Side"], horizontal=True)
    
    if st.session_state.drones_data:
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for idx, (d_id, data) in enumerate(st.session_state.drones_data.items()):
            color = colors[idx % len(colors)]
            name = st.session_state.drone_names.get(d_id, d_id)
            if data["flight_history"]:
                fh = np.array(data["flight_history"])
                fig.add_trace(go.Scatter3d(x=fh[:, 0], y=fh[:, 1], z=fh[:, 2],
                    mode='lines', line=dict(width=4, color=color), name=f"{name} Path"))
                fig.add_trace(go.Scatter3d(
                    x=[data["telemetry"]["longitude"]], y=[data["telemetry"]["latitude"]],
                    z=[data["telemetry"]["altitude"]], mode='markers+text',
                    marker=dict(size=15, color=color), text=[f"🚁{name}"], name=f"{name}"))
            home = data["home_position"]
            fig.add_trace(go.Scatter3d(x=[home[1]], y=[home[0]], z=[0],
                mode='markers', marker=dict(size=20, color='cyan'), name=f"{name} Home"))
        
        camera = dict(eye=dict(x=0, y=0, z=2.5)) if view_mode == "Top-Down" else \
                 dict(eye=dict(x=2.5, y=0, z=0.5)) if view_mode == "Side" else \
                 dict(eye=dict(x=1.5, y=1.5, z=1.2))
        fig.update_layout(scene=dict(xaxis_title='Lon', yaxis_title='Lat', zaxis_title='Alt(m)', camera=camera),
            height=600, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Connect drones to view")

# ANALYTICS TAB
with tab2:
    st.header("📊 Analytics")
    if st.session_state.drones_data:
        total_dist = sum(d["statistics"]["total_distance"] for d in st.session_state.drones_data.values())
        total_obj = sum(d["statistics"]["objects_detected"] for d in st.session_state.drones_data.values())
        c1, c2, c3 = st.columns(3)
        c1.metric("Distance", f"{total_dist:.1f}m")
        c2.metric("Detections", total_obj)
        c3.metric("Missions", st.session_state.missions_completed)
        
        st.divider()
        st.subheader("Drone Comparison")
        comp_data = []
        for d_id, data in st.session_state.drones_data.items():
            comp_data.append({
                "Drone": st.session_state.drone_names.get(d_id, d_id),
                "Distance": data["statistics"]["total_distance"],
                "Objects": data["statistics"]["objects_detected"],
                "Battery": data["telemetry"]["battery"]
            })
        df = pd.DataFrame(comp_data)
        st.dataframe(df, use_container_width=True)
        
        fig = px.bar(df, x="Drone", y="Distance", title="Distance by Drone")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data")

# PLANNER TAB (NOW MAP & TRACKING)
with tab3:
    st.header("🗺️ Real-Time Map & Drone Tracking")
    
    # Map view selector
    map_type = st.radio("Map Type:", ["Live Tracking", "Mission Planning", "Flight History"], horizontal=True)
    
    if map_type == "Live Tracking":
        st.subheader("📍 Live Drone Locations")
        
        try:
            import folium
            from folium.plugins import Draw, MarkerCluster
            from streamlit_folium import st_folium
            
            # Google Maps API configuration
            GOOGLE_MAPS_API_KEY = 'AIzaSyBjaToM_WMrY3r3N2nnKrTc1vqF6_Bvhuk'
            
            # Create map centered on average drone position
            if st.session_state.drones_data:
                lats = [d["telemetry"]["latitude"] for d in st.session_state.drones_data.values()]
                lons = [d["telemetry"]["longitude"] for d in st.session_state.drones_data.values()]
                center_lat = np.mean(lats)
                center_lon = np.mean(lons)
            else:
                center_lat, center_lon = 6.9271, 79.8612
            
            # Create folium map with Google Satellite view
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=17,
                tiles=f"https://mt1.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}&key={GOOGLE_MAPS_API_KEY}",
                attr="Google Satellite",
            )
            
            # Add drone markers and flight paths
            colors = ['blue', 'red', 'green', 'purple', 'orange']
            for idx, (drone_id, data) in enumerate(st.session_state.drones_data.items()):
                color = colors[idx % len(colors)]
                drone_name = st.session_state.drone_names.get(drone_id, drone_id)
                
                # Current position marker
                lat = data["telemetry"]["latitude"]
                lon = data["telemetry"]["longitude"]
                alt = data["telemetry"]["altitude"]
                battery = data["telemetry"]["battery"]
                
                # Create popup with drone info
                popup_html = f"""
                <div style="font-family: Arial; width: 200px;">
                    <h4 style="margin: 0;">🚁 {drone_name}</h4>
                    <hr style="margin: 5px 0;">
                    <b>Battery:</b> {battery:.1f}%<br>
                    <b>Altitude:</b> {alt:.1f}m<br>
                    <b>Speed:</b> {data["telemetry"]["speed"]:.1f}m/s<br>
                    <b>Heading:</b> {data["telemetry"]["heading"]:.1f}°<br>
                    <b>Status:</b> {data["telemetry"]["status"]}
                </div>
                """
                
                # Add current position marker
                folium.Marker(
                    [lat, lon],
                    popup=folium.Popup(popup_html, max_width=250),
                    icon=folium.Icon(color=color, icon='plane', prefix='fa'),
                    tooltip=f"{drone_name} - {battery:.1f}%"
                ).add_to(m)
                
                # Add flight path
                if len(data["flight_history"]) > 1:
                    path_coords = [(point[1], point[0]) for point in data["flight_history"][-100:]]
                    folium.PolyLine(
                        path_coords,
                        color=color,
                        weight=3,
                        opacity=0.7,
                        popup=f"{drone_name} Flight Path"
                    ).add_to(m)
                
                # Add home position
                home = data["home_position"]
                folium.Marker(
                    [home[0], home[1]],
                    popup=f"🏠 {drone_name} Home",
                    icon=folium.Icon(color='green', icon='home')
                ).add_to(m)
                
                # Add detected objects
                for obj in data["detected_objects"][-20:]:  # Last 20 objects
                    folium.CircleMarker(
                        [obj["lat"], obj["lon"]],
                        radius=3,
                        color='yellow',
                        fill=True,
                        popup=f"Object detected (Class {obj['class']})",
                        tooltip="Detected Object"
                    ).add_to(m)
            
            # Display map
            st_folium(m, height=600, width=None)
            
            # Live telemetry table
            st.divider()
            st.subheader("📊 Live Telemetry Data")
            telemetry_data = []
            for drone_id, data in st.session_state.drones_data.items():
                telemetry_data.append({
                    "Drone": st.session_state.drone_names.get(drone_id, drone_id),
                    "Lat": f"{data['telemetry']['latitude']:.6f}",
                    "Lon": f"{data['telemetry']['longitude']:.6f}",
                    "Alt(m)": f"{data['telemetry']['altitude']:.1f}",
                    "Battery": f"{data['telemetry']['battery']:.1f}%",
                    "Speed": f"{data['telemetry']['speed']:.1f}m/s",
                    "Heading": f"{data['telemetry']['heading']:.0f}°",
                    "Status": data['telemetry']['status']
                })
            
            if telemetry_data:
                st.dataframe(pd.DataFrame(telemetry_data), use_container_width=True)
            
        except ImportError:
            st.error("⚠️ Map feature requires: pip install folium streamlit-folium")
    
    elif map_type == "Mission Planning":
        st.subheader("📍 Mission Planning & Area Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.mission_type = st.selectbox(
                "Mission Type",
                ["Survey", "Security Patrol", "Infrastructure Inspection", "Emergency Response"]
            )
        with col2:
            st.session_state.weather_condition = st.selectbox(
                "Weather", ["Clear", "Cloudy", "Rainy", "Windy"]
            )
        
        # Area selection map (keeping original functionality)
        try:
            import folium
            from folium.plugins import Draw
            from streamlit_folium import st_folium
            
            GOOGLE_MAPS_API_KEY = 'AIzaSyBjaToM_WMrY3r3N2nnKrTc1vqF6_Bvhuk'
            
            m = folium.Map(
                location=[6.9271, 79.8612],
                zoom_start=16,
                tiles=f"https://mt1.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}&key={GOOGLE_MAPS_API_KEY}",
                attr="Google Satellite",
            )
            
            Draw(
                export=True,
                draw_options={
                    "polygon": True,
                    "rectangle": True,
                    "circle": False,
                    "marker": False,
                    "polyline": False,
                    "circlemarker": False
                },
                edit_options={"edit": True, "remove": True}
            ).add_to(m)
            
            if st.session_state.drones_data:
                for drone_id, data in st.session_state.drones_data.items():
                    home = data["home_position"]
                    folium.Marker(
                        [home[0], home[1]],
                        popup=f"🏠 {st.session_state.drone_names.get(drone_id, drone_id)} Home",
                        icon=folium.Icon(color='green', icon='home')
                    ).add_to(m)
            
            map_data = st_folium(m, height=500, width=None, returned_objects=["last_active_drawing"])
            
            if map_data and map_data.get("last_active_drawing"):
                geometry = map_data["last_active_drawing"]["geometry"]
                if geometry["type"] in ["Polygon", "Rectangle"]:
                    coords = geometry["coordinates"][0]
                    st.session_state.selected_area = [(c[1], c[0]) for c in coords]
                    st.success("✅ Working area selected successfully!")
            
            if st.session_state.selected_area:
                st.info(f"📐 Area Points: {len(st.session_state.selected_area)} | Area defined and ready for mission")
                
                if len(st.session_state.selected_area) > 2:
                    area_approx = len(st.session_state.selected_area) * 100
                    st.metric("Estimated Area", f"~{area_approx}m²")
            
        except ImportError:
            st.warning("⚠️ Map feature requires: pip install folium streamlit-folium")
        
        st.divider()
        
        # Mission planning controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 Generate Mission Plan", use_container_width=True, type="primary"):
                if st.session_state.selected_area:
                    st.success(f"✅ Mission plan generated for {st.session_state.mission_type}")
                    st.info(f"🌦️ Weather: {st.session_state.weather_condition}")
                    
                    num_waypoints = st.slider("Number of Waypoints", 3, 20, 8)
                    if len(st.session_state.selected_area) >= 3:
                        lats = [coord[0] for coord in st.session_state.selected_area]
                        lons = [coord[1] for coord in st.session_state.selected_area]
                        
                        min_lat, max_lat = min(lats), max(lats)
                        min_lon, max_lon = min(lons), max(lons)
                        
                        waypoints = []
                        for i in range(num_waypoints):
                            waypoints.append({
                                "Waypoint": f"WP{i+1}",
                                "Latitude": min_lat + (max_lat - min_lat) * np.random.random(),
                                "Longitude": min_lon + (max_lon - min_lon) * np.random.random(),
                                "Altitude": st.session_state.flight_altitude,
                                "Status": "Pending"
                            })
                        
                        st.session_state.generated_waypoints = pd.DataFrame(waypoints)
                        st.success(f"Generated {num_waypoints} waypoints!")
                else:
                    st.warning("⚠️ Please select a working area first!")
        
        with col2:
            if st.button("📋 Auto-Assign to Drones", use_container_width=True):
                if st.session_state.drone_ids and 'generated_waypoints' in st.session_state:
                    num_drones = len(st.session_state.drone_ids)
                    waypoints_per_drone = len(st.session_state.generated_waypoints) // num_drones
                    
                    for idx, drone_id in enumerate(st.session_state.drone_ids):
                        start_idx = idx * waypoints_per_drone
                        end_idx = start_idx + waypoints_per_drone if idx < num_drones - 1 else len(st.session_state.generated_waypoints)
                        
                        st.session_state.drones_data[drone_id]["waypoints"] = st.session_state.generated_waypoints.iloc[start_idx:end_idx].copy()
                    
                    st.success(f"Waypoints distributed among {num_drones} drones!")
                else:
                    st.warning("Need drones connected and mission plan generated!")
        
        with col3:
            if st.button("🔄 Reset Area", use_container_width=True):
                st.session_state.selected_area = []
                if 'generated_waypoints' in st.session_state:
                    del st.session_state.generated_waypoints
                st.info("Area reset!")
    
    else:  # Flight History
        st.subheader("📜 Flight History & Coverage")
        
        selected_drone = st.selectbox("Select Drone for History", 
                                     list(st.session_state.drones_data.keys()) if st.session_state.drones_data else [])
        
        if selected_drone and st.session_state.drones_data:
            data = st.session_state.drones_data[selected_drone]
            drone_name = st.session_state.drone_names.get(selected_drone, selected_drone)
            
            st.write(f"**Flight History for {drone_name}**")
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Distance", f"{data['statistics']['total_distance']:.1f}m")
            col2.metric("Flight Points", len(data['flight_history']))
            col3.metric("Objects Detected", data['statistics']['objects_detected'])
            col4.metric("Waypoints Done", data['statistics']['waypoints_completed'])
            
            # Plot flight path on map
            try:
                import folium
                from streamlit_folium import st_folium
                
                GOOGLE_MAPS_API_KEY = 'AIzaSyBjaToM_WMrY3r3N2nnKrTc1vqF6_Bvhuk'
                
                if data["flight_history"]:
                    # Center on flight path
                    lats = [point[1] for point in data["flight_history"]]
                    lons = [point[0] for point in data["flight_history"]]
                    center_lat = np.mean(lats)
                    center_lon = np.mean(lons)
                    
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=17,
                        tiles=f"https://mt1.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}&key={GOOGLE_MAPS_API_KEY}",
                        attr="Google Satellite",
                    )
                    
                    # Add complete flight path
                    path_coords = [(point[1], point[0]) for point in data["flight_history"]]
                    folium.PolyLine(
                        path_coords,
                        color='blue',
                        weight=3,
                        opacity=0.7,
                        popup=f"{drone_name} Complete Flight Path"
                    ).add_to(m)
                    
                    # Add start and end markers
                    if len(path_coords) > 0:
                        folium.Marker(
                            path_coords[0],
                            popup="Start",
                            icon=folium.Icon(color='green', icon='play')
                        ).add_to(m)
                        
                        folium.Marker(
                            path_coords[-1],
                            popup="Current Position",
                            icon=folium.Icon(color='red', icon='plane', prefix='fa')
                        ).add_to(m)
                    
                    # Add detected objects
                    for obj in data["detected_objects"]:
                        folium.CircleMarker(
                            [obj["lat"], obj["lon"]],
                            radius=5,
                            color='yellow',
                            fill=True,
                            popup=f"Object (Class {obj['class']})"
                        ).add_to(m)
                    
                    st_folium(m, height=500, width=None)
                else:
                    st.info("No flight history available yet")
                    
            except ImportError:
                st.error("⚠️ Map feature requires: pip install folium streamlit-folium")
    st.header("🗺️ Mission Planner & Area Selection")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.mission_type = st.selectbox(
            "Mission Type",
            ["Survey", "Security Patrol", "Infrastructure Inspection", "Emergency Response"]
        )
    with col2:
        st.session_state.weather_condition = st.selectbox(
            "Weather", ["Clear", "Cloudy", "Rainy", "Windy"]
        )
    
    st.divider()
    st.subheader("📍 Select Working Area")
    
    # Google Maps API configuration
    GOOGLE_MAPS_API_KEY = 'AIzaSyBjaToM_WMrY3r3N2nnKrTc1vqF6_Bvhuk'
    
    # Interactive map with drawing tools
    try:
        import folium
        from folium.plugins import Draw
        from streamlit_folium import st_folium
        
        # Initialize selected area in session state
        if 'selected_area' not in st.session_state:
            st.session_state.selected_area = []
        
        # Create folium map with Google Satellite view using API key
        m = folium.Map(
            location=[6.9271, 79.8612],
            zoom_start=16,
            tiles=f"https://mt1.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}&key={GOOGLE_MAPS_API_KEY}",
            attr="Google Satellite",
        )
        
        # Add drawing tools
        Draw(
            export=True,
            draw_options={
                "polygon": True,
                "rectangle": True,
                "circle": False,
                "marker": False,
                "polyline": False,
                "circlemarker": False
            },
            edit_options={"edit": True, "remove": True}
        ).add_to(m)
        
        # Add drone home positions if available
        if st.session_state.drones_data:
            for drone_id, data in st.session_state.drones_data.items():
                home = data["home_position"]
                folium.Marker(
                    [home[0], home[1]],
                    popup=f"🏠 {st.session_state.drone_names.get(drone_id, drone_id)} Home",
                    icon=folium.Icon(color='green', icon='home')
                ).add_to(m)
        
        # Display map
        map_data = st_folium(m, height=500, width=None, returned_objects=["last_active_drawing"])
        
        # Process drawn area
        if map_data and map_data.get("last_active_drawing"):
            geometry = map_data["last_active_drawing"]["geometry"]
            if geometry["type"] in ["Polygon", "Rectangle"]:
                coords = geometry["coordinates"][0]
                st.session_state.selected_area = [(c[1], c[0]) for c in coords]  # lat, lon format
                st.success("✅ Working area selected successfully!")
        
        # Display area information
        if st.session_state.selected_area:
            st.info(f"📐 Area Points: {len(st.session_state.selected_area)} | Area defined and ready for mission")
            
            # Calculate approximate area
            if len(st.session_state.selected_area) > 2:
                # Simple area calculation
                area_approx = len(st.session_state.selected_area) * 100  # Rough estimate
                st.metric("Estimated Area", f"~{area_approx}m²")
        
    except ImportError:
        st.warning("⚠️ Map feature requires: pip install folium streamlit-folium")
        st.info("Using simplified area selection...")
        
        # Fallback: Manual coordinate input
        st.write("**Manual Area Definition**")
        col1, col2 = st.columns(2)
        with col1:
            lat_center = st.number_input("Center Latitude", value=6.9271, format="%.6f")
        with col2:
            lon_center = st.number_input("Center Longitude", value=79.8612, format="%.6f")
        
        area_size = st.slider("Area Size (meters)", 50, 500, 200)
        
        if st.button("Define Area"):
            # Create simple square area
            offset = area_size / 111000  # rough conversion to degrees
            st.session_state.selected_area = [
                (lat_center + offset, lon_center - offset),
                (lat_center + offset, lon_center + offset),
                (lat_center - offset, lon_center + offset),
                (lat_center - offset, lon_center - offset),
            ]
            st.success("✅ Area defined!")
    
    st.divider()
    
    # Mission planning controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Generate Mission Plan", use_container_width=True, type="primary"):
            if st.session_state.selected_area:
                st.success(f"✅ Mission plan generated for {st.session_state.mission_type}")
                st.info(f"🌦️ Weather: {st.session_state.weather_condition}")
                
                # Generate waypoints based on area
                num_waypoints = st.slider("Number of Waypoints", 3, 20, 8)
                if len(st.session_state.selected_area) >= 3:
                    # Calculate waypoints within the selected area
                    lats = [coord[0] for coord in st.session_state.selected_area]
                    lons = [coord[1] for coord in st.session_state.selected_area]
                    
                    min_lat, max_lat = min(lats), max(lats)
                    min_lon, max_lon = min(lons), max(lons)
                    
                    # Generate grid waypoints
                    waypoints = []
                    for i in range(num_waypoints):
                        waypoints.append({
                            "Waypoint": f"WP{i+1}",
                            "Latitude": min_lat + (max_lat - min_lat) * np.random.random(),
                            "Longitude": min_lon + (max_lon - min_lon) * np.random.random(),
                            "Altitude": st.session_state.flight_altitude,
                            "Status": "Pending"
                        })
                    
                    st.session_state.generated_waypoints = pd.DataFrame(waypoints)
                    st.success(f"Generated {num_waypoints} waypoints!")
            else:
                st.warning("⚠️ Please select a working area first!")
    
    with col2:
        if st.button("📋 Auto-Assign to Drones", use_container_width=True):
            if st.session_state.drone_ids and 'generated_waypoints' in st.session_state:
                # Distribute waypoints among drones
                num_drones = len(st.session_state.drone_ids)
                waypoints_per_drone = len(st.session_state.generated_waypoints) // num_drones
                
                for idx, drone_id in enumerate(st.session_state.drone_ids):
                    start_idx = idx * waypoints_per_drone
                    end_idx = start_idx + waypoints_per_drone if idx < num_drones - 1 else len(st.session_state.generated_waypoints)
                    
                    st.session_state.drones_data[drone_id]["waypoints"] = st.session_state.generated_waypoints.iloc[start_idx:end_idx].copy()
                
                st.success(f"Waypoints distributed among {num_drones} drones!")
            else:
                st.warning("Need drones connected and mission plan generated!")
    
    with col3:
        if st.button("🔄 Reset Area", use_container_width=True):
            st.session_state.selected_area = []
            if 'generated_waypoints' in st.session_state:
                del st.session_state.generated_waypoints
            st.info("Area reset!")
    
    # Display generated waypoints
    if 'generated_waypoints' in st.session_state and st.session_state.generated_waypoints is not None and not st.session_state.generated_waypoints.empty:
        st.divider()
        st.subheader("📍 Generated Waypoints")
        st.dataframe(st.session_state.generated_waypoints, use_container_width=True)
        
        # Download waypoints
        try:
            csv = st.session_state.generated_waypoints.to_csv(index=False)
            st.download_button(
                "💾 Download Waypoints CSV",
                csv,
                "mission_waypoints.csv",
                "text/csv",
                key="download_waypoints"
            )
        except Exception as e:
            st.error(f"Error generating CSV: {e}")
    
    # Show drone-specific waypoints
    if st.session_state.drones_data:
        st.divider()
        st.subheader("🚁 Drone Waypoint Assignments")
        
        selected_drone = st.selectbox("View Drone Waypoints", list(st.session_state.drones_data.keys()))
        if selected_drone:
            drone_name = st.session_state.drone_names.get(selected_drone, selected_drone)
            st.write(f"**Waypoints for {drone_name}:**")
            st.dataframe(st.session_state.drones_data[selected_drone]["waypoints"], use_container_width=True)
            
            # Waypoint progress
            total_wp = len(st.session_state.drones_data[selected_drone]["waypoints"])
            completed_wp = len(st.session_state.drones_data[selected_drone]["waypoints"][
                st.session_state.drones_data[selected_drone]["waypoints"]["Status"] == "Completed"
            ])
            
            if total_wp > 0:
                progress = completed_wp / total_wp
                st.progress(progress)
                st.caption(f"Progress: {completed_wp}/{total_wp} waypoints completed ({progress*100:.1f}%)")

# LIVE CAMERAS TAB (NEW)
with tab4:
    st.header("📹 Live Drone Cameras")
    
    if not st.session_state.drones_data:
        st.info("No drones connected. Connect drones to view camera feeds.")
    else:
        # Camera grid layout selector
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Live Camera Feeds")
        with col2:
            grid_layout = st.selectbox("Layout", ["1x1", "2x2", "3x3"], index=1)
        
        st.divider()
        
        # Determine grid size
        if grid_layout == "1x1":
            cols_count = 1
        elif grid_layout == "2x2":
            cols_count = 2
        else:
            cols_count = 3
        
        # Display cameras in grid
        drone_list = list(st.session_state.drones_data.items())
        
        for row_idx in range(0, len(drone_list), cols_count):
            cols = st.columns(cols_count)
            
            for col_idx, col in enumerate(cols):
                drone_idx = row_idx + col_idx
                if drone_idx < len(drone_list):
                    drone_id, data = drone_list[drone_idx]
                    drone_name = st.session_state.drone_names.get(drone_id, drone_id)
                    
                    with col:
                        st.markdown(f"### 🚁 {drone_name}")
                        
                        # Camera URL input
                        camera_url = st.text_input(
                            f"Camera URL",
                            value=data.get("camera_url", ""),
                            key=f"cam_url_{drone_id}",
                            placeholder="rtsp://192.168.1.100:8554/stream or http://..."
                        )
                        
                        # Update camera URL in session state
                        if camera_url != data.get("camera_url", ""):
                            st.session_state.drones_data[drone_id]["camera_url"] = camera_url
                        
                        # Display camera feed
                        if camera_url:
                            # Check if it's HTTP/HTTPS URL (image stream)
                            if camera_url.startswith("http://") or camera_url.startswith("https://"):
                                try:
                                    st.image(camera_url, use_container_width=True)
                                    st.caption("🟢 Live Feed Active")
                                except:
                                    st.error("❌ Failed to load camera feed")
                                    st.caption("Verify URL is accessible")
                            
                            # RTSP stream info
                            elif camera_url.startswith("rtsp://"):
                                st.info("📡 RTSP Stream Detected")
                                st.write("**Stream URL:**", camera_url)
                                st.caption("⚠️ RTSP requires video conversion. Use HTTP/HTTPS for direct viewing in browser.")
                                
                                # Instructions for RTSP
                                with st.expander("ℹ️ How to use RTSP streams"):
                                    st.write("""
                                    **Option 1: Convert RTSP to HTTP using FFmpeg**
                                    ```bash
                                    ffmpeg -i rtsp://camera_ip:port/stream -f mjpeg http://localhost:8080/stream.mjpg
                                    ```
                                    
                                    **Option 2: Use VLC or dedicated player**
                                    - Open RTSP URL in VLC Media Player
                                    - Use IP Webcam apps that provide HTTP streams
                                    
                                    **Option 3: Use camera manufacturer's web interface**
                                    - Most IP cameras provide HTTP snapshot URLs
                                    - Format: `http://camera_ip/snapshot.jpg`
                                    """)
                            else:
                                st.warning("⚠️ Unsupported URL format")
                                st.caption("Use HTTP/HTTPS URLs for live viewing")
                        else:
                            # Placeholder when no camera configured
                            st.image("https://via.placeholder.com/640x480/1f77b4/ffffff?text=No+Camera+Feed", 
                                   use_container_width=True)
                            st.caption("⚪ No camera configured")
                        
                        # Drone telemetry overlay
                        st.divider()
                        tel_col1, tel_col2, tel_col3 = st.columns(3)
                        tel_col1.metric("Battery", f"{data['telemetry']['battery']:.1f}%")
                        tel_col2.metric("Alt", f"{data['telemetry']['altitude']:.1f}m")
                        tel_col3.metric("Speed", f"{data['telemetry']['speed']:.1f}m/s")
                        
                        # Camera controls
                        st.divider()
                        cam_col1, cam_col2, cam_col3 = st.columns(3)
                        
                        with cam_col1:
                            if st.button("📸 Snapshot", key=f"snap_{drone_id}", use_container_width=True):
                                st.success("Snapshot saved!")
                        
                        with cam_col2:
                            if st.button("🔄 Refresh", key=f"refresh_{drone_id}", use_container_width=True):
                                st.rerun()
                        
                        with cam_col3:
                            if st.button("⚙️ Settings", key=f"settings_{drone_id}", use_container_width=True):
                                st.info("Camera settings")
        
        st.divider()
        
        # Camera configuration help
        with st.expander("📖 Camera Configuration Guide"):
            st.markdown("""
            ### How to Connect Drone IP Cameras
            
            **Supported URL Formats:**
            - HTTP/HTTPS: `http://192.168.1.100/snapshot.jpg` (Direct image)
            - HTTP Stream: `http://192.168.1.100:8080/stream.mjpg` (MJPEG stream)
            - RTSP: `rtsp://192.168.1.100:8554/stream` (Requires conversion)
            
            **Common Drone Camera URLs:**
            1. **DJI Drones**: Use DJI Mobile SDK or third-party apps
            2. **IP Webcams**: `http://phone_ip:8080/video`
            3. **ESP32-CAM**: `http://esp32_ip/cam-hi.jpg`
            4. **Raspberry Pi Camera**: Set up with motion or picamera web server
            
            **Testing Your Camera:**
            - Open the URL in a web browser first
            - Ensure the drone and computer are on the same network
            - Check firewall settings
            - Use camera manufacturer's app to verify stream
            
            **Troubleshooting:**
            - If RTSP, convert to HTTP using FFmpeg or similar tools
            - Check network connectivity between drone and computer
            - Verify camera is powered on and streaming
            - Use static IP for reliable connection
            """)
        
        # Quick camera test
        st.divider()
        st.subheader("🧪 Camera Connection Test")
        col1, col2 = st.columns([3, 1])
        with col1:
            test_url = st.text_input("Test Camera URL", placeholder="http://192.168.1.100/snapshot.jpg")
        with col2:
            st.write("")
            st.write("")
            if st.button("Test Connection", use_container_width=True):
                if test_url:
                    try:
                        import requests
                        response = requests.get(test_url, timeout=5)
                        if response.status_code == 200:
                            st.success("✅ Connection successful!")
                            st.image(test_url)
                        else:
                            st.error(f"❌ Failed: Status {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
                else:
                    st.warning("Enter a URL to test")

# SETTINGS TAB
with tab5:
    st.header("⚙️ Settings")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Flight Parameters")
        st.session_state.flight_altitude = st.slider("Altitude (m)", 5, 100, 20)
        st.session_state.mission_speed = st.slider("Speed (m/s)", 1, 20, 7)
        st.session_state.min_battery_threshold = st.slider("Min Battery (%)", 10, 40, 20)
    with c2:
        st.subheader("Safety")
        st.session_state.auto_rth_low_battery = st.checkbox("Auto RTH", True)
    
    st.divider()
    st.subheader("🔌 Drone Fleet")
    c1, c2, c3, c4 = st.columns([2,2,2,1])
    with c1:
        new_id = st.text_input("ID", "", placeholder="Drone1")
    with c2:
        new_name = st.text_input("Name", "", placeholder="Alpha")
    with c3:
        st.write(""); st.write("")
    with c4:
        st.write(""); st.write("")
        if st.button("➕", use_container_width=True):
            if new_id:
                st.session_state.drone_configs.append({'id': new_id, 'connected': False})
                if new_name:
                    st.session_state.drone_names[new_id] = new_name
                st.success(f"Added {new_id}")
                st.rerun()
    
    if st.session_state.drone_configs:
        for idx, cfg in enumerate(st.session_state.drone_configs):
            c1, c2, c3, c4 = st.columns([2,2,2,1])
            with c1:
                st.write(f"**{cfg['id']}**")
            with c2:
                st.write(st.session_state.drone_names.get(cfg['id'], ""))
            with c3:
                if cfg['connected']:
                    if st.button("Disconnect", key=f"dc_{idx}"):
                        disconnect_drone(cfg)
                        st.rerun()
                else:
                    if st.button("Connect", key=f"cn_{idx}", type="primary"):
                        connect_drone(cfg)
                        st.rerun()
            with c4:
                if st.button("🗑️", key=f"del_{idx}"):
                    st.session_state.drone_configs.pop(idx)
                    st.rerun()

# REPORTS TAB
with tab5:
    st.header("📄 Reports")
    if st.session_state.drones_data:
        if st.button("Export CSV"):
            csv_data = []
            for d_id, data in st.session_state.drones_data.items():
                for point in data["flight_history"]:
                    csv_data.append({"Drone": d_id, "Lon": point[0], "Lat": point[1], "Alt": point[2]})
            df = pd.DataFrame(csv_data)
            st.download_button("Download", df.to_csv(index=False), "mission.csv", "text/csv")
        
        st.divider()
        st.subheader("Mission Summary")
        summary = []
        for d_id, data in st.session_state.drones_data.items():
            summary.append({
                "Drone": st.session_state.drone_names.get(d_id, d_id),
                "Battery": f"{data['telemetry']['battery']:.1f}%",
                "Distance": f"{data['statistics']['total_distance']:.1f}m",
                "Objects": data['statistics']['objects_detected']
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
    else:
        st.info("No data")

# Auto-refresh
if st.session_state.mission_active:
    time.sleep(2)
    st.rerun()