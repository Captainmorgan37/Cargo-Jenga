import streamlit as st
import pandas as pd
import itertools
import plotly.graph_objects as go
import math

# ----------------- Predefined Containers -----------------
containers = {
    "CJ": {
        "door": {"width_min": 24, "width_max": 26, "height": 20, "diag": 31},
        "interior": {
            "height": 22,
            "depth": 45,
            "width": 84,
            "restricted": {"width": 20, "depth": 20}
        }
    },
    "Legacy": {
        "door": {"width": 34, "height": 22, "diag": 38},
        "interior": {
            "height": 36,
            "depth": 89,
            "width_min": 36,
            "width_max": 54
        }
    }
}

# ----------------- Standard Baggage Presets -----------------
standard_baggage = {
    "Small Carry-on": {"dims": (22, 14, 9), "flex": 1.0},
    "Standard Suitcase": {"dims": (26, 18, 10), "flex": 1.0},
    "Large Suitcase": {"dims": (30, 19, 11), "flex": 1.0},
    "Golf Clubs (Hard Case)": {"dims": (50, 14, 14), "flex": 1.0},
    "Golf Clubs (Soft Bag)": {"dims": (50, 14, 14), "flex": 0.9},
    "Ski Bag (Soft)": {"dims": (70, 12, 7), "flex": 0.9},
    "Custom": {"dims": None, "flex": 1.0}
}

# ----------------- Helpers -----------------
def fits_through_door(box_dims, door):
    l, w, h = box_dims
    for bw, bh, _ in itertools.permutations([l, w, h]):
        diag = (bw**2 + bh**2) ** 0.5
        if "width_min" in door:  # CJ
            if (bw <= door["width_max"] and bh <= door["height"]) or diag <= door["diag"]:
                return True
        else:  # Legacy
            if (bw <= door["width"] and bh <= door["height"]) or diag <= door["diag"]:
                return True
    return False

def legacy_width_at_height(interior, z):
    h = interior["height"]
    wmin, wmax = interior["width_min"], interior["width_max"]
    return wmin + (wmax - wmin) * (z / h)

def apply_flex(dims, flex):
    l, w, h = dims
    return (l * flex, w * flex, h * flex)

def aabb_overlap(a, b):
    ax0, ay0, az0, ax1, ay1, az1 = a
    bx0, by0, bz0, bx1, by1, bz1 = b
    return not (ax1 <= bx0 or ax0 >= bx1 or
                ay1 <= by0 or ay0 >= by1 or
                az1 <= bz0 or az0 >= bz1)

def point_in_cj_restricted(x, y, interior, cargo_W):
    r = interior["restricted"]
    rx0, rx1 = 0, r["depth"]
    ry0, ry1 = cargo_W - r["width"], cargo_W
    return (rx0 <= x <= rx1) and (ry0 <= y <= ry1)

def dedupe_points(points):
    seen = set()
    out = []
    for p in points:
        key = (round(p[0], 4), round(p[1], 4), round(p[2], 4))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out

def prune_dominated_points(points):
    pruned = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j: continue
            if q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2] and (q[0] < p[0] or q[1] < p[1] or q[2] < p[2]):
                dominated = True
                break
        if not dominated:
            pruned.append(p)
    return pruned

def get_orientations(dims):
    return set(itertools.permutations(dims, 3))

# ----------------- Extreme-Points Packer with Diagonal -----------------
def extreme_points_packing(baggage_list, container_type, interior):
    if container_type == "CJ":
        L, W, H = interior["depth"], interior["width"], interior["height"]
    else:
        L, W, H = interior["depth"], interior["width_max"], interior["height"]
    cargo_dims = (L, W, H)

    points = [(0.0, 0.0, 0.0)]
    placed_boxes = []
    placements = []

    for item in baggage_list:
        dims_real = item["Dims"]
        dims_flex = apply_flex(dims_real, item.get("Flex", 1.0))
        max_dim = max(dims_flex)
        min_dim = min(dims_flex)
        is_long = (max_dim / min_dim > 3)

        best = None
        for (px, py, pz) in sorted(points, key=lambda p: (p[2], p[1], p[0])):
            if container_type == "CJ" and point_in_cj_restricted(px, py, interior, W):
                continue
            if container_type == "Legacy" and py > legacy_width_at_height(interior, pz):
                continue

            for (l, w, h) in get_orientations(dims_flex):
                x0, y0, z0 = px, py, pz
                x1, y1, z1 = x0+l, y0+w, z0+h
                if x1 > L or z1 > H: continue
                if container_type == "CJ" and y1 > W: continue
                if container_type == "Legacy":
                    w_avail = min(legacy_width_at_height(interior, z0), legacy_width_at_height(interior, z1))
                    if y1 > w_avail: continue
                candidate = (x0,y0,z0,x1,y1,z1)
                if any(aabb_overlap(candidate,b) for b in placed_boxes):
                    continue
                score = (z1,y1,x1)
                if (best is None) or (score < best[0]):
                    best = (score,(x0,y0,z0,l,w,h))

        # try diagonal in CJ if no axis-aligned works
        if best is None and container_type == "CJ" and is_long:
            l,w,h = dims_real
            diag = math.sqrt(L**2 + W**2)
            if l <= diag and h <= H:
                placements.append({"Item": len(placements)+1,"Type": item["Type"],"Dims": dims_real,
                                   "Position": (0,0,0),"Diagonal": True})
                continue
            else:
                return False, placements

        if best is None:
            return False, placements

        _, (x0,y0,z0,l,w,h) = best
        placed_boxes.append((x0,y0,z0,x0+l,y0+w,z0+h))
        placements.append({"Item": len(placements)+1,"Type": item["Type"],"Dims": dims_real,"Position": (x0,y0,z0)})

        new_pts = [(x0+l,y0,z0),(x0,y0+w,z0),(x0,y0,z0+h)]
        filt = []
        for (nx,ny,nz) in new_pts:
            if nx > L or nz > H: continue
            if container_type == "CJ" and ny > W: continue
            if container_type == "Legacy" and ny > legacy_width_at_height(interior,nz): continue
            inside_existing = False
            for bx in placed_boxes:
                if bx[0] <= nx <= bx[3] and bx[1] <= ny <= bx[4] and bx[2] <= nz <= bx[5]:
                    inside_existing=True
                    break
            if not inside_existing:
                filt.append((nx,ny,nz))
        points.extend(filt)
        points = dedupe_points(points)
        points = prune_dominated_points(points)

    return True, placements

# ----------------- Visualization -----------------
def plot_cargo(cargo_dims, placements, container_type=None, interior=None):
    cargo_L,cargo_W,cargo_H = cargo_dims
    fig = go.Figure()
    if container_type=="CJ":
        corners=[(0,0,0),(cargo_L,0,0),(cargo_L,cargo_W,0),(0,cargo_W,0),
                 (0,0,cargo_H),(cargo_L,0,cargo_H),(cargo_L,cargo_W,cargo_H),(0,cargo_W,cargo_H)]
        edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            x=[corners[e[0]][0],corners[e[1]][0]]
            y=[corners[e[0]][1],corners[e[1]][1]]
            z=[corners[e[0]][2],corners[e[1]][2]]
            fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',line=dict(color='black',width=4),name='Cargo Hold'))
    colors=['red','green','blue','orange','purple','cyan','magenta']
    for idx,item in enumerate(placements):
        l,w,h=item["Dims"]
        x0,y0,z0=item["Position"]
        color=colors[idx%len(colors)]
        if "Diagonal" in item:
            fig.add_trace(go.Scatter3d(
                x=[0,cargo_L],y=[0,cargo_W],z=[0,0],
                mode='lines',line=dict(color=color,width=8),name=item["Type"]+" (Diagonal)"
            ))
        else:
            x=[x0,x0+l,x0+l,x0,x0,x0+l,x0+l,x0]
            y=[y0,y0,y0+w,y0+w,y0,y0,y0+w,y0+w]
            z=[z0,z0,z0,z0,z0+h,z0+h,z0+h,z0+h]
            fig.add_trace(go.Mesh3d(x=x,y=y,z=z,color=color,opacity=0.5,name=item["Type"]))
    fig.update_layout(scene=dict(xaxis_title='Depth (in)',yaxis_title='Width (in)',zaxis_title='Height (in)',aspectmode='data'),
                      margin=dict(l=0,r=0,b=0,t=0))
    return fig

# ----------------- Streamlit UI -----------------
st.title("Aircraft Cargo Fit Checker — Extreme-Points + Diagonal")

container_choice = st.selectbox("Select Aircraft Cargo Hold", list(containers.keys()))
container=containers[container_choice]
if "baggage_list" not in st.session_state: st.session_state["baggage_list"]=[]
if st.button("Clear Baggage List"): st.session_state["baggage_list"]=[]; st.success("✅ Baggage list cleared.")
st.write("### Add Baggage")
baggage_type=st.selectbox("Baggage Type",list(standard_baggage.keys()))
if baggage_type=="Custom":
    length=st.number_input("Length (in)",min_value=1)
    width=st.number_input("Width (in)",min_value=1)
    height=st.number_input("Height (in)",min_value=1)
    dims=(length,width,height); flex=1.0
else: dims=standard_baggage[baggage_type]["dims"]; flex=standard_baggage[baggage_type]["flex"]
qty=st.number_input("Quantity",min_value=1,value=1)
if st.button("Add Item"):
    if dims is None: st.warning("Please enter dimensions for custom item.")
    else:
        for _ in range(qty): st.session_state["baggage_list"].append({"Type":baggage_type,"Dims":dims,"Flex":flex})
        st.success(f"Added {qty} × {baggage_type}")
if st.session_state["baggage_list"]:
    df=pd.DataFrame(st.session_state["baggage_list"]).reset_index(drop=True)
    df.index=df.index+1; df.index.name="Item"
    st.write("### Current Baggage Load"); st.table(df)
    if st.button("Check Fit"):
        results=[]
        for item in st.session_state["baggage_list"]:
            box_dims=item["Dims"]
            door_fit=fits_through_door(box_dims,container["door"])
            interior_fit=True
            results.append({"Type":item["Type"],"Dims":box_dims,
                            "Result":"✅ Fits" if door_fit and interior_fit else "❌ Fail"})
        results_df=pd.DataFrame(results).reset_index(drop=True); results_df.index=results_df.index+1; results_df.index.name="Item"
        st.write("### Fit Results"); st.table(results_df)
        success,placements=extreme_points_packing(st.session_state["baggage_list"],container_choice,container["interior"])
        st.write("### Overall Cargo Packing Feasibility")
        if success: st.success("✅ Packing possible.")
        else: st.error("❌ Packing failed.")
        if placements:
            placements_df=pd.DataFrame(placements).reset_index(drop=True)
            placements_df.index=placements_df.index+1; placements_df.index.name="Item"
            st.write("### Suggested Placement Positions"); st.table(placements_df)
            if container_choice=="CJ": cargo_dims=(container["interior"]["depth"],container["interior"]["width"],container["interior"]["height"])
            else: cargo_dims=(container["interior"]["depth"],container["interior"]["width_max"],container["interior"]["height"])
            st.write("### Cargo Load Visualization"); fig=plot_cargo(cargo_dims,placements,container_choice,container["interior"])
            st.plotly_chart(fig,use_container_width=True)
