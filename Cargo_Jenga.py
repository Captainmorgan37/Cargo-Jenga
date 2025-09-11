import streamlit as st
import pandas as pd
import itertools
import plotly.graph_objects as go

# ----------------- Predefined Containers -----------------
containers = {
    "CJ": {
        "door": {"width_min": 24, "width_max": 26, "height": 20, "diag": 31},
        "interior": {
            "height": 22,
            "depth": 45,   # main box depth (x)
            "width": 84,   # main box width (y)
            "restricted": {"width": 20, "depth": 20},  # near the door, right side
            # Long tunnel section: against back wall (x), runs left-to-right (y)
            "tunnel": {"depth": 24, "width": 84}
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
    "Golf Clubs (Hard Case)": {"dims": (55, 12, 12), "flex": 1.0},
    "Golf Clubs (Soft Bag)": {"dims": (55, 12, 12), "flex": 0.9},
    "Ski Bag (Soft)": {"dims": (70, 12, 7), "flex": 0.9},
    "Custom": {"dims": None, "flex": 1.0}
}

# ----------------- Helper Functions -----------------
def fits_through_door(box_dims, door):
    l, w, h = box_dims
    for dims in itertools.permutations([l, w, h]):
        bw, bh = dims[0], dims[1]
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

def fits_in_space(box_dims, space_dims):
    l, w, h = box_dims
    for dims in itertools.permutations([l, w, h]):
        bl, bw, bh = dims
        if bl <= space_dims[0] and bw <= space_dims[1] and bh <= space_dims[2]:
            return dims
    return None

def fits_inside(box_dims, interior, container_type, flex=1.0):
    l, w, h = apply_flex(box_dims, flex)
    for dims in itertools.permutations([l, w, h]):
        bl, bw, bh = dims
        if container_type == "CJ":
            if bh <= interior["height"] and bl <= interior["depth"] and bw <= interior["width"]:
                return True
        elif container_type == "Legacy":
            if bl <= interior["depth"] and bh <= interior["height"]:
                if bw <= min(
                    legacy_width_at_height(interior, 0),
                    legacy_width_at_height(interior, bh)
                ):
                    return True
    return False

# ----------------- Greedy 3D Packing -----------------
def greedy_3d_packing(baggage_list, container_type, interior):
    placements = []

    if container_type == "CJ":
        cargo_L = interior["depth"]    # x axis (front→back)
        cargo_W = interior["width"]    # y axis (left→right)
        cargo_H = interior["height"]   # z axis

        # Tunnel geometry (back wall, left side)
        t_depth = interior["tunnel"]["depth"]  # along x
        t_width = interior["tunnel"]["width"]  # along y
        t_x0 = cargo_L - t_depth               # back wall origin in x
        t_y0 = 0.0                             # left edge origin in y
        t_z0 = 0.0

        # Tunnel cursors (lay along y, then layer in z)
        t_y_cursor = 0.0
        t_z_cursor = 0.0
        t_row_height = 0.0

    else:  # Legacy
        cargo_L = interior["depth"]
        cargo_W = interior["width_min"]
        cargo_H = interior["height"]

    # Main box cursors
    x_cursor = y_cursor = z_cursor = 0.0
    row_height = 0.0
    max_y_in_row = 0.0

    for i, item in enumerate(baggage_list):
        # Always compute dims_flex first (fixes UnboundLocalError)
        dims_flex = apply_flex(item["Dims"], item.get("Flex", 1.0))
        placed = False

        # ---------- CJ tunnel for long items (force orientation) ----------
        if container_type == "CJ" and max(dims_flex) >= 50:  # heuristic for long bags
            long_dim = max(dims_flex)                        # must go along y in the tunnel
            # other two dims are thickness & height; order them to fit (x = thinner)
            others = sorted([d for d in dims_flex if d != long_dim] or [dims_flex[0], dims_flex[1]])
            x_size, z_size = others[0], others[-1]           # x <= z
            # Check remaining tunnel space
            rem_depth = t_depth
            rem_width = max(0.0, t_width - t_y_cursor)
            rem_height = max(0.0, cargo_H - t_z_cursor)
            if (x_size <= rem_depth) and (long_dim <= rem_width) and (z_size <= rem_height):
                x0 = t_x0
                y0 = t_y0 + t_y_cursor
                z0 = t_z0 + t_z_cursor
                placements.append({
                    "Item": i + 1, "Type": item["Type"], "Dims": item["Dims"],
                    "Position": (x0, y0, z0)
                })
                # advance along tunnel width (y)
                t_y_cursor += long_dim
                t_row_height = max(t_row_height, z_size)
                if t_y_cursor > t_width + 1e-6:
                    t_y_cursor = 0.0
                    t_z_cursor += t_row_height
                    t_row_height = 0.0
                placed = True

        if placed:
            continue

        # ---------- Normal greedy placement in main hold ----------
        for attempt in range(3):  # try extend row, new row, new layer
            oriented = fits_in_space(
                dims_flex,
                (max(0.0, cargo_L - x_cursor), cargo_W, max(0.0, cargo_H - z_cursor))
            )
            if oriented:
                l, w, h = oriented
                x0, y0, z0 = x_cursor, y_cursor, z_cursor
                x1, y1, z1 = x0 + l, y0 + w, z0 + h

                if container_type == "CJ":
                    # restricted block (near door, right side)
                    r = interior["restricted"]
                    rx0, rx1 = 0.0, r["depth"]
                    ry0, ry1 = cargo_W - r["width"], cargo_W
                    rz0, rz1 = 0.0, cargo_H
                    overlap = not (x1 <= rx0 or x0 >= rx1 or y1 <= ry0 or y0 >= ry1 or z1 <= rz0 or z0 >= rz1)
                    if not overlap:
                        placements.append({"Item": i+1, "Type": item["Type"], "Dims": item["Dims"],
                                           "Position": (x0, y0, z0)})
                        x_cursor += l
                        row_height = max(row_height, h)
                        max_y_in_row = max(max_y_in_row, w)
                        placed = True
                        break
                else:  # Legacy taper check
                    from_bottom = legacy_width_at_height(interior, z0)
                    to_top = legacy_width_at_height(interior, z1)
                    w_avail = min(from_bottom, to_top)
                    if y1 <= w_avail:
                        placements.append({"Item": i+1, "Type": item["Type"], "Dims": item["Dims"],
                                           "Position": (x0, y0, z0)})
                        x_cursor += l
                        row_height = max(row_height, h)
                        max_y_in_row = max(max_y_in_row, w)
                        placed = True
                        break

            if not placed:
                if attempt == 0:  # new row
                    x_cursor = 0.0
                    y_cursor += max_y_in_row
                    max_y_in_row = 0.0
                elif attempt == 1:  # new layer
                    x_cursor = y_cursor = 0.0
                    z_cursor += row_height
                    row_height = 0.0

        if not placed:
            return False, placements

    return True, placements

# ----------------- Multi-Strategy Packing -----------------
def bag_volume(dims):
    l, w, h = dims
    return l * w * h

def multi_strategy_packing(baggage_list, container_type, interior):
    strategies = {
        "Original Order": baggage_list,
        "Largest Volume First": sorted(baggage_list, key=lambda x: bag_volume(x["Dims"]), reverse=True),
        "Largest Dimension First": sorted(baggage_list, key=lambda x: max(x["Dims"]), reverse=True),
        "Smallest First": sorted(baggage_list, key=lambda x: bag_volume(x["Dims"]))
    }

    best_result = {"success": False, "placements": [], "strategy": None, "fit_count": 0}

    for name, bags in strategies.items():
        success, placements = greedy_3d_packing(bags, container_type, interior)
        if success:
            return {"success": True, "placements": placements, "strategy": name, "fit_count": len(placements)}
        else:
            if len(placements) > best_result["fit_count"]:
                best_result = {"success": False, "placements": placements, "strategy": name, "fit_count": len(placements)}

    return best_result

def cargo_volume(interior, container_type):
    if container_type == "CJ":
        main_vol = interior["depth"] * interior["width"] * interior["height"]
        tunnel_vol = interior["tunnel"]["depth"] * interior["tunnel"]["width"] * interior["height"]
        return main_vol + tunnel_vol
    else:
        return interior["depth"] * ((interior["width_min"] + interior["width_max"]) / 2) * interior["height"]

# ----------------- Visualization -----------------
def plot_cargo(cargo_dims, placements, container_type=None, interior=None):
    cargo_L, cargo_W, cargo_H = cargo_dims
    fig = go.Figure()

    if container_type == "CJ":
        # Main hold wireframe (no legend spam)
        corners = [
            (0,0,0),(cargo_L,0,0),(cargo_L,cargo_W,0),(0,cargo_W,0),
            (0,0,cargo_H),(cargo_L,0,cargo_H),(cargo_L,cargo_W,cargo_H),(0,cargo_W,cargo_H)
        ]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            x = [corners[e[0]][0], corners[e[1]][0]]
            y = [corners[e[0]][1], corners[e[1]][1]]
            z = [corners[e[0]][2], corners[e[1]][2]]
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                                       line=dict(color='black', width=3),
                                       showlegend=False))

        # Restricted block (near door, right side)
        r = interior["restricted"]
        x0, y0, z0 = 0, cargo_W - r["width"], 0
        x1, y1, z1 = r["depth"], cargo_W, cargo_H
        vertices = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
        ]
        x, y, z = zip(*vertices)
        faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),
                 (0,1,5),(0,5,4),(1,2,6),(1,6,5),
                 (2,3,7),(2,7,6),(3,0,4),(3,4,7)]
        i, j, k = zip(*faces)
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                                color='gray', opacity=0.35,
                                name='Restricted Area'))

        # Long tunnel (back wall, left side)
        t_depth = interior["tunnel"]["depth"]
        t_width = interior["tunnel"]["width"]
        t_x0 = cargo_L - t_depth
        t_y0 = 0
        vertices = [
            [t_x0,           t_y0,          0],
            [t_x0+t_depth,   t_y0,          0],
            [t_x0+t_depth,   t_y0+t_width,  0],
            [t_x0,           t_y0+t_width,  0],
            [t_x0,           t_y0,          cargo_H],
            [t_x0+t_depth,   t_y0,          cargo_H],
            [t_x0+t_depth,   t_y0+t_width,  cargo_H],
            [t_x0,           t_y0+t_width,  cargo_H],
        ]
        x, y, z = zip(*vertices)
        faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),
                 (0,1,5),(0,5,4),(1,2,6),(1,6,5),
                 (2,3,7),(2,7,6),(3,0,4),(3,4,7)]
        i, j, k = zip(*faces)
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                                color='lightblue', opacity=0.2,
                                name='Long Tunnel'))


    if container_type == "Legacy":
        d = interior["depth"]
        wmin, wmax = interior["width_min"], interior["width_max"]
        h = interior["height"]
        vertices = [
            [0,0,0],[d,0,0],[d,wmin,0],[0,wmin,0],
            [0,0,h],[d,0,h],[d,wmax,h],[0,wmax,h]
        ]
        x, y, z = zip(*vertices)
        faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),
                 (0,1,5),(0,5,4),(1,2,6),(1,6,5),
                 (2,3,7),(2,7,6),(3,0,4),(3,4,7)]
        i, j, k = zip(*faces)
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                                color='lightblue', opacity=0.15,
                                name='Legacy Hold'))

    # Add baggage
    colors = ['red','green','blue','orange','purple','cyan','magenta']
    for idx, item in enumerate(placements):
        l, w, h = item["Dims"]
        x0, y0, z0 = item["Position"]
        color = colors[idx % len(colors)]
        x = [x0, x0+l, x0+l, x0, x0, x0+l, x0+l, x0]
        y = [y0, y0, y0+w, y0+w, y0, y0, y0+w, y0+w]
        z = [z0, z0, z0, z0, z0+h, z0+h, z0+h, z0+h]
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color=color, opacity=0.5, name=item["Type"]))

    fig.update_layout(
        scene=dict(
            xaxis_title='Depth (in)', yaxis_title='Width (in)', zaxis_title='Height (in)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    return fig

# ----------------- Streamlit UI -----------------
st.title("Aircraft Cargo Fit Checker")

container_choice = st.selectbox("Select Aircraft Cargo Hold", list(containers.keys()))
container = containers[container_choice]

if "baggage_list" not in st.session_state:
    st.session_state["baggage_list"] = []

if st.button("Clear Baggage List"):
    st.session_state["baggage_list"] = []
    st.success("✅ Baggage list cleared.")

st.write("### Add Baggage")
baggage_type = st.selectbox("Baggage Type", list(standard_baggage.keys()))

if baggage_type == "Custom":
    length = st.number_input("Length (in)", min_value=1)
    width = st.number_input("Width (in)", min_value=1)
    height = st.number_input("Height (in)", min_value=1)
    dims = (length, width, height)
    flex = 1.0
else:
    dims = standard_baggage[baggage_type]["dims"]
    flex = standard_baggage[baggage_type]["flex"]

qty = st.number_input("Quantity", min_value=1, value=1)

if st.button("Add Item"):
    if dims is None:
        st.warning("Please enter dimensions for custom item.")
    else:
        for _ in range(qty):
            st.session_state["baggage_list"].append({
                "Type": baggage_type,
                "Dims": dims,
                "Flex": flex
            })
        st.success(f"Added {qty} × {baggage_type}")

if st.session_state["baggage_list"]:
    df = pd.DataFrame(st.session_state["baggage_list"]).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Item"
    st.write("### Current Baggage Load")
    st.table(df)

    if st.button("Check Fit"):
        results = []
        for i, item in enumerate(st.session_state["baggage_list"], 1):
            box_dims = item["Dims"]
            door_fit = fits_through_door(box_dims, container["door"])
            interior_fit = fits_inside(box_dims, container["interior"], container_choice, item.get("Flex", 1.0))
            status = "✅ Fits" if door_fit and interior_fit else "❌ Door Fail" if not door_fit else "❌ Interior Fail"
            results.append({"Type": item["Type"], "Dims": box_dims, "Result": status})

        results_df = pd.DataFrame(results).reset_index(drop=True)
        results_df.index = results_df.index + 1
        results_df.index.name = "Item"
        st.write("### Fit Results")
        st.table(results_df)

        result = multi_strategy_packing(
            st.session_state["baggage_list"], container_choice, container["interior"]
        )

        st.write("### Overall Cargo Packing Feasibility")
        if result["success"]:
            st.success(f"✅ Packing possible using **{result['strategy']}** strategy.")
        else:
            st.warning(
                f"⚠️ Full packing failed. Best strategy was **{result['strategy']}**, "
                f"which fit {result['fit_count']} out of {len(st.session_state['baggage_list'])} items."
            )

        placements = result["placements"]

        if placements:
            placements_df = pd.DataFrame(placements).reset_index(drop=True)
            placements_df.index = placements_df.index + 1
            placements_df.index.name = "Item"
            st.write("### Suggested Placement Positions")
            st.table(placements_df)

            # Volume utilization
            def bag_volume(d): return d[0]*d[1]*d[2]
            total_bag_vol = sum(bag_volume(item["Dims"]) for item in st.session_state["baggage_list"])
            hold_vol = cargo_volume(container["interior"], container_choice)
            utilization = (total_bag_vol / hold_vol) * 100
            st.info(f"📦 Estimated Volume Utilization: {utilization:.1f}%")

            st.write("### Cargo Load Visualization")
            if container_choice == "CJ":
                cargo_dims = (container["interior"]["depth"],
                              container["interior"]["width"],
                              container["interior"]["height"])
            else:
                cargo_dims = (container["interior"]["depth"],
                              container["interior"]["width_max"],
                              container["interior"]["height"])
            fig = plot_cargo(cargo_dims, placements, container_choice, container["interior"])
            st.plotly_chart(fig, use_container_width=True)


