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

# ----------------- Standard Baggage Presets (with Flexibility) -----------------
standard_baggage = {
    "Small Carry-on": {"dims": (22, 14, 9), "flex": 1.0},
    "Standard Suitcase": {"dims": (26, 18, 10), "flex": 1.0},
    "Large Suitcase": {"dims": (30, 19, 11), "flex": 1.0},
    "Golf Clubs (Hard Case)": {"dims": (50, 14, 14), "flex": 1.0},
    "Golf Clubs (Soft Bag)": {"dims": (50, 14, 14), "flex": 0.95},  # squishable
    "Ski Bag (Soft)": {"dims": (70, 12, 7), "flex": 0.95},
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
    """Linearly interpolate available width between bottom and top at height z."""
    h = interior["height"]
    wmin, wmax = interior["width_min"], interior["width_max"]
    return wmin + (wmax - wmin) * (z / h)

def apply_flex(dims, flex):
    """Apply flexibility factor to squish bag dimensions (used for fit checks)."""
    l, w, h = dims
    return (l * flex, w * flex, h * flex)

def fits_inside(box_dims, interior, container_type, flex=1.0):
    l, w, h = apply_flex(box_dims, flex)
    for dims in itertools.permutations([l, w, h]):
        bl, bw, bh = dims
        if container_type == "CJ":
            if bh <= interior["height"] and bl <= interior["depth"] and bw <= interior["width"]:
                return True
        elif container_type == "Legacy":
            if bl <= interior["depth"] and bh <= interior["height"]:
                # check against taper at bottom and top
                if bw <= min(
                    legacy_width_at_height(interior, 0),
                    legacy_width_at_height(interior, bh)
                ):
                    return True
    return False

# ----------------- Greedy 3D Packing -----------------
def fits_in_space(box_dims, space_dims):
    l, w, h = box_dims
    for dims in itertools.permutations([l, w, h]):
        bl, bw, bh = dims
        if bl <= space_dims[0] and bw <= space_dims[1] and bh <= space_dims[2]:
            return dims
    return None

def greedy_3d_packing(baggage_list, container_type, interior):
    if container_type == "CJ":
        cargo_L = interior["depth"]
        cargo_W = interior["width"]
        cargo_H = interior["height"]
    else:  # Legacy
        cargo_L = interior["depth"]
        cargo_W = interior["width_min"]  # baseline: narrowest width
        cargo_H = interior["height"]

    placements = []
    x_cursor = y_cursor = z_cursor = 0
    row_height = 0
    max_y_in_row = 0

    for i, item in enumerate(baggage_list):
        # Apply flex factor
        dims_flex = apply_flex(item["Dims"], item.get("Flex", 1.0))
        placed = False

        for attempt in range(3):  # try row, new row, new layer
            oriented = fits_in_space(dims_flex, (cargo_L - x_cursor, cargo_W, cargo_H - z_cursor))
            if oriented:
                l, w, h = oriented
                x0, y0, z0 = x_cursor, y_cursor, z_cursor
                x1, y1, z1 = x0 + l, y0 + w, z0 + h

                if container_type == "CJ":
                    # restricted block check
                    r = interior["restricted"]
                    rx0, rx1 = 0, r["depth"]
                    ry0, ry1 = cargo_W - r["width"], cargo_W
                    rz0, rz1 = 0, interior["height"]

                    overlap = not (x1 <= rx0 or x0 >= rx1 or
                                   y1 <= ry0 or y0 >= ry1 or
                                   z1 <= rz0 or z0 >= rz1)
                    if overlap:
                        oriented = None
                    else:
                        placements.append({"Item": i+1, "Type": item["Type"],
                                           "Dims": item["Dims"],  # show true size
                                           "Position": (x0, y0, z0)})
                        x_cursor += l
                        row_height = max(row_height, h)
                        max_y_in_row = max(max_y_in_row, w)
                        placed = True
                        break
                else:  # Legacy taper check
                    w_avail_bottom = legacy_width_at_height(interior, z0)
                    w_avail_top = legacy_width_at_height(interior, z1)
                    w_avail = min(w_avail_bottom, w_avail_top)

                    if y1 <= w_avail:  # ensure bag stays inside taper
                        placements.append({"Item": i+1, "Type": item["Type"],
                                           "Dims": item["Dims"],  # show true size
                                           "Position": (x0, y0, z0)})
                        x_cursor += l
                        row_height = max(row_height, h)
                        max_y_in_row = max(max_y_in_row, w)
                        placed = True
                        break
                    else:
                        oriented = None

            if not placed:
                if attempt == 0:
                    x_cursor = 0
                    y_cursor += max_y_in_row
                    max_y_in_row = 0
                elif attempt == 1:
                    x_cursor = y_cursor = 0
                    z_cursor += row_height
                    row_height = 0

        if not placed:
            return False, placements

    return True, placements

# ----------------- Visualization -----------------
def plot_cargo(cargo_dims, placements, container_type=None, interior=None):
    cargo_L, cargo_W, cargo_H = cargo_dims
    fig = go.Figure()

    # Cargo hold for CJ
    if container_type == "CJ":
        corners = [
            (0,0,0), (cargo_L,0,0), (cargo_L,cargo_W,0), (0,cargo_W,0),
            (0,0,cargo_H), (cargo_L,0,cargo_H), (cargo_L,cargo_W,cargo_H), (0,cargo_W,cargo_H)
        ]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            x = [corners[e[0]][0], corners[e[1]][0]]
            y = [corners[e[0]][1], corners[e[1]][1]]
            z = [corners[e[0]][2], corners[e[1]][2]]
            fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',
                                       line=dict(color='black',width=4),
                                       name='Cargo Hold'))

        # Restricted block
        r = interior["restricted"]
        x0, y0, z0 = 0, cargo_W - r["width"], 0
        x1, y1, z1 = r["depth"], cargo_W, interior["height"]

        vertices = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
        ]
        x, y, z = zip(*vertices)
        faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
                 (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
        i, j, k = zip(*faces)
        fig.add_trace(go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k,
                                color='gray', opacity=0.4,
                                name='Restricted Area'))

    # Cargo hold for Legacy (rectangular prism with taper)
    if container_type == "Legacy":
        d = interior["depth"]
        wmin, wmax = interior["width_min"], interior["width_max"]
        h = interior["height"]

        vertices = [
            [0, 0, 0], [d, 0, 0], [d, wmin, 0], [0, wmin, 0],
            [0, 0, h], [d, 0, h], [d, wmax, h], [0, wmax, h]
        ]
        x, y, z = zip(*vertices)
        faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),
                 (0,1,5),(0,5,4),(1,2,6),(1,6,5),
                 (2,3,7),(2,7,6),(3,0,4),(3,4,7)]
        i, j, k = zip(*faces)
        fig.add_trace(go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k,
                                color='lightblue', opacity=0.15,
                                name='Legacy Cargo Hold'))

    # Add baggage
    colors = ['red','green','blue','orange','purple','cyan','magenta']
    for idx, item in enumerate(placements):
        l, w, h = item["Dims"]  # show real size, not flexed size
        x0, y0, z0 = item["Position"]
        color = colors[idx % len(colors)]
        x = [x0, x0+l, x0+l, x0, x0, x0+l, x0+l, x0]
        y = [y0, y0, y0+w, y0+w, y0, y0, y0+w, y0+w]
        z = [z0, z0, z0, z0, z0+h, z0+h, z0+h, z0+h]
        fig.add_trace(go.Mesh3d(x=x,y=y,z=z,color=color,opacity=0.5,name=item["Type"]))

    fig.update_layout(
        scene=dict(
            xaxis_title='Depth (in)', yaxis_title='Width (in)', zaxis_title='Height (in)',
            aspectmode='data'
        ),
        margin=dict(l=0,r=0,b=0,t=0)
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
    df = pd.DataFrame(st.session_state["baggage_list"])
    st.write("### Current Baggage Load")
    st.table(df.reset_index(drop=True))  # 🔹 drop index

    if st.button("Check Fit"):
        results = []
        for i, item in enumerate(st.session_state["baggage_list"], 1):
            box_dims = item["Dims"]
            door_fit = fits_through_door(box_dims, container["door"])
            interior_fit = fits_inside(box_dims, container["interior"], container_choice, item.get("Flex", 1.0))
            status = "✅ Fits" if door_fit and interior_fit else "❌ Door Fail" if not door_fit else "❌ Interior Fail"
            results.append({"Item": i, "Type": item["Type"], "Dims": box_dims, "Result": status})

        st.write("### Fit Results")
        st.table(pd.DataFrame(results).reset_index(drop=True))  # 🔹 drop index

        success, placements = greedy_3d_packing(
            st.session_state["baggage_list"], container_choice, container["interior"]
        )
        st.write("### Overall Cargo Packing Feasibility")
        if success:
            st.success("✅ Packing possible.")
        else:
            st.error("❌ Packing failed.")

        if placements:
            st.write("### Suggested Placement Positions")
            st.table(pd.DataFrame(placements).reset_index(drop=True))  # 🔹 drop index

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

