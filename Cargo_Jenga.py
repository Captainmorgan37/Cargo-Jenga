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
            "depth_min": 36,
            "depth_max": 54,
            "width_min": 36,
            "width_max": 89
        }
    }
}

# ----------------- Standard Baggage Presets -----------------
standard_baggage = {
    "Small Carry-on": (22, 14, 9),
    "Medium Suitcase": (24, 16, 10),
    "Large Suitcase": (28, 20, 12),
    "Golf Clubs": (50, 15, 12),
    "Ski Bag": (72, 10, 7),
    "Custom": None
}

# ----------------- Helper Functions -----------------
def fits_through_door(box_dims, door):
    l, w, h = box_dims
    for dims in itertools.permutations([l, w, h]):
        bw, bh = dims[0], dims[1]  # orientation
        diag = (bw**2 + bh**2) ** 0.5
        if "width_min" in door:  # CJ
            if (bw <= door["width_max"] and bh <= door["height"]) or diag <= door["diag"]:
                return True
        else:  # Legacy
            if (bw <= door["width"] and bh <= door["height"]) or diag <= door["diag"]:
                return True
    return False

def fits_inside(box_dims, interior, container_type):
    l, w, h = box_dims
    for dims in itertools.permutations([l, w, h]):
        bl, bw, bh = dims
        if container_type == "CJ":
            if bh <= interior["height"] and bl <= interior["depth"] and bw <= interior["width"]:
                return True
        elif container_type == "Legacy":
            if bh <= interior["height"] and bl <= interior["depth_max"] and bw <= interior["width_max"]:
                return True
    return False

def cargo_volume(container_type, interior):
    if container_type == "CJ":
        total = interior["height"] * interior["depth"] * interior["width"]
        restricted = interior["height"] * interior["restricted"]["depth"] * interior["restricted"]["width"]
        return total - restricted
    elif container_type == "Legacy":
        front = interior["height"] * interior["depth_min"] * interior["width_max"]
        back = interior["height"] * (interior["depth_max"] - interior["depth_min"]) * interior["width_min"]
        return front + back

def box_volume(dims):
    l, w, h = dims
    return l * w * h

def check_total_fit(baggage_list, container, container_type):
    total_baggage = sum(box_volume(item["Dims"]) for item in baggage_list)
    cargo_cap = cargo_volume(container_type, container["interior"])
    return total_baggage, cargo_cap, total_baggage <= cargo_cap

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
        cargo_L = interior["depth_max"]
        cargo_W = interior["width_max"]
        cargo_H = interior["height"]

    placements = []
    x_cursor = y_cursor = z_cursor = 0
    row_height = 0
    max_y_in_row = 0

    for i, item in enumerate(baggage_list):
        box = item["Dims"]
        oriented = fits_in_space(box, (cargo_L - x_cursor, cargo_W - y_cursor, cargo_H - z_cursor))
        if not oriented:
            # Next row Y
            x_cursor = 0
            y_cursor += max_y_in_row
            max_y_in_row = 0
            oriented = fits_in_space(box, (cargo_L - x_cursor, cargo_W - y_cursor, cargo_H - z_cursor))
            if not oriented:
                # Next layer Z
                x_cursor = y_cursor = 0
                z_cursor += row_height
                row_height = 0
                oriented = fits_in_space(box, (cargo_L - x_cursor, cargo_W - y_cursor, cargo_H - z_cursor))
                if not oriented:
                    return False, placements

        l, w, h = oriented
        placements.append({"Item": i+1, "Type": item["Type"], "Dims": (l, w, h),
                           "Position": (x_cursor, y_cursor, z_cursor)})
        x_cursor += l
        row_height = max(row_height, h)
        max_y_in_row = max(max_y_in_row, w)

    return True, placements

# ----------------- Visualization -----------------
def plot_cargo(cargo_dims, placements, container_type=None, interior=None):
    cargo_L, cargo_W, cargo_H = cargo_dims
    fig = go.Figure()

    # Cargo hold edges
    fig.add_trace(go.Scatter3d(
        x=[0, cargo_L, cargo_L, 0, 0, 0, cargo_L, cargo_L],
        y=[0, 0, cargo_W, cargo_W, 0, 0, cargo_W, cargo_W],
        z=[0, 0, 0, 0, cargo_H, cargo_H, cargo_H, cargo_H],
        mode='lines',
        line=dict(color='black', width=4),
        name='Cargo Hold'
    ))

    # CJ restricted area
    if container_type == "CJ" and interior is not None:
        r = interior["restricted"]
        x = [0, r["depth"], r["depth"], 0, 0, r["depth"], r["depth"], 0]
        y = [0, 0, r["width"], r["width"], 0, 0, r["width"], r["width"]]
        z = [0,0,0,0, interior["height"], interior["height"], interior["height"], interior["height"]]
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            color='gray',
            opacity=0.4,
            name='Restricted Area'
        ))

        # Door outline at the front wall
        fig.add_trace(go.Scatter3d(
            x=[0, 0, 0, 0, 0],
            y=[0, 24, 26, 0, 24],
            z=[0, 0, 20, 20, 0],
            mode="lines",
            line=dict(color="red", width=6),
            name="Door Opening"
        ))

    # Add baggage
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
    for idx, item in enumerate(placements):
        l, w, h = item["Dims"]
        x0, y0, z0 = item["Position"]
        color = colors[idx % len(colors)]
        x = [x0, x0+l, x0+l, x0, x0, x0+l, x0+l, x0]
        y = [y0, y0, y0+w, y0+w, y0, y0, y0+w, y0+w]
        z = [z0, z0, z0, z0, z0+h, z0+h, z0+h, z0+h]
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            color=color,
            opacity=0.5,
            name=item["Type"]
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Depth (in)',
            yaxis_title='Width (in)',
            zaxis_title='Height (in)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    return fig

# ----------------- Streamlit UI -----------------
st.title("Aircraft Cargo Fit Checker")

container_choice = st.selectbox("Select Aircraft Cargo Hold", list(containers.keys()))
container = containers[container_choice]

# Initialize baggage list
if "baggage_list" not in st.session_state:
    st.session_state["baggage_list"] = []

# Clear baggage list button
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
else:
    dims = standard_baggage[baggage_type]

qty = st.number_input("Quantity", min_value=1, value=1)

if st.button("Add Item"):
    if dims is None:
        st.warning("Please enter dimensions for custom item.")
    else:
        for _ in range(qty):
            st.session_state["baggage_list"].append({"Type": baggage_type, "Dims": dims})
        st.success(f"Added {qty} × {baggage_type}")

# Show current baggage list
if st.session_state["baggage_list"]:
    df = pd.DataFrame(st.session_state["baggage_list"])
    st.write("### Current Baggage Load")
    st.table(df)

    if st.button("Check Fit"):
        results = []
        for i, item in enumerate(st.session_state["baggage_list"], 1):
            box_dims = item["Dims"]
            door_fit = fits_through_door(box_dims, container["door"])
            interior_fit = fits_inside(box_dims, container["interior"], container_choice)
            if door_fit and interior_fit:
                status = "✅ Fits"
            elif not door_fit:
                status = "❌ Door Fail"
            else:
                status = "❌ Interior Fail"
            results.append({"Item": i, "Type": item["Type"], "Dims": box_dims, "Result": status})

        st.write("### Fit Results")
        st.table(pd.DataFrame(results))

        # Overall volume check
        total_baggage, cargo_cap, feasible = check_total_fit(
            st.session_state["baggage_list"], container, container_choice
        )
        st.write("### Overall Cargo Volume Check")
        st.write(f"Total Baggage Volume: {total_baggage:,} in³")
        st.write(f"Cargo Capacity: {cargo_cap:,} in³")
        if feasible:
            st.success("✅ Total baggage volume can fit inside the cargo hold.")
        else:
            st.error("❌ Total baggage volume exceeds cargo hold capacity.")

        # Greedy 3D placement
        success, placements = greedy_3d_packing(st.session_state["baggage_list"], container_choice, container["interior"])
        st.write("### Overall Cargo Packing Feasibility")
        if success:
            st.success("✅ All baggage items can be placed in the cargo hold.")
        else:
            st.error("❌ Some items cannot fit together in the cargo hold.")

        if placements:
            st.write("### Suggested Placement Positions")
            df_place = pd.DataFrame(placements)
            st.table(df_place)

            st.write("### Cargo Load Visualization")
            if container_choice == "CJ":
                cargo_dims = (container["interior"]["depth"], container["interior"]["width"], container["interior"]["height"])
            else:
                cargo_dims = (container["interior"]["depth_max"], container["interior"]["width_max"], container["interior"]["height"])
            fig = plot_cargo(cargo_dims, placements, container_choice, container["interior"])
            st.plotly_chart(fig, use_container_width=True)
